#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python src/eval/eval_full_valid_fdr.py \
  --model-dir /root/aipc/models/lgbm_v1 \
  --manifest /root/autodl-tmp/datasets/aipc/eval_folds/fold_manifest.csv \
  --fold 0 \
  --out-dir /root/aipc/eval/lgbm_v1_fold0_full_valid

对完整 valid 文件进行全量预测，并用尽量贴近官方 target-decoy 逻辑的指标评估模型。

主调参目标是 unique peptide @ 1% FDR；AUC/AP 只作为诊断信息。
输出内容包括：PSM@1% FDR、unique peptide@1% FDR、按文件/按仪器拆分、
FDP proxy 风险、top1 target rate。
"""

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import os
import hashlib
from typing import Dict, Iterable, List, Optional, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score


INSTRUMENTS = ["mzml", "tims", "wiff"]

ID_COLUMNS = [
    "file_id",
    "instrument",
    "scan_number",
    "group_key",
    "peptide_key",
    "precursor_sequence",
]

CATEGORICAL_FEATURES = {
    "instrument_id",
    "charge",
    "has_ion_mobility",
    "has_mod",
    "parse_ok",
    "fragment_parse_ok",
    "best_isotope_offset",
    "is_lgbm_v1_group_top1",
    "is_lgbm_v1_group_top3",
}


def instrument_to_id_expr() -> pl.Expr:
    return (
        pl.when(pl.col("instrument") == "mzml")
        .then(pl.lit(0))
        .when(pl.col("instrument") == "tims")
        .then(pl.lit(1))
        .when(pl.col("instrument") == "wiff")
        .then(pl.lit(2))
        .otherwise(pl.lit(-1))
        .cast(pl.Int8)
        .alias("instrument_id")
    )


def is_candidate_parquet(path: Path) -> bool:
    name = path.name
    return (
        path.suffix == ".parquet"
        and ".tmp" not in name
        and not name.endswith(".bak")
        and not name.endswith(".bak_fragment")
        and not name.endswith(".tmp_group.parquet")
    )


def prediction_part_name(path: Path) -> str:
    digest = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}__{digest}__pred.parquet"


def list_valid_root_files(valid_root: Path) -> List[Path]:
    out: List[Path] = []
    for instrument in INSTRUMENTS:
        d = valid_root / instrument
        if not d.exists():
            print(f"valid instrument dir not found, skip: {d}")
            continue
        files = [p for p in sorted(d.glob("*.parquet")) if is_candidate_parquet(p)]
        print(f"{d}: {len(files)} files")
        out.extend(files)
    return out


def list_manifest_files(manifest: Path, fold: Optional[int], train_folds: Optional[Sequence[int]]) -> List[Path]:
    mf = pl.read_csv(manifest)
    if "path" not in mf.columns:
        raise RuntimeError(f"manifest must contain a path column: {manifest}")
    if "fold" not in mf.columns:
        raise RuntimeError(f"manifest must contain a fold column: {manifest}")

    if fold is not None and train_folds is not None:
        raise ValueError("Use either --fold or --train-folds, not both.")

    if fold is not None:
        mf = mf.filter(pl.col("fold") == int(fold))
    elif train_folds is not None:
        mf = mf.filter(pl.col("fold").is_in([int(x) for x in train_folds]))
    else:
        print("No --fold supplied; evaluating every file in manifest.")

    files = [Path(x) for x in mf["path"].to_list()]
    print(f"manifest selected files: {len(files)}")
    return files


def load_model_and_features(model_dir: Path):
    model_path = model_dir / "model.txt"
    feature_path = model_dir / "feature_columns.json"
    if not model_path.exists():
        raise FileNotFoundError(f"missing LightGBM model: {model_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"missing feature_columns.json: {feature_path}")
    model = lgb.Booster(model_file=str(model_path))
    with open(feature_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise RuntimeError(f"feature_columns.json is invalid: {feature_path}")
    return model, feature_cols


def get_schema_cols(path: Path) -> List[str]:
    return pl.scan_parquet(path).collect_schema().names()


def prepare_eval_frame(path: Path, feature_cols: List[str]) -> pl.DataFrame:
    schema = set(get_schema_cols(path))

    required_base = ["label", "instrument"]
    missing_base = [c for c in required_base if c not in schema]
    if missing_base:
        raise RuntimeError(f"{path} missing required columns: {missing_base}")

    missing_features = [c for c in feature_cols if c != "instrument_id" and c not in schema]
    if missing_features:
        raise RuntimeError(f"{path} missing model features: {missing_features[:30]}")

    if "instrument_id" in feature_cols and "instrument" not in schema:
        raise RuntimeError(f"{path} needs instrument to build instrument_id")

    read_cols = []
    for c in feature_cols:
        if c == "instrument_id":
            read_cols.append("instrument")
        else:
            read_cols.append(c)
    read_cols += ["label"]
    read_cols += [c for c in ID_COLUMNS if c in schema]
    read_cols = list(dict.fromkeys(read_cols))

    df = pl.read_parquet(path, columns=read_cols)

    if "instrument_id" in feature_cols:
        df = df.with_columns([instrument_to_id_expr()])

    cast_exprs = [pl.col("label").cast(pl.Int8)]
    for c in feature_cols:
        if c not in df.columns:
            continue
        if c in CATEGORICAL_FEATURES:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))
    df = df.with_columns(cast_exprs)

    return df


def predict_one_file(model: lgb.Booster, feature_cols: List[str], path: Path, out_path: Path, force: bool) -> Dict:
    if out_path.exists() and not force:
        try:
            existing = pl.scan_parquet(out_path).select(pl.len().alias("n")).collect().item()
            return {"path": str(path), "pred_path": str(out_path), "rows": int(existing), "status": "reused"}
        except Exception:
            print(f"Existing prediction is unreadable; recomputing: {out_path}")

    df = prepare_eval_frame(path, feature_cols)

    X = df.select(feature_cols).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and best_iter > 0:
        pred = model.predict(X, num_iteration=best_iter)
    else:
        pred = model.predict(X)
    pred = np.asarray(pred, dtype=np.float32)

    keep_cols = [c for c in ID_COLUMNS + ["label"] if c in df.columns]
    pred_df = df.select(keep_cols).with_columns(pl.Series("score", pred).cast(pl.Float32))

    # Make sure every row has stable strings for grouping/reporting.
    if "file_id" not in pred_df.columns:
        pred_df = pred_df.with_columns(pl.lit(path.stem).alias("file_id"))
    if "instrument" not in pred_df.columns:
        pred_df = pred_df.with_columns(pl.lit("unknown").alias("instrument"))
    if "peptide_key" not in pred_df.columns and "precursor_sequence" in pred_df.columns:
        pred_df = pred_df.with_columns(pl.col("precursor_sequence").cast(pl.Utf8).alias("peptide_key"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    pred_df.write_parquet(tmp_path)
    os.replace(tmp_path, out_path)

    rows = pred_df.height
    del df, X, pred_df, pred
    gc.collect()
    return {"path": str(path), "pred_path": str(out_path), "rows": int(rows), "status": "computed"}


def safe_auc_ap(df: pl.DataFrame) -> Dict:
    y = df["label"].to_numpy()
    score = df["score"].to_numpy()
    out: Dict[str, Optional[float]] = {}
    try:
        out["auc"] = float(roc_auc_score(y, score))
    except Exception:
        out["auc"] = None
    try:
        out["average_precision"] = float(average_precision_score(y, score))
    except Exception:
        out["average_precision"] = None
    out["label_mean"] = float(np.mean(y)) if len(y) else None
    out["score_mean"] = float(np.mean(score)) if len(score) else None
    out["score_std"] = float(np.std(score)) if len(score) else None
    return out


def fdr_table(df: pl.DataFrame, fdr_threshold: float, conservative_tdc: bool = True) -> Dict:
    if df.height == 0:
        return {
            "rows": 0,
            "target_rows": 0,
            "decoy_rows": 0,
            "accepted_target_psm_at_fdr": 0,
            "accepted_unique_peptide_at_fdr": 0,
            "accepted_decoy_rows_at_fdr": 0,
            "accepted_total_rows_at_fdr": 0,
            "score_threshold_at_fdr": None,
            "rank_cutoff_at_fdr": 0,
            "estimated_fdr_at_cutoff": None,
            "raw_decoy_over_target_at_cutoff": None,
            "fdp_proxy_decoy_over_accepted_at_cutoff": None,
            "cutoff_target_rows": 0,
            "cutoff_decoy_rows": 0,
            "conservative_tdc": conservative_tdc,
        }

    needed = ["label", "score"]
    if "peptide_key" in df.columns:
        needed.append("peptide_key")
    if "instrument" in df.columns:
        needed.append("instrument")
    if "file_id" in df.columns:
        needed.append("file_id")
    if "group_key" in df.columns:
        needed.append("group_key")

    df = df.select([c for c in needed if c in df.columns]).sort("score", descending=True)

    labels = df["label"].to_numpy()
    is_target = labels == 1
    is_decoy = labels == 0
    cum_target = np.cumsum(is_target)
    cum_decoy = np.cumsum(is_decoy)

    # Official helper code in eval_fdr/utils.py uses a conservative convention:
    # before the first observed decoy, decoy count is treated as 1 instead of 0.
    # This avoids unrealistically optimistic q=0 at the very top of the ranking.
    decoy_for_fdr = np.maximum(cum_decoy, 1) if conservative_tdc else cum_decoy
    fdr = decoy_for_fdr / np.maximum(cum_target, 1)
    raw_fdr = cum_decoy / np.maximum(cum_target, 1)
    q_value = np.minimum.accumulate(fdr[::-1])[::-1]
    keep = q_value <= fdr_threshold

    dfq = df.with_columns(
        [
            pl.Series("q_value", q_value).cast(pl.Float32),
            pl.Series("accepted", keep).cast(pl.Boolean),
            pl.Series("rank", np.arange(1, df.height + 1, dtype=np.int64)),
            pl.Series("cum_target", cum_target).cast(pl.Int64),
            pl.Series("cum_decoy", cum_decoy).cast(pl.Int64),
            pl.Series("fdr", fdr).cast(pl.Float32),
            pl.Series("raw_fdr", raw_fdr).cast(pl.Float32),
        ]
    )

    accepted_all = dfq.filter(pl.col("accepted"))
    accepted_targets = accepted_all.filter(pl.col("label") == 1)
    accepted_decoys = accepted_all.filter(pl.col("label") == 0)

    if accepted_all.height > 0:
        last = accepted_all.tail(1).row(0, named=True)
        score_threshold = float(last["score"])
        rank_cutoff = int(last["rank"])
        estimated_fdr = float(last["fdr"])
        raw_cutoff_fdr = float(last["raw_fdr"])
        cutoff_target_rows = int(last["cum_target"])
        cutoff_decoy_rows = int(last["cum_decoy"])
        fdp_proxy = cutoff_decoy_rows / max(cutoff_target_rows + cutoff_decoy_rows, 1)
        accepted_unique = int(accepted_targets["peptide_key"].n_unique()) if "peptide_key" in accepted_targets.columns else None
    else:
        score_threshold = None
        rank_cutoff = 0
        estimated_fdr = None
        raw_cutoff_fdr = None
        cutoff_target_rows = 0
        cutoff_decoy_rows = 0
        fdp_proxy = None
        accepted_unique = 0 if "peptide_key" in dfq.columns else None

    return {
        "rows": int(df.height),
        "target_rows": int(is_target.sum()),
        "decoy_rows": int(is_decoy.sum()),
        "target_rate": float(is_target.mean()) if df.height else None,
        "accepted_target_psm_at_fdr": int(accepted_targets.height),
        "accepted_unique_peptide_at_fdr": accepted_unique,
        "accepted_decoy_rows_at_fdr": int(accepted_decoys.height),
        "accepted_total_rows_at_fdr": int(accepted_all.height),
        "score_threshold_at_fdr": score_threshold,
        "rank_cutoff_at_fdr": rank_cutoff,
        "estimated_fdr_at_cutoff": estimated_fdr,
        "raw_decoy_over_target_at_cutoff": raw_cutoff_fdr,
        "fdp_proxy_decoy_over_accepted_at_cutoff": fdp_proxy,
        "cutoff_target_rows": cutoff_target_rows,
        "cutoff_decoy_rows": cutoff_decoy_rows,
        "conservative_tdc": conservative_tdc,
    }


def rename_fdr_keys(metric: Dict, fdr_threshold: float) -> Dict:
    suffix = f"{int(round(fdr_threshold * 10000))}bp"
    # For 0.01, use the conventional 1pct names used in the existing repo.
    if abs(fdr_threshold - 0.01) < 1e-12:
        return {
            **{k: v for k, v in metric.items() if not k.endswith("_at_fdr")},
            "accepted_target_psm_at_1pct": metric["accepted_target_psm_at_fdr"],
            "accepted_unique_peptide_at_1pct": metric["accepted_unique_peptide_at_fdr"],
            "accepted_decoy_rows_at_1pct": metric.get("accepted_decoy_rows_at_fdr"),
            "accepted_total_rows_at_1pct": metric.get("accepted_total_rows_at_fdr"),
        }
    return {
        **{k: v for k, v in metric.items() if not k.endswith("_at_fdr")},
        f"accepted_target_psm_at_{suffix}": metric["accepted_target_psm_at_fdr"],
        f"accepted_unique_peptide_at_{suffix}": metric["accepted_unique_peptide_at_fdr"],
        f"accepted_decoy_rows_at_{suffix}": metric.get("accepted_decoy_rows_at_fdr"),
        f"accepted_total_rows_at_{suffix}": metric.get("accepted_total_rows_at_fdr"),
    }


def add_rank_within_group(df: pl.DataFrame) -> pl.DataFrame:
    if "group_key" not in df.columns:
        return df
    return df.with_columns(
        pl.col("score").rank(method="ordinal", descending=True).over("group_key").cast(pl.Int32).alias("score_rank_in_group")
    )


def add_rank_within_peptide(df: pl.DataFrame) -> pl.DataFrame:
    if "peptide_key" not in df.columns:
        return df
    return df.with_columns(
        pl.col("score").rank(method="ordinal", descending=True).over("peptide_key").cast(pl.Int32).alias("score_rank_in_peptide")
    )


def primary_metric_name(fdr_threshold: float) -> str:
    if abs(fdr_threshold - 0.01) < 1e-12:
        return "accepted_unique_peptide_at_1pct"
    suffix = f"{int(round(fdr_threshold * 10000))}bp"
    return f"accepted_unique_peptide_at_{suffix}"


def compute_all_metrics(pred_df: pl.DataFrame, fdr_threshold: float, conservative_tdc: bool) -> Dict:
    metrics: Dict = {}
    metrics["basic_metrics"] = safe_auc_ap(pred_df)

    official_all = rename_fdr_keys(
        fdr_table(pred_df, fdr_threshold, conservative_tdc=conservative_tdc),
        fdr_threshold,
    )
    metrics["official_like_global_all_rows"] = official_all

    objective_name = primary_metric_name(fdr_threshold)
    metrics["primary_tuning_objective"] = {
        "name": objective_name,
        "value": official_all.get(objective_name),
        "source": "official_like_global_all_rows",
        "rule": "Use this unique-peptide-at-FDR value as the main tuning objective; AUC/AP are diagnostics only.",
    }

    # Diagnostic: after global FDR, where do accepted targets come from?
    globally_sorted = pred_df.select([c for c in ["label", "score", "peptide_key", "instrument", "file_id"] if c in pred_df.columns])
    globally_sorted = globally_sorted.sort("score", descending=True)
    labels = globally_sorted["label"].to_numpy()
    cum_target = np.cumsum(labels == 1)
    cum_decoy = np.cumsum(labels == 0)
    decoy_for_fdr = np.maximum(cum_decoy, 1) if conservative_tdc else cum_decoy
    fdr = decoy_for_fdr / np.maximum(cum_target, 1)
    q = np.minimum.accumulate(fdr[::-1])[::-1]
    accepted_global_all = globally_sorted.with_columns(pl.Series("q_value", q).cast(pl.Float32)).filter(
        pl.col("q_value") <= fdr_threshold
    )
    accepted_global = accepted_global_all.filter(pl.col("label") == 1)

    by_inst_global = {}
    if "instrument" in accepted_global.columns:
        for inst in INSTRUMENTS:
            sub = accepted_global.filter(pl.col("instrument") == inst)
            sub_all = accepted_global_all.filter(pl.col("instrument") == inst)
            sub_decoys = int((sub_all["label"] == 0).sum()) if sub_all.height else 0
            by_inst_global[inst] = {
                "accepted_target_psm_at_1pct": int(sub.height),
                "accepted_unique_peptide_at_1pct": int(sub["peptide_key"].n_unique()) if "peptide_key" in sub.columns and sub.height else 0,
                "accepted_decoy_rows_at_1pct": sub_decoys,
                "fdp_proxy_decoy_over_accepted_at_1pct": sub_decoys / max(int(sub_all.height), 1) if sub_all.height else None,
            }
    metrics["accepted_by_instrument_under_global_fdr"] = by_inst_global

    by_file_global = []
    if "file_id" in accepted_global.columns:
        if abs(fdr_threshold - 0.01) < 1e-12:
            target_key = "accepted_target_psm_at_1pct"
            unique_key = "accepted_unique_peptide_at_1pct"
            decoy_key = "accepted_decoy_rows_at_1pct"
        else:
            suffix = f"{int(round(fdr_threshold * 10000))}bp"
            target_key = f"accepted_target_psm_at_{suffix}"
            unique_key = f"accepted_unique_peptide_at_{suffix}"
            decoy_key = f"accepted_decoy_rows_at_{suffix}"

        all_file_ids = sorted(pred_df["file_id"].unique().to_list())
        for file_id in all_file_ids:
            sub_all_rows = pred_df.filter(pl.col("file_id") == file_id)
            sub_target = accepted_global.filter(pl.col("file_id") == file_id)
            sub_all = accepted_global_all.filter(pl.col("file_id") == file_id)
            out = {
                "file_id": file_id,
                "rows": int(sub_all_rows.height),
                "target_rows": int((sub_all_rows["label"] == 1).sum()) if sub_all_rows.height else 0,
                "decoy_rows": int((sub_all_rows["label"] == 0).sum()) if sub_all_rows.height else 0,
                target_key: int(sub_target.height),
                unique_key: int(sub_target["peptide_key"].n_unique()) if "peptide_key" in sub_target.columns and sub_target.height else 0,
                decoy_key: int((sub_all["label"] == 0).sum()) if sub_all.height else 0,
                "fdp_proxy_decoy_over_accepted_at_1pct": int((sub_all["label"] == 0).sum()) / max(int(sub_all.height), 1) if sub_all.height else None,
            }
            if "instrument" in sub_target.columns and sub_target.height:
                out["instrument"] = sub_target["instrument"][0]
            elif "instrument" in sub_all_rows.columns and sub_all_rows.height:
                out["instrument"] = sub_all_rows["instrument"][0]
            by_file_global.append(out)
    metrics["accepted_by_file_under_global_fdr"] = by_file_global

    # Diagnostic: each instrument evaluated independently. This is not necessarily official.
    by_inst_independent = {}
    if "instrument" in pred_df.columns:
        for inst in INSTRUMENTS:
            sub = pred_df.filter(pl.col("instrument") == inst)
            by_inst_independent[inst] = rename_fdr_keys(
                fdr_table(sub, fdr_threshold, conservative_tdc=conservative_tdc),
                fdr_threshold,
            )
    metrics["independent_fdr_by_instrument"] = by_inst_independent

    # Diagnostic: each file independently. Useful for fold debugging and bad-file detection.
    by_file = []
    if "file_id" in pred_df.columns:
        for row in pred_df.group_by("file_id").agg(pl.len().alias("n")).sort("file_id").iter_rows(named=True):
            file_id = row["file_id"]
            sub = pred_df.filter(pl.col("file_id") == file_id)
            m = rename_fdr_keys(
                fdr_table(sub, fdr_threshold, conservative_tdc=conservative_tdc),
                fdr_threshold,
            )
            m["file_id"] = file_id
            if "instrument" in sub.columns and sub.height:
                m["instrument"] = sub["instrument"][0]
            by_file.append(m)
    metrics["independent_fdr_by_file"] = by_file

    # Diagnostic: group-level top1 quality and FDR using only the top-scoring candidate per scan/group.
    if "group_key" in pred_df.columns:
        ranked_group = add_rank_within_group(pred_df)
        top1 = ranked_group.filter(pl.col("score_rank_in_group") == 1)
        metrics["top1_group_diagnostics"] = {
            "num_groups": int(top1.height),
            "top1_target_rate": float(top1["label"].mean()) if top1.height else None,
            "top1_official_like_fdr": rename_fdr_keys(
                fdr_table(top1, fdr_threshold, conservative_tdc=conservative_tdc),
                fdr_threshold,
            ),
        }

    # Diagnostic: peptide-level best PSM. Useful when optimizing unique peptide count.
    if "peptide_key" in pred_df.columns:
        ranked_pep = add_rank_within_peptide(pred_df)
        best_pep = ranked_pep.filter(pl.col("score_rank_in_peptide") == 1)
        metrics["best_psm_per_peptide_diagnostics"] = {
            "num_peptides": int(best_pep.height),
            "best_peptide_target_rate": float(best_pep["label"].mean()) if best_pep.height else None,
            "best_peptide_official_like_fdr": rename_fdr_keys(
                fdr_table(best_pep, fdr_threshold, conservative_tdc=conservative_tdc),
                fdr_threshold,
            ),
        }

    return metrics


def load_prediction_parts(prediction_dir: Path) -> pl.DataFrame:
    files = sorted(prediction_dir.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"No prediction parquet files in {prediction_dir}")
    print(f"Loading prediction parts: {len(files)}")
    lf = pl.scan_parquet([str(p) for p in files])
    keep_cols = [c for c in ID_COLUMNS + ["label", "score"] if c in lf.collect_schema().names()]
    return lf.select(keep_cols).collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=None, help="Optional shortcut; uses data-root/processed_split/valid when --valid-root is omitted.")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None, help="Validation fold to evaluate from --manifest.")
    parser.add_argument(
        "--train-folds",
        type=int,
        nargs="+",
        default=None,
        help="Optional: evaluate a set of folds. Usually you want --fold for valid.",
    )
    parser.add_argument("--valid-root", type=str, default=None, help="Fallback: processed_split/valid root with mzml/tims/wiff subdirs.")
    parser.add_argument("--fdr-threshold", type=float, default=0.01)
    parser.add_argument(
        "--non-conservative-zero-decoy",
        action="store_true",
        help="Use cum_decoy/cum_target directly. Default is conservative TDC with max(cum_decoy, 1), matching eval_fdr/utils.py more closely.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Debug only.")
    parser.add_argument("--force", action="store_true", help="Recompute prediction parts even when they exist.")
    parser.add_argument("--reuse-predictions-only", action="store_true", help="Skip model prediction and only recompute metrics from out_dir/pred_parts.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "pred_parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if args.valid_root is None and args.data_root is not None:
        args.valid_root = str(Path(args.data_root) / "processed_split" / "valid")

    if args.reuse_predictions_only:
        files: List[Path] = []
    elif args.manifest:
        files = list_manifest_files(Path(args.manifest), fold=args.fold, train_folds=args.train_folds)
    elif args.valid_root:
        files = list_valid_root_files(Path(args.valid_root))
    else:
        raise RuntimeError("Provide either --manifest + --fold or --valid-root.")

    if args.max_files is not None:
        files = files[: args.max_files]
    if not args.reuse_predictions_only and not files:
        raise RuntimeError("No validation files selected.")

    file_reports: List[Dict] = []

    if not args.reuse_predictions_only:
        model_dir = Path(args.model_dir)
        model, feature_cols = load_model_and_features(model_dir)
        with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2, ensure_ascii=False)

        print("\n========== Predict full validation files ==========")
        for path in tqdm(files):
            safe_name = prediction_part_name(path)
            out_path = pred_dir / safe_name
            try:
                report = predict_one_file(model, feature_cols, path, out_path, force=args.force)
            except Exception as e:
                report = {"path": str(path), "pred_path": str(out_path), "rows": 0, "status": "failed", "error": str(e)}
                print(f"Prediction failed: {path}\n  {e}")
            file_reports.append(report)
            gc.collect()

        with open(out_dir / "prediction_file_report.json", "w", encoding="utf-8") as f:
            json.dump(file_reports, f, indent=2, ensure_ascii=False)

        failed = [x for x in file_reports if x.get("status") == "failed"]
        if failed:
            raise RuntimeError(f"{len(failed)} validation files failed. See prediction_file_report.json")

    print("\n========== Load predictions and compute official-like FDR ==========")
    pred_df = load_prediction_parts(pred_dir)
    print("prediction rows:", pred_df.height)
    print(pred_df.group_by("label").len().sort("label"))
    if "instrument" in pred_df.columns:
        print(pred_df.group_by("instrument").len().sort("instrument"))

    conservative_tdc = not args.non_conservative_zero_decoy
    metrics = compute_all_metrics(
        pred_df,
        fdr_threshold=args.fdr_threshold,
        conservative_tdc=conservative_tdc,
    )

    meta = {
        "model_dir": str(Path(args.model_dir)),
        "out_dir": str(out_dir),
        "manifest": args.manifest,
        "fold": args.fold,
        "train_folds": args.train_folds,
        "valid_root": args.valid_root,
        "fdr_threshold": args.fdr_threshold,
        "conservative_tdc": conservative_tdc,
        "num_prediction_rows": int(pred_df.height),
    }
    output = {"meta": meta, "metrics": metrics}

    metrics_path = out_dir / "full_valid_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # File-level table is easier to compare as CSV.
    by_file = metrics.get("independent_fdr_by_file", [])
    if by_file:
        pl.DataFrame(by_file).write_csv(out_dir / "full_valid_metrics_by_file.csv")

    by_file_global = metrics.get("accepted_by_file_under_global_fdr", [])
    if by_file_global:
        pl.DataFrame(by_file_global).write_csv(out_dir / "full_valid_global_fdr_by_file.csv")

    by_inst_global = metrics.get("accepted_by_instrument_under_global_fdr", {})
    if by_inst_global:
        rows = []
        for inst, vals in by_inst_global.items():
            rows.append({"instrument": inst, **vals})
        pl.DataFrame(rows).write_csv(out_dir / "full_valid_global_fdr_by_instrument.csv")

    by_inst = metrics.get("independent_fdr_by_instrument", {})
    if by_inst:
        rows = []
        for inst, vals in by_inst.items():
            rows.append({"instrument": inst, **vals})
        pl.DataFrame(rows).write_csv(out_dir / "full_valid_metrics_by_instrument.csv")

    print("\n========== Main metrics ==========")
    print("Primary tuning objective:")
    print(json.dumps(metrics["primary_tuning_objective"], indent=2, ensure_ascii=False))
    print("\nOfficial-like global all rows:")
    print(json.dumps(metrics["official_like_global_all_rows"], indent=2, ensure_ascii=False))
    if "top1_group_diagnostics" in metrics:
        print("\nTop1 group diagnostics:")
        print(json.dumps(metrics["top1_group_diagnostics"], indent=2, ensure_ascii=False))
    print(f"\nWrote: {metrics_path}")


if __name__ == "__main__":
    main()
