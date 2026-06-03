# 训练 K 折 LightGBM v1，用于后续生成 OOF v1 分数

# python src/model/train_lightGBM_v1_folds.py \
#   --data-root /root/autodl-tmp/datasets/aipc \
#   --out-dir ~/aipc/models/lgbm_v1_oof \
#   --n-folds 5 \
#   --balance-by-rows \
#   --reuse-folds \
#   --train-max-rows 30000000 \
#   --valid-max-rows 3000000 \
#   --max-rows-per-file 150000 \
#   --num-boost-round 3000 \
#   --early-stopping-rounds 150 \
#   --semi-target-fdr 0.01 \
#   --semi-neg-pos-ratio 2.0 \
#   --semi-hard-decoy-frac 0.5 \
#   --semi-high-score-decoy-frac 0.3 \
#   --num-threads 12

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import gc
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

from fdr_training_metric import (
    FdrEvalMetadata,
    UniquePeptideFdrEarlyStopping,
    resolve_best_iteration,
)


INSTRUMENTS = ["mzml", "tims", "wiff"]

CATEGORICAL_FEATURES = [
    "instrument_id",
    "charge",
    "has_ion_mobility",
    "has_mod",
    "parse_ok",
    "fragment_parse_ok",
    "best_isotope_offset",
    "is_first_candidate_in_scan",
    "is_top3_candidate_in_scan",
    "is_best_abs_delta_rt_in_scan",
    "candidate_order_matches_abs_delta_rank",
    "scan_has_multiple_charges",
    "is_first_candidate_for_charge_in_scan",
]

ID_COLUMNS_FOR_VALID = [
    "file_id",
    "instrument",
    "scan_number",
    "group_key",
    "peptide_key",
    "precursor_sequence",
]

EXCLUDE_FEATURE_COLUMNS = {
    "label",
    "file_id",
    "instrument",
    "scan_number",
    "group_key",
    "peptide_key",
    "precursor_sequence",
    "mz_array",
    "intensity_array",
    "proteins",
    "peptide",
    "aux_feature_done",
    "fragment_feature_done",
    "group_feature_done",
    "sage_discriminant_score",
    "sage_score_filled",
    "spectrum_q",
    "spectrum_q_filled",
    "fp_q_value",
    "fp_q_value_filled",
    "has_fp_q_value",
    "in_fp",
    "lgbm_v1_score",
    "lgbm_v1_group_size",
    "lgbm_v1_group_rank",
    "lgbm_v1_group_rank_pct",
    "lgbm_v1_group_max",
    "lgbm_v1_group_min",
    "lgbm_v1_group_mean",
    "lgbm_v1_group_std",
    "lgbm_v1_group_top2",
    "lgbm_v1_gap_to_top1",
    "lgbm_v1_gap_to_top2",
    "lgbm_v1_top1_margin",
    "lgbm_v1_minus_group_mean",
    "lgbm_v1_z_in_group",
    "lgbm_v1_softmax_in_group",
    "is_lgbm_v1_group_top1",
    "is_lgbm_v1_group_top3",
}

NUMERIC_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
    pl.Boolean,
}

RT_QVALUE_ANOMALY_INSTRUMENTS = {"tims", "wiff"}
RT_QVALUE_ANOMALY_FEATURES = [
    "predicted_rt",
    "delta_rt",
    "spectrum_q",
    "spectrum_q_filled",
    "abs_delta_rt",
    "delta_rt_z_in_file",
    "predicted_rt_z_in_file",
    "abs_delta_rt_rank_in_scan",
    "abs_delta_rt_rank_pct_in_scan",
    "is_best_abs_delta_rt_in_scan",
    "abs_delta_rt_min_in_scan",
    "abs_delta_rt_mean_in_scan",
    "abs_delta_rt_std_in_scan",
    "abs_delta_rt_gap_to_best_in_scan",
    "abs_delta_rt_z_in_scan",
    "candidate_order_rt_rank_gap",
    "abs_candidate_order_rt_rank_gap",
    "candidate_order_matches_abs_delta_rank",
]
RT_QVALUE_ANOMALY_CATEGORICAL_FEATURES = {
    "is_best_abs_delta_rt_in_scan",
    "candidate_order_matches_abs_delta_rank",
}


# ============================================================
# 1. fold 划分工具
# ============================================================

def stable_hash(text: str, seed: int) -> int:
    raw = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.md5(raw).hexdigest(), 16)


def infer_instrument(path: Path) -> str:
    lower_parts = [p.lower() for p in path.parts]

    for inst in INSTRUMENTS:
        if inst in lower_parts:
            return inst

    lower = str(path).lower()

    for inst in INSTRUMENTS:
        if inst in lower:
            return inst

    return "unknown"


def file_key(path: Path, train_root: Path) -> str:
    try:
        return path.relative_to(train_root).as_posix()
    except ValueError:
        inst = infer_instrument(path)
        return f"{inst}/{path.name}"


def get_row_count(path: Path) -> int:
    try:
        return int(
            pl.scan_parquet(path)
            .select(pl.len())
            .collect()
            .item()
        )
    except Exception:
        return 1


def list_parquet_files(root: Path) -> List[Path]:
    return [
        p for p in sorted(root.rglob("*.parquet"))
        if ".tmp" not in p.name
        and not p.name.endswith(".bak")
        and not p.name.endswith(".bak_fragment")
        and not p.name.endswith(".tmp_group.parquet")
    ]


def list_split_files(split_root: Path) -> List[Path]:
    files: List[Path] = []
    for instrument in INSTRUMENTS:
        instrument_dir = split_root / instrument
        if instrument_dir.exists():
            instrument_files = list_parquet_files(instrument_dir)
            print(f"{split_root.name}/{instrument}: {len(instrument_files)} files")
            files.extend(instrument_files)
        else:
            print(f"目录不存在: {instrument_dir}")
    return files


def get_schema_cols(path: Path) -> List[str]:
    return pl.scan_parquet(path).collect_schema().names()


def check_file_ready(path: Path) -> bool:
    cols = set(get_schema_cols(path))
    required = {"label", "instrument", "aux_feature_done", "fragment_feature_done"}
    missing = required - cols
    if missing:
        print(f"跳过未完成文件: {path}, 缺少: {missing}")
        return False
    return True


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


def resolve_feature_columns(files: List[Path]) -> List[str]:
    for path in files:
        if not check_file_ready(path):
            continue
        schema = pl.scan_parquet(path).collect_schema()
        feature_cols = []
        for col_name, dtype in schema.items():
            if col_name in EXCLUDE_FEATURE_COLUMNS:
                continue
            if col_name in ID_COLUMNS_FOR_VALID:
                continue
            if col_name == "instrument_id":
                continue
            if dtype in NUMERIC_DTYPES:
                feature_cols.append(col_name)
        if "instrument_id" not in feature_cols:
            feature_cols.append("instrument_id")
        return feature_cols
    raise RuntimeError("没有找到可用于确定 v1 特征列的文件")


def polars_to_pandas_xy(df: pl.DataFrame, feature_cols: List[str]):
    y = df["label"].to_numpy()
    X = df.select(feature_cols).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y


def mask_rt_qvalue_anomaly_features(
    features: pd.DataFrame,
    source_df: pl.DataFrame,
    feature_cols: List[str],
    context: str,
) -> pd.DataFrame:
    if "instrument" not in source_df.columns:
        print(f"withoutRT: skip masking for {context}, missing instrument column")
        return features
    columns_to_mask = [
        col for col in RT_QVALUE_ANOMALY_FEATURES
        if col in feature_cols and col in features.columns
    ]
    if not columns_to_mask:
        return features
    instruments = source_df["instrument"].cast(pl.Utf8).to_numpy()
    mask = np.isin(instruments, list(RT_QVALUE_ANOMALY_INSTRUMENTS))
    masked_rows = int(mask.sum())
    if masked_rows == 0:
        return features
    features = features.copy()
    categorical_to_mask = [
        col for col in columns_to_mask
        if col in RT_QVALUE_ANOMALY_CATEGORICAL_FEATURES
    ]
    numeric_to_mask = [col for col in columns_to_mask if col not in categorical_to_mask]
    if numeric_to_mask:
        features.loc[mask, numeric_to_mask] = np.nan
    if categorical_to_mask:
        features.loc[mask, categorical_to_mask] = -1
    print(
        f"withoutRT: {context} masked {len(columns_to_mask)} RT/q-value columns "
        f"for {masked_rows:,} tims/wiff labeled rows"
    )
    return features


def build_fold_manifest(
    files: List[Path],
    train_root: Path,
    n_folds: int,
    seed: int,
    balance_by_rows: bool,
) -> Dict:
    rows = []

    for path in files:
        inst = infer_instrument(path)
        row_count = get_row_count(path) if balance_by_rows else 1

        rows.append({
            "key": file_key(path, train_root),
            "path": str(path),
            "file_name": path.name,
            "instrument": inst,
            "rows": int(row_count),
            "fold": None,
        })

    by_inst: Dict[str, List[Dict]] = {inst: [] for inst in INSTRUMENTS}
    by_inst["unknown"] = []

    for item in rows:
        by_inst.setdefault(item["instrument"], []).append(item)

    for inst, items in by_inst.items():
        fold_load = [0 for _ in range(n_folds)]

        items = sorted(
            items,
            key=lambda x: (-x["rows"], stable_hash(x["key"], seed)),
        )

        for item in items:
            fold_id = min(range(n_folds), key=lambda i: (fold_load[i], i))
            item["fold"] = int(fold_id)
            fold_load[fold_id] += int(item["rows"])

        print(f"{inst}: files={len(items)}, fold_load={fold_load}")

    rows = sorted(rows, key=lambda x: (x["fold"], x["instrument"], x["key"]))

    return {
        "schema_version": 1,
        "description": (
            "OOF v1 folds for generating out-of-fold "
            "lgbm_v1_score/group features on processed_split/train."
        ),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "balance_by_rows": bool(balance_by_rows),
        "train_root": str(train_root),
        "files": rows,
    }


def load_or_create_manifest(
    args,
    train_root: Path,
    train_files: List[Path],
    out_dir: Path,
) -> Dict:
    manifest_path = Path(args.folds_json).expanduser() if args.folds_json else out_dir / "folds.json"

    if manifest_path.exists() and args.reuse_folds:
        print(f"读取已有 folds: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("生成新的 OOF folds")

    manifest = build_fold_manifest(
        files=train_files,
        train_root=train_root,
        n_folds=args.n_folds,
        seed=args.seed,
        balance_by_rows=args.balance_by_rows,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"folds 已保存: {manifest_path}")

    return manifest


# ============================================================
# 2. v1 专用三仪器均衡采样
#    注意：这里不要求 group_feature_done
# ============================================================

def split_files_by_instrument(files: List[Path]) -> Dict[str, List[Path]]:
    by_inst = {
        "mzml": [],
        "tims": [],
        "wiff": [],
    }

    for p in files:
        inst = infer_instrument(p)

        if inst in by_inst:
            by_inst[inst].append(p)
        else:
            print(f"警告：无法判断仪器类型，跳过: {p}")

    return by_inst


def normalize_part_schema_v1(
    df: pl.DataFrame,
    feature_cols: List[str],
    keep_id_cols: bool,
) -> pl.DataFrame:
    """
    统一每个采样 part 的 schema。
    这个函数是 v1 专用版，不依赖 group 特征。
    """

    cast_exprs = []

    if "label" in df.columns:
        cast_exprs.append(pl.col("label").cast(pl.Int8))

    if "instrument" in df.columns:
        cast_exprs.append(pl.col("instrument").cast(pl.Utf8))

    id_cast_map = {
        "file_id": pl.Utf8,
        "scan_number": pl.Int64,
        "group_key": pl.Utf8,
        "peptide_key": pl.Utf8,
        "precursor_sequence": pl.Utf8,
    }

    for c, dtype in id_cast_map.items():
        if c in df.columns:
            cast_exprs.append(pl.col(c).cast(dtype))

    for c in feature_cols:
        if c not in df.columns:
            continue

        if c in CATEGORICAL_FEATURES:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    final_cols = []

    for c in feature_cols:
        if c in df.columns:
            final_cols.append(c)

    if "label" in df.columns:
        final_cols.append("label")

    if "instrument" in df.columns:
        final_cols.append("instrument")

    if keep_id_cols:
        for c in ID_COLUMNS_FOR_VALID:
            if c in df.columns:
                final_cols.append(c)

    final_cols = list(dict.fromkeys(final_cols))

    return df.select(final_cols)


def clean_polars_df(df: pl.DataFrame, feature_cols: List[str], keep_id_cols: bool) -> pl.DataFrame:
    if "instrument_id" in feature_cols and "instrument" in df.columns:
        df = df.with_columns([instrument_to_id_expr()])

    required_cols = list(feature_cols) + ["label"]
    if "instrument" in df.columns:
        required_cols.append("instrument")
    if keep_id_cols:
        required_cols.extend([c for c in ID_COLUMNS_FOR_VALID if c in df.columns])
    required_cols = [c for c in list(dict.fromkeys(required_cols)) if c in df.columns]
    df = df.select(required_cols)
    return normalize_part_schema_v1(df, feature_cols=feature_cols, keep_id_cols=keep_id_cols)


def sample_one_file(
    path: Path,
    feature_cols: List[str],
    max_rows_per_file: int,
    neg_pos_ratio: float,
    seed: int,
    keep_id_cols: bool = False,
) -> pl.DataFrame:
    cols = set(get_schema_cols(path))
    read_cols = [c for c in feature_cols if c in cols and c != "instrument_id"]
    read_cols += ["label", "instrument"]
    if keep_id_cols:
        read_cols += [c for c in ID_COLUMNS_FOR_VALID if c in cols]
    read_cols = list(dict.fromkeys(read_cols))

    df = pl.read_parquet(path, columns=read_cols)
    if df.height == 0:
        return clean_polars_df(df, feature_cols, keep_id_cols=keep_id_cols)

    df = df.with_columns(pl.col("label").cast(pl.Int8))
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)

    if pos.height == 0 or neg.height == 0:
        n = min(max_rows_per_file, df.height)
        out = df.sample(n=n, seed=seed, shuffle=True) if df.height > n else df
        return clean_polars_df(out, feature_cols, keep_id_cols=keep_id_cols)

    max_pos = int(max_rows_per_file / (1.0 + neg_pos_ratio))
    take_pos = min(pos.height, max(1, max_pos))
    take_neg = min(neg.height, max_rows_per_file - take_pos, int(take_pos * neg_pos_ratio))

    pos_s = pos.sample(n=take_pos, seed=seed, shuffle=True) if pos.height > take_pos else pos
    neg_s = neg.sample(n=take_neg, seed=seed + 17, shuffle=True) if neg.height > take_neg else neg
    out = pl.concat([pos_s, neg_s], how="vertical", rechunk=False)
    if out.height > 0:
        out = out.sample(n=out.height, seed=seed + 33, shuffle=True)
    return clean_polars_df(out, feature_cols, keep_id_cols=keep_id_cols)


def collect_sample_dataset_balanced_v1(
    files: List[Path],
    feature_cols: List[str],
    max_total_rows: int,
    max_rows_per_file: int,
    neg_pos_ratio: float,
    seed: int,
    keep_id_cols: bool = False,
) -> pl.DataFrame:
    """
    v1 专用三仪器均衡采样。
    不从 files 前面开始顺序采到 max_total_rows，而是 mzML / TIMS / WIFF 分别采样。
    """

    by_inst = split_files_by_instrument(files)
    instruments = [inst for inst in INSTRUMENTS if len(by_inst.get(inst, [])) > 0]
    if not instruments:
        raise RuntimeError("没有可用于 v1 采样的仪器文件")

    target_per_inst = max_total_rows // len(instruments)

    parts = []

    for inst_idx, inst in enumerate(instruments):
        inst_files = list(by_inst.get(inst, []))

        print()
        print(f"========== 采样 {inst} ==========")
        print(f"{inst} files: {len(inst_files)}")
        print(f"{inst} target rows: {target_per_inst}")

        if len(inst_files) == 0:
            print(f"{inst} 没有文件，跳过")
            continue

        rng = np.random.default_rng(seed + inst_idx * 1000)
        rng.shuffle(inst_files)

        inst_parts = []
        inst_rows = 0

        for i, path in enumerate(inst_files):
            if inst_rows >= target_per_inst:
                break

            # v1 的 check_file_ready 只应要求：
            # label, instrument, aux_feature_done, fragment_feature_done
            # 不应要求 group_feature_done
            if not check_file_ready(path):
                continue

            remain = target_per_inst - inst_rows
            per_file = min(max_rows_per_file, remain)

            try:
                part = sample_one_file(
                    path=path,
                    feature_cols=feature_cols,
                    max_rows_per_file=per_file,
                    neg_pos_ratio=neg_pos_ratio,
                    seed=seed + inst_idx * 100000 + i,
                    keep_id_cols=keep_id_cols,
                )

                if part.height == 0:
                    continue

                if "instrument" not in part.columns:
                    part = part.with_columns(pl.lit(inst).alias("instrument"))

                part = normalize_part_schema_v1(
                    part,
                    feature_cols=feature_cols,
                    keep_id_cols=keep_id_cols,
                )

                inst_parts.append(part)
                inst_rows += part.height

                del part
                gc.collect()

            except Exception as e:
                print(f"采样失败: {path}")
                print(f"错误: {e}")

        if len(inst_parts) == 0:
            print(f"{inst} 没有采样到数据")
            continue

        inst_df = pl.concat(inst_parts, how="vertical", rechunk=False)

        print(f"{inst} sampled rows: {inst_df.height}")
        print(inst_df.group_by("label").len().sort("label"))

        parts.append(inst_df)

        del inst_parts
        gc.collect()

    if len(parts) == 0:
        raise RuntimeError("没有采样到任何数据")

    df = pl.concat(parts, how="vertical", rechunk=False)

    if df.height > max_total_rows:
        df = df.sample(n=max_total_rows, seed=seed + 999, shuffle=True)

    print()
    print("========== 总采样结果 ==========")
    print("sampled rows:", df.height)

    if "instrument" in df.columns:
        print("instrument counts:")
        print(df.group_by("instrument").len().sort("instrument"))

    print("label counts:")
    print(df.group_by("label").len().sort("label"))

    return df


def value_counts_dict(df: pl.DataFrame, col: str) -> Dict[str, int]:
    if col not in df.columns:
        return {}

    rows = (
        df.group_by(col)
        .len()
        .sort(col)
        .to_dicts()
    )

    return {
        str(row[col]): int(row["len"])
        for row in rows
    }


def compute_basic_metrics(y_true, pred) -> Dict:
    out = {}
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        out["auc"] = float(roc_auc_score(y_true, pred))
    except Exception:
        out["auc"] = None
    try:
        from sklearn.metrics import average_precision_score
        out["average_precision"] = float(average_precision_score(y_true, pred))
    except Exception:
        out["average_precision"] = None
    out["pred_mean"] = float(np.mean(pred)) if len(pred) else None
    out["pred_std"] = float(np.std(pred)) if len(pred) else None
    out["label_mean"] = float(np.mean(y_true)) if len(y_true) else None
    return out


def compute_sampled_fdr_metric(
    valid_df: pl.DataFrame,
    pred: np.ndarray,
    conservative_tdc: bool = True,
) -> Dict:
    df = valid_df.select([
        c for c in ["label", "peptide_key", "instrument"]
        if c in valid_df.columns
    ]).with_columns([
        pl.Series("score", pred).cast(pl.Float32)
    ])
    df = df.sort("score", descending=True)
    labels = df["label"].to_numpy()
    is_target = labels == 1
    is_decoy = labels == 0
    cum_target = np.cumsum(is_target)
    cum_decoy = np.cumsum(is_decoy)
    decoy_for_fdr = np.maximum(cum_decoy, 1) if conservative_tdc else cum_decoy
    fdr = decoy_for_fdr / np.maximum(cum_target, 1)
    q_value = np.minimum.accumulate(fdr[::-1])[::-1]
    accepted = df.with_columns([
        pl.Series("q_value", q_value).cast(pl.Float32),
        pl.Series("accepted_1pct", q_value <= 0.01).cast(pl.Int8),
    ]).filter((pl.col("accepted_1pct") == 1) & (pl.col("label") == 1))
    metric = {
        "sample_valid_rows": int(df.height),
        "accepted_target_psm_at_1pct": int(accepted.height),
        "accepted_unique_peptide_at_1pct": int(accepted["peptide_key"].n_unique()) if "peptide_key" in accepted.columns and accepted.height > 0 else 0,
        "conservative_tdc": bool(conservative_tdc),
    }
    by_instrument = {}
    if "instrument" in accepted.columns:
        for instrument in INSTRUMENTS:
            sub = accepted.filter(pl.col("instrument") == instrument)
            by_instrument[instrument] = {
                "accepted_target_psm_at_1pct": int(sub.height),
                "accepted_unique_peptide_at_1pct": int(sub["peptide_key"].n_unique()) if "peptide_key" in sub.columns and sub.height > 0 else 0,
            }
    metric["by_instrument"] = by_instrument
    return metric


def add_q_value_by_instrument(
    df: pl.DataFrame,
    score_col: str,
    q_col: str = "semi_q_value",
    conservative_tdc: bool = True,
) -> pl.DataFrame:
    """Compute target-decoy q-values independently inside each instrument.

    Unknown / missing instruments are preserved with null q-values instead of being
    silently dropped. They will not become pseudo-positive samples.
    """
    if "instrument" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.Float32).alias(q_col))

    parts = []
    seen_instruments = []
    for instrument in INSTRUMENTS:
        sub = df.filter(pl.col("instrument") == instrument)
        if sub.height == 0:
            continue
        seen_instruments.append(instrument)
        sub = sub.sort(score_col, descending=True)
        labels = sub["label"].to_numpy().astype(np.int8)
        is_target = labels == 1
        is_decoy = labels == 0
        cum_target = np.cumsum(is_target)
        cum_decoy = np.cumsum(is_decoy)
        decoy_for_fdr = np.maximum(cum_decoy, 1) if conservative_tdc else cum_decoy
        fdr = decoy_for_fdr / np.maximum(cum_target, 1)
        q_value = np.minimum.accumulate(fdr[::-1])[::-1]
        parts.append(sub.with_columns(pl.Series(q_col, q_value).cast(pl.Float32)))

    rest = df.filter(~pl.col("instrument").is_in(seen_instruments))
    if rest.height > 0:
        parts.append(rest.with_columns(pl.lit(None).cast(pl.Float32).alias(q_col)))

    if not parts:
        return df.with_columns(pl.lit(None).cast(pl.Float32).alias(q_col))
    return pl.concat(parts, how="vertical", rechunk=False)


def add_group_hard_flags(df: pl.DataFrame, score_col: str, hard_margin: float) -> pl.DataFrame:
    """Mark hard decoys inside the same scan/group.

    Preferred group id is group_key. If it is absent, fall back to
    file_id + scan_number when both are available.
    """
    if "group_key" in df.columns:
        group_col = "group_key"
        working_df = df
    elif {"file_id", "scan_number"}.issubset(set(df.columns)):
        group_col = "__semi_group_key"
        working_df = df.with_columns(
            (pl.col("file_id").cast(pl.Utf8) + pl.lit("|") + pl.col("scan_number").cast(pl.Utf8))
            .alias(group_col)
        )
    else:
        return df.with_columns([
            pl.lit(0).cast(pl.Int8).alias("semi_is_hard_decoy"),
            pl.lit(0).cast(pl.Int8).alias("semi_is_top_decoy"),
        ])

    required = {group_col, "label", score_col}
    if not required.issubset(set(working_df.columns)):
        return df.with_columns([
            pl.lit(0).cast(pl.Int8).alias("semi_is_hard_decoy"),
            pl.lit(0).cast(pl.Int8).alias("semi_is_top_decoy"),
        ])

    group_stats = (
        working_df.group_by(group_col)
        .agg([
            pl.col(score_col).filter(pl.col("label") == 1).max().alias("semi_group_best_target"),
            pl.col(score_col).filter(pl.col("label") == 0).max().alias("semi_group_best_decoy"),
        ])
        .with_columns([
            (pl.col("semi_group_best_decoy") - pl.col("semi_group_best_target"))
            .alias("semi_decoy_minus_target"),
        ])
    )
    working_df = working_df.join(group_stats, on=group_col, how="left")
    working_df = working_df.with_columns([
        (
            (pl.col("label") == 0)
            & pl.col("semi_group_best_target").is_not_null()
            & pl.col("semi_group_best_decoy").is_not_null()
            & (pl.col(score_col) >= pl.col("semi_group_best_target") - hard_margin)
        ).cast(pl.Int8).alias("semi_is_hard_decoy"),
        (
            (pl.col("label") == 0)
            & pl.col("semi_group_best_decoy").is_not_null()
            & (pl.col(score_col) >= pl.col("semi_group_best_decoy") - 1e-12)
        ).cast(pl.Int8).alias("semi_is_top_decoy"),
    ])
    if group_col == "__semi_group_key":
        working_df = working_df.drop(group_col)
    return working_df


def sample_semisupervised_from_scored_df(
    scored_df: pl.DataFrame,
    feature_cols: List[str],
    max_rows: int,
    neg_pos_ratio: float,
    q_threshold: float,
    seed: int,
    high_score_decoy_frac: float,
    hard_decoy_frac: float,
    hard_margin: float,
) -> Tuple[pl.DataFrame, Dict]:
    """Build a cleaner training set from noisy target/decoy labels.

    Positive samples are only high-confidence targets whose instrument-level
    q-value is <= q_threshold. Negative samples are decoys, enriched with
    high-scoring decoys and same-scan hard decoys.
    """
    if scored_df.height == 0:
        return scored_df, {"input_rows": 0, "sampled_rows": 0}

    scored_df = add_q_value_by_instrument(
        scored_df,
        score_col="semi_score",
        q_col="semi_q_value",
        conservative_tdc=True,
    )
    scored_df = add_group_hard_flags(scored_df, score_col="semi_score", hard_margin=hard_margin)

    positives_all = scored_df.filter((pl.col("label") == 1) & (pl.col("semi_q_value") <= q_threshold))
    decoys = scored_df.filter(pl.col("label") == 0)
    high_decoys = decoys.sort("semi_score", descending=True)
    hard_decoys = decoys.filter(pl.col("semi_is_hard_decoy") == 1).sort("semi_score", descending=True)
    top_decoys = decoys.filter(pl.col("semi_is_top_decoy") == 1).sort("semi_score", descending=True)
    normal_decoys = decoys.filter(pl.col("semi_is_hard_decoy") != 1)

    pseudo_positive_candidates = int(positives_all.height)
    hard_decoy_candidates = int(hard_decoys.height)
    top_decoy_candidates = int(top_decoys.height)
    high_score_decoy_candidates = int(high_decoys.height)

    max_pos = int(max_rows / (1.0 + neg_pos_ratio)) if neg_pos_ratio > 0 else max_rows
    take_pos = min(positives_all.height, max(1, max_pos)) if positives_all.height > 0 else 0

    if take_pos == 0:
        empty = scored_df.head(0)
        empty = normalize_part_schema_v1(empty, feature_cols=feature_cols, keep_id_cols=False)
        return empty, {
            "input_rows": int(scored_df.height),
            "pseudo_positive_candidates": 0,
            "sampled_rows": 0,
            "sampled_positive_rows": 0,
            "sampled_decoy_rows": 0,
            "target_neg_rows": 0,
            "hard_decoy_candidates": hard_decoy_candidates,
            "top_decoy_candidates": top_decoy_candidates,
            "high_score_decoy_candidates": high_score_decoy_candidates,
            "q_threshold": float(q_threshold),
            "warning": "no high-confidence targets were found; increase --semi-target-fdr or inspect stage1 scores",
        }

    rng_seed = int(seed)
    positives = (
        positives_all.sample(n=take_pos, seed=rng_seed + 11, shuffle=True)
        if positives_all.height > take_pos
        else positives_all
    )

    target_neg = min(decoys.height, max_rows - take_pos, int(take_pos * neg_pos_ratio))
    selected_decoys = []
    selected_ids = set()

    def append_decoy_part(part: pl.DataFrame, n: int) -> None:
        if n <= 0 or part.height == 0:
            return
        sub = part.head(n)
        selected_decoys.append(sub)
        if "__row_id" in sub.columns:
            selected_ids.update(sub["__row_id"].to_list())

    hard_take = min(hard_decoys.height, int(target_neg * hard_decoy_frac))
    append_decoy_part(hard_decoys, hard_take)

    if selected_ids and "__row_id" in high_decoys.columns:
        high_decoys = high_decoys.filter(~pl.col("__row_id").is_in(list(selected_ids)))
    high_take = min(high_decoys.height, max(0, target_neg - sum(part.height for part in selected_decoys)), int(target_neg * high_score_decoy_frac))
    append_decoy_part(high_decoys, high_take)

    selected_decoy_rows = sum(part.height for part in selected_decoys)
    remaining_neg = max(0, target_neg - selected_decoy_rows)
    if selected_ids and "__row_id" in normal_decoys.columns:
        normal_decoys = normal_decoys.filter(~pl.col("__row_id").is_in(list(selected_ids)))
    if remaining_neg > 0 and normal_decoys.height > 0:
        n = min(remaining_neg, normal_decoys.height)
        selected_decoys.append(
            normal_decoys.sample(n=n, seed=rng_seed + 29, shuffle=True)
            if normal_decoys.height > n
            else normal_decoys
        )

    parts = [positives]
    parts.extend(selected_decoys)
    sampled = pl.concat(parts, how="vertical", rechunk=False)
    if sampled.height > max_rows:
        sampled = sampled.sample(n=max_rows, seed=rng_seed + 41, shuffle=True)
    elif sampled.height > 0:
        sampled = sampled.sample(n=sampled.height, seed=rng_seed + 43, shuffle=True)

    helper_cols = [
        "semi_score",
        "semi_q_value",
        "semi_group_best_target",
        "semi_group_best_decoy",
        "semi_decoy_minus_target",
        "semi_is_hard_decoy",
        "semi_is_top_decoy",
        "__row_id",
    ]
    sampled = sampled.drop([col for col in helper_cols if col in sampled.columns])
    sampled = normalize_part_schema_v1(sampled, feature_cols=feature_cols, keep_id_cols=False)

    summary = {
        "input_rows": int(scored_df.height),
        "pseudo_positive_candidates": pseudo_positive_candidates,
        "sampled_rows": int(sampled.height),
        "sampled_positive_rows": int((sampled["label"] == 1).sum()) if sampled.height else 0,
        "sampled_decoy_rows": int((sampled["label"] == 0).sum()) if sampled.height else 0,
        "target_neg_rows": int(target_neg),
        "hard_decoy_candidates": hard_decoy_candidates,
        "top_decoy_candidates": top_decoy_candidates,
        "high_score_decoy_candidates": high_score_decoy_candidates,
        "q_threshold": float(q_threshold),
        "hard_margin": float(hard_margin),
        "hard_decoy_frac": float(hard_decoy_frac),
        "high_score_decoy_frac": float(high_score_decoy_frac),
    }
    if "instrument" in scored_df.columns:
        summary["pseudo_positive_by_instrument"] = value_counts_dict(positives_all, "instrument")
        summary["sampled_by_instrument"] = value_counts_dict(sampled, "instrument")
    return sampled, summary


# ============================================================
# 3. LightGBM 参数
# ============================================================

def params_from_args(args, fold_seed: int) -> Dict:
    return {
        "objective": "binary",
        "metric": "None",
        "boosting_type": "gbdt",

        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,

        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,

        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,

        "max_bin": args.max_bin,
        "verbosity": -1,
        "seed": fold_seed,
        "num_threads": args.num_threads,
    }


def train_lgbm_stage(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    feature_cols: List[str],
    args,
    fold_seed: int,
    stage_name: str,
    model_path: Path,
    params_path: Optional[Path] = None,
) -> Tuple[lgb.Booster, int, np.ndarray, Dict, pd.DataFrame, np.ndarray]:
    print()
    print(f"========== 转换为 LightGBM 输入: {stage_name} ==========")

    X_train, y_train = polars_to_pandas_xy(train_df, feature_cols)
    X_valid, y_valid = polars_to_pandas_xy(valid_df, feature_cols)
    X_train = mask_rt_qvalue_anomaly_features(X_train, train_df, feature_cols, context=f"{stage_name} train")
    X_valid = mask_rt_qvalue_anomaly_features(X_valid, valid_df, feature_cols, context=f"{stage_name} valid")

    categorical_features = [col for col in CATEGORICAL_FEATURES if col in feature_cols]
    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("categorical_features:", categorical_features)

    lgb_train = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_cols,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    lgb_valid = lgb.Dataset(
        X_valid,
        label=y_valid,
        feature_name=feature_cols,
        categorical_feature=categorical_features,
        reference=lgb_train,
        free_raw_data=False,
    )

    params = params_from_args(args, fold_seed)
    if params_path is not None:
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

    print()
    print(f"========== 开始训练 LightGBM: {stage_name} ==========")
    fdr_stopper = UniquePeptideFdrEarlyStopping(
        valid_features=X_valid,
        valid_meta=FdrEvalMetadata.from_frame(
            valid_df,
            fdr_threshold=args.fdr_threshold,
            conservative_tdc=not args.non_conservative_zero_decoy,
        ),
        stopping_rounds=args.early_stopping_rounds,
        eval_period=args.fdr_eval_period,
        min_delta=args.fdr_min_delta,
    )

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=args.num_boost_round,
        valid_sets=[lgb_valid],
        valid_names=["valid"],
        callbacks=[fdr_stopper],
    )

    best_iteration = resolve_best_iteration(model, fdr_stopper, args.num_boost_round)
    model.save_model(str(model_path), num_iteration=best_iteration)
    valid_pred = model.predict(X_valid, num_iteration=best_iteration)

    stage_metrics = {
        "stage": stage_name,
        "model_path": str(model_path),
        "best_iteration": int(best_iteration),
        "early_stopping_metric": fdr_stopper.metric_name,
        "early_stopping_best_score": int(fdr_stopper.best_score) if np.isfinite(fdr_stopper.best_score) else None,
        "early_stopping_best_metric": fdr_stopper.best_metric,
        "fdr_threshold": float(args.fdr_threshold),
        "fdr_eval_period": int(args.fdr_eval_period),
        "fdr_conservative_tdc": bool(not args.non_conservative_zero_decoy),
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "positive_rate_train": float(np.mean(y_train)) if len(y_train) else None,
        "positive_rate_valid": float(np.mean(y_valid)) if len(y_valid) else None,
        "basic_metrics": compute_basic_metrics(y_valid, valid_pred),
        "sampled_fdr_metrics": compute_sampled_fdr_metric(
            valid_df,
            valid_pred,
            conservative_tdc=not args.non_conservative_zero_decoy,
        ),
        "fdr_training_history": fdr_stopper.history,
    }

    del lgb_train
    del lgb_valid
    del X_valid
    gc.collect()

    return model, best_iteration, valid_pred, stage_metrics, X_train, y_train


# ============================================================
# 4. 单 fold 训练
# ============================================================

def train_one_fold(
    fold_id: int,
    manifest: Dict,
    feature_cols: List[str],
    args,
    out_dir: Path,
) -> None:
    fold_dir = out_dir / f"fold_{fold_id}"
    model_path = fold_dir / "model.txt"

    if model_path.exists() and args.skip_existing:
        print(f"fold_{fold_id} 已存在，跳过: {model_path}")
        return

    fold_dir.mkdir(parents=True, exist_ok=True)

    valid_items = [
        x for x in manifest["files"]
        if int(x["fold"]) == fold_id
    ]

    train_items = [
        x for x in manifest["files"]
        if int(x["fold"]) != fold_id
    ]

    train_files = [Path(x["path"]) for x in train_items]
    valid_files = [Path(x["path"]) for x in valid_items]

    if not train_files or not valid_files:
        raise RuntimeError(f"fold_{fold_id} train/valid 文件为空")

    fold_seed = args.seed + fold_id * 1000

    with open(fold_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    with open(fold_dir / "train_files.json", "w", encoding="utf-8") as f:
        json.dump([str(p) for p in train_files], f, indent=2, ensure_ascii=False)

    with open(fold_dir / "valid_files.json", "w", encoding="utf-8") as f:
        json.dump([str(p) for p in valid_files], f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print(f"开始训练 fold_{fold_id}")
    print(f"train files: {len(train_files)}")
    print(f"valid files: {len(valid_files)}")
    print(f"fold_dir: {fold_dir}")

    # --------------------------------------------------------
    # 采样训练集
    # --------------------------------------------------------
    print()
    print("========== 采样 fold train ==========")

    train_df = collect_sample_dataset_balanced_v1(
        files=train_files,
        feature_cols=feature_cols,
        max_total_rows=args.train_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=fold_seed,
        keep_id_cols=True,
    )

    # --------------------------------------------------------
    # 采样验证集
    # --------------------------------------------------------
    print()
    print("========== 采样 fold valid ==========")

    valid_df = collect_sample_dataset_balanced_v1(
        files=valid_files,
        feature_cols=feature_cols,
        max_total_rows=args.valid_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=fold_seed + 10000,
        keep_id_cols=True,
    )

    print()
    print("========== fold 采样分布检查 ==========")

    if "instrument" in train_df.columns:
        print("train instrument counts:")
        print(train_df.group_by("instrument").len().sort("instrument"))
    else:
        print("警告：train_df 中没有 instrument 列")

    if "instrument" in valid_df.columns:
        print("valid instrument counts:")
        print(valid_df.group_by("instrument").len().sort("instrument"))
    else:
        print("警告：valid_df 中没有 instrument 列")

    print("train label counts:")
    print(train_df.group_by("label").len().sort("label"))

    print("valid label counts:")
    print(valid_df.group_by("label").len().sort("label"))

    # --------------------------------------------------------
    # Stage 1: 原始标签 baseline 训练
    # --------------------------------------------------------
    stage1_model_path = fold_dir / "stage1_model.txt"
    stage1_params_path = fold_dir / "stage1_params.json"
    stage1_model, stage1_best_iteration, stage1_valid_pred, stage1_metrics, stage1_X_train, _ = train_lgbm_stage(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        args=args,
        fold_seed=fold_seed,
        stage_name=f"fold_{fold_id}_stage1_raw_label",
        model_path=stage1_model_path,
        params_path=stage1_params_path,
    )

    print()
    print("========== Stage 1: 给训练采样集打分并按仪器计算 q-value ==========")
    train_stage1_pred = stage1_model.predict(stage1_X_train, num_iteration=stage1_best_iteration)
    scored_train_df = train_df.with_row_count("__row_id").with_columns([
        pl.Series("semi_score", train_stage1_pred).cast(pl.Float32)
    ])
    del stage1_X_train
    del train_stage1_pred
    del stage1_model
    gc.collect()

    # --------------------------------------------------------
    # Stage 2: 半监督采样后重训最终 fold 模型
    # --------------------------------------------------------
    print()
    print("========== Stage 2: 半监督筛选高置信 target + hard decoy ==========")
    semi_train_df, semi_summary = sample_semisupervised_from_scored_df(
        scored_df=scored_train_df,
        feature_cols=feature_cols,
        max_rows=args.train_max_rows,
        neg_pos_ratio=args.semi_neg_pos_ratio,
        q_threshold=args.semi_target_fdr,
        seed=fold_seed + 20000,
        high_score_decoy_frac=args.semi_high_score_decoy_frac,
        hard_decoy_frac=args.semi_hard_decoy_frac,
        hard_margin=args.semi_hard_margin,
    )
    print("semi-supervised sampling summary:")
    print(json.dumps(semi_summary, indent=2, ensure_ascii=False))
    if semi_train_df.height == 0:
        raise RuntimeError(f"fold_{fold_id} 半监督采样为空，请调大 --semi-target-fdr 或检查 stage1 分数")
    semi_label_counts = value_counts_dict(semi_train_df, "label")
    if semi_label_counts.get("1", 0) == 0 or semi_label_counts.get("0", 0) == 0:
        raise RuntimeError(
            f"fold_{fold_id} 半监督采样后不是二分类数据: {semi_label_counts}; "
            "请调大 --semi-target-fdr、增大采样量或检查 stage1 分数"
        )
    print("semi train label counts:")
    print(semi_train_df.group_by("label").len().sort("label"))

    model, best_iteration, valid_pred, final_stage_metrics, _, _ = train_lgbm_stage(
        train_df=semi_train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        args=args,
        fold_seed=fold_seed + 50000,
        stage_name=f"fold_{fold_id}_stage2_semisupervised",
        model_path=model_path,
        params_path=fold_dir / "params.json",
    )

    print()
    print(f"fold_{fold_id} 最终半监督模型已保存: {model_path}")
    print("best_iteration:", best_iteration)

    # --------------------------------------------------------
    # 验证
    # --------------------------------------------------------
    print()
    print("========== 验证集评估 ==========")

    y_train = semi_train_df["label"].to_numpy()
    y_valid = valid_df["label"].to_numpy()
    basic_metrics = final_stage_metrics["basic_metrics"]
    sampled_fdr_metrics = final_stage_metrics["sampled_fdr_metrics"]

    metrics = {
        "fold": int(fold_id),
        "training_mode": "semisupervised_two_stage_withoutRT",
        "stage1": {k: v for k, v in stage1_metrics.items() if k != "fdr_training_history"},
        "stage2": {k: v for k, v in final_stage_metrics.items() if k != "fdr_training_history"},
        "semi_sampling_summary": semi_summary,
        "basic_metrics": basic_metrics,
        "sampled_fdr_metrics": sampled_fdr_metrics,
        "best_iteration": int(best_iteration),
        "early_stopping_metric": final_stage_metrics["early_stopping_metric"],
        "early_stopping_best_score": final_stage_metrics["early_stopping_best_score"],
        "early_stopping_best_metric": final_stage_metrics["early_stopping_best_metric"],
        "fdr_threshold": float(args.fdr_threshold),
        "fdr_eval_period": int(args.fdr_eval_period),
        "fdr_conservative_tdc": bool(not args.non_conservative_zero_decoy),
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "positive_rate_train": float(np.mean(y_train)) if len(y_train) else None,
        "positive_rate_valid": float(np.mean(y_valid)) if len(y_valid) else None,

        "train_instrument_counts": value_counts_dict(semi_train_df, "instrument"),
        "valid_instrument_counts": value_counts_dict(valid_df, "instrument"),
        "train_label_counts": value_counts_dict(semi_train_df, "label"),
        "valid_label_counts": value_counts_dict(valid_df, "label"),
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(fold_dir / "stage1_fdr_training_history.json", "w", encoding="utf-8") as f:
        json.dump(stage1_metrics["fdr_training_history"], f, indent=2, ensure_ascii=False)
    with open(fold_dir / "fdr_training_history.json", "w", encoding="utf-8") as f:
        json.dump(final_stage_metrics["fdr_training_history"], f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 特征重要性
    # --------------------------------------------------------
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    importance.to_csv(fold_dir / "feature_importance.csv", index=False)

    # --------------------------------------------------------
    # 保存采样验证集预测
    # --------------------------------------------------------
    pred_df = valid_df.select([
        c for c in ID_COLUMNS_FOR_VALID + ["label"]
        if c in valid_df.columns
    ]).with_columns([
        pl.Series("score", valid_pred).cast(pl.Float32)
    ])

    pred_df.write_parquet(fold_dir / "valid_sample_pred.parquet")

    print()
    print(f"metrics 已保存: {fold_dir / 'metrics.json'}")
    print(f"特征重要性已保存: {fold_dir / 'feature_importance.csv'}")
    print(f"验证集采样预测已保存: {fold_dir / 'valid_sample_pred.parquet'}")
    print(f"fold_{fold_id} 训练完成")

    # --------------------------------------------------------
    # 清理内存
    # --------------------------------------------------------
    del train_df
    del valid_df
    del scored_train_df
    del semi_train_df
    del stage1_valid_pred
    del valid_pred
    del model
    gc.collect()


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="/root/autodl-tmp/datasets/aipc",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="~/aipc/models/lgbm_v1_oof",
    )

    parser.add_argument(
        "--only-instrument",
        choices=INSTRUMENTS,
        default=None,
        help="Only use one instrument split for v1 training. Default keeps the existing all-instrument flow.",
    )

    parser.add_argument(
        "--folds-json",
        type=str,
        default="",
        help="可选：已有 folds.json 路径。如果为空，默认使用 out_dir/folds.json。",
    )

    parser.add_argument(
        "--reuse-folds",
        action="store_true",
        help="如果 folds.json 已存在，是否复用。",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--balance-by-rows",
        action="store_true",
        help="fold 划分时是否按文件行数均衡。强烈建议开启。",
    )

    parser.add_argument(
        "--only-fold",
        type=int,
        default=None,
        help="只训练某一个 fold，调试时推荐先用 --only-fold 0。",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果 fold/model.txt 已存在，是否跳过该 fold。",
    )

    parser.add_argument(
        "--train-max-rows",
        type=int,
        default=3_000_000,
    )

    parser.add_argument(
        "--valid-max-rows",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=80_000,
    )

    parser.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--semi-target-fdr",
        type=float,
        default=0.01,
        help="半监督采样时，高置信 target 的 q-value 阈值。可尝试 0.01 或 0.05。",
    )

    parser.add_argument(
        "--semi-neg-pos-ratio",
        type=float,
        default=2.0,
        help="半监督重训数据中的 decoy:target 采样比例。",
    )

    parser.add_argument(
        "--semi-high-score-decoy-frac",
        type=float,
        default=0.30,
        help="半监督 decoy 样本中优先选择高分 decoy 的目标占比。",
    )

    parser.add_argument(
        "--semi-hard-decoy-frac",
        type=float,
        default=0.50,
        help="半监督 decoy 样本中优先选择同 scan hard decoy 的目标占比。",
    )

    parser.add_argument(
        "--semi-hard-margin",
        type=float,
        default=0.02,
        help="同 scan 内 decoy 分数 >= best_target_score - margin 时视作 hard decoy。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260519,
    )

    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.01,
        help="早停主指标使用的 FDR 阈值。比赛主目标固定为 0.01。",
    )

    parser.add_argument(
        "--fdr-eval-period",
        type=int,
        default=50,
        help="每多少轮计算一次 unique peptide @ FDR；全量排序较慢，不建议设为 1。",
    )

    parser.add_argument(
        "--fdr-min-delta",
        type=float,
        default=0.0,
        help="unique peptide @ FDR 至少提升多少才算早停改进。",
    )

    parser.add_argument(
        "--non-conservative-zero-decoy",
        action="store_true",
        help="使用 cum_decoy/cum_target；默认使用 max(cum_decoy, 1)/cum_target，更贴近官方工具。",
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--num-leaves",
        type=int,
        default=127,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--feature-fraction",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--bagging-fraction",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--lambda-l2",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--max-bin",
        type=int,
        default=255,
    )

    args = parser.parse_args()

    if args.n_folds < 2:
        raise ValueError("--n-folds 必须 >= 2")
    if not (0.0 < args.semi_target_fdr <= 1.0):
        raise ValueError("--semi-target-fdr 必须在 (0, 1] 内")
    if args.semi_neg_pos_ratio <= 0:
        raise ValueError("--semi-neg-pos-ratio 必须 > 0")
    if not (0.0 <= args.semi_high_score_decoy_frac <= 1.0):
        raise ValueError("--semi-high-score-decoy-frac 必须在 [0, 1] 内")
    if not (0.0 <= args.semi_hard_decoy_frac <= 1.0):
        raise ValueError("--semi-hard-decoy-frac 必须在 [0, 1] 内")
    if args.semi_high_score_decoy_frac + args.semi_hard_decoy_frac > 1.0:
        print("警告：--semi-high-score-decoy-frac + --semi-hard-decoy-frac > 1，后续会按剩余可采样数量截断")

    data_root = Path(args.data_root)
    split_root = data_root / "processed_split"
    train_root = split_root / "train"

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== 路径 ==========")
    print("data_root:", data_root)
    print("train_root:", train_root)
    print("out_dir:", out_dir)
    print("only_instrument:", args.only_instrument or "all")

    train_files = list_split_files(train_root)
    if args.only_instrument is not None:
        train_files = split_files_by_instrument(train_files)[args.only_instrument]

    print()
    print(f"train files: {len(train_files)}")

    if len(train_files) == 0:
        raise RuntimeError(
            f"未找到 train parquet: {train_root}, only_instrument={args.only_instrument}"
        )

    feature_cols = resolve_feature_columns(train_files)

    print()
    print("========== v1 使用特征 ==========")
    print(f"feature count: {len(feature_cols)}")
    for c in feature_cols:
        print(c)

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    manifest = load_or_create_manifest(
        args=args,
        train_root=train_root,
        train_files=train_files,
        out_dir=out_dir,
    )

    fold_ids = list(range(int(manifest["n_folds"])))

    if args.only_fold is not None:
        if args.only_fold not in fold_ids:
            raise ValueError(f"--only-fold 必须在 {fold_ids} 内")

        fold_ids = [args.only_fold]

    for fold_id in fold_ids:
        train_one_fold(
            fold_id=fold_id,
            manifest=manifest,
            feature_cols=feature_cols,
            args=args,
            out_dir=out_dir,
        )

    print()
    print("全部 OOF v1 fold 模型训练完成")
    print(f"folds.json: {out_dir / 'folds.json'}")
    print(f"feature_columns.json: {out_dir / 'feature_columns.json'}")


if __name__ == "__main__":
    main()
