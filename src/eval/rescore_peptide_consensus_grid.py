#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Peptide-consensus rescoring for AIPC validation and Basic submissions.

This is deliberately different from exact peptide lookup.  It uses only
prediction-time evidence available in both validation and Basic:

    - base PSM score
    - repeated evidence for the same cleaned peptide inside the same file
    - the rank of repeated PSMs of the same peptide

The official objective is unique peptide count at 1% FDR, so a useful
encoding should promote peptides that have multiple independent high-score
observations while demoting duplicate PSMs that do not add new unique peptides.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

DEFAULT_CPU_THREADS = 16
for _env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "ARROW_NUM_THREADS",
):
    os.environ.setdefault(_env_name, str(DEFAULT_CPU_THREADS))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import pyarrow as pa
except Exception:  # pragma: no cover
    pa = None

from tqdm import tqdm

from offline_leaderboard import (
    BASE_EVAL_COLUMNS,
    PRIMARY_METRIC,
    SCORED_EVAL_COLUMNS,
    EvalConfig,
    PredictionInput,
    add_leaderboard_deltas,
    aggregate_by_instrument,
    aggregate_metrics,
    clean_peptide_sequence,
    compute_one_file_metrics,
    discover_scored_parquets,
    existing_columns,
    infer_instrument,
    is_candidate_parquet,
    parquet_columns,
    standardize_eval_frame,
    write_outputs,
)


EPS = 1e-12


def configure_cpu_threads(num_threads: int) -> None:
    if num_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "ARROW_NUM_THREADS",
    ):
        os.environ[env_name] = str(num_threads)
    if pa is not None:
        pa.set_cpu_count(num_threads)
        pa.set_io_thread_count(num_threads)


def parse_float_list(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError(f"empty float list: {text!r}")
    return values


def parse_int_list(text: str) -> List[int]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError(f"empty int list: {text!r}")
    return values


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value).strip("_")


def num_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    pos = values >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-values[pos]))
    exp_values = np.exp(values[~pos])
    out[~pos] = exp_values / (1.0 + exp_values)
    return out


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def transform_score(score: pd.Series | np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    if method == "identity":
        return values
    if method == "logit":
        return logit(values)
    if method == "zscore":
        finite = np.isfinite(values)
        out = np.zeros(values.shape, dtype=np.float64)
        if finite.any():
            valid = values[finite]
            std = float(valid.std())
            out[finite] = valid - float(valid.mean()) if std <= EPS else (valid - float(valid.mean())) / std
        return out
    if method == "rank_pct":
        ranks = pd.Series(values).rank(method="first", ascending=True).to_numpy(dtype=np.float64)
        return ranks / max(len(ranks) - 1, 1)
    raise ValueError(f"unsupported score transform: {method}")


def infer_sequence_series(df: pd.DataFrame) -> pd.Series:
    for column in ("modified_sequence", "precursor_sequence", "peptide", "peptide_key"):
        if column in df.columns:
            return df[column].fillna("").astype(str)
    raise RuntimeError("missing peptide sequence column")


def read_parquet_existing(path: Path, wanted: Iterable[str]) -> pd.DataFrame:
    columns = existing_columns(path, wanted)
    if not columns:
        raise RuntimeError(f"No requested columns found in parquet: {path}")
    return pq.read_table(path, columns=columns).to_pandas()


def _scan_key_series(df: pd.DataFrame) -> pd.Series:
    if "top1_key" in df.columns:
        return df["top1_key"].fillna("").astype(str)
    if "scan_number" in df.columns:
        return df["scan_number"].fillna("").astype(str)
    if "group_key" in df.columns:
        return df["group_key"].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def build_consensus_features(
    df: pd.DataFrame,
    base_score: np.ndarray,
    topk: int,
    sequence_col: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    if topk <= 0:
        raise ValueError("topk must be positive")

    if sequence_col is not None and sequence_col in df.columns:
        sequence = df[sequence_col].fillna("").astype(str)
    else:
        sequence = infer_sequence_series(df)
    cleaned_sequence = clean_peptide_sequence(sequence)

    if "file_id" in df.columns:
        file_id = df["file_id"].fillna("").astype(str)
    else:
        file_id = pd.Series(["__single_file__"] * len(df), index=df.index)

    work = pd.DataFrame(
        {
            "__row_id": np.arange(len(df), dtype=np.int64),
            "file_id": file_id.to_numpy(),
            "cleaned_sequence": cleaned_sequence.to_numpy(),
            "scan_key": _scan_key_series(df).to_numpy(),
            "base_score": np.asarray(base_score, dtype=np.float64),
        }
    )
    work = work.sort_values(
        ["file_id", "cleaned_sequence", "base_score"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    group_cols = ["file_id", "cleaned_sequence"]
    work["peptide_rank0"] = work.groupby(group_cols, sort=False).cumcount()
    top = work[work["peptide_rank0"] < int(topk)].copy()
    top_prob = np.clip(sigmoid(top["base_score"].to_numpy(dtype=np.float64)), EPS, 1.0 - EPS)
    top["log_survival"] = np.log1p(-top_prob)
    top["prob"] = top_prob

    agg = top.groupby(group_cols, sort=False).agg(
        peptide_topk_count=("base_score", "size"),
        peptide_scan_count=("scan_key", "nunique"),
        peptide_top1_score=("base_score", "max"),
        peptide_mean_topk_score=("base_score", "mean"),
        peptide_sum_topk_prob=("prob", "sum"),
        peptide_log_survival=("log_survival", "sum"),
    )
    noisy_prob = 1.0 - np.exp(np.clip(agg["peptide_log_survival"].to_numpy(dtype=np.float64), -100.0, 0.0))
    agg["peptide_noisy_or_logit"] = logit(noisy_prob)
    agg = agg.drop(columns=["peptide_log_survival"]).reset_index()
    work = work.merge(agg, on=group_cols, how="left", validate="many_to_one")
    work = work.sort_values("__row_id", kind="mergesort")

    out = {
        "peptide_rank0": work["peptide_rank0"].to_numpy(dtype=np.float64),
        "peptide_topk_count": work["peptide_topk_count"].to_numpy(dtype=np.float64),
        "peptide_scan_count": work["peptide_scan_count"].to_numpy(dtype=np.float64),
        "peptide_top1_score": work["peptide_top1_score"].to_numpy(dtype=np.float64),
        "peptide_mean_topk_score": work["peptide_mean_topk_score"].to_numpy(dtype=np.float64),
        "peptide_sum_topk_prob": work["peptide_sum_topk_prob"].to_numpy(dtype=np.float64),
        "peptide_noisy_or_logit": work["peptide_noisy_or_logit"].to_numpy(dtype=np.float64),
    }
    return out


def consensus_score(
    base_score: np.ndarray,
    features: Dict[str, np.ndarray],
    alpha: float,
    beta: float,
    gamma: float,
    mode: str,
) -> np.ndarray:
    if mode == "noisy_or":
        evidence = features["peptide_noisy_or_logit"]
    elif mode == "top1":
        evidence = features["peptide_top1_score"]
    elif mode == "mean_topk":
        evidence = features["peptide_mean_topk_score"]
    else:
        raise ValueError(f"unsupported consensus mode: {mode}")

    support = np.log1p(np.maximum(features["peptide_scan_count"], 1.0) - 1.0)
    repeat_penalty = np.log1p(features["peptide_rank0"])
    return (
        np.asarray(base_score, dtype=np.float64)
        + float(beta) * (np.asarray(evidence, dtype=np.float64) - np.asarray(base_score, dtype=np.float64))
        + float(gamma) * support
        - float(alpha) * repeat_penalty
    )


def model_name(mode: str, topk: int, alpha: float, beta: float, gamma: float) -> str:
    return (
        f"consensus_{safe_name(mode)}"
        f"_k{topk}_a{num_tag(alpha)}_b{num_tag(beta)}_g{num_tag(gamma)}"
    )


def evaluate_grid(
    pred_root: Path,
    topk_grid: List[int],
    alpha_grid: List[float],
    beta_grid: List[float],
    gamma_grid: List[float],
    mode_grid: List[str],
    score_transform: str,
    config: EvalConfig,
    out_dir: Path,
) -> pd.DataFrame:
    prediction = PredictionInput("benchmark", pred_root)
    parquet_files = discover_scored_parquets(prediction.path)
    if not parquet_files:
        raise RuntimeError(f"no scored parquet files found: {pred_root}")

    combos: List[Tuple[str, int, float, float, float]] = []
    for mode in mode_grid:
        for topk in topk_grid:
            for alpha in alpha_grid:
                for beta in beta_grid:
                    for gamma in gamma_grid:
                        combos.append((mode, int(topk), float(alpha), float(beta), float(gamma)))
    if not combos:
        raise RuntimeError("empty grid")

    metrics_by_combo: Dict[Tuple[str, int, float, float, float], List[Dict]] = {
        combo: [] for combo in combos
    }
    feature_cache: Dict[int, Dict[str, np.ndarray]] = {}

    for parquet_path in tqdm(parquet_files, desc="validation parquet"):
        raw = read_parquet_existing(parquet_path, SCORED_EVAL_COLUMNS)
        standardized = standardize_eval_frame(raw, parquet_path.stem, config)
        base_score = transform_score(standardized["score"].to_numpy(), score_transform)
        feature_cache.clear()

        for combo in combos:
            mode, topk, alpha, beta, gamma = combo
            if topk not in feature_cache:
                feature_cache[topk] = build_consensus_features(
                    standardized,
                    base_score,
                    topk=topk,
                    sequence_col="modified_sequence",
                )
            final_score = consensus_score(
                base_score,
                feature_cache[topk],
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                mode=mode,
            )
            scored = standardized.copy()
            scored["score"] = final_score
            for file_id, part in scored.groupby("file_id", sort=False):
                metrics_by_combo[combo].append(compute_one_file_metrics(part, str(file_id), config))

    summaries = []
    details = {}
    for combo in combos:
        mode, topk, alpha, beta, gamma = combo
        name = model_name(mode, topk, alpha, beta, gamma)
        summary, by_file = aggregate_metrics(metrics_by_combo[combo], name)
        summary["mode"] = mode
        summary["topk"] = int(topk)
        summary["alpha"] = float(alpha)
        summary["beta"] = float(beta)
        summary["gamma"] = float(gamma)
        summary["score_transform"] = score_transform
        summaries.append(summary)
        details[name] = {
            "by_file": by_file,
            "by_instrument": aggregate_by_instrument(by_file),
        }

    leaderboard = add_leaderboard_deltas(pd.DataFrame(summaries))
    meta = {
        "primary_metric": PRIMARY_METRIC,
        "method": (
            "base + beta*(peptide_consensus - base) "
            "+ gamma*log1p(peptide_scan_count-1) "
            "- alpha*log1p(rank_within_file_cleaned_peptide)"
        ),
        "score_transform": score_transform,
        "topk_grid": topk_grid,
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "gamma_grid": gamma_grid,
        "mode_grid": mode_grid,
        "pred_root": str(pred_root),
        "parquet_files": [str(path) for path in parquet_files],
    }
    write_outputs(out_dir, leaderboard, details, meta)
    return leaderboard


def discover_test_parquets(parquet_dir: Path) -> List[Path]:
    if parquet_dir.is_file():
        return [parquet_dir]
    files = sorted(path for path in parquet_dir.glob("*.parquet") if is_candidate_parquet(path))
    if not files:
        files = sorted(path for path in parquet_dir.rglob("*.parquet") if is_candidate_parquet(path))
    if not files:
        raise FileNotFoundError(f"no test parquet files found: {parquet_dir}")
    return files


def _read_aligned_model_score(df: pd.DataFrame, parquet_path: Path, model_pred_dir: Path) -> np.ndarray:
    pred_file = model_pred_dir / f"{parquet_path.stem}_pred.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"missing benchmark prediction file: {pred_file}")
    pred = pd.read_csv(pred_file, header=None, names=["index", "score"])
    pred["index"] = pd.to_numeric(pred["index"], errors="raise").astype(np.int64)
    if pred["index"].duplicated().any():
        raise RuntimeError(f"duplicated index in {pred_file}")
    aligned = pd.DataFrame(
        {"index": pd.to_numeric(df["index"], errors="raise").astype(np.int64)}
    ).merge(pred, on="index", how="left", validate="one_to_one")
    if aligned["score"].isna().any():
        missing_scores = int(aligned["score"].isna().sum())
        raise RuntimeError(f"{pred_file} missing {missing_scores} scores after index alignment")
    return pd.to_numeric(aligned["score"], errors="coerce").to_numpy()


def build_test_submission(
    parquet_dir: Path,
    score_col: str,
    model_pred_dir: Optional[Path],
    mode: str,
    topk: int,
    alpha: float,
    beta: float,
    gamma: float,
    score_transform: str,
    out_dir: Path,
    expected_rows: Optional[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = discover_test_parquets(parquet_dir)
    merged_parts = []
    total_rows = 0
    instruments = Counter()

    for parquet_path in tqdm(parquet_files, desc="write submission"):
        columns = parquet_columns(parquet_path)
        wanted = [
            "index",
            "file_id",
            "instrument",
            "scan_number",
            "group_key",
            "precursor_sequence",
            "peptide_key",
        ]
        if model_pred_dir is None:
            wanted.append(score_col)
        read_cols = [column for column in wanted if column in columns]
        missing = {"index"} - set(read_cols)
        if model_pred_dir is None:
            missing |= {score_col} - set(read_cols)
        if missing:
            raise RuntimeError(f"{parquet_path} missing required columns: {sorted(missing)}")

        df = pq.read_table(parquet_path, columns=read_cols).to_pandas()
        if "file_id" not in df.columns:
            df["file_id"] = parquet_path.stem
        if "instrument" not in df.columns:
            df["instrument"] = infer_instrument(parquet_path.name)

        if model_pred_dir is None:
            raw_score = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
            score_source = score_col
        else:
            raw_score = _read_aligned_model_score(df, parquet_path, model_pred_dir)
            score_source = str(model_pred_dir)

        base_score = transform_score(raw_score, score_transform)
        features = build_consensus_features(df, base_score, topk=topk)
        final_score = consensus_score(
            base_score,
            features,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            mode=mode,
        ).astype(np.float32)

        part = pd.DataFrame(
            {
                "index": pd.to_numeric(df["index"], errors="raise").astype(np.int64),
                "score": final_score,
            }
        )
        part.to_csv(out_dir / f"{parquet_path.stem}_pred.csv", header=False, index=False)
        merged_parts.append(part)
        total_rows += len(part)
        instruments.update(df["instrument"].fillna("unknown").astype(str).tolist())

    merged = pd.concat(merged_parts, ignore_index=True)
    if total_rows != len(merged):
        raise RuntimeError(f"internal row mismatch: parts={total_rows:,}, merged={len(merged):,}")
    if expected_rows is not None and len(merged) != expected_rows:
        raise RuntimeError(f"submission row mismatch: got {len(merged):,}, expected {expected_rows:,}")
    unique_index = int(merged["index"].nunique())
    if expected_rows is not None and unique_index != expected_rows:
        raise RuntimeError(f"submission unique index mismatch: got {unique_index:,}, expected {expected_rows:,}")

    all_pred = out_dir / "all_pred.tsv"
    merged.to_csv(all_pred, sep="\t", index=False)

    zip_path = out_dir / "all_pred.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(all_pred, arcname="all_pred.tsv")

    metadata = {
        "method": "peptide_consensus_rescore",
        "score_col": score_col if model_pred_dir is None else None,
        "model_pred_dir": str(model_pred_dir) if model_pred_dir is not None else None,
        "score_source": score_source,
        "mode": mode,
        "topk": int(topk),
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "score_transform": score_transform,
        "rows": int(len(merged)),
        "unique_index": unique_index,
        "score_min": float(merged["score"].min()),
        "score_max": float(merged["score"].max()),
        "score_mean": float(merged["score"].mean()),
        "instrument_row_counts": dict(instruments),
        "zip_path": str(zip_path),
    }
    with open(out_dir / "rescore_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search peptide-consensus rescoring.")
    parser.add_argument("--pred-root", help="Validation scored parquet file/dir to rescore and evaluate")
    parser.add_argument("--out-dir", required=True, help="Evaluation output directory")
    parser.add_argument("--topk-grid", default="1,2,3,5")
    parser.add_argument("--alpha-grid", default="0,1,2,3,4")
    parser.add_argument("--beta-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--gamma-grid", default="0,0.25,0.5,1")
    parser.add_argument("--mode-grid", default="noisy_or,top1,mean_topk")
    parser.add_argument("--score-transform", choices=["identity", "logit", "zscore", "rank_pct"], default="logit")
    parser.add_argument("--fdr-threshold", type=float, default=0.01)
    parser.add_argument("--pre-fdr-dedup", choices=["scan", "scan_precursor", "none"], default="scan")
    parser.add_argument("--top1-key", choices=["scan", "group"], default="scan")
    parser.add_argument("--non-conservative-zero-decoy", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--best-mode", default=None)
    parser.add_argument("--best-topk", type=int, default=None)
    parser.add_argument("--best-alpha", type=float, default=None)
    parser.add_argument("--best-beta", type=float, default=None)
    parser.add_argument("--best-gamma", type=float, default=None)
    parser.add_argument("--test-parquet-dir", help="Optional Basic test parquet dir for submission generation")
    parser.add_argument("--test-score-col", default="lgbm_v1_score")
    parser.add_argument("--test-model-pred-dir", help="Optional per-file *_pred.csv dir used as base score source for test submission")
    parser.add_argument("--submission-out-dir", help="Output directory for rescored Basic submission")
    parser.add_argument("--expected-test-rows", type=int, default=10_768_114)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cpu_threads(int(args.cpu_threads))
    config = EvalConfig(
        fdr_threshold=float(args.fdr_threshold),
        conservative_tdc=not args.non_conservative_zero_decoy,
        top1_key=args.top1_key,
        pre_fdr_dedup=args.pre_fdr_dedup,
    )
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {
        "mode": args.best_mode,
        "topk": args.best_topk,
        "alpha": args.best_alpha,
        "beta": args.best_beta,
        "gamma": args.best_gamma,
    }
    if args.pred_root:
        leaderboard = evaluate_grid(
            Path(args.pred_root).expanduser().resolve(),
            topk_grid=parse_int_list(args.topk_grid),
            alpha_grid=parse_float_list(args.alpha_grid),
            beta_grid=parse_float_list(args.beta_grid),
            gamma_grid=parse_float_list(args.gamma_grid),
            mode_grid=[item.strip() for item in args.mode_grid.split(",") if item.strip()],
            score_transform=args.score_transform,
            config=config,
            out_dir=out_dir,
        )
        best = leaderboard.iloc[0]
        for key in selected:
            if selected[key] is None:
                selected[key] = best[key]
        display_cols = [
            "rank",
            "model",
            "primary_score",
            "delta_vs_worst",
            "mode",
            "topk",
            "alpha",
            "beta",
            "gamma",
        ]
        print(leaderboard[display_cols].head(30).to_string(index=False))
        print(
            "\nBest params: "
            f"mode={selected['mode']}, topk={int(selected['topk'])}, "
            f"alpha={float(selected['alpha']):g}, beta={float(selected['beta']):g}, "
            f"gamma={float(selected['gamma']):g}"
        )

    missing_selected = [key for key, value in selected.items() if value is None]
    if missing_selected and (args.test_parquet_dir or args.submission_out_dir):
        raise RuntimeError(
            "Submission generation needs selected params; missing: "
            + ", ".join(missing_selected)
        )

    if args.test_parquet_dir or args.submission_out_dir:
        if not args.test_parquet_dir or not args.submission_out_dir:
            raise RuntimeError("--test-parquet-dir and --submission-out-dir must be used together")
        expected_rows = None if args.expected_test_rows <= 0 else int(args.expected_test_rows)
        build_test_submission(
            parquet_dir=Path(args.test_parquet_dir).expanduser().resolve(),
            score_col=args.test_score_col,
            model_pred_dir=Path(args.test_model_pred_dir).expanduser().resolve() if args.test_model_pred_dir else None,
            mode=str(selected["mode"]),
            topk=int(selected["topk"]),
            alpha=float(selected["alpha"]),
            beta=float(selected["beta"]),
            gamma=float(selected["gamma"]),
            score_transform=args.score_transform,
            out_dir=Path(args.submission_out_dir).expanduser().resolve(),
            expected_rows=expected_rows,
        )

    print(f"\nWrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
