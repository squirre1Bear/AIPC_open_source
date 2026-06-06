#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import numpy as np
import polars as pl


PRIMARY_METRIC = "official_like_unique_clean_peptide_at_1pct"


def clean_expr(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .fill_null("")
        .cast(pl.Utf8)
        .str.replace_all(r"n\[42\]", "")
        .str.replace_all(r"N\[.98\]", "N")
        .str.replace_all(r"Q\[.98\]", "Q")
        .str.replace_all(r"M\[15.99\]", "M")
        .str.replace_all(r"C\[57.02\]", "C")
    )


def stripped_expr(column: str) -> pl.Expr:
    return (
        clean_expr(column)
        .str.replace_all(r"\[[^\]]+\]", "")
        .str.replace_all(r"[^A-Z]", "")
    )


def logit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, 1e-8, 1.0 - 1e-8)
    return np.log(values / (1.0 - values))


def logit_expr(column: str) -> pl.Expr:
    clipped = pl.col(column).cast(pl.Float64).clip(1e-8, 1.0 - 1e-8)
    return (clipped / (1.0 - clipped)).log()


def discover_train_files(train_root: Path, max_files_per_instrument: Optional[int]) -> List[Path]:
    files: List[Path] = []
    for instrument in ("mzml", "tims", "wiff"):
        inst_files = sorted((train_root / instrument).glob("*.parquet"))
        if max_files_per_instrument is not None:
            inst_files = inst_files[: int(max_files_per_instrument)]
        files.extend(inst_files)
    return files


def build_lexicon_part(args: Tuple[str, str, str]) -> Tuple[str, int, int]:
    path_text, out_text, key_mode = args
    path = Path(path_text)
    out_path = Path(out_text)
    expr = clean_expr("precursor_sequence") if key_mode == "clean" else stripped_expr("precursor_sequence")
    frame = (
        pl.scan_parquet(path)
        .select(
            expr.alias("pep_key"),
            (pl.col("label") == 1).cast(pl.Int64).alias("target"),
            (pl.col("label") != 1).cast(pl.Int64).alias("decoy"),
        )
        .filter(pl.col("pep_key") != "")
        .group_by("pep_key")
        .agg(
            pl.sum("target").alias("target_count"),
            pl.sum("decoy").alias("decoy_count"),
            pl.len().alias("row_count"),
        )
        .collect(streaming=True)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    return str(path), int(frame.height), int(frame["row_count"].sum()) if frame.height else 0


def build_lexicon(
    train_root: Path,
    work_dir: Path,
    key_mode: str,
    max_files_per_instrument: Optional[int],
    workers: int,
) -> Path:
    files = discover_train_files(train_root, max_files_per_instrument)
    if not files:
        raise RuntimeError(f"no train files under {train_root}")
    part_dir = work_dir / f"lexicon_parts_{key_mode}"
    part_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(path), str(part_dir / f"{i:05d}_{path.stem}.parquet"), key_mode)
        for i, path in enumerate(files)
    ]
    print(f"build lexicon key_mode={key_mode} files={len(files)} workers={workers}", flush=True)
    done = 0
    if workers <= 1:
        for task in tasks:
            path, unique_count, row_count = build_lexicon_part(task)
            done += 1
            print(f"part {done}/{len(tasks)} unique={unique_count:,} rows={row_count:,} {path}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(build_lexicon_part, task) for task in tasks]
            for future in as_completed(futures):
                path, unique_count, row_count = future.result()
                done += 1
                print(f"part {done}/{len(tasks)} unique={unique_count:,} rows={row_count:,} {path}", flush=True)

    lex_path = work_dir / f"lexicon_{key_mode}.parquet"
    merged = (
        pl.scan_parquet(str(part_dir / "*.parquet"))
        .group_by("pep_key")
        .agg(
            pl.sum("target_count").alias("target_count"),
            pl.sum("decoy_count").alias("decoy_count"),
            pl.sum("row_count").alias("row_count"),
        )
        .with_columns(
            (
                ((pl.col("target_count") + 1.0) / (pl.col("decoy_count") + 1.0)).log()
            ).alias("lex_logodds"),
            (pl.col("target_count") > 0).alias("seen_target"),
            (pl.col("decoy_count") > 0).alias("seen_decoy"),
        )
        .collect(streaming=True)
    )
    merged.write_parquet(lex_path)
    print(
        f"lexicon merged rows={merged.height:,} target_seen={int(merged['seen_target'].sum()):,} "
        f"decoy_seen={int(merged['seen_decoy'].sum()):,} path={lex_path}",
        flush=True,
    )
    return lex_path


def q_values_for_sorted_labels(labels: np.ndarray) -> np.ndarray:
    is_target = labels == 1
    is_decoy = labels != 1
    cum_target = np.cumsum(is_target, dtype=np.int64)
    cum_decoy = np.cumsum(is_decoy, dtype=np.int64)
    fdr = np.maximum(cum_decoy, 1) / np.maximum(cum_target, 1)
    return np.minimum.accumulate(fdr[::-1])[::-1]


def compute_file_metric(part: pl.DataFrame, score_col: str) -> Dict[str, object]:
    if part.height == 0:
        return {"file_id": "", PRIMARY_METRIC: 0, "rows": 0}
    file_id = str(part["file_id"][0])
    instrument = str(part["instrument"][0]) if "instrument" in part.columns else "unknown"
    sorted_part = part.sort([score_col, "__row_id"], descending=[True, False])
    top1 = sorted_part.unique(subset=["scan_number"], keep="first", maintain_order=True)
    labels = top1["label"].to_numpy()
    q = q_values_for_sorted_labels(labels)
    accepted = top1.with_columns(pl.Series("q_value", q)).filter(pl.col("q_value") <= 0.01)
    accepted_targets = accepted.filter(pl.col("label") == 1)
    peptide_targets = accepted_targets.unique(subset=["cleaned_sequence"], keep="first", maintain_order=True)
    return {
        "file_id": file_id,
        "instrument": instrument,
        "rows": int(part.height),
        "top1_rows": int(top1.height),
        "accepted_target_psm_at_1pct": int(accepted_targets.height),
        "accepted_total_rows_at_1pct": int(accepted.height),
        "accepted_decoy_rows_at_1pct": int(accepted.height - accepted_targets.height),
        PRIMARY_METRIC: int(peptide_targets.height),
    }


def evaluate_scored_frame(df: pl.DataFrame, score_col: str) -> Tuple[int, List[Dict[str, object]]]:
    metrics: List[Dict[str, object]] = []
    for _, part in df.group_by("file_id", maintain_order=True):
        metrics.append(compute_file_metric(part, score_col))
    score = sum(int(m[PRIMARY_METRIC]) for m in metrics)
    return score, metrics


def evaluate_with_lexicon(
    valid_pred_root: Path,
    lex_path: Path,
    key_mode: str,
    weights: Sequence[float],
    out_dir: Path,
) -> Tuple[pl.DataFrame, Dict[str, List[Dict[str, object]]]]:
    valid_files = sorted(valid_pred_root.rglob("valid_sample_pred.parquet"))
    if not valid_files:
        raise RuntimeError(f"no valid_sample_pred.parquet under {valid_pred_root}")
    lex = pl.scan_parquet(lex_path).select(["pep_key", "lex_logodds", "target_count", "decoy_count"])
    summaries: List[Dict[str, object]] = []
    details: Dict[str, List[Dict[str, object]]] = {f"lex_w{w:g}": [] for w in weights}
    details["base_raw"] = []
    details["base_logit"] = []

    totals = {f"lex_w{w:g}": 0 for w in weights}
    totals["base_raw"] = 0
    totals["base_logit"] = 0
    coverage_rows = 0
    coverage_target_rows = 0
    total_rows = 0

    for valid_i, path in enumerate(valid_files, start=1):
        expr = clean_expr("precursor_sequence") if key_mode == "clean" else stripped_expr("precursor_sequence")
        df = (
            pl.scan_parquet(path)
            .with_row_index("__row_id")
            .select(
                "__row_id",
                pl.col("file_id").cast(pl.Utf8),
                pl.col("instrument").cast(pl.Utf8),
                pl.col("scan_number"),
                pl.col("precursor_sequence"),
                pl.col("label").cast(pl.Int8),
                pl.col("score").cast(pl.Float64).alias("base_raw"),
                clean_expr("precursor_sequence").alias("cleaned_sequence"),
                expr.alias("pep_key"),
            )
            .join(lex, on="pep_key", how="left")
            .with_columns(
                pl.col("lex_logodds").fill_null(0.0),
                pl.col("target_count").fill_null(0).cast(pl.Int64),
                pl.col("decoy_count").fill_null(0).cast(pl.Int64),
            )
            .with_columns(logit_expr("base_raw").alias("base_logit"))
            .collect(streaming=True)
        )
        total_rows += int(df.height)
        seen = (df["target_count"] + df["decoy_count"]) > 0
        coverage_rows += int(seen.sum())
        coverage_target_rows += int(df.filter(seen & (df["label"] == 1)).height)

        for score_col in ("base_raw", "base_logit"):
            score, metrics = evaluate_scored_frame(df, score_col)
            totals[score_col] += int(score)
            details[score_col].extend(metrics)
        for weight in weights:
            name = f"lex_w{weight:g}"
            score_col = f"score_{name}"
            scored = df.with_columns((pl.col("base_logit") + float(weight) * pl.col("lex_logodds")).alias(score_col))
            score, metrics = evaluate_scored_frame(scored, score_col)
            totals[name] += int(score)
            details[name].extend(metrics)
        print(
            f"valid {valid_i}/{len(valid_files)} rows={df.height:,} "
            + " ".join(f"{name}={totals[name]:,}" for name in totals),
            flush=True,
        )

    for name, score in totals.items():
        summaries.append(
            {
                "model": name,
                "files": len(details[name]),
                "rows": total_rows,
                PRIMARY_METRIC: int(score),
                "primary_score": int(score),
                "coverage_rows": coverage_rows,
                "coverage_target_rows": coverage_target_rows,
                "coverage_rate": coverage_rows / total_rows if total_rows else 0.0,
            }
        )
    leaderboard = pl.DataFrame(summaries).sort("primary_score", descending=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.write_csv(out_dir / "leaderboard.csv")
    with open(out_dir / "details.json", "w", encoding="utf-8") as handle:
        json.dump(details, handle, ensure_ascii=False)
    print(leaderboard, flush=True)
    return leaderboard, details


def parse_weights(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="/root/autodl-tmp/datasets/aipc/processed_split/train")
    parser.add_argument("--valid-pred-root", default="/root/aipc/models/exp_by_instrument/v1")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--key-mode", choices=["clean", "stripped"], default="clean")
    parser.add_argument("--max-files-per-instrument", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--weights", default="0.25,0.5,1,2,4,8,12,16,24")
    parser.add_argument("--lexicon", default=None, help="Use an existing lexicon parquet instead of rebuilding it")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.lexicon:
        lex_path = Path(args.lexicon).expanduser().resolve()
    else:
        lex_path = build_lexicon(
            train_root=Path(args.train_root).expanduser().resolve(),
            work_dir=out_dir,
            key_mode=args.key_mode,
            max_files_per_instrument=args.max_files_per_instrument,
            workers=int(args.workers),
        )
    leaderboard, _ = evaluate_with_lexicon(
        valid_pred_root=Path(args.valid_pred_root).expanduser().resolve(),
        lex_path=lex_path,
        key_mode=args.key_mode,
        weights=parse_weights(args.weights),
        out_dir=out_dir,
    )
    meta = {
        "lex_path": str(lex_path),
        "key_mode": args.key_mode,
        "max_files_per_instrument": args.max_files_per_instrument,
        "weights": parse_weights(args.weights),
        "leaderboard": leaderboard.to_dicts(),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
