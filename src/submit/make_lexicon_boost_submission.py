#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("ARROW_NUM_THREADS", "1")
os.environ.setdefault("POLARS_MAX_THREADS", "1")

import polars as pl


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


def discover_test_parquets(test_dir: Path) -> List[Path]:
    files = sorted(path for path in test_dir.glob("*.parquet") if path.is_file())
    if not files:
        raise RuntimeError(f"no parquet files found: {test_dir}")
    return files


def find_benchmark_part(benchmark_dir: Path, parquet_path: Path) -> Path:
    candidates = [
        benchmark_dir / f"{parquet_path.stem}_pred.csv",
        benchmark_dir / f"{parquet_path.stem}_pred.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(benchmark_dir.glob(f"{parquet_path.stem}*pred.csv"))
    if len(matches) == 1:
        return matches[0]
    raise RuntimeError(f"cannot find benchmark prediction for {parquet_path.name} in {benchmark_dir}")


def scan_benchmark_part(path: Path) -> pl.LazyFrame:
    suffix = path.suffix.lower()
    separator = "\t" if suffix in {".tsv", ".txt"} else ","
    return pl.scan_csv(
        path,
        has_header=False,
        new_columns=["index", "benchmark_score"],
        separator=separator,
        schema_overrides={"index": pl.Int64, "benchmark_score": pl.Float64},
    )


def logit_expr(column: str) -> pl.Expr:
    clipped = pl.col(column).cast(pl.Float64).clip(1e-8, 1.0 - 1e-8)
    return (clipped / (1.0 - clipped)).log()


def score_one_file(args: Tuple[str, Optional[str], str, str, str, float, str, str, str]) -> Tuple[str, Dict[str, object], str]:
    parquet_text, benchmark_text, lex_text, out_text, key_mode, weight, score_mode, base_source, score_col = args
    parquet_path = Path(parquet_text)
    benchmark_path = Path(benchmark_text) if benchmark_text else None
    lex_path = Path(lex_text)
    out_dir = Path(out_text)
    out_dir.mkdir(parents=True, exist_ok=True)

    key_expr = clean_expr("precursor_sequence") if key_mode == "clean" else stripped_expr("precursor_sequence")
    select_exprs = [
        pl.col("index").cast(pl.Int64),
        pl.col("instrument").fill_null("unknown").cast(pl.Utf8).alias("instrument"),
        key_expr.alias("pep_key"),
    ]
    if base_source == "test_score":
        select_exprs.append(pl.col(score_col).cast(pl.Float64).alias("raw_base_score"))
    test_lf = pl.scan_parquet(parquet_path).select(select_exprs)
    lex_lf = pl.scan_parquet(lex_path).select(["pep_key", "lex_logodds", "target_count", "decoy_count"])
    if base_source == "benchmark":
        if benchmark_path is None:
            raise RuntimeError("benchmark source needs a benchmark prediction path")
        bench_lf = scan_benchmark_part(benchmark_path)
        joined = test_lf.join(bench_lf, on="index", how="inner")
        base_col = "benchmark_score"
    else:
        joined = test_lf.with_columns(logit_expr("raw_base_score").alias("test_base_score"))
        base_col = "test_base_score"

    df = joined.join(lex_lf, on="pep_key", how="left").with_columns(
        pl.col("lex_logodds").fill_null(0.0),
        pl.col("target_count").fill_null(0).cast(pl.Int64),
        pl.col("decoy_count").fill_null(0).cast(pl.Int64),
    )
    if score_mode == "add":
        df = df.with_columns((pl.col(base_col) + float(weight) * pl.col("lex_logodds")).alias("score"))
    else:
        df = df.with_columns((float(weight) * pl.col("lex_logodds") + 0.01 * pl.col(base_col)).alias("score"))
    df = df.select(
        ["index", "score", "instrument", "target_count", "decoy_count", "lex_logodds", base_col]
    ).collect(streaming=True)

    seen = (df["target_count"] + df["decoy_count"]) > 0
    seen_target = df["target_count"] > 0
    seen_decoy = df["decoy_count"] > 0
    pred_path = out_dir / f"{parquet_path.stem}_pred.csv"
    df.select(["index", "score"]).write_csv(pred_path, include_header=False)
    metadata = {
        "parquet": str(parquet_path),
        "benchmark_pred": str(benchmark_path) if benchmark_path is not None else None,
        "pred_path": str(pred_path),
        "base_source": base_source,
        "base_score_column": base_col,
        "rows": int(df.height),
        "seen_rows": int(seen.sum()),
        "seen_target_rows": int(seen_target.sum()),
        "seen_decoy_rows": int(seen_decoy.sum()),
        "score_min": float(df["score"].min()),
        "score_max": float(df["score"].max()),
        "score_mean": float(df["score"].mean()),
        "base_score_min": float(df[base_col].min()),
        "base_score_max": float(df[base_col].max()),
        "lex_logodds_min": float(df["lex_logodds"].min()),
        "lex_logodds_max": float(df["lex_logodds"].max()),
        "instrument_counts": df.group_by("instrument").agg(pl.len().alias("n")).to_dicts(),
    }
    return parquet_path.name, metadata, str(pred_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="/root/autodl-tmp/datasets/aipc/processed/bas_merged")
    parser.add_argument("--benchmark-dir", default=None)
    parser.add_argument("--lexicon", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--weight", type=float, default=24.0)
    parser.add_argument("--key-mode", choices=["clean", "stripped"], default="clean")
    parser.add_argument("--score-mode", choices=["add", "lex_dominant"], default="add")
    parser.add_argument("--base-source", choices=["benchmark", "test_score"], default="benchmark")
    parser.add_argument("--score-col", default="lgbm_v1_score")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--expected-rows", type=int, default=10768114)
    args = parser.parse_args()

    test_dir = Path(args.test_dir).expanduser().resolve()
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve() if args.benchmark_dir else None
    if args.base_source == "benchmark" and benchmark_dir is None:
        raise RuntimeError("--benchmark-dir is required when --base-source benchmark")
    lex_path = Path(args.lexicon).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = discover_test_parquets(test_dir)
    tasks = [
        (
            str(path),
            str(find_benchmark_part(benchmark_dir, path)) if benchmark_dir is not None else None,
            str(lex_path),
            str(out_dir),
            args.key_mode,
            float(args.weight),
            args.score_mode,
            args.base_source,
            args.score_col,
        )
        for path in parquet_files
    ]
    print(
        f"make submission files={len(tasks)} workers={args.workers} "
        f"weight={args.weight:g} score_mode={args.score_mode} base_source={args.base_source}",
        flush=True,
    )

    metadata_by_name: Dict[str, Dict[str, object]] = {}
    pred_by_name: Dict[str, str] = {}
    workers = max(1, int(args.workers))
    if workers <= 1:
        for task in tasks:
            name, meta, pred_path = score_one_file(task)
            metadata_by_name[name] = meta
            pred_by_name[name] = pred_path
            print(f"done {name} rows={meta['rows']:,} seen={meta['seen_rows']:,}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(score_one_file, task) for task in tasks]
            done = 0
            for future in as_completed(futures):
                name, meta, pred_path = future.result()
                metadata_by_name[name] = meta
                pred_by_name[name] = pred_path
                done += 1
                print(
                    f"done {done}/{len(tasks)} {name} rows={meta['rows']:,} "
                    f"seen={meta['seen_rows']:,}",
                    flush=True,
                )

    parts = []
    for path in parquet_files:
        pred_path = pred_by_name[path.name]
        parts.append(
            pl.scan_csv(
                pred_path,
                has_header=False,
                new_columns=["index", "score"],
                schema_overrides={"index": pl.Int64, "score": pl.Float64},
            )
        )
    merged = pl.concat(parts).collect(streaming=True)
    rows = int(merged.height)
    unique_index = int(merged["index"].n_unique())
    if args.expected_rows and rows != int(args.expected_rows):
        raise RuntimeError(f"row mismatch: got {rows:,}, expected {int(args.expected_rows):,}")
    if args.expected_rows and unique_index != int(args.expected_rows):
        raise RuntimeError(f"unique index mismatch: got {unique_index:,}, expected {int(args.expected_rows):,}")

    all_pred = out_dir / "all_pred.tsv"
    merged.write_csv(all_pred, separator="\t")
    zip_path = out_dir / "all_pred.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(all_pred, arcname="all_pred.tsv")

    total_seen = sum(int(meta["seen_rows"]) for meta in metadata_by_name.values())
    total_seen_target = sum(int(meta["seen_target_rows"]) for meta in metadata_by_name.values())
    total_seen_decoy = sum(int(meta["seen_decoy_rows"]) for meta in metadata_by_name.values())
    metadata = {
        "method": "benchmark_score_plus_train_label_clean_peptide_lexicon",
        "base_source": args.base_source,
        "score_col": args.score_col,
        "benchmark_dir": str(benchmark_dir) if benchmark_dir is not None else None,
        "lexicon": str(lex_path),
        "weight": float(args.weight),
        "key_mode": args.key_mode,
        "score_mode": args.score_mode,
        "rows": rows,
        "unique_index": unique_index,
        "seen_rows": total_seen,
        "seen_target_rows": total_seen_target,
        "seen_decoy_rows": total_seen_decoy,
        "seen_rate": total_seen / rows if rows else 0.0,
        "zip_path": str(zip_path),
        "file_metadata": [metadata_by_name[path.name] for path in parquet_files],
    }
    with open(out_dir / "rescore_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2)[:4000], flush=True)


if __name__ == "__main__":
    main()
