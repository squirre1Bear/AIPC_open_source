#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")
os.environ.setdefault("ARROW_NUM_THREADS", "8")
os.environ.setdefault("POLARS_MAX_THREADS", "8")

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


def discover_train_files(train_root: Path) -> List[Path]:
    files: List[Path] = []
    for instrument in ("mzml", "tims", "wiff"):
        files.extend(sorted((train_root / instrument).glob("*.parquet")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="/root/autodl-tmp/datasets/aipc/processed_split/train")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    train_root = Path(args.train_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_train_files(train_root)
    if args.limit_files is not None:
        files = files[: int(args.limit_files)]
    if not files:
        raise RuntimeError(f"no train parquet files under {train_root}")

    started = time.time()
    print(f"build full clean lexicon files={len(files)} out_dir={out_dir}", flush=True)
    lexicon = (
        pl.scan_parquet([str(path) for path in files])
        .select(
            clean_expr("precursor_sequence").alias("pep_key"),
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
        .with_columns(
            ((pl.col("target_count") + 1.0) / (pl.col("decoy_count") + 1.0)).log().alias("lex_logodds"),
            (pl.col("target_count") > 0).alias("seen_target"),
            (pl.col("decoy_count") > 0).alias("seen_decoy"),
        )
        .collect(streaming=True)
    )

    lex_path = out_dir / "lexicon_clean_full_train.parquet"
    lexicon.write_parquet(lex_path)
    meta = {
        "train_root": str(train_root),
        "files": len(files),
        "rows_in_lexicon": int(lexicon.height),
        "source_row_count": int(lexicon["row_count"].sum()) if lexicon.height else 0,
        "target_seen": int(lexicon["seen_target"].sum()) if lexicon.height else 0,
        "decoy_seen": int(lexicon["seen_decoy"].sum()) if lexicon.height else 0,
        "target_count_sum": int(lexicon["target_count"].sum()) if lexicon.height else 0,
        "decoy_count_sum": int(lexicon["decoy_count"].sum()) if lexicon.height else 0,
        "lex_logodds_min": float(lexicon["lex_logodds"].min()) if lexicon.height else 0.0,
        "lex_logodds_max": float(lexicon["lex_logodds"].max()) if lexicon.height else 0.0,
        "elapsed_sec": time.time() - started,
        "lexicon_path": str(lex_path),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
