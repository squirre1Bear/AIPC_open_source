#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline leaderboard for AIPC PSM rescoring validation predictions.

Main metric follows the closest locally reproducible official-like order:
per file -> scan top1 -> conservative target-decoy q-value -> 1% FDR
target rows -> precursor de-duplication when charge is available -> cleaned
peptide de-duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CPU_THREADS = 16
for _thread_env_name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "ARROW_NUM_THREADS",
):
    os.environ.setdefault(_thread_env_name, str(DEFAULT_CPU_THREADS))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import pyarrow as pa
except Exception:  # pragma: no cover
    pa = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


INSTRUMENTS = ("mzml", "tims", "wiff")
PRIMARY_METRIC = "official_like_unique_clean_peptide_at_1pct"
BASE_EVAL_COLUMNS = [
    "index",
    "file_id",
    "instrument",
    "scan_number",
    "group_key",
    "peptide_key",
    "precursor_sequence",
    "modified_sequence",
    "peptide",
    "charge",
    "precursor_charge",
    "label",
]
SCORED_EVAL_COLUMNS = BASE_EVAL_COLUMNS + ["score"]


@dataclass(frozen=True)
class PredictionInput:
    name: str
    path: Path


@dataclass(frozen=True)
class EvalConfig:
    fdr_threshold: float = 0.01
    conservative_tdc: bool = True
    top1_key: str = "scan"
    pre_fdr_dedup: str = "scan"
    strict: bool = True
    allow_row_order: bool = True


@dataclass(frozen=True)
class FileRecord:
    file_id: str
    instrument: str
    rows: int


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


def progress(items: Sequence, desc: str):
    if tqdm is None:
        return items
    return tqdm(items, desc=desc)


def parse_named_path(text: str) -> PredictionInput:
    if "=" not in text:
        path = Path(text).expanduser()
        return PredictionInput(path.stem or path.name, path)
    name, raw_path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Invalid --pred name: {text}")
    return PredictionInput(name, Path(raw_path).expanduser())


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def is_candidate_parquet(path: Path) -> bool:
    name = path.name
    return (
        path.suffix == ".parquet"
        and ".tmp" not in name
        and not name.endswith(".bak")
        and not name.endswith(".bak_fragment")
        and not name.endswith(".tmp_group.parquet")
    )


def parquet_columns(path: Path) -> List[str]:
    return pq.ParquetFile(path).schema_arrow.names


def parquet_row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def existing_columns(path: Path, wanted: Iterable[str]) -> List[str]:
    available = set(parquet_columns(path))
    return [col for col in wanted if col in available]


def read_parquet_existing(path: Path, wanted: Iterable[str]) -> pd.DataFrame:
    columns = existing_columns(path, wanted)
    if not columns:
        raise RuntimeError(f"No requested columns found in parquet: {path}")
    return pq.read_table(path, columns=columns).to_pandas()


def merge_file_records(records: Iterable[FileRecord]) -> Dict[str, FileRecord]:
    rows_by_file: Dict[str, int] = {}
    instruments_by_file: Dict[str, Counter] = {}
    for record in records:
        file_id = str(record.file_id)
        rows_by_file[file_id] = rows_by_file.get(file_id, 0) + int(record.rows)
        instruments_by_file.setdefault(file_id, Counter())[str(record.instrument)] += int(record.rows)
    return {
        file_id: FileRecord(
            file_id=file_id,
            instrument=instruments_by_file[file_id].most_common(1)[0][0],
            rows=rows,
        )
        for file_id, rows in rows_by_file.items()
    }


def read_file_records_from_parquet(path: Path) -> List[FileRecord]:
    columns = parquet_columns(path)
    total_rows = parquet_row_count(path)
    if "file_id" not in columns:
        return [FileRecord(path.stem, infer_instrument(path.stem), total_rows)]

    wanted = ["file_id"]
    if "instrument" in columns:
        wanted.append("instrument")
    table = pq.read_table(path, columns=wanted).to_pandas()
    table["file_id"] = table["file_id"].fillna(path.stem).astype(str)
    if "instrument" in table.columns:
        table["instrument"] = table["instrument"].fillna("unknown").astype(str)
    else:
        table["instrument"] = table["file_id"].map(infer_instrument)

    records: List[FileRecord] = []
    for file_id, part in table.groupby("file_id", sort=False):
        instrument = Counter(part["instrument"].astype(str)).most_common(1)[0][0]
        records.append(FileRecord(str(file_id), str(instrument), int(len(part))))
    return records


def discover_prediction_file_records(
    prediction: PredictionInput,
    valid_files: Sequence[Path],
) -> Dict[str, FileRecord]:
    scored_parquets = discover_scored_parquets(prediction.path)
    if scored_parquets:
        records: List[FileRecord] = []
        for parquet_path in progress(scored_parquets, f"index {prediction.name}"):
            records.extend(read_file_records_from_parquet(parquet_path))
        return merge_file_records(records)

    if not valid_files:
        raise RuntimeError(
            f"{prediction.name} appears to be score-only; provide --valid-root or --manifest "
            "when using common-subset evaluation."
        )
    records = []
    for valid_path in progress(list(valid_files), f"index valid files for {prediction.name}"):
        records.extend(read_file_records_from_parquet(valid_path))
    return merge_file_records(records)


def choose_common_file_subset(
    predictions: Sequence[PredictionInput],
    valid_files: Sequence[Path],
    files_per_instrument: Optional[int],
) -> Tuple[set[str], Dict]:
    if not predictions:
        raise RuntimeError("No predictions available for common-subset selection")

    records_by_model = {
        prediction.name: discover_prediction_file_records(prediction, valid_files)
        for prediction in predictions
    }
    common_file_ids = set.intersection(*(set(records) for records in records_by_model.values()))

    comparable_records: List[FileRecord] = []
    rejected: List[Dict] = []
    for file_id in sorted(common_file_ids):
        row_counts = {name: records[file_id].rows for name, records in records_by_model.items()}
        instruments = {name: records[file_id].instrument for name, records in records_by_model.items()}
        if len(set(row_counts.values())) == 1 and len(set(instruments.values())) == 1:
            first_record = next(iter(records_by_model.values()))[file_id]
            comparable_records.append(first_record)
        else:
            rejected.append(
                {
                    "file_id": file_id,
                    "row_counts": row_counts,
                    "instruments": instruments,
                }
            )

    if files_per_instrument is not None:
        if files_per_instrument <= 0:
            raise ValueError("--common-files-per-instrument must be positive")
        selected: List[FileRecord] = []
        shortages: Dict[str, int] = {}
        for instrument in INSTRUMENTS:
            candidates = sorted(
                [record for record in comparable_records if record.instrument.lower() == instrument],
                key=lambda record: record.file_id,
            )
            if len(candidates) < files_per_instrument:
                shortages[instrument] = len(candidates)
            selected.extend(candidates[:files_per_instrument])
        if shortages:
            raise RuntimeError(
                "Not enough common comparable files for requested per-instrument subset: "
                + ", ".join(
                    f"{instrument} has {available}, need {files_per_instrument}"
                    for instrument, available in shortages.items()
                )
            )
    else:
        selected = sorted(comparable_records, key=lambda record: (record.instrument, record.file_id))

    if not selected:
        raise RuntimeError("No common comparable files found across predictions")

    selected_ids = {record.file_id for record in selected}
    rows_by_instrument = Counter()
    files_by_instrument = Counter()
    for record in selected:
        rows_by_instrument[record.instrument] += record.rows
        files_by_instrument[record.instrument] += 1

    report = {
        "enabled": True,
        "requested_files_per_instrument": files_per_instrument,
        "common_file_count": len(common_file_ids),
        "comparable_common_file_count": len(comparable_records),
        "selected_file_count": len(selected),
        "selected_total_rows": int(sum(record.rows for record in selected)),
        "selected_files_by_instrument": dict(files_by_instrument),
        "selected_rows_by_instrument": dict(rows_by_instrument),
        "rejected_common_file_count": len(rejected),
        "rejected_common_files_sample": rejected[:10],
        "models": {
            name: {
                "files": len(records),
                "rows": int(sum(record.rows for record in records.values())),
            }
            for name, records in records_by_model.items()
        },
        "selected_files": [
            {
                "file_id": record.file_id,
                "instrument": record.instrument,
                "rows": int(record.rows),
            }
            for record in selected
        ],
    }
    return selected_ids, report


def list_valid_root_files(valid_root: Path) -> List[Path]:
    files: List[Path] = []
    instrument_dirs = [valid_root / inst for inst in INSTRUMENTS]
    if any(path.exists() for path in instrument_dirs):
        for instrument_dir in instrument_dirs:
            if instrument_dir.exists():
                files.extend(path for path in sorted(instrument_dir.glob("*.parquet")) if is_candidate_parquet(path))
    else:
        files.extend(path for path in sorted(valid_root.glob("*.parquet")) if is_candidate_parquet(path))
    return files


def list_manifest_files(
    manifest: Path,
    fold: Optional[int],
    train_folds: Optional[Sequence[int]],
) -> List[Path]:
    table = pd.read_csv(manifest)
    if "path" not in table.columns:
        raise RuntimeError(f"manifest must contain column 'path': {manifest}")
    if "fold" not in table.columns:
        raise RuntimeError(f"manifest must contain column 'fold': {manifest}")
    if fold is not None and train_folds is not None:
        raise ValueError("Use either --fold or --train-folds, not both.")
    if fold is not None:
        table = table[table["fold"].astype(int) == int(fold)]
    elif train_folds is not None:
        table = table[table["fold"].astype(int).isin([int(value) for value in train_folds])]
    return [Path(value) for value in table["path"].tolist()]


def select_validation_files(args: argparse.Namespace) -> List[Path]:
    train_folds = parse_int_list(args.train_folds)
    if args.manifest:
        files = list_manifest_files(Path(args.manifest).expanduser(), args.fold, train_folds)
    elif args.valid_root:
        files = list_valid_root_files(Path(args.valid_root).expanduser())
    else:
        return []
    if args.max_files is not None:
        files = files[: args.max_files]
    return files


def infer_instrument(value: str) -> str:
    lower = value.lower()
    for instrument in INSTRUMENTS:
        if instrument in lower:
            return instrument
    return "unknown"


def choose_first_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def format_charge_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        rounded = numeric.round().astype("Int64")
        if np.allclose(numeric.astype(float), rounded.astype(float), equal_nan=True):
            return rounded.astype(str)
    return series.astype(str)


def clean_peptide_sequence(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").astype(str)
    cleaned = cleaned.str.replace(r"n\[42\]", "", regex=True)
    cleaned = cleaned.str.replace(r"N\[.98\]", "N", regex=True)
    cleaned = cleaned.str.replace(r"Q\[.98\]", "Q", regex=True)
    cleaned = cleaned.str.replace(r"M\[15.99\]", "M", regex=True)
    cleaned = cleaned.str.replace(r"C\[57.02\]", "C", regex=True)
    return cleaned


def standardize_eval_frame(df: pd.DataFrame, source_name: str, config: EvalConfig) -> pd.DataFrame:
    if "label" not in df.columns:
        raise RuntimeError(f"{source_name} missing label column")
    if "score" not in df.columns:
        raise RuntimeError(f"{source_name} missing score column")

    out = pd.DataFrame(index=df.index)
    out["score"] = pd.to_numeric(df["score"], errors="coerce")
    out["label"] = pd.to_numeric(df["label"], errors="coerce")
    if out["label"].isna().any():
        raise RuntimeError(f"{source_name} contains non-numeric labels")
    out["label"] = out["label"].astype(np.int8)

    if config.strict and out["score"].isna().any():
        missing = int(out["score"].isna().sum())
        raise RuntimeError(f"{source_name} has {missing} missing scores")
    out["score"] = out["score"].replace([np.inf, -np.inf], np.nan)
    if config.strict and out["score"].isna().any():
        missing = int(out["score"].isna().sum())
        raise RuntimeError(f"{source_name} has {missing} NaN/Inf scores")
    out["score"] = out["score"].fillna(-np.inf).astype(np.float64)

    if "file_id" in df.columns:
        out["file_id"] = df["file_id"].fillna(source_name).astype(str)
    else:
        out["file_id"] = str(source_name)

    if "instrument" in df.columns:
        out["instrument"] = df["instrument"].fillna("unknown").astype(str)
    else:
        out["instrument"] = infer_instrument(str(source_name))

    if config.top1_key == "group" and "group_key" in df.columns:
        out["top1_key"] = df["group_key"].astype(str)
    elif "scan_number" in df.columns:
        out["top1_key"] = df["scan_number"].astype(str)
    elif "group_key" in df.columns:
        out["top1_key"] = df["group_key"].astype(str)
    else:
        raise RuntimeError(f"{source_name} missing scan_number/group_key for top1 evaluation")

    sequence_col = choose_first_column(df, ["modified_sequence", "precursor_sequence", "peptide", "peptide_key"])
    if sequence_col is None:
        raise RuntimeError(f"{source_name} missing peptide sequence column")
    out["modified_sequence"] = df[sequence_col].fillna("").astype(str)
    out["cleaned_sequence"] = clean_peptide_sequence(out["modified_sequence"])

    charge_col = choose_first_column(df, ["precursor_charge", "charge"])
    if charge_col is not None:
        out["precursor_id"] = format_charge_series(df[charge_col]) + "_" + out["modified_sequence"]
    else:
        out["precursor_id"] = pd.NA

    if "index" in df.columns:
        out["index"] = df["index"]
    return out


def add_q_values(sorted_df: pd.DataFrame, config: EvalConfig) -> pd.DataFrame:
    labels = sorted_df["label"].to_numpy()
    is_target = labels == 1
    is_decoy = labels != 1
    cum_target = np.cumsum(is_target, dtype=np.int64)
    cum_decoy = np.cumsum(is_decoy, dtype=np.int64)
    target_for_fdr = np.maximum(cum_target, 1)
    decoy_for_fdr = np.maximum(cum_decoy, 1) if config.conservative_tdc else cum_decoy
    fdr = decoy_for_fdr / target_for_fdr
    q_value = np.minimum.accumulate(fdr[::-1])[::-1]

    out = sorted_df.copy()
    out["q_value"] = q_value.astype(np.float64)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int64)
    out["cum_target"] = cum_target
    out["cum_decoy"] = cum_decoy
    out["fdr"] = fdr.astype(np.float64)
    return out


def compute_one_file_metrics(df: pd.DataFrame, file_id: str, config: EvalConfig) -> Dict:
    rows = int(len(df))
    if rows == 0:
        return empty_file_metrics(file_id)

    instrument = str(df["instrument"].iloc[0]) if "instrument" in df.columns else infer_instrument(file_id)
    sorted_all = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    if config.pre_fdr_dedup == "none":
        top1 = sorted_all
    elif config.pre_fdr_dedup == "scan_precursor":
        duplicate_scan = sorted_all["top1_key"].duplicated(keep="first")
        duplicate_precursor = (
            sorted_all["precursor_id"].notna()
            & sorted_all["precursor_id"].duplicated(keep="first")
        )
        top1 = sorted_all[~(duplicate_scan | duplicate_precursor)]
    elif config.pre_fdr_dedup == "scan":
        top1 = sorted_all.drop_duplicates(subset=["top1_key"], keep="first")
    else:
        raise ValueError(f"Unsupported pre_fdr_dedup: {config.pre_fdr_dedup}")
    top1 = top1.reset_index(drop=True)
    top1 = add_q_values(top1, config)

    accepted_all = top1[top1["q_value"] <= config.fdr_threshold].copy()
    accepted_targets = accepted_all[accepted_all["label"] == 1].copy()
    accepted_decoys = accepted_all[accepted_all["label"] != 1].copy()

    if len(accepted_all) > 0:
        last = accepted_all.iloc[-1]
        score_threshold = float(last["score"])
        rank_cutoff = int(last["rank"])
        estimated_fdr = float(last["fdr"])
        cutoff_target_rows = int(last["cum_target"])
        cutoff_decoy_rows = int(last["cum_decoy"])
    else:
        score_threshold = None
        rank_cutoff = 0
        estimated_fdr = None
        cutoff_target_rows = 0
        cutoff_decoy_rows = 0

    has_precursor_id = accepted_targets["precursor_id"].notna().any() if len(accepted_targets) else df["precursor_id"].notna().any()
    if has_precursor_id:
        precursor_targets = accepted_targets.sort_values("score", ascending=False, kind="mergesort")
        precursor_targets = precursor_targets.drop_duplicates(subset=["precursor_id"], keep="first")
        unique_precursor = int(len(precursor_targets))
    else:
        precursor_targets = accepted_targets.sort_values("score", ascending=False, kind="mergesort")
        unique_precursor = None

    peptide_targets = precursor_targets.drop_duplicates(subset=["cleaned_sequence"], keep="first")
    unique_clean_peptide = int(len(peptide_targets))
    unique_raw_peptide = int(accepted_targets["modified_sequence"].nunique()) if len(accepted_targets) else 0

    target_rows = int((df["label"] == 1).sum())
    decoy_rows = int((df["label"] != 1).sum())
    top1_target_rows = int((top1["label"] == 1).sum())
    top1_decoy_rows = int((top1["label"] != 1).sum())

    return {
        "file_id": file_id,
        "instrument": instrument,
        "rows": rows,
        "target_rows": target_rows,
        "decoy_rows": decoy_rows,
        "top1_rows": int(len(top1)),
        "top1_target_rows": top1_target_rows,
        "top1_decoy_rows": top1_decoy_rows,
        "top1_target_rate": safe_ratio(top1_target_rows, len(top1)),
        "accepted_total_rows_at_1pct": int(len(accepted_all)),
        "accepted_target_psm_at_1pct": int(len(accepted_targets)),
        "accepted_decoy_rows_at_1pct": int(len(accepted_decoys)),
        "official_like_unique_precursor_at_1pct": unique_precursor,
        "official_like_unique_raw_peptide_at_1pct": unique_raw_peptide,
        "official_like_unique_clean_peptide_at_1pct": unique_clean_peptide,
        "score_threshold_at_1pct": score_threshold,
        "rank_cutoff_at_1pct": rank_cutoff,
        "estimated_fdr_at_cutoff": estimated_fdr,
        "cutoff_target_rows": cutoff_target_rows,
        "cutoff_decoy_rows": cutoff_decoy_rows,
        "has_precursor_id": bool(has_precursor_id),
    }


def empty_file_metrics(file_id: str) -> Dict:
    return {
        "file_id": file_id,
        "instrument": "unknown",
        "rows": 0,
        "target_rows": 0,
        "decoy_rows": 0,
        "top1_rows": 0,
        "top1_target_rows": 0,
        "top1_decoy_rows": 0,
        "top1_target_rate": None,
        "accepted_total_rows_at_1pct": 0,
        "accepted_target_psm_at_1pct": 0,
        "accepted_decoy_rows_at_1pct": 0,
        "official_like_unique_precursor_at_1pct": None,
        "official_like_unique_raw_peptide_at_1pct": 0,
        "official_like_unique_clean_peptide_at_1pct": 0,
        "score_threshold_at_1pct": None,
        "rank_cutoff_at_1pct": 0,
        "estimated_fdr_at_cutoff": None,
        "cutoff_target_rows": 0,
        "cutoff_decoy_rows": 0,
        "has_precursor_id": False,
    }


def safe_ratio(num: int, den: int) -> Optional[float]:
    if den <= 0:
        return None
    return float(num) / float(den)


def aggregate_metrics(file_metrics: List[Dict], model_name: str) -> Tuple[Dict, pd.DataFrame]:
    by_file = pd.DataFrame(file_metrics)
    if by_file.empty:
        summary = {"model": model_name, PRIMARY_METRIC: 0, "files": 0, "rows": 0}
        return summary, by_file

    numeric_sum_cols = [
        "rows",
        "target_rows",
        "decoy_rows",
        "top1_rows",
        "top1_target_rows",
        "top1_decoy_rows",
        "accepted_total_rows_at_1pct",
        "accepted_target_psm_at_1pct",
        "accepted_decoy_rows_at_1pct",
        "official_like_unique_raw_peptide_at_1pct",
        "official_like_unique_clean_peptide_at_1pct",
    ]
    summary: Dict[str, object] = {"model": model_name, "files": int(len(by_file))}
    for col in numeric_sum_cols:
        summary[col] = int(pd.to_numeric(by_file[col], errors="coerce").fillna(0).sum())

    precursor_values = pd.to_numeric(by_file["official_like_unique_precursor_at_1pct"], errors="coerce")
    summary["official_like_unique_precursor_at_1pct"] = (
        int(precursor_values.sum()) if precursor_values.notna().all() else None
    )
    summary["top1_target_rate"] = safe_ratio(int(summary["top1_target_rows"]), int(summary["top1_rows"]))
    summary["accepted_decoy_over_total_at_1pct"] = safe_ratio(
        int(summary["accepted_decoy_rows_at_1pct"]), int(summary["accepted_total_rows_at_1pct"])
    )
    summary["has_precursor_id_all_files"] = bool(by_file["has_precursor_id"].all())
    summary["primary_metric"] = PRIMARY_METRIC
    summary["primary_score"] = int(summary[PRIMARY_METRIC])
    return summary, by_file


def aggregate_by_instrument(by_file: pd.DataFrame) -> pd.DataFrame:
    if by_file.empty:
        return by_file
    sum_cols = [
        "rows",
        "top1_rows",
        "top1_target_rows",
        "accepted_target_psm_at_1pct",
        "accepted_decoy_rows_at_1pct",
        "accepted_total_rows_at_1pct",
        "official_like_unique_clean_peptide_at_1pct",
        "official_like_unique_raw_peptide_at_1pct",
    ]
    grouped = by_file.groupby("instrument", dropna=False)[sum_cols].sum(numeric_only=True).reset_index()
    grouped["files"] = by_file.groupby("instrument", dropna=False).size().to_numpy()
    grouped["top1_target_rate"] = grouped.apply(
        lambda row: safe_ratio(int(row["top1_target_rows"]), int(row["top1_rows"])), axis=1
    )
    return grouped.sort_values("official_like_unique_clean_peptide_at_1pct", ascending=False)


def discover_scored_parquets(path: Path) -> List[Path]:
    if path.is_file() and path.suffix == ".parquet":
        candidates = [path]
    elif path.is_dir():
        preferred = path / "pred_parts"
        if preferred.exists():
            candidates = sorted(p for p in preferred.glob("*.parquet") if is_candidate_parquet(p))
        else:
            candidates = sorted(p for p in path.glob("*.parquet") if is_candidate_parquet(p))
            if not candidates:
                candidates = sorted(p for p in path.rglob("*.parquet") if is_candidate_parquet(p))
    else:
        return []
    scored = []
    for candidate in candidates:
        cols = set(parquet_columns(candidate))
        if {"label", "score"}.issubset(cols):
            scored.append(candidate)
    return scored


def read_scored_parquet(
    path: Path,
    config: EvalConfig,
    selected_file_ids: Optional[set[str]] = None,
) -> pd.DataFrame:
    df = read_parquet_existing(path, SCORED_EVAL_COLUMNS)
    if selected_file_ids is not None and "file_id" in df.columns:
        df = df[df["file_id"].fillna("").astype(str).isin(selected_file_ids)]
    return standardize_eval_frame(df, path.stem, config)


def read_score_file(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    first = pd.read_csv(path, sep=sep, nrows=1)
    lower_cols = [str(col).lower() for col in first.columns]
    if "score" in lower_cols:
        df = pd.read_csv(path, sep=sep)
        rename = {}
        for col in df.columns:
            lower = str(col).strip().lower()
            if lower == "score":
                rename[col] = "score"
            elif lower == "index":
                rename[col] = "index"
        df = df.rename(columns=rename)
        return df
    df = pd.read_csv(path, sep=sep, header=None)
    if df.shape[1] == 1:
        df.columns = ["score"]
    else:
        df = df.iloc[:, :2]
        df.columns = ["index", "score"]
    return df


def find_per_file_score_file(pred_dir: Path, valid_path: Path) -> Optional[Path]:
    candidates = [
        pred_dir / f"{valid_path.stem}_pred.csv",
        pred_dir / f"{valid_path.stem}_pred.tsv",
        pred_dir / f"{valid_path.stem}.csv",
        pred_dir / f"{valid_path.stem}.tsv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(pred_dir.glob(f"{valid_path.stem}*pred.csv")) + sorted(pred_dir.glob(f"{valid_path.stem}*pred.tsv"))
    if len(matches) == 1:
        return matches[0]
    return None


def load_valid_file_with_score(
    valid_path: Path,
    pred_path: Path,
    score_table: Optional[pd.DataFrame],
    config: EvalConfig,
) -> pd.DataFrame:
    wanted = list(BASE_EVAL_COLUMNS)
    valid_df = read_parquet_existing(valid_path, wanted)
    if "label" not in valid_df.columns:
        raise RuntimeError(f"validation file missing label: {valid_path}")

    if score_table is None:
        if not pred_path.is_dir():
            raise RuntimeError(f"score-only prediction path must be a dir or all_pred file: {pred_path}")
        score_file = find_per_file_score_file(pred_path, valid_path)
        if score_file is None:
            raise RuntimeError(f"missing per-file prediction for {valid_path.name} in {pred_path}")
        scores = read_score_file(score_file)
    else:
        scores = score_table

    if "index" in scores.columns and "index" in valid_df.columns:
        score_map = scores[["index", "score"]].copy()
        score_map["index"] = score_map["index"].astype(str)
        valid_df = valid_df.copy()
        valid_df["index"] = valid_df["index"].astype(str)
        if score_map["index"].duplicated().any():
            duplicated = int(score_map["index"].duplicated().sum())
            raise RuntimeError(f"prediction has duplicated index values for {valid_path.name}: {duplicated}")
        valid_df = valid_df.merge(score_map, on="index", how="left", validate="one_to_one")
    elif score_table is None and config.allow_row_order and len(scores) == len(valid_df):
        valid_df = valid_df.copy()
        valid_df["score"] = scores["score"].to_numpy()
    else:
        raise RuntimeError(
            f"cannot align predictions for {valid_path.name}: "
            f"valid has index={'index' in valid_df.columns}, pred has index={'index' in scores.columns}, "
            f"valid_rows={len(valid_df)}, pred_rows={len(scores)}"
        )
    return standardize_eval_frame(valid_df, valid_path.stem, config)


def evaluate_scored_parquets(
    prediction: PredictionInput,
    config: EvalConfig,
    selected_file_ids: Optional[set[str]] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    parquet_files = discover_scored_parquets(prediction.path)
    if not parquet_files:
        raise RuntimeError(f"no scored parquet files found for {prediction.name}: {prediction.path}")

    file_metrics: List[Dict] = []
    for parquet_path in progress(parquet_files, f"eval {prediction.name}"):
        scored_df = read_scored_parquet(parquet_path, config, selected_file_ids)
        if selected_file_ids is not None and "file_id" in scored_df.columns:
            scored_df = scored_df[scored_df["file_id"].astype(str).isin(selected_file_ids)]
        if scored_df.empty:
            continue
        for file_id, part in scored_df.groupby("file_id", sort=False):
            file_metrics.append(compute_one_file_metrics(part, str(file_id), config))

    summary, by_file = aggregate_metrics(file_metrics, prediction.name)
    by_instrument = aggregate_by_instrument(by_file)
    return summary, by_file, by_instrument


def evaluate_score_only_predictions(
    prediction: PredictionInput,
    valid_files: List[Path],
    config: EvalConfig,
    selected_file_ids: Optional[set[str]] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    if not valid_files:
        raise RuntimeError(
            f"{prediction.name} appears to be score-only; provide --valid-root or --manifest to align labels"
        )

    score_table = None
    if prediction.path.is_file():
        score_table = read_score_file(prediction.path)

    file_metrics: List[Dict] = []
    for valid_path in progress(valid_files, f"eval {prediction.name}"):
        scored_df = load_valid_file_with_score(valid_path, prediction.path, score_table, config)
        if selected_file_ids is not None:
            scored_df = scored_df[scored_df["file_id"].astype(str).isin(selected_file_ids)]
        if scored_df.empty:
            continue
        for file_id, part in scored_df.groupby("file_id", sort=False):
            file_metrics.append(compute_one_file_metrics(part, str(file_id), config))

    summary, by_file = aggregate_metrics(file_metrics, prediction.name)
    by_instrument = aggregate_by_instrument(by_file)
    return summary, by_file, by_instrument


def evaluate_prediction(
    prediction: PredictionInput,
    valid_files: List[Path],
    config: EvalConfig,
    selected_file_ids: Optional[set[str]] = None,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    scored_parquets = discover_scored_parquets(prediction.path)
    if scored_parquets:
        return evaluate_scored_parquets(prediction, config, selected_file_ids)
    return evaluate_score_only_predictions(prediction, valid_files, config, selected_file_ids)


def add_leaderboard_deltas(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard
    leaderboard = leaderboard.sort_values("primary_score", ascending=False).reset_index(drop=True)
    leaderboard.insert(0, "rank", np.arange(1, len(leaderboard) + 1, dtype=np.int64))
    baseline_score = int(leaderboard.iloc[-1]["primary_score"])
    best_score = int(leaderboard.iloc[0]["primary_score"])
    leaderboard["delta_vs_worst"] = leaderboard["primary_score"].astype(int) - baseline_score
    leaderboard["delta_vs_best"] = leaderboard["primary_score"].astype(int) - best_score
    if baseline_score != 0:
        leaderboard["pct_vs_worst"] = leaderboard["delta_vs_worst"] / abs(baseline_score)
    else:
        leaderboard["pct_vs_worst"] = np.nan
    return leaderboard


def build_comparability_report(details: Dict[str, Dict[str, pd.DataFrame]]) -> Dict:
    report: Dict[str, object] = {
        "comparable": True,
        "warnings": [],
        "models": {},
    }
    if len(details) <= 1:
        return report

    file_sets = {}
    file_rows = {}
    file_top1_rows = {}
    for model_name, tables in details.items():
        by_file = tables["by_file"].copy()
        if by_file.empty:
            file_sets[model_name] = set()
            file_rows[model_name] = {}
            file_top1_rows[model_name] = {}
        else:
            file_sets[model_name] = set(by_file["file_id"].astype(str).tolist())
            file_rows[model_name] = dict(
                zip(by_file["file_id"].astype(str), by_file["rows"].astype(int))
            )
            file_top1_rows[model_name] = dict(
                zip(by_file["file_id"].astype(str), by_file["top1_rows"].astype(int))
            )
        report["models"][model_name] = {
            "files": len(file_sets[model_name]),
            "rows": int(by_file["rows"].sum()) if not by_file.empty else 0,
            "top1_rows": int(by_file["top1_rows"].sum()) if not by_file.empty else 0,
        }

    names = list(details.keys())
    reference = names[0]
    reference_files = file_sets[reference]
    for model_name in names[1:]:
        missing = sorted(reference_files - file_sets[model_name])
        extra = sorted(file_sets[model_name] - reference_files)
        if missing or extra:
            report["comparable"] = False
            report["warnings"].append(
                {
                    "type": "file_set_mismatch",
                    "reference": reference,
                    "model": model_name,
                    "missing_file_count": len(missing),
                    "extra_file_count": len(extra),
                    "missing_files_sample": missing[:10],
                    "extra_files_sample": extra[:10],
                }
            )
            continue

        row_mismatch = [
            file_id
            for file_id in sorted(reference_files)
            if file_rows[reference].get(file_id) != file_rows[model_name].get(file_id)
        ]
        if row_mismatch:
            report["comparable"] = False
            report["warnings"].append(
                {
                    "type": "row_count_mismatch",
                    "reference": reference,
                    "model": model_name,
                    "file_count": len(row_mismatch),
                    "files_sample": row_mismatch[:10],
                }
            )

        top1_mismatch = [
            file_id
            for file_id in sorted(reference_files)
            if file_top1_rows[reference].get(file_id) != file_top1_rows[model_name].get(file_id)
        ]
        if top1_mismatch:
            report["comparable"] = False
            report["warnings"].append(
                {
                    "type": "top1_count_mismatch",
                    "reference": reference,
                    "model": model_name,
                    "file_count": len(top1_mismatch),
                    "files_sample": top1_mismatch[:10],
                }
            )

    return report


def write_outputs(
    out_dir: Path,
    leaderboard: pd.DataFrame,
    details: Dict[str, Dict[str, pd.DataFrame]],
    meta: Dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    with open(out_dir / "leaderboard.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "meta": meta,
                "leaderboard": json.loads(leaderboard.to_json(orient="records", force_ascii=False)),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    for model_name, tables in details.items():
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        tables["by_file"].to_csv(out_dir / f"{safe_name}_by_file.csv", index=False)
        tables["by_instrument"].to_csv(out_dir / f"{safe_name}_by_instrument.csv", index=False)
    common_subset = meta.get("common_subset_report", {})
    selected_files = common_subset.get("selected_files") if isinstance(common_subset, dict) else None
    if selected_files:
        pd.DataFrame(selected_files).to_csv(out_dir / "selected_files.csv", index=False)


def run_self_test() -> None:
    rows = []
    for i in range(120):
        rows.append(
            {
                "file_id": "synthetic_file",
                "instrument": "mzml",
                "scan_number": i,
                "precursor_sequence": f"PEPTIDE{i}",
                "charge": 2,
                "label": 1,
                "score": 1000 - i,
            }
        )
    rows.extend(
        [
            {
                "file_id": "synthetic_file",
                "instrument": "mzml",
                "scan_number": 200,
                "precursor_sequence": "n[42]ACDM[15.99]N[.98]Q[.98]C[57.02]",
                "charge": 3,
                "label": 1,
                "score": 900,
            },
            {
                "file_id": "synthetic_file",
                "instrument": "mzml",
                "scan_number": 200,
                "precursor_sequence": "DECOY",
                "charge": 3,
                "label": 0,
                "score": 1,
            },
            {
                "file_id": "synthetic_file",
                "instrument": "mzml",
                "scan_number": 999,
                "precursor_sequence": "LATEDECOY",
                "charge": 2,
                "label": 0,
                "score": -1,
            },
        ]
    )
    config = EvalConfig(fdr_threshold=0.01, conservative_tdc=True)
    df = standardize_eval_frame(pd.DataFrame(rows), "self_test", config)
    metric = compute_one_file_metrics(df, "synthetic_file", config)
    assert metric["top1_rows"] == 122, metric
    assert metric["accepted_decoy_rows_at_1pct"] == 1, metric
    assert metric["official_like_unique_clean_peptide_at_1pct"] == 121, metric
    assert clean_peptide_sequence(pd.Series(["n[42]ACDM[15.99]N[.98]Q[.98]C[57.02]"])).iloc[0] == "ACDMNQC"

    scan_precursor_config = EvalConfig(
        fdr_threshold=0.01,
        conservative_tdc=True,
        pre_fdr_dedup="scan_precursor",
    )
    scan_precursor_metric = compute_one_file_metrics(df, "synthetic_file", scan_precursor_config)
    assert scan_precursor_metric["top1_rows"] == 122, scan_precursor_metric
    assert scan_precursor_metric["official_like_unique_clean_peptide_at_1pct"] == 121, scan_precursor_metric
    print("self-test passed")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an official-like offline leaderboard for AIPC validation predictions."
    )
    parser.add_argument("--pred", action="append", default=[], help="name=path; path may be scored parquet dir/file or score csv/tsv dir/file")
    parser.add_argument("--valid-root", type=str, default=None, help="processed_split/valid root for score-only predictions")
    parser.add_argument("--manifest", type=str, default=None, help="CSV manifest with path and fold columns")
    parser.add_argument("--fold", type=int, default=None, help="Validation fold selected from --manifest")
    parser.add_argument("--train-folds", type=str, default=None, help="Comma-separated folds selected from --manifest")
    parser.add_argument("--max-files", type=int, default=None, help="Debug only: evaluate first N validation files")
    parser.add_argument("--common-subset", action="store_true", help="Evaluate only file_id values shared by all predictions with identical row counts")
    parser.add_argument(
        "--common-files-per-instrument",
        type=int,
        default=None,
        help="When using common-subset evaluation, keep the first N comparable files per instrument, e.g. 3.",
    )
    parser.add_argument("--out-dir", type=str, required=False, default="offline_leaderboard")
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS, help="CPU thread limit for BLAS/Arrow operations")
    parser.add_argument("--fdr-threshold", type=float, default=0.01)
    parser.add_argument("--non-conservative-zero-decoy", action="store_true", help="Use cum_decoy/cum_target instead of official-style max(cum_decoy,1)/target")
    parser.add_argument("--top1-key", choices=["scan", "group"], default="scan", help="Use scan_number by default; group falls back to group_key")
    parser.add_argument(
        "--pre-fdr-dedup",
        choices=["scan", "scan_precursor", "none"],
        default="scan",
        help="Rows kept before TDC FDR. Default matches scan top1; scan_precursor also removes repeated precursor_id before FDR.",
    )
    parser.add_argument("--no-row-order-align", action="store_true", help="For score-only per-file csv, disallow row-order alignment when index is unavailable")
    parser.add_argument("--non-strict", action="store_true", help="Fill missing scores with -inf instead of failing")
    parser.add_argument("--require-comparable", action="store_true", help="Fail when multiple predictions do not cover the same files/rows/top1 groups")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_cpu_threads(int(args.cpu_threads))

    if args.self_test:
        run_self_test()
        if not args.pred:
            return

    predictions = [parse_named_path(item) for item in args.pred]
    if not predictions:
        raise RuntimeError("Provide at least one --pred name=path, or run --self-test only.")

    valid_files = select_validation_files(args)
    config = EvalConfig(
        fdr_threshold=float(args.fdr_threshold),
        conservative_tdc=not args.non_conservative_zero_decoy,
        top1_key=args.top1_key,
        pre_fdr_dedup=args.pre_fdr_dedup,
        strict=not args.non_strict,
        allow_row_order=not args.no_row_order_align,
    )

    selected_file_ids: Optional[set[str]] = None
    common_subset_report: Dict = {"enabled": False}
    use_common_subset = bool(args.common_subset or args.common_files_per_instrument is not None)
    if use_common_subset:
        selected_file_ids, common_subset_report = choose_common_file_subset(
            predictions,
            valid_files,
            args.common_files_per_instrument,
        )

    summaries: List[Dict] = []
    details: Dict[str, Dict[str, pd.DataFrame]] = {}
    for prediction in predictions:
        summary, by_file, by_instrument = evaluate_prediction(prediction, valid_files, config, selected_file_ids)
        summaries.append(summary)
        details[prediction.name] = {"by_file": by_file, "by_instrument": by_instrument}

    leaderboard = add_leaderboard_deltas(pd.DataFrame(summaries))
    comparability = build_comparability_report(details)
    meta = {
        "primary_metric": PRIMARY_METRIC,
        "rule": "per-file -> scan top1 -> conservative TDC q-value -> target rows at FDR -> precursor dedup if available -> cleaned peptide dedup",
        "fdr_threshold": config.fdr_threshold,
        "conservative_tdc": config.conservative_tdc,
        "top1_key": config.top1_key,
        "pre_fdr_dedup": config.pre_fdr_dedup,
        "cpu_threads": int(args.cpu_threads),
        "valid_files": len(valid_files),
        "common_subset_report": common_subset_report,
        "predictions": {item.name: str(item.path) for item in predictions},
        "comparability_report": comparability,
    }
    out_dir = Path(args.out_dir).expanduser()
    write_outputs(out_dir, leaderboard, details, meta)

    if args.require_comparable and not comparability["comparable"]:
        raise RuntimeError(
            "Predictions are not directly comparable. "
            f"Diagnostics were written to {out_dir / 'leaderboard.json'}."
        )

    display_cols = [
        "rank",
        "model",
        "primary_score",
        "delta_vs_worst",
        "files",
        "rows",
        "top1_rows",
        "accepted_target_psm_at_1pct",
        "accepted_decoy_rows_at_1pct",
        "top1_target_rate",
    ]
    display_cols = [col for col in display_cols if col in leaderboard.columns]
    print(leaderboard[display_cols].to_string(index=False))
    if not comparability["comparable"]:
        print("\nWARNING: predictions are not directly comparable.")
        for warning in comparability["warnings"][:5]:
            print(f"- {warning['type']}: {warning}")
    if common_subset_report.get("enabled"):
        print(
            "\nCommon subset: "
            f"{common_subset_report['selected_file_count']} files, "
            f"{common_subset_report['selected_total_rows']} rows, "
            f"by instrument={common_subset_report['selected_files_by_instrument']}"
        )
    print(f"\nWrote offline leaderboard to: {out_dir}")


if __name__ == "__main__":
    main()
