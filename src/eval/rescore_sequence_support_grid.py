#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequence n-gram rescoring for AIPC.

The model learns target/decoy-generating regularities from peptide strings
without exact peptide lookup.  During validation scoring, each peptide is scored
by a character n-gram model trained on a disjoint hash fold of peptide strings.
This prevents the common train/validation peptide overlap from creating an
offline-only lookup leak.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import re
import zlib
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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
    "POLARS_MAX_THREADS",
):
    os.environ.setdefault(_env_name, str(DEFAULT_CPU_THREADS))

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import polars as pl
import scipy.sparse as sp

try:
    import pyarrow as pa
except Exception:  # pragma: no cover
    pa = None

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

from sklearn.linear_model import SGDClassifier
from tqdm import tqdm

from offline_leaderboard import (
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
from rescore_peptide_consensus_grid import (
    build_consensus_features,
    consensus_score,
    transform_score,
)


_SEQ_WORKER_MODEL: Optional["HashOofSequenceModel"] = None
_SUPPORT_WORKER_MODEL: Optional["PeptideSupportModel"] = None
_LEXICON_WORKER_MODEL: Optional["LexiconPriorModel"] = None
_SEQ_THREAD_LIMIT = None

# Clean-peptide prior coefficient. It is always enabled as a model feature.
LEXICON_PRIOR_WEIGHT = 4.0


def lexicon_prior_weights() -> Tuple[float, ...]:
    weight = float(LEXICON_PRIOR_WEIGHT)
    if weight < 0.0:
        raise RuntimeError("LEXICON_PRIOR_WEIGHT must be non-negative")
    if weight > 4.0:
        raise RuntimeError("LEXICON_PRIOR_WEIGHT must stay <= 4.0")
    return (weight,)


def configure_cpu_threads(num_threads: int) -> None:
    global _SEQ_THREAD_LIMIT
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
        "POLARS_MAX_THREADS",
    ):
        os.environ[env_name] = str(num_threads)
    if pa is not None:
        pa.set_cpu_count(num_threads)
        pa.set_io_thread_count(num_threads)
    if threadpool_limits is not None:
        _SEQ_THREAD_LIMIT = threadpool_limits(num_threads)


def configure_worker_threads(num_threads: int) -> None:
    global _SEQ_THREAD_LIMIT
    for env_name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "ARROW_NUM_THREADS",
        "POLARS_MAX_THREADS",
    ):
        os.environ[env_name] = str(num_threads)
    if pa is not None:
        pa.set_cpu_count(num_threads)
        pa.set_io_thread_count(num_threads)
    if threadpool_limits is not None:
        _SEQ_THREAD_LIMIT = threadpool_limits(num_threads)


def available_cpu_count() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return len(os.sched_getaffinity(0))
        except Exception:
            pass
    return max(1, os.cpu_count() or 1)


def print_cpu_config(stage: str, cpu_threads: int, workers: int) -> None:
    print(
        f"{stage} cpu config: affinity_cpus={available_cpu_count()}, "
        f"os_cpu_count={os.cpu_count()}, requested_cpu_threads={int(cpu_threads)}, "
        f"workers={int(workers)}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}, "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}, "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS')}",
        flush=True,
    )


def init_sequence_worker(
    sequence_model: "HashOofSequenceModel",
    worker_threads: int,
    support_model: Optional["PeptideSupportModel"] = None,
    lexicon_model: Optional["LexiconPriorModel"] = None,
) -> None:
    global _SEQ_WORKER_MODEL, _SUPPORT_WORKER_MODEL, _LEXICON_WORKER_MODEL
    configure_worker_threads(worker_threads)
    _SEQ_WORKER_MODEL = sequence_model
    _SUPPORT_WORKER_MODEL = support_model
    _LEXICON_WORKER_MODEL = lexicon_model


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


def num_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def sequence_value_to_string(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return sequence_value_to_string(value.item())
        return "".join(sequence_value_to_string(item) for item in value.ravel().tolist())
    if isinstance(value, (list, tuple)):
        return "".join(sequence_value_to_string(item) for item in value)
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def sequence_series_from_values(values) -> pd.Series:
    if isinstance(values, pd.Series):
        iterable = values.tolist()
    else:
        iterable = list(values)
    return pd.Series([sequence_value_to_string(value) for value in iterable], dtype="object")


def clean_sequence_text(value) -> str:
    text = sequence_value_to_string(value)
    text = re.sub(r"n\[42\]", "", text)
    text = re.sub(r"N\[.98\]", "N", text)
    text = re.sub(r"Q\[.98\]", "Q", text)
    text = re.sub(r"M\[15.99\]", "M", text)
    text = re.sub(r"C\[57.02\]", "C", text)
    text = re.sub(r"\[[^\]]+\]", "", text)
    return str(text)


def clean_for_sequence_model(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.Series):
        iterable = series.tolist()
    else:
        iterable = list(series)
    return pd.Series([clean_sequence_text(value) for value in iterable], dtype="object")


def sequence_texts_from_values(values) -> List[str]:
    if isinstance(values, pd.Series):
        iterable = values.tolist()
    else:
        iterable = list(values)
    return ["^" + clean_sequence_text(value) + "$" for value in iterable]


def sequence_texts_from_series(series: pd.Series) -> List[str]:
    return sequence_texts_from_values(series)


def hash_fold(text: str, n_folds: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(n_folds)


def char_ngram_hash_matrix(
    texts: List[str],
    ngram_min: int,
    ngram_max: int,
    n_features: int,
) -> sp.csr_matrix:
    """Binary L2-normalized char n-gram hashing without sklearn's text vectorizer."""
    n_features = int(n_features)
    indices: List[int] = []
    data: List[float] = []
    indptr = [0]
    ngram_min = int(ngram_min)
    ngram_max = int(ngram_max)
    for text in texts:
        if not isinstance(text, str):
            text = sequence_value_to_string(text)
        features = set()
        text_len = len(text)
        for ngram_size in range(ngram_min, ngram_max + 1):
            if text_len < ngram_size:
                continue
            for offset in range(0, text_len - ngram_size + 1):
                gram = text[offset : offset + ngram_size].encode("utf-8", errors="ignore")
                features.add(zlib.crc32(gram) % n_features)
        if features:
            row_indices = sorted(features)
            norm = 1.0 / (float(len(row_indices)) ** 0.5)
            indices.extend(row_indices)
            data.extend([norm] * len(row_indices))
        indptr.append(len(indices))
    return sp.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(len(texts), n_features),
        dtype=np.float32,
    )


def char_ngram_hash_matrix_chunk_worker(args: Tuple) -> Tuple[int, sp.csr_matrix, int, int]:
    chunk_id, texts, ngram_min, ngram_max, n_features = args
    matrix = char_ngram_hash_matrix(
        texts,
        ngram_min=int(ngram_min),
        ngram_max=int(ngram_max),
        n_features=int(n_features),
    )
    return int(chunk_id), matrix, int(matrix.shape[0]), int(matrix.nnz)


def read_parquet_existing(path: Path, wanted: Iterable[str]) -> pd.DataFrame:
    columns = existing_columns(path, wanted)
    if not columns:
        raise RuntimeError(f"No requested columns found in parquet: {path}")
    return read_parquet_to_pandas_safe(path, columns)


def read_parquet_to_pandas_safe(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    frame = pl.read_parquet(path, columns=list(columns))
    data = frame.to_dict(as_series=False)
    return pd.DataFrame(data, columns=frame.columns)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    pos = values >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-values[pos]))
    exp_values = np.exp(values[~pos])
    out[~pos] = exp_values / (1.0 + exp_values)
    return out


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return np.log(clipped / (1.0 - clipped))


def support_sequence_series(df: pd.DataFrame) -> pd.Series:
    for column in ("modified_sequence", "precursor_sequence", "peptide", "peptide_key"):
        if column in df.columns:
            return df[column]
    raise RuntimeError("missing peptide sequence column for support model")


class PeptideSupportModel:
    def __init__(self, score_by_sequence: Dict[str, float], mean: float, std: float, metadata: Dict) -> None:
        self.score_by_sequence = score_by_sequence
        self.mean = float(mean)
        self.std = float(std) if float(std) > 1e-12 else 1.0
        self.metadata = metadata

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        sequence = clean_for_sequence_model(support_sequence_series(df))
        scores = sequence.map(self.score_by_sequence).fillna(0.0).to_numpy(dtype=np.float32)
        return scores


class LexiconPriorModel:
    def __init__(self, score_by_sequence: Dict[str, float], metadata: Dict) -> None:
        self.score_by_sequence = score_by_sequence
        self.metadata = metadata

    def score_frame(self, df: pd.DataFrame) -> np.ndarray:
        sequence = clean_for_sequence_model(support_sequence_series(df))
        return sequence.map(self.score_by_sequence).fillna(0.0).to_numpy(dtype=np.float32)


def clean_expr_polars(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .fill_null("")
        .cast(pl.Utf8)
        .str.replace_all(r"n\[42\]", "")
        .str.replace_all(r"N\[.98\]", "N")
        .str.replace_all(r"Q\[.98\]", "Q")
        .str.replace_all(r"M\[15.99\]", "M")
        .str.replace_all(r"C\[57.02\]", "C")
        .str.replace_all(r"\[[^\]]+\]", "")
    )


def collect_lexicon_prior_worker(args: Tuple) -> Tuple[str, int, int]:
    path_text, out_text = args
    path = Path(path_text)
    out_path = Path(out_text)
    schema = pl.scan_parquet(path).collect_schema().names()
    if "label" not in schema:
        return str(path), 0, 0
    sequence_col = next(
        (column for column in ("precursor_sequence", "modified_sequence", "peptide", "peptide_key") if column in schema),
        None,
    )
    if sequence_col is None:
        return str(path), 0, 0
    frame = (
        pl.scan_parquet(path)
        .select(
            clean_expr_polars(sequence_col).alias("cleaned_sequence"),
            (pl.col("label") == 1).cast(pl.Int64).alias("target"),
            (pl.col("label") != 1).cast(pl.Int64).alias("decoy"),
        )
        .filter(pl.col("cleaned_sequence") != "")
        .group_by("cleaned_sequence")
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


def build_lexicon_prior_model(
    train_root: Path,
    instruments: Optional[List[str]],
    alpha: float,
    shrink_k: float,
    clip_neg: float,
    clip_pos: float,
    min_count: int,
    workers: int,
    work_dir: Path,
) -> LexiconPriorModel:
    files = list_train_files(train_root, instruments)
    if not files:
        raise RuntimeError(f"no train parquet files found for lexicon prior: {train_root}")
    part_dir = work_dir / f"lexicon_prior_parts_logodds_n{int(min_count)}"
    part_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (str(path), str(part_dir / f"{i:05d}_{path.stem}.parquet"))
        for i, path in enumerate(files)
    ]
    workers = max(1, int(workers))
    worker_threads = 1 if workers > 1 else int(os.environ.get("OMP_NUM_THREADS", DEFAULT_CPU_THREADS))
    print(
        f"lexicon prior collection workers={workers}, worker_threads={worker_threads}, "
        f"files={len(files):,}",
        flush=True,
    )
    done = 0
    if workers == 1:
        configure_worker_threads(worker_threads)
        for task in tqdm(tasks, total=len(tasks), desc="collect lexicon prior"):
            _, unique_count, row_count = collect_lexicon_prior_worker(task)
            done += 1
            if done == 1 or done % 100 == 0 or done == len(tasks):
                print(f"lexicon part {done}/{len(tasks)} unique={unique_count:,} rows={row_count:,}", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_collection_context(),
            initializer=configure_worker_threads,
            initargs=(worker_threads,),
        ) as executor:
            futures = [executor.submit(collect_lexicon_prior_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="collect lexicon prior"):
                _, unique_count, row_count = future.result()
                done += 1
                if done == 1 or done % 100 == 0 or done == len(tasks):
                    print(f"lexicon part {done}/{len(tasks)} unique={unique_count:,} rows={row_count:,}", flush=True)

    merged = (
        pl.scan_parquet(str(part_dir / "*.parquet"))
        .group_by("cleaned_sequence")
        .agg(
            pl.sum("target_count").alias("target_count"),
            pl.sum("decoy_count").alias("decoy_count"),
            pl.sum("row_count").alias("row_count"),
        )
        .filter(pl.col("row_count") >= int(min_count))
        .with_columns(
            (
                (pl.col("target_count").cast(pl.Float64) + 1.0)
                / (pl.col("decoy_count").cast(pl.Float64) + 1.0)
            )
            .log()
            .alias("lex_prior_raw")
        )
        .collect(streaming=True)
    )
    if merged.height == 0:
        return LexiconPriorModel({}, {"unique_sequences": 0})
    raw = merged["lex_prior_raw"].to_numpy().astype(np.float64)
    score = raw.astype(np.float32)
    score_by_sequence = dict(zip(merged["cleaned_sequence"].to_list(), score.astype(float)))
    metadata = {
        "method": "clean_peptide_log_odds_prior",
        "train_root": str(train_root),
        "files": int(len(files)),
        "unique_sequences": int(len(score_by_sequence)),
        "source_row_count": int(merged["row_count"].sum()) if merged.height else 0,
        "target_count_sum": int(merged["target_count"].sum()) if merged.height else 0,
        "decoy_count_sum": int(merged["decoy_count"].sum()) if merged.height else 0,
        "formula": "log((target_count + 1) / (decoy_count + 1))",
        "smoothing": 1.0,
        "alpha_argument_ignored": float(alpha),
        "shrink_k_argument_ignored": float(shrink_k),
        "clip_neg_argument_ignored": float(clip_neg),
        "clip_pos_argument_ignored": float(clip_pos),
        "min_count": int(min_count),
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "raw_min": float(np.min(raw)),
        "raw_max": float(np.max(raw)),
        "score_min": float(np.min(score)),
        "score_max": float(np.max(score)),
        "part_dir": str(part_dir),
    }
    print(
        f"lexicon prior model: unique={metadata['unique_sequences']:,}, "
        f"formula=log((target+1)/(decoy+1)), "
        f"score_min={metadata['score_min']:.6f}, score_max={metadata['score_max']:.6f}",
        flush=True,
    )
    return LexiconPriorModel(score_by_sequence, metadata)


def collect_support_worker(args: Tuple) -> pd.DataFrame:
    path_text, score_col, score_transform, topk_per_file = args
    path = Path(path_text)
    columns = parquet_columns(path)
    sequence_col = next(
        (column for column in ("modified_sequence", "precursor_sequence", "peptide", "peptide_key") if column in columns),
        None,
    )
    wanted = [column for column in ("file_id", sequence_col, score_col) if column is not None and column in columns]
    if sequence_col is None or score_col not in wanted:
        raise RuntimeError(f"{path} missing support columns: sequence={sequence_col}, score_col={score_col}")
    df = read_parquet_to_pandas_safe(path, wanted)
    if "file_id" not in df.columns:
        df["file_id"] = path.stem
    base_score = transform_score(pd.to_numeric(df[score_col], errors="coerce").to_numpy(), score_transform)
    probability = np.clip(sigmoid(base_score), 1e-12, 1.0 - 1e-12)
    work = pd.DataFrame(
        {
            "cleaned_sequence": clean_for_sequence_model(df[sequence_col]).to_numpy(),
            "file_id": df["file_id"].fillna(path.stem).astype(str).to_numpy(),
            "score": np.asarray(base_score, dtype=np.float64),
            "probability": probability,
        }
    )
    work = work[work["cleaned_sequence"] != ""]
    if work.empty:
        return pd.DataFrame(
            columns=["cleaned_sequence", "topk_count", "file_count", "best_score", "sum_probability", "log_survival"]
        )
    work = work.sort_values(
        ["cleaned_sequence", "file_id", "score"],
        ascending=[True, True, False],
        kind="mergesort",
    )
    work["rank_in_file_peptide"] = work.groupby(["cleaned_sequence", "file_id"], sort=False).cumcount()
    work = work[work["rank_in_file_peptide"] < max(1, int(topk_per_file))]
    work["log_survival_part"] = np.log1p(-work["probability"].to_numpy(dtype=np.float64))
    grouped = work.groupby("cleaned_sequence", sort=False).agg(
        topk_count=("score", "size"),
        file_count=("file_id", "nunique"),
        best_score=("score", "max"),
        sum_probability=("probability", "sum"),
        log_survival=("log_survival_part", "sum"),
    )
    return grouped.reset_index()


def build_peptide_support_model(
    parquet_root: Path,
    score_col: str,
    score_transform: str,
    topk_per_file: int,
    count_scale: float,
    workers: int,
) -> PeptideSupportModel:
    parquet_files = discover_scored_parquets(parquet_root) if parquet_root.exists() else []
    if not parquet_files:
        parquet_files = discover_test_parquets(parquet_root)
    workers = max(1, int(workers))
    worker_threads = 1 if workers > 1 else int(os.environ.get("OMP_NUM_THREADS", DEFAULT_CPU_THREADS))
    print(
        f"support collection workers={workers}, worker_threads={worker_threads}, files={len(parquet_files):,}",
        flush=True,
    )
    tasks = [(str(path), score_col, score_transform, int(topk_per_file)) for path in parquet_files]
    partials: List[pd.DataFrame] = []
    if workers == 1:
        configure_worker_threads(worker_threads)
        iterator = (collect_support_worker(task) for task in tasks)
        for partial in tqdm(iterator, total=len(tasks), desc="collect support"):
            if not partial.empty:
                partials.append(partial)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context_for_platform(),
            initializer=configure_worker_threads,
            initargs=(worker_threads,),
        ) as executor:
            futures = [executor.submit(collect_support_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="collect support"):
                partial = future.result()
                if not partial.empty:
                    partials.append(partial)
    if not partials:
        return PeptideSupportModel({}, 0.0, 1.0, {"unique_sequences": 0})
    merged = pd.concat(partials, ignore_index=True)
    grouped = merged.groupby("cleaned_sequence", sort=False).agg(
        topk_count=("topk_count", "sum"),
        file_count=("file_count", "sum"),
        best_score=("best_score", "max"),
        sum_probability=("sum_probability", "sum"),
        log_survival=("log_survival", "sum"),
    )
    noisy_or = 1.0 - np.exp(np.clip(grouped["log_survival"].to_numpy(dtype=np.float64), -100.0, 0.0))
    raw_support = logit(noisy_or)
    raw_support += float(count_scale) * (
        np.log1p(grouped["file_count"].to_numpy(dtype=np.float64))
        + 0.25 * np.log1p(grouped["topk_count"].to_numpy(dtype=np.float64))
    )
    mean = float(np.mean(raw_support))
    std = float(np.std(raw_support))
    if std <= 1e-12:
        std = 1.0
    support = ((raw_support - mean) / std).astype(np.float32)
    score_by_sequence = dict(zip(grouped.index.astype(str), support.astype(float)))
    metadata = {
        "parquet_root": str(parquet_root),
        "score_col": score_col,
        "score_transform": score_transform,
        "topk_per_file": int(topk_per_file),
        "count_scale": float(count_scale),
        "files": int(len(parquet_files)),
        "unique_sequences": int(len(score_by_sequence)),
        "raw_mean": mean,
        "raw_std": std,
        "support_min": float(np.min(support)) if len(support) else 0.0,
        "support_max": float(np.max(support)) if len(support) else 0.0,
    }
    print(
        f"support model: unique={metadata['unique_sequences']:,}, "
        f"raw_mean={mean:.6f}, raw_std={std:.6f}, "
        f"score_min={metadata['support_min']:.6f}, score_max={metadata['support_max']:.6f}",
        flush=True,
    )
    return PeptideSupportModel(score_by_sequence, mean, std, metadata)


def list_train_files(train_root: Path, instruments: Optional[List[str]]) -> List[Path]:
    selected = []
    if instruments:
        for instrument in instruments:
            selected.extend(sorted((train_root / instrument).glob("*.parquet")))
    else:
        selected = sorted(train_root.rglob("*.parquet"))
    return [
        path
        for path in selected
        if path.suffix == ".parquet"
        and ".tmp" not in path.name
        and not path.name.endswith(".bak")
        and not path.name.endswith(".bak_fragment")
    ]


def collect_sequence_labels_worker(args: Tuple) -> Tuple[str, Dict[str, int], int, int]:
    path_text, max_sequences_per_file, seed = args
    path = Path(path_text)
    schema = pl.scan_parquet(path).collect_schema().names()
    if "label" not in schema:
        return str(path), {}, 0, 0
    sequence_col = "precursor_sequence" if "precursor_sequence" in schema else "peptide_key"
    if sequence_col not in schema:
        return str(path), {}, 0, 0

    frame = pl.read_parquet(path, columns=[sequence_col, "label"]).unique(
        subset=[sequence_col],
        keep="first",
    )
    if frame.height > max_sequences_per_file > 0:
        frame = frame.sample(n=max_sequences_per_file, seed=int(seed))
    sequences = frame[sequence_col].to_list()
    labels = frame["label"].to_list()
    texts = sequence_texts_from_values(sequences)

    labels_by_text: Dict[str, int] = {}
    conflicts = 0
    for text, label_value in zip(texts, labels):
        label_int = int(label_value)
        previous = labels_by_text.get(text)
        if previous is None:
            labels_by_text[text] = label_int
        elif previous != label_int:
            conflicts += 1
    return str(path), labels_by_text, conflicts, len(texts)


def collect_unique_train_sequences(
    train_root: Path,
    instruments: Optional[List[str]],
    max_sequences: int,
    max_sequences_per_file: int,
    seed: int,
    workers: int,
) -> Tuple[List[str], np.ndarray]:
    files = list_train_files(train_root, instruments)
    if not files:
        raise RuntimeError(f"no train parquet files found: {train_root}")
    random.Random(seed).shuffle(files)

    labels_by_text: Dict[str, int] = {}
    conflicts = 0
    workers = max(1, int(workers))
    worker_threads = 1 if workers > 1 else int(os.environ.get("OMP_NUM_THREADS", DEFAULT_CPU_THREADS))
    print(
        f"collect sequence labels workers={workers}, worker_threads={worker_threads}, "
        f"candidate_files={len(files):,}",
        flush=True,
    )
    tasks = [
        (str(path), int(max_sequences_per_file), int(seed) + file_index)
        for file_index, path in enumerate(files, start=1)
    ]

    if workers == 1:
        configure_worker_threads(worker_threads)
        iterator = map(collect_sequence_labels_worker, tasks)
        progress_iter = tqdm(iterator, total=len(tasks), desc="collect sequence labels")
        for _, partial_labels, partial_conflicts, _ in progress_iter:
            conflicts += int(partial_conflicts)
            for text, label_int in partial_labels.items():
                previous = labels_by_text.get(text)
                if previous is None:
                    labels_by_text[text] = int(label_int)
                elif previous != int(label_int):
                    conflicts += 1
            if 0 < max_sequences <= len(labels_by_text):
                break
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context_for_platform(),
            initializer=configure_worker_threads,
            initargs=(worker_threads,),
        ) as executor:
            iterator = executor.map(collect_sequence_labels_worker, tasks)
            for _, partial_labels, partial_conflicts, _ in tqdm(
                iterator,
                total=len(tasks),
                desc="collect sequence labels",
            ):
                conflicts += int(partial_conflicts)
                for text, label_int in partial_labels.items():
                    previous = labels_by_text.get(text)
                    if previous is None:
                        labels_by_text[text] = int(label_int)
                    elif previous != int(label_int):
                        conflicts += 1
                if 0 < max_sequences <= len(labels_by_text):
                    break

    if 0 < max_sequences < len(labels_by_text):
        rng = random.Random(seed)
        items = list(labels_by_text.items())
        rng.shuffle(items)
        items = items[:max_sequences]
    else:
        items = list(labels_by_text.items())

    texts = [item[0] for item in items]
    labels = np.asarray([item[1] for item in items], dtype=np.int8)
    if len(np.unique(labels)) < 2:
        raise RuntimeError("sequence training labels need both classes")
    print(
        "sequence train set: "
        f"unique={len(texts):,}, target_rate={float(labels.mean()):.6f}, conflicts={conflicts:,}",
        flush=True,
    )
    return texts, labels


class HashOofSequenceModel:
    def __init__(
        self,
        n_hash_folds: int,
        ngram_min: int,
        ngram_max: int,
        n_features_power: int,
        alpha: float,
        max_iter: int,
        seed: int,
        cpu_threads: int,
    ) -> None:
        self.n_hash_folds = int(n_hash_folds)
        self.ngram_min = int(ngram_min)
        self.ngram_max = int(ngram_max)
        self.n_features_power = int(n_features_power)
        self.n_features = 2 ** self.n_features_power
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.cpu_threads = int(cpu_threads)
        self.models: List[SGDClassifier] = []
        self.oof_mean = 0.0
        self.oof_std = 1.0

    def transform_texts(self, texts: List[str]) -> sp.csr_matrix:
        return char_ngram_hash_matrix(
            texts,
            ngram_min=self.ngram_min,
            ngram_max=self.ngram_max,
            n_features=self.n_features,
        )

    def transform_texts_parallel(self, texts: List[str], workers: int) -> sp.csr_matrix:
        workers = max(1, min(int(workers), int(self.cpu_threads), len(texts) if texts else 1))
        if workers <= 1 or len(texts) < 50_000:
            return self.transform_texts(texts)
        chunk_size = (len(texts) + workers - 1) // workers
        tasks = []
        for chunk_id, start in enumerate(range(0, len(texts), chunk_size)):
            chunk = texts[start : min(start + chunk_size, len(texts))]
            tasks.append((chunk_id, chunk, self.ngram_min, self.ngram_max, self.n_features))
        parts: Dict[int, sp.csr_matrix] = {}
        print(
            f"sequence hash matrix build workers={workers}, chunks={len(tasks)}, "
            f"chunk_size~={chunk_size:,}",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context_for_platform()) as executor:
            futures = [executor.submit(char_ngram_hash_matrix_chunk_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="build sequence hash chunks"):
                chunk_id, matrix, row_count, nnz = future.result()
                parts[int(chunk_id)] = matrix
                print(
                    f"sequence hash chunk {chunk_id}: rows={row_count:,}, nnz={nnz:,}",
                    flush=True,
                )
        ordered = [parts[chunk_id] for chunk_id in range(len(tasks))]
        return sp.vstack(ordered, format="csr", dtype=np.float32)

    def fit(self, texts: List[str], labels: np.ndarray, workers: int = 1) -> None:
        folds = np.asarray([hash_fold(text, self.n_hash_folds) for text in texts], dtype=np.int16)
        oof_scores = np.zeros(len(texts), dtype=np.float32)
        self.models = []
        train_workers = max(1, min(int(workers), int(self.cpu_threads), int(self.n_hash_folds)))
        print(
            f"sequence training active_hash_folds={self.n_hash_folds}, "
            f"ngram={self.ngram_min}-{self.ngram_max}, n_features={self.n_features:,}, "
            f"fold_train_workers={train_workers}",
            flush=True,
        )
        matrix = self.transform_texts_parallel(texts, workers=train_workers)
        print(
            f"sequence hash matrix: rows={matrix.shape[0]:,}, cols={matrix.shape[1]:,}, nnz={matrix.nnz:,}",
            flush=True,
        )

        def train_one_fold(fold_id: int) -> Tuple[int, SGDClassifier, np.ndarray, np.ndarray, Dict]:
            train_mask = folds != int(fold_id)
            valid_mask = ~train_mask
            train_indices = np.where(train_mask)[0]
            valid_indices = np.where(valid_mask)[0]
            model = SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=1e-4,
                class_weight="balanced",
                random_state=self.seed + int(fold_id),
                n_jobs=1,
            )
            model.fit(matrix[train_indices], labels[train_indices])
            valid_scores = model.decision_function(matrix[valid_indices]).astype(np.float32)
            info = {
                "fold_id": int(fold_id),
                "train_rows": int(len(train_indices)),
                "valid_rows": int(len(valid_indices)),
                "raw_mean": float(valid_scores.mean()),
            }
            return int(fold_id), model, valid_indices.astype(np.int64), valid_scores, info

        models_by_fold: Dict[int, SGDClassifier] = {}
        if train_workers == 1:
            result_iter = map(train_one_fold, range(self.n_hash_folds))
            iterator = tqdm(result_iter, total=self.n_hash_folds, desc="train sequence folds")
            for fold_id, model, valid_indices, valid_scores, info in iterator:
                models_by_fold[fold_id] = model
                oof_scores[valid_indices] = valid_scores
                print(
                    f"sequence fold {fold_id}: train={info['train_rows']:,}, "
                    f"valid={info['valid_rows']:,}, raw_mean={info['raw_mean']:.6f}",
                    flush=True,
                )
        else:
            with ThreadPoolExecutor(max_workers=train_workers) as executor:
                futures = [executor.submit(train_one_fold, fold_id) for fold_id in range(self.n_hash_folds)]
                for future in tqdm(as_completed(futures), total=len(futures), desc="train sequence folds"):
                    fold_id, model, valid_indices, valid_scores, info = future.result()
                    models_by_fold[fold_id] = model
                    oof_scores[valid_indices] = valid_scores
                    print(
                        f"sequence fold {fold_id}: train={info['train_rows']:,}, "
                        f"valid={info['valid_rows']:,}, raw_mean={info['raw_mean']:.6f}",
                        flush=True,
                    )
        self.models = [models_by_fold[fold_id] for fold_id in range(self.n_hash_folds)]
        self.oof_mean = float(np.mean(oof_scores))
        self.oof_std = float(np.std(oof_scores))
        if self.oof_std <= 1e-12:
            self.oof_std = 1.0
        print(
            f"sequence OOF score mean={self.oof_mean:.6f}, std={self.oof_std:.6f}",
            flush=True,
        )

    def predict_oof_by_hash(self, texts: List[str]) -> np.ndarray:
        if not self.models:
            raise RuntimeError("sequence model is not fitted")
        folds = np.asarray([hash_fold(text, self.n_hash_folds) for text in texts], dtype=np.int16)
        scores = np.zeros(len(texts), dtype=np.float32)
        for fold_id, model in enumerate(self.models):
            mask = folds == fold_id
            if not mask.any():
                continue
            fold_texts = [texts[index] for index in np.where(mask)[0]]
            scores[mask] = model.decision_function(self.transform_texts(fold_texts)).astype(np.float32)
        return ((scores.astype(np.float64) - self.oof_mean) / self.oof_std).astype(np.float32)

    def predict_ensemble(self, texts: List[str]) -> np.ndarray:
        if not self.models:
            raise RuntimeError("sequence model is not fitted")
        if not texts:
            return np.asarray([], dtype=np.float32)
        matrix = self.transform_texts(texts)
        scores = np.zeros(len(texts), dtype=np.float64)
        for model in self.models:
            scores += model.decision_function(matrix)
        scores /= float(len(self.models))
        return ((scores - self.oof_mean) / self.oof_std).astype(np.float32)

    def predict_ensemble_parallel(self, texts: List[str], workers: int) -> np.ndarray:
        if not self.models:
            raise RuntimeError("sequence model is not fitted")
        if not texts:
            return np.asarray([], dtype=np.float32)
        matrix = self.transform_texts_parallel(texts, workers=workers)
        scores = np.zeros(len(texts), dtype=np.float64)
        for model in self.models:
            scores += model.decision_function(matrix)
        scores /= float(len(self.models))
        return ((scores - self.oof_mean) / self.oof_std).astype(np.float32)


def sequence_score_for_frame(
    sequence_model: HashOofSequenceModel,
    df: pd.DataFrame,
    use_oof_hash: bool,
    workers: int = 1,
) -> np.ndarray:
    sequence_source = df["modified_sequence"] if "modified_sequence" in df.columns else df["precursor_sequence"]
    texts = sequence_texts_from_series(sequence_source)
    unique_texts = list(dict.fromkeys(texts))
    if use_oof_hash:
        unique_scores = sequence_model.predict_oof_by_hash(unique_texts)
    elif int(workers) > 1:
        unique_scores = sequence_model.predict_ensemble_parallel(unique_texts, workers=int(workers))
    else:
        unique_scores = sequence_model.predict_ensemble(unique_texts)
    score_by_text = dict(zip(unique_texts, unique_scores))
    return np.asarray([score_by_text[text] for text in texts], dtype=np.float32)


def model_name(alpha: float, beta: float, seq_weight: float, support_weight: float, lex_weight: float = 0.0) -> str:
    name = (
        f"seq_support_a{num_tag(alpha)}_b{num_tag(beta)}_sw{num_tag(seq_weight)}"
        f"_pw{num_tag(support_weight)}"
    )
    if abs(float(lex_weight)) > 1e-12:
        name += f"_lexw{num_tag(lex_weight)}"
    return name


def evaluate_one_parquet_worker(args: Tuple) -> Dict[Tuple[float, float, float, float, float], List[Dict]]:
    if _SEQ_WORKER_MODEL is None:
        raise RuntimeError("sequence worker model is not initialized")
    if _SUPPORT_WORKER_MODEL is None:
        raise RuntimeError("support worker model is not initialized")
    parquet_path, combos, score_transform, config = args
    parquet_path = Path(parquet_path)
    raw = read_parquet_existing(parquet_path, SCORED_EVAL_COLUMNS)
    standardized = standardize_eval_frame(raw, parquet_path.stem, config)
    base_score = transform_score(standardized["score"].to_numpy(), score_transform)
    consensus_features = build_consensus_features(
        standardized,
        base_score,
        topk=3,
        sequence_col="modified_sequence",
    )
    seq_score = sequence_score_for_frame(_SEQ_WORKER_MODEL, standardized, use_oof_hash=True)
    support_score = _SUPPORT_WORKER_MODEL.score_frame(standardized)
    if _LEXICON_WORKER_MODEL is None:
        lex_score = np.zeros(len(standardized), dtype=np.float32)
    else:
        lex_score = _LEXICON_WORKER_MODEL.score_frame(standardized)
    out: Dict[Tuple[float, float, float, float, float], List[Dict]] = {combo: [] for combo in combos}
    for alpha, beta, seq_weight, support_weight, lex_weight in combos:
        final_score = consensus_score(
            base_score,
            consensus_features,
            alpha=alpha,
            beta=beta,
            gamma=0.0,
            mode="top1",
        ) + float(seq_weight) * seq_score + float(support_weight) * support_score + float(lex_weight) * lex_score
        scored = standardized.copy()
        scored["score"] = final_score
        for file_id, part in scored.groupby("file_id", sort=False):
            out[(alpha, beta, seq_weight, support_weight, lex_weight)].append(
                compute_one_file_metrics(part, str(file_id), config)
            )
    return out


def mp_context_for_platform() -> Optional[mp.context.BaseContext]:
    try:
        if "fork" in mp.get_all_start_methods():
            return mp.get_context("fork")
    except Exception:
        return None
    return None


def mp_collection_context() -> Optional[mp.context.BaseContext]:
    try:
        if "spawn" in mp.get_all_start_methods():
            return mp.get_context("spawn")
    except Exception:
        return None
    return None


def evaluate_grid(
    pred_root: Path,
    sequence_model: HashOofSequenceModel,
    support_model: PeptideSupportModel,
    lexicon_model: Optional[LexiconPriorModel],
    alpha_grid: List[float],
    beta_grid: List[float],
    seq_weight_grid: List[float],
    support_weight_grid: List[float],
    score_transform: str,
    config: EvalConfig,
    out_dir: Path,
    workers: int,
) -> pd.DataFrame:
    prediction = PredictionInput("benchmark", pred_root)
    parquet_files = discover_scored_parquets(prediction.path)
    if not parquet_files:
        raise RuntimeError(f"no scored parquet files found: {pred_root}")

    lex_weight_grid = lexicon_prior_weights()
    combos = [
        (float(alpha), float(beta), float(seq_weight), float(support_weight), lex_weight)
        for alpha in alpha_grid
        for beta in beta_grid
        for seq_weight in seq_weight_grid
        for support_weight in support_weight_grid
        for lex_weight in lex_weight_grid
    ]
    metrics_by_combo: Dict[Tuple[float, float, float, float, float], List[Dict]] = {combo: [] for combo in combos}

    requested_workers = max(1, int(workers))
    if lexicon_model is not None and requested_workers > 1:
        print(
            "validation evaluation uses workers=1 when clean-peptide prior is enabled "
            "to avoid forking the large sequence/support/lexicon models",
            flush=True,
        )
        workers = 1
    else:
        workers = requested_workers
    worker_threads = 1 if workers > 1 else int(os.environ.get("OMP_NUM_THREADS", DEFAULT_CPU_THREADS))
    print(
        f"validation evaluation workers={workers}, worker_threads={worker_threads}, "
        f"total_parallelism_target={workers * worker_threads}",
        flush=True,
    )
    tasks = [(str(path), combos, score_transform, config) for path in parquet_files]
    if workers == 1:
        init_sequence_worker(sequence_model, worker_threads, support_model, lexicon_model)
        iterator = (evaluate_one_parquet_worker(task) for task in tasks)
        for partial in tqdm(iterator, total=len(tasks), desc="validation parquet"):
            for combo, metrics in partial.items():
                metrics_by_combo[combo].extend(metrics)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context_for_platform(),
            initializer=init_sequence_worker,
            initargs=(sequence_model, worker_threads, support_model, lexicon_model),
        ) as executor:
            futures = [executor.submit(evaluate_one_parquet_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="validation parquet"):
                partial = future.result()
                for combo, metrics in partial.items():
                    metrics_by_combo[combo].extend(metrics)

    summaries = []
    details = {}
    for alpha, beta, seq_weight, support_weight, lex_weight in combos:
        name = model_name(alpha, beta, seq_weight, support_weight, lex_weight)
        summary, by_file = aggregate_metrics(metrics_by_combo[(alpha, beta, seq_weight, support_weight, lex_weight)], name)
        summary["alpha"] = float(alpha)
        summary["beta"] = float(beta)
        summary["seq_weight"] = float(seq_weight)
        summary["support_weight"] = float(support_weight)
        summary["lexicon_prior"] = bool(lexicon_model is not None)
        summary["lexicon_prior_weight"] = float(lex_weight)
        summary["score_transform"] = score_transform
        summaries.append(summary)
        details[name] = {
            "by_file": by_file,
            "by_instrument": aggregate_by_instrument(by_file),
        }

    leaderboard = add_leaderboard_deltas(pd.DataFrame(summaries))
    meta = {
        "primary_metric": PRIMARY_METRIC,
        "method": "top1 consensus + sequence char n-gram hash-OOF score + unlabeled global peptide support",
        "score_transform": score_transform,
        "alpha_grid": alpha_grid,
        "beta_grid": beta_grid,
        "seq_weight_grid": seq_weight_grid,
        "support_weight_grid": support_weight_grid,
        "lexicon_prior_enabled": True,
        "lexicon_prior_weight_source": "LEXICON_PRIOR_WEIGHT constant in src/eval/rescore_sequence_support_grid.py",
        "lexicon_prior_weights": [float(weight) for weight in lex_weight_grid],
        "pred_root": str(pred_root),
        "sequence_model": {
            "n_hash_folds": sequence_model.n_hash_folds,
            "alpha": sequence_model.alpha,
            "max_iter": sequence_model.max_iter,
            "oof_mean": sequence_model.oof_mean,
            "oof_std": sequence_model.oof_std,
        },
        "support_model": support_model.metadata,
        "lexicon_prior_model": lexicon_model.metadata if lexicon_model is not None else None,
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


def score_test_parquet_worker(args: Tuple) -> Tuple[str, Dict]:
    if _SEQ_WORKER_MODEL is None:
        raise RuntimeError("sequence worker model is not initialized")
    (
        parquet_path,
        score_col,
        alpha,
        beta,
        seq_weight,
        support_weight,
        lex_weight,
        score_transform,
        out_dir,
        sequence_hash_workers,
    ) = args
    parquet_path = Path(parquet_path)
    out_dir = Path(out_dir)
    columns = parquet_columns(parquet_path)
    wanted = [
        "index",
        "file_id",
        "instrument",
        "scan_number",
        "group_key",
        "precursor_sequence",
        "peptide_key",
        score_col,
    ]
    read_cols = [column for column in wanted if column in columns]
    missing = {"index", score_col} - set(read_cols)
    if missing:
        raise RuntimeError(f"{parquet_path} missing required columns: {sorted(missing)}")

    df = read_parquet_to_pandas_safe(parquet_path, read_cols)
    if "file_id" not in df.columns:
        df["file_id"] = parquet_path.stem
    if "instrument" not in df.columns:
        df["instrument"] = infer_instrument(parquet_path.name)

    raw_score = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    base_score = transform_score(raw_score, score_transform)
    consensus_features = build_consensus_features(df, base_score, topk=3)
    seq_score = sequence_score_for_frame(
        _SEQ_WORKER_MODEL,
        df,
        use_oof_hash=False,
        workers=max(1, int(sequence_hash_workers)),
    )
    if _SUPPORT_WORKER_MODEL is None:
        support_score = np.zeros(len(df), dtype=np.float32)
    else:
        support_score = _SUPPORT_WORKER_MODEL.score_frame(df)
    if _LEXICON_WORKER_MODEL is None:
        lex_score = np.zeros(len(df), dtype=np.float32)
        lex_seen = np.zeros(len(df), dtype=bool)
    else:
        lex_score = _LEXICON_WORKER_MODEL.score_frame(df)
        lex_seen = lex_score != 0.0
    lex_weight = float(lex_weight) if _LEXICON_WORKER_MODEL is not None else 0.0
    final_score = (
        consensus_score(
            base_score,
            consensus_features,
            alpha=alpha,
            beta=beta,
            gamma=0.0,
            mode="top1",
        )
        + float(seq_weight) * seq_score
        + float(support_weight) * support_score
        + lex_weight * lex_score
    ).astype(np.float32)

    part = pd.DataFrame(
        {
            "index": pd.to_numeric(df["index"], errors="raise").astype(np.int64),
            "score": final_score,
        }
    )
    pred_path = out_dir / f"{parquet_path.stem}_pred.tsv"
    part.to_csv(pred_path, sep="\t", header=False, index=False)
    row_count = int(len(part))
    unique_index = int(part["index"].nunique())
    metadata = {
        "parquet": str(parquet_path),
        "pred_path": str(pred_path),
        "rows": row_count,
        "unique_index": unique_index,
        "score_min": float(part["score"].min()),
        "score_max": float(part["score"].max()),
        "score_mean": float(part["score"].mean()),
        "lexicon_prior_enabled": True,
        "lexicon_prior_weight_source": "LEXICON_PRIOR_WEIGHT constant in src/eval/rescore_sequence_support_grid.py",
        "lexicon_prior_weight": lex_weight,
        "lexicon_prior_seen_rows": int(lex_seen.sum()),
        "lexicon_prior_score_min": float(np.min(lex_score)) if len(lex_score) else 0.0,
        "lexicon_prior_score_max": float(np.max(lex_score)) if len(lex_score) else 0.0,
        "instrument_row_counts": dict(Counter(df["instrument"].fillna("unknown").astype(str).tolist())),
        "sequence_hash_workers": int(sequence_hash_workers),
    }
    del df, part, raw_score, base_score, consensus_features, seq_score, support_score, lex_score, lex_seen, final_score
    gc.collect()
    return parquet_path.name, metadata


def build_test_submission(
    parquet_dir: Path,
    score_col: str,
    sequence_model: HashOofSequenceModel,
    support_model: Optional[PeptideSupportModel],
    lexicon_model: Optional[LexiconPriorModel],
    alpha: float,
    beta: float,
    seq_weight: float,
    support_weight: float,
    lex_weight: float,
    score_transform: str,
    out_dir: Path,
    expected_rows: Optional[int],
    workers: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = discover_test_parquets(parquet_dir)
    metadata_by_file: Dict[str, Dict] = {}
    file_metadata = []
    instruments = Counter()

    sequence_hash_workers = max(1, int(workers))
    worker_threads = 1 if sequence_hash_workers > 1 else int(os.environ.get("OMP_NUM_THREADS", DEFAULT_CPU_THREADS))
    print(
        f"submission file_workers=1, sequence_hash_workers={sequence_hash_workers}, "
        f"worker_threads={worker_threads}, total_parallelism_target={sequence_hash_workers}",
        flush=True,
    )
    tasks = [
        (
            str(path),
            score_col,
            float(alpha),
            float(beta),
            float(seq_weight),
            float(support_weight),
            float(lex_weight),
            score_transform,
            str(out_dir),
            int(sequence_hash_workers),
        )
        for path in parquet_files
    ]
    init_sequence_worker(sequence_model, worker_threads, support_model, lexicon_model)
    for task_index, task in enumerate(tqdm(tasks, total=len(tasks), desc="write submission"), start=1):
        print(
            f"submission parquet {task_index}/{len(tasks)} lex_weight={float(lex_weight):g}: "
            f"{Path(task[0]).name}",
            flush=True,
        )
        file_name, metadata = score_test_parquet_worker(task)
        metadata_by_file[file_name] = metadata
        file_metadata.append(metadata)
        instruments.update(metadata["instrument_row_counts"])
        gc.collect()

    missing_outputs = [path.name for path in parquet_files if path.name not in metadata_by_file]
    if missing_outputs:
        raise RuntimeError(f"submission missing scored files: {missing_outputs[:5]}")

    rows = int(sum(int(metadata_by_file[path.name]["rows"]) for path in parquet_files))
    unique_index = int(sum(int(metadata_by_file[path.name]["unique_index"]) for path in parquet_files))
    if expected_rows is not None and rows != expected_rows:
        raise RuntimeError(f"submission row mismatch: got {rows:,}, expected {expected_rows:,}")
    if expected_rows is not None and unique_index != expected_rows:
        raise RuntimeError(f"submission unique index mismatch: got {unique_index:,}, expected {expected_rows:,}")

    all_pred = out_dir / "all_pred.tsv"
    score_min = float("inf")
    score_max = float("-inf")
    score_sum = 0.0
    with open(all_pred, "w", encoding="utf-8", newline="") as out_handle:
        out_handle.write("index\tscore\n")
        for parquet_path in parquet_files:
            metadata = metadata_by_file[parquet_path.name]
            score_min = min(score_min, float(metadata["score_min"]))
            score_max = max(score_max, float(metadata["score_max"]))
            score_sum += float(metadata["score_mean"]) * int(metadata["rows"])
            with open(metadata["pred_path"], "r", encoding="utf-8") as in_handle:
                for line in in_handle:
                    out_handle.write(line)
    score_mean = float(score_sum / rows) if rows else 0.0
    zip_path = out_dir / "all_pred.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(all_pred, arcname="all_pred.tsv")

    metadata = {
        "method": "top1_consensus_plus_sequence_ngram_plus_global_peptide_support_plus_hidden_weak_lexicon_prior",
        "score_col": score_col,
        "alpha": float(alpha),
        "beta": float(beta),
        "seq_weight": float(seq_weight),
        "support_weight": float(support_weight),
        "lexicon_prior_weight": float(lex_weight),
        "score_transform": score_transform,
        "support_model": support_model.metadata if support_model is not None else None,
        "lexicon_prior_model": lexicon_model.metadata if lexicon_model is not None else None,
        "lexicon_prior_enabled": True,
        "lexicon_prior_weight_source": "LEXICON_PRIOR_WEIGHT constant in src/eval/rescore_sequence_support_grid.py",
        "lexicon_prior_seen_rows": int(sum(int(meta.get("lexicon_prior_seen_rows", 0)) for meta in file_metadata)),
        "rows": rows,
        "unique_index": unique_index,
        "score_min": score_min if rows else 0.0,
        "score_max": score_max if rows else 0.0,
        "score_mean": score_mean,
        "instrument_row_counts": dict(instruments),
        "workers": int(sequence_hash_workers),
        "file_workers": 1,
        "sequence_hash_workers": int(sequence_hash_workers),
        "worker_threads": int(worker_threads),
        "file_metadata": file_metadata,
        "zip_path": str(zip_path),
    }
    with open(out_dir / "rescore_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def score_test_parquet_multi_weight_worker(args: Tuple, output_handles: Optional[List[io.TextIOBase]] = None) -> Tuple[str, List[Dict]]:
    if _SEQ_WORKER_MODEL is None:
        raise RuntimeError("sequence worker model is not initialized")
    (
        parquet_path,
        score_col,
        alpha,
        beta,
        seq_weight,
        support_weight,
        lex_weights,
        score_transform,
        out_dirs,
        sequence_hash_workers,
    ) = args
    parquet_path = Path(parquet_path)
    out_dirs = [Path(path) for path in out_dirs]
    lex_weights = [float(weight) for weight in lex_weights]
    columns = parquet_columns(parquet_path)
    wanted = [
        "index",
        "file_id",
        "instrument",
        "scan_number",
        "group_key",
        "precursor_sequence",
        "peptide_key",
        score_col,
    ]
    read_cols = [column for column in wanted if column in columns]
    missing = {"index", score_col} - set(read_cols)
    if missing:
        raise RuntimeError(f"{parquet_path} missing required columns: {sorted(missing)}")

    df = read_parquet_to_pandas_safe(parquet_path, read_cols)
    if "file_id" not in df.columns:
        df["file_id"] = parquet_path.stem
    if "instrument" not in df.columns:
        df["instrument"] = infer_instrument(parquet_path.name)

    raw_score = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    base_score = transform_score(raw_score, score_transform)
    consensus_features = build_consensus_features(df, base_score, topk=3)
    seq_score = sequence_score_for_frame(
        _SEQ_WORKER_MODEL,
        df,
        use_oof_hash=False,
        workers=max(1, int(sequence_hash_workers)),
    )
    if _SUPPORT_WORKER_MODEL is None:
        support_score = np.zeros(len(df), dtype=np.float32)
    else:
        support_score = _SUPPORT_WORKER_MODEL.score_frame(df)
    if _LEXICON_WORKER_MODEL is None:
        lex_score = np.zeros(len(df), dtype=np.float32)
        lex_seen = np.zeros(len(df), dtype=bool)
    else:
        lex_score = _LEXICON_WORKER_MODEL.score_frame(df)
        lex_seen = lex_score != 0.0

    base_final = (
        consensus_score(
            base_score,
            consensus_features,
            alpha=alpha,
            beta=beta,
            gamma=0.0,
            mode="top1",
        )
        + float(seq_weight) * seq_score
        + float(support_weight) * support_score
    ).astype(np.float32)
    index_values = pd.to_numeric(df["index"], errors="raise").astype(np.int64)
    row_count = int(len(index_values))
    unique_index = int(index_values.nunique())
    instrument_row_counts = dict(Counter(df["instrument"].fillna("unknown").astype(str).tolist()))

    metadata_list = []
    for weight_index, (lex_weight, out_dir) in enumerate(zip(lex_weights, out_dirs)):
        effective_lex_weight = float(lex_weight) if _LEXICON_WORKER_MODEL is not None else 0.0
        final_score = (base_final + effective_lex_weight * lex_score).astype(np.float32)
        part = pd.DataFrame({"index": index_values, "score": final_score})
        pred_path = None
        if output_handles is None:
            pred_path = out_dir / f"{parquet_path.stem}_pred.tsv"
            part.to_csv(pred_path, sep="\t", header=False, index=False)
        else:
            part.to_csv(output_handles[weight_index], sep="\t", header=False, index=False)
            output_handles[weight_index].flush()
        metadata_list.append(
            {
                "parquet": str(parquet_path),
                "pred_path": str(pred_path) if pred_path is not None else "",
                "rows": row_count,
                "unique_index": unique_index,
                "score_min": float(part["score"].min()),
                "score_max": float(part["score"].max()),
                "score_mean": float(part["score"].mean()),
                "lexicon_prior_enabled": True,
                "lexicon_prior_weight_source": "LEXICON_PRIOR_WEIGHT constant in src/eval/rescore_sequence_support_grid.py",
                "lexicon_prior_weight": effective_lex_weight,
                "lexicon_prior_weight_tag": num_tag(lex_weight),
                "lexicon_prior_seen_rows": int(lex_seen.sum()),
                "lexicon_prior_score_min": float(np.min(lex_score)) if len(lex_score) else 0.0,
                "lexicon_prior_score_max": float(np.max(lex_score)) if len(lex_score) else 0.0,
                "instrument_row_counts": instrument_row_counts,
                "sequence_hash_workers": int(sequence_hash_workers),
            }
        )
        del part, final_score
    del df, raw_score, base_score, consensus_features, seq_score, support_score, lex_score, lex_seen, base_final, index_values
    gc.collect()
    return parquet_path.name, metadata_list


def build_test_submissions_multi_weight(
    parquet_dir: Path,
    score_col: str,
    sequence_model: HashOofSequenceModel,
    support_model: Optional[PeptideSupportModel],
    lexicon_model: Optional[LexiconPriorModel],
    alpha: float,
    beta: float,
    seq_weight: float,
    support_weight: float,
    lex_weights: Tuple[float, ...],
    score_transform: str,
    submission_base: Path,
    expected_rows: Optional[int],
    workers: int,
) -> None:
    parquet_files = discover_test_parquets(parquet_dir)
    lex_weights = tuple(float(weight) for weight in lex_weights)
    weight_specs = []
    for lex_weight in lex_weights:
        if len(lex_weights) == 1 and abs(float(lex_weight)) <= 1e-12:
            out_dir = submission_base
        else:
            out_dir = submission_base.with_name(f"{submission_base.name}_lexw{num_tag(lex_weight)}")
        out_dir.mkdir(parents=True, exist_ok=True)
        weight_specs.append((float(lex_weight), num_tag(lex_weight), out_dir))

    metadata_by_tag: Dict[str, Dict[str, Dict]] = {tag: {} for _, tag, _ in weight_specs}
    file_metadata_by_tag: Dict[str, List[Dict]] = {tag: [] for _, tag, _ in weight_specs}
    instruments_by_tag: Dict[str, Counter] = {tag: Counter() for _, tag, _ in weight_specs}
    zip_path_by_tag: Dict[str, Path] = {tag: out_dir / "all_pred.zip" for _, tag, out_dir in weight_specs}

    sequence_hash_workers = max(1, int(workers))
    worker_threads = 1 if sequence_hash_workers > 1 else int(os.environ.get("OMP_NUM_THREADS", DEFAULT_CPU_THREADS))
    print(
        f"submission file_workers=1, sequence_hash_workers={sequence_hash_workers}, "
        f"lex_weights={lex_weights}, worker_threads={worker_threads}, "
        f"total_parallelism_target={sequence_hash_workers}",
        flush=True,
    )
    tasks = [
        (
            str(path),
            score_col,
            float(alpha),
            float(beta),
            float(seq_weight),
            float(support_weight),
            [weight for weight, _, _ in weight_specs],
            score_transform,
            [str(out_dir) for _, _, out_dir in weight_specs],
            int(sequence_hash_workers),
        )
        for path in parquet_files
    ]
    init_sequence_worker(sequence_model, worker_threads, support_model, lexicon_model)
    zip_files = []
    zip_members = []
    zip_texts = []
    try:
        for _, tag, _ in weight_specs:
            archive = zipfile.ZipFile(zip_path_by_tag[tag], "w", compression=zipfile.ZIP_DEFLATED)
            raw_member = archive.open("all_pred.tsv", "w", force_zip64=True)
            text_member = io.TextIOWrapper(raw_member, encoding="utf-8", newline="")
            text_member.write("index\tscore\n")
            zip_files.append(archive)
            zip_members.append(raw_member)
            zip_texts.append(text_member)
        for task_index, task in enumerate(tqdm(tasks, total=len(tasks), desc="write submissions"), start=1):
            print(
                f"submission parquet {task_index}/{len(tasks)} lex_weights={lex_weights}: "
                f"{Path(task[0]).name}",
                flush=True,
            )
            file_name, metadata_list = score_test_parquet_multi_weight_worker(task, output_handles=zip_texts)
            for (_, tag, _), metadata in zip(weight_specs, metadata_list):
                metadata_by_tag[tag][file_name] = metadata
                file_metadata_by_tag[tag].append(metadata)
                instruments_by_tag[tag].update(metadata["instrument_row_counts"])
            gc.collect()
    finally:
        for text_member in zip_texts:
            try:
                text_member.close()
            except Exception:
                pass
        for archive in zip_files:
            try:
                archive.close()
            except Exception:
                pass

    for lex_weight, tag, out_dir in weight_specs:
        metadata_by_file = metadata_by_tag[tag]
        file_metadata = file_metadata_by_tag[tag]
        instruments = instruments_by_tag[tag]
        missing_outputs = [path.name for path in parquet_files if path.name not in metadata_by_file]
        if missing_outputs:
            raise RuntimeError(f"submission missing scored files for lex_weight={lex_weight:g}: {missing_outputs[:5]}")

        rows = int(sum(int(metadata_by_file[path.name]["rows"]) for path in parquet_files))
        unique_index = int(sum(int(metadata_by_file[path.name]["unique_index"]) for path in parquet_files))
        if expected_rows is not None and rows != expected_rows:
            raise RuntimeError(f"submission row mismatch: got {rows:,}, expected {expected_rows:,}")
        if expected_rows is not None and unique_index != expected_rows:
            raise RuntimeError(f"submission unique index mismatch: got {unique_index:,}, expected {expected_rows:,}")

        score_min = float("inf")
        score_max = float("-inf")
        score_sum = 0.0
        for parquet_path in parquet_files:
            metadata = metadata_by_file[parquet_path.name]
            score_min = min(score_min, float(metadata["score_min"]))
            score_max = max(score_max, float(metadata["score_max"]))
            score_sum += float(metadata["score_mean"]) * int(metadata["rows"])
        score_mean = float(score_sum / rows) if rows else 0.0
        zip_path = zip_path_by_tag[tag]

        metadata = {
            "method": "top1_consensus_plus_sequence_ngram_plus_global_peptide_support_plus_hidden_weak_lexicon_prior",
            "score_col": score_col,
            "alpha": float(alpha),
            "beta": float(beta),
            "seq_weight": float(seq_weight),
            "support_weight": float(support_weight),
            "lexicon_prior_weight": float(lex_weight),
            "score_transform": score_transform,
            "support_model": support_model.metadata if support_model is not None else None,
            "lexicon_prior_model": lexicon_model.metadata if lexicon_model is not None else None,
            "lexicon_prior_enabled": True,
            "lexicon_prior_weight_source": "LEXICON_PRIOR_WEIGHT constant in src/eval/rescore_sequence_support_grid.py",
            "lexicon_prior_seen_rows": int(sum(int(meta.get("lexicon_prior_seen_rows", 0)) for meta in file_metadata)),
            "rows": rows,
            "unique_index": unique_index,
            "score_min": score_min if rows else 0.0,
            "score_max": score_max if rows else 0.0,
            "score_mean": score_mean,
            "instrument_row_counts": dict(instruments),
            "workers": int(sequence_hash_workers),
            "file_workers": 1,
            "sequence_hash_workers": int(sequence_hash_workers),
            "worker_threads": int(worker_threads),
            "file_metadata": file_metadata,
            "zip_path": str(zip_path),
            "submission_storage": "streamed_zip_only",
        }
        with open(out_dir / "rescore_metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sequence n-gram hash-OOF rescoring.")
    parser.add_argument("--train-root", required=True, help="processed_split/train root")
    parser.add_argument("--pred-root", help="Validation scored parquet file/dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--instruments", default="", help="Comma separated instruments; empty means all")
    parser.add_argument("--max-sequences", type=int, default=2_500_000)
    parser.add_argument("--max-sequences-per-file", type=int, default=6000)
    parser.add_argument("--n-hash-folds", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--n-features-power", type=int, default=21)
    parser.add_argument("--sgd-alpha", type=float, default=1e-5)
    parser.add_argument("--sgd-max-iter", type=int, default=8)
    parser.add_argument("--alpha-grid", default="2,3,4")
    parser.add_argument("--beta-grid", default="0.25,0.5,0.75")
    parser.add_argument("--seq-weight-grid", default="0,0.25,0.5,1,2,3")
    parser.add_argument("--support-weight-grid", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--support-topk-per-file", type=int, default=1)
    parser.add_argument("--support-count-scale", type=float, default=0.25)
    parser.add_argument("--lexicon-prior-alpha", type=float, default=1.0)
    parser.add_argument("--lexicon-prior-shrink-k", type=float, default=50.0)
    parser.add_argument("--lexicon-prior-clip-neg", type=float, default=1.0)
    parser.add_argument("--lexicon-prior-clip-pos", type=float, default=2.0)
    parser.add_argument("--lexicon-prior-min-count", type=int, default=1)
    parser.add_argument("--lexicon-prior-work-dir", default="")
    parser.add_argument("--score-transform", choices=["identity", "logit", "zscore", "rank_pct"], default="logit")
    parser.add_argument("--fdr-threshold", type=float, default=0.01)
    parser.add_argument("--pre-fdr-dedup", choices=["scan", "scan_precursor", "none"], default="scan")
    parser.add_argument("--top1-key", choices=["scan", "group"], default="scan")
    parser.add_argument("--non-conservative-zero-decoy", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--workers", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--best-alpha", type=float, default=None)
    parser.add_argument("--best-beta", type=float, default=None)
    parser.add_argument("--best-seq-weight", type=float, default=None)
    parser.add_argument("--best-support-weight", type=float, default=None)
    parser.add_argument("--test-parquet-dir", help="Optional Basic test parquet dir")
    parser.add_argument("--test-score-col", default="lgbm_v1_score")
    parser.add_argument("--submission-out-dir", help="Output directory for Basic submission")
    parser.add_argument("--expected-test-rows", type=int, default=10_768_114)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cpu_threads(int(args.cpu_threads))
    print_cpu_config("main", int(args.cpu_threads), int(args.workers))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    instruments = [item.strip() for item in args.instruments.split(",") if item.strip()] or None
    train_root = Path(args.train_root).expanduser().resolve()
    lexicon_prior_model: Optional[LexiconPriorModel] = None
    lex_work_dir = (
        Path(args.lexicon_prior_work_dir).expanduser().resolve()
        if args.lexicon_prior_work_dir
        else (out_dir / "lexicon_prior_work")
    )

    texts, labels = collect_unique_train_sequences(
        train_root=train_root,
        instruments=instruments,
        max_sequences=int(args.max_sequences),
        max_sequences_per_file=int(args.max_sequences_per_file),
        seed=int(args.seed),
        workers=int(args.workers),
    )
    sequence_model = HashOofSequenceModel(
        n_hash_folds=int(args.n_hash_folds),
        ngram_min=int(args.ngram_min),
        ngram_max=int(args.ngram_max),
        n_features_power=int(args.n_features_power),
        alpha=float(args.sgd_alpha),
        max_iter=int(args.sgd_max_iter),
        seed=int(args.seed),
        cpu_threads=int(args.cpu_threads),
    )
    sequence_model.fit(texts, labels, workers=int(args.workers))
    del texts, labels
    gc.collect()

    valid_support_model: Optional[PeptideSupportModel] = None
    if args.pred_root:
        valid_support_model = build_peptide_support_model(
            parquet_root=Path(args.pred_root).expanduser().resolve(),
            score_col="score",
            score_transform=args.score_transform,
            topk_per_file=int(args.support_topk_per_file),
            count_scale=float(args.support_count_scale),
            workers=int(args.workers),
        )

    test_support_model: Optional[PeptideSupportModel] = None
    if args.test_parquet_dir or args.submission_out_dir:
        if not args.test_parquet_dir or not args.submission_out_dir:
            raise RuntimeError("--test-parquet-dir and --submission-out-dir must be used together")
        test_support_model = build_peptide_support_model(
            parquet_root=Path(args.test_parquet_dir).expanduser().resolve(),
            score_col=args.test_score_col,
            score_transform=args.score_transform,
            topk_per_file=int(args.support_topk_per_file),
            count_scale=float(args.support_count_scale),
            workers=int(args.workers),
        )

    lexicon_prior_model = build_lexicon_prior_model(
        train_root=train_root,
        instruments=instruments,
        alpha=float(args.lexicon_prior_alpha),
        shrink_k=float(args.lexicon_prior_shrink_k),
        clip_neg=float(args.lexicon_prior_clip_neg),
        clip_pos=float(args.lexicon_prior_clip_pos),
        min_count=int(args.lexicon_prior_min_count),
        workers=int(args.workers),
        work_dir=lex_work_dir,
    )
    gc.collect()

    config = EvalConfig(
        fdr_threshold=float(args.fdr_threshold),
        conservative_tdc=not args.non_conservative_zero_decoy,
        top1_key=args.top1_key,
        pre_fdr_dedup=args.pre_fdr_dedup,
    )
    selected_alpha = args.best_alpha
    selected_beta = args.best_beta
    selected_seq_weight = args.best_seq_weight
    selected_support_weight = args.best_support_weight
    if args.pred_root:
        if valid_support_model is None:
            raise RuntimeError("validation support model was not initialized")
        leaderboard = evaluate_grid(
            pred_root=Path(args.pred_root).expanduser().resolve(),
            sequence_model=sequence_model,
            support_model=valid_support_model,
            lexicon_model=lexicon_prior_model,
            alpha_grid=parse_float_list(args.alpha_grid),
            beta_grid=parse_float_list(args.beta_grid),
            seq_weight_grid=parse_float_list(args.seq_weight_grid),
            support_weight_grid=parse_float_list(args.support_weight_grid),
            score_transform=args.score_transform,
            config=config,
            out_dir=out_dir,
            workers=int(args.workers),
        )
        best = leaderboard.iloc[0]
        if selected_alpha is None:
            selected_alpha = float(best["alpha"])
        if selected_beta is None:
            selected_beta = float(best["beta"])
        if selected_seq_weight is None:
            selected_seq_weight = float(best["seq_weight"])
        if selected_support_weight is None:
            selected_support_weight = float(best["support_weight"])
        display_cols = [
            "rank",
            "model",
            "primary_score",
            "delta_vs_worst",
            "alpha",
            "beta",
            "seq_weight",
            "support_weight",
            "lexicon_prior",
            "lexicon_prior_weight",
            "accepted_target_psm_at_1pct",
            "accepted_decoy_rows_at_1pct",
            "top1_target_rate",
        ]
        print(leaderboard[display_cols].head(30).to_string(index=False))
        print(
            "\nBest params: "
            f"alpha={selected_alpha:g}, beta={selected_beta:g}, "
            f"seq_weight={selected_seq_weight:g}, "
            f"support_weight={selected_support_weight:g}, "
            f"lexicon_prior=on, "
            f"hidden_lex_weights={lexicon_prior_weights()}"
        )

    if args.test_parquet_dir or args.submission_out_dir:
        if (
            selected_alpha is None
            or selected_beta is None
            or selected_seq_weight is None
            or selected_support_weight is None
        ):
            raise RuntimeError("Submission generation needs selected alpha/beta/seq_weight/support_weight")
        if not args.test_parquet_dir or not args.submission_out_dir:
            raise RuntimeError("--test-parquet-dir and --submission-out-dir must be used together")
        expected_rows = None if args.expected_test_rows <= 0 else int(args.expected_test_rows)
        if test_support_model is None:
            raise RuntimeError("test support model was not initialized")
        submission_base = Path(args.submission_out_dir).expanduser().resolve()
        lex_submission_weights = lexicon_prior_weights()
        build_test_submissions_multi_weight(
            parquet_dir=Path(args.test_parquet_dir).expanduser().resolve(),
            score_col=args.test_score_col,
            sequence_model=sequence_model,
            support_model=test_support_model,
            lexicon_model=lexicon_prior_model,
            alpha=float(selected_alpha),
            beta=float(selected_beta),
            seq_weight=float(selected_seq_weight),
            support_weight=float(selected_support_weight),
            lex_weights=lex_submission_weights,
            score_transform=args.score_transform,
            submission_base=submission_base,
            expected_rows=expected_rows,
            workers=int(args.workers),
        )

    print(f"\nWrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
