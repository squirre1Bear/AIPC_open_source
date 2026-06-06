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
import json
import multiprocessing as mp
import os
import random
import re
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
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

try:
    import pyarrow as pa
except Exception:  # pragma: no cover
    pa = None

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

from sklearn.feature_extraction.text import HashingVectorizer
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
_SEQ_THREAD_LIMIT = None
_SEQ_TRAIN_TEXTS: Optional[List[str]] = None
_SEQ_TRAIN_LABELS: Optional[np.ndarray] = None
_SEQ_TRAIN_FOLDS: Optional[np.ndarray] = None
_SEQ_TRAIN_VECTOR_PARAMS: Optional[Dict] = None


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
) -> None:
    global _SEQ_WORKER_MODEL, _SUPPORT_WORKER_MODEL
    configure_worker_threads(worker_threads)
    _SEQ_WORKER_MODEL = sequence_model
    _SUPPORT_WORKER_MODEL = support_model


def init_sequence_train_worker(
    texts: List[str],
    labels: np.ndarray,
    folds: np.ndarray,
    vector_params: Dict,
    worker_threads: int,
) -> None:
    global _SEQ_TRAIN_TEXTS, _SEQ_TRAIN_LABELS, _SEQ_TRAIN_FOLDS, _SEQ_TRAIN_VECTOR_PARAMS
    configure_worker_threads(worker_threads)
    _SEQ_TRAIN_TEXTS = texts
    _SEQ_TRAIN_LABELS = labels
    _SEQ_TRAIN_FOLDS = folds
    _SEQ_TRAIN_VECTOR_PARAMS = vector_params


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


def clean_for_sequence_model(series: pd.Series) -> pd.Series:
    cleaned = clean_peptide_sequence(series)
    cleaned = cleaned.str.replace(r"n\[42\]", "", regex=True)
    cleaned = cleaned.str.replace(r"\[[^\]]+\]", "", regex=True)
    return cleaned.fillna("").astype(str)


def sequence_texts_from_series(series: pd.Series) -> List[str]:
    cleaned = clean_for_sequence_model(series)
    return ("^" + cleaned + "$").tolist()


def hash_fold(text: str, n_folds: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(n_folds)


def read_parquet_existing(path: Path, wanted: Iterable[str]) -> pd.DataFrame:
    columns = existing_columns(path, wanted)
    if not columns:
        raise RuntimeError(f"No requested columns found in parquet: {path}")
    return pq.read_table(path, columns=columns).to_pandas()


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
            return df[column].fillna("").astype(str)
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
    df = pq.read_table(path, columns=wanted).to_pandas()
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
    texts = sequence_texts_from_series(pd.Series(sequences))

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


def train_sequence_fold_worker(args: Tuple) -> Tuple[int, SGDClassifier, np.ndarray, np.ndarray, Dict]:
    if (
        _SEQ_TRAIN_TEXTS is None
        or _SEQ_TRAIN_LABELS is None
        or _SEQ_TRAIN_FOLDS is None
        or _SEQ_TRAIN_VECTOR_PARAMS is None
    ):
        raise RuntimeError("sequence train worker is not initialized")
    fold_id, n_hash_folds, alpha, max_iter, seed = args
    vectorizer = HashingVectorizer(**_SEQ_TRAIN_VECTOR_PARAMS)
    train_mask = _SEQ_TRAIN_FOLDS != int(fold_id)
    valid_mask = ~train_mask
    train_indices = np.where(train_mask)[0]
    valid_indices = np.where(valid_mask)[0]
    train_texts = [_SEQ_TRAIN_TEXTS[index] for index in train_indices]
    valid_texts = [_SEQ_TRAIN_TEXTS[index] for index in valid_indices]
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=float(alpha),
        max_iter=int(max_iter),
        tol=1e-4,
        class_weight="balanced",
        random_state=int(seed) + int(fold_id),
        n_jobs=1,
    )
    model.fit(vectorizer.transform(train_texts), _SEQ_TRAIN_LABELS[train_indices])
    valid_scores = model.decision_function(vectorizer.transform(valid_texts)).astype(np.float32)
    info = {
        "fold_id": int(fold_id),
        "train_rows": int(len(train_indices)),
        "valid_rows": int(len(valid_indices)),
        "raw_mean": float(valid_scores.mean()),
    }
    return int(fold_id), model, valid_indices.astype(np.int64), valid_scores, info


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
        self.vectorizer = HashingVectorizer(
            analyzer="char",
            ngram_range=(int(ngram_min), int(ngram_max)),
            n_features=2 ** int(n_features_power),
            alternate_sign=False,
            norm="l2",
            lowercase=False,
            binary=True,
        )
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.cpu_threads = int(cpu_threads)
        self.models: List[SGDClassifier] = []
        self.oof_mean = 0.0
        self.oof_std = 1.0
        self.vector_params = {
            "analyzer": "char",
            "ngram_range": (int(ngram_min), int(ngram_max)),
            "n_features": 2 ** int(n_features_power),
            "alternate_sign": False,
            "norm": "l2",
            "lowercase": False,
            "binary": True,
        }

    def fit(self, texts: List[str], labels: np.ndarray, workers: int = 1) -> None:
        folds = np.asarray([hash_fold(text, self.n_hash_folds) for text in texts], dtype=np.int16)
        oof_scores = np.zeros(len(texts), dtype=np.float32)
        self.models = []
        workers = max(1, min(int(workers), int(self.n_hash_folds)))
        worker_threads = 1 if workers > 1 else max(1, int(self.cpu_threads))
        print(
            f"sequence training workers={workers}, worker_threads={worker_threads}, "
            f"active_hash_folds={self.n_hash_folds}",
            flush=True,
        )
        train_args = [
            (fold_id, self.n_hash_folds, self.alpha, self.max_iter, self.seed)
            for fold_id in range(self.n_hash_folds)
        ]
        models_by_fold: Dict[int, SGDClassifier] = {}
        if workers == 1:
            init_sequence_train_worker(texts, labels, folds, self.vector_params, worker_threads)
            for result in map(train_sequence_fold_worker, train_args):
                fold_id, model, valid_indices, valid_scores, info = result
                models_by_fold[fold_id] = model
                oof_scores[valid_indices] = valid_scores
                print(
                    f"sequence fold {fold_id}: train={info['train_rows']:,}, "
                    f"valid={info['valid_rows']:,}, raw_mean={info['raw_mean']:.6f}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_context_for_platform(),
                initializer=init_sequence_train_worker,
                initargs=(texts, labels, folds, self.vector_params, worker_threads),
            ) as executor:
                futures = [executor.submit(train_sequence_fold_worker, item) for item in train_args]
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
            scores[mask] = model.decision_function(
                self.vectorizer.transform(fold_texts)
            ).astype(np.float32)
        return ((scores.astype(np.float64) - self.oof_mean) / self.oof_std).astype(np.float32)

    def predict_ensemble(self, texts: List[str]) -> np.ndarray:
        if not self.models:
            raise RuntimeError("sequence model is not fitted")
        if not texts:
            return np.asarray([], dtype=np.float32)
        matrix = self.vectorizer.transform(texts)
        scores = np.zeros(len(texts), dtype=np.float64)
        for model in self.models:
            scores += model.decision_function(matrix)
        scores /= float(len(self.models))
        return ((scores - self.oof_mean) / self.oof_std).astype(np.float32)


def sequence_score_for_frame(
    sequence_model: HashOofSequenceModel,
    df: pd.DataFrame,
    use_oof_hash: bool,
) -> np.ndarray:
    sequence_source = df["modified_sequence"] if "modified_sequence" in df.columns else df["precursor_sequence"]
    texts = sequence_texts_from_series(sequence_source)
    unique_texts = list(dict.fromkeys(texts))
    if use_oof_hash:
        unique_scores = sequence_model.predict_oof_by_hash(unique_texts)
    else:
        unique_scores = sequence_model.predict_ensemble(unique_texts)
    score_by_text = dict(zip(unique_texts, unique_scores))
    return np.asarray([score_by_text[text] for text in texts], dtype=np.float32)


def model_name(alpha: float, beta: float, seq_weight: float, support_weight: float) -> str:
    return (
        f"seq_support_a{num_tag(alpha)}_b{num_tag(beta)}_sw{num_tag(seq_weight)}"
        f"_pw{num_tag(support_weight)}"
    )


def evaluate_one_parquet_worker(args: Tuple) -> Dict[Tuple[float, float, float, float], List[Dict]]:
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
    out: Dict[Tuple[float, float, float, float], List[Dict]] = {combo: [] for combo in combos}
    for alpha, beta, seq_weight, support_weight in combos:
        final_score = consensus_score(
            base_score,
            consensus_features,
            alpha=alpha,
            beta=beta,
            gamma=0.0,
            mode="top1",
        ) + float(seq_weight) * seq_score + float(support_weight) * support_score
        scored = standardized.copy()
        scored["score"] = final_score
        for file_id, part in scored.groupby("file_id", sort=False):
            out[(alpha, beta, seq_weight, support_weight)].append(
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


def evaluate_grid(
    pred_root: Path,
    sequence_model: HashOofSequenceModel,
    support_model: PeptideSupportModel,
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

    combos = [
        (float(alpha), float(beta), float(seq_weight), float(support_weight))
        for alpha in alpha_grid
        for beta in beta_grid
        for seq_weight in seq_weight_grid
        for support_weight in support_weight_grid
    ]
    metrics_by_combo: Dict[Tuple[float, float, float, float], List[Dict]] = {combo: [] for combo in combos}

    workers = max(1, int(workers))
    worker_threads = 1 if workers > 1 else int(os.environ.get("OMP_NUM_THREADS", DEFAULT_CPU_THREADS))
    print(
        f"validation evaluation workers={workers}, worker_threads={worker_threads}, "
        f"total_parallelism_target={workers * worker_threads}",
        flush=True,
    )
    tasks = [(str(path), combos, score_transform, config) for path in parquet_files]
    if workers == 1:
        init_sequence_worker(sequence_model, worker_threads, support_model)
        iterator = (evaluate_one_parquet_worker(task) for task in tasks)
        for partial in tqdm(iterator, total=len(tasks), desc="validation parquet"):
            for combo, metrics in partial.items():
                metrics_by_combo[combo].extend(metrics)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context_for_platform(),
            initializer=init_sequence_worker,
            initargs=(sequence_model, worker_threads, support_model),
        ) as executor:
            futures = [executor.submit(evaluate_one_parquet_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="validation parquet"):
                partial = future.result()
                for combo, metrics in partial.items():
                    metrics_by_combo[combo].extend(metrics)

    summaries = []
    details = {}
    for alpha, beta, seq_weight, support_weight in combos:
        name = model_name(alpha, beta, seq_weight, support_weight)
        summary, by_file = aggregate_metrics(metrics_by_combo[(alpha, beta, seq_weight, support_weight)], name)
        summary["alpha"] = float(alpha)
        summary["beta"] = float(beta)
        summary["seq_weight"] = float(seq_weight)
        summary["support_weight"] = float(support_weight)
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
        "pred_root": str(pred_root),
        "sequence_model": {
            "n_hash_folds": sequence_model.n_hash_folds,
            "alpha": sequence_model.alpha,
            "max_iter": sequence_model.max_iter,
            "oof_mean": sequence_model.oof_mean,
            "oof_std": sequence_model.oof_std,
        },
        "support_model": support_model.metadata,
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


def score_test_parquet_worker(args: Tuple) -> Tuple[str, Dict, pd.DataFrame]:
    if _SEQ_WORKER_MODEL is None:
        raise RuntimeError("sequence worker model is not initialized")
    parquet_path, score_col, alpha, beta, seq_weight, support_weight, score_transform, out_dir = args
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

    df = pq.read_table(parquet_path, columns=read_cols).to_pandas()
    if "file_id" not in df.columns:
        df["file_id"] = parquet_path.stem
    if "instrument" not in df.columns:
        df["instrument"] = infer_instrument(parquet_path.name)

    raw_score = pd.to_numeric(df[score_col], errors="coerce").to_numpy()
    base_score = transform_score(raw_score, score_transform)
    consensus_features = build_consensus_features(df, base_score, topk=3)
    seq_score = sequence_score_for_frame(_SEQ_WORKER_MODEL, df, use_oof_hash=False)
    if _SUPPORT_WORKER_MODEL is None:
        support_score = np.zeros(len(df), dtype=np.float32)
    else:
        support_score = _SUPPORT_WORKER_MODEL.score_frame(df)
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
    ).astype(np.float32)

    part = pd.DataFrame(
        {
            "index": pd.to_numeric(df["index"], errors="raise").astype(np.int64),
            "score": final_score,
        }
    )
    pred_path = out_dir / f"{parquet_path.stem}_pred.csv"
    part.to_csv(pred_path, header=False, index=False)
    metadata = {
        "parquet": str(parquet_path),
        "pred_path": str(pred_path),
        "rows": int(len(part)),
        "score_min": float(part["score"].min()),
        "score_max": float(part["score"].max()),
        "score_mean": float(part["score"].mean()),
        "instrument_row_counts": dict(Counter(df["instrument"].fillna("unknown").astype(str).tolist())),
    }
    return parquet_path.name, metadata, part


def build_test_submission(
    parquet_dir: Path,
    score_col: str,
    sequence_model: HashOofSequenceModel,
    support_model: Optional[PeptideSupportModel],
    alpha: float,
    beta: float,
    seq_weight: float,
    support_weight: float,
    score_transform: str,
    out_dir: Path,
    expected_rows: Optional[int],
    workers: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_files = discover_test_parquets(parquet_dir)
    merged_by_file: Dict[str, pd.DataFrame] = {}
    file_metadata = []
    instruments = Counter()

    workers = max(1, int(workers))
    worker_threads = 1 if workers > 1 else int(os.environ.get("OMP_NUM_THREADS", DEFAULT_CPU_THREADS))
    print(
        f"submission workers={workers}, worker_threads={worker_threads}, "
        f"total_parallelism_target={workers * worker_threads}",
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
            score_transform,
            str(out_dir),
        )
        for path in parquet_files
    ]
    if workers == 1:
        init_sequence_worker(sequence_model, worker_threads, support_model)
        iterator = (score_test_parquet_worker(task) for task in tasks)
        for file_name, metadata, part in tqdm(iterator, total=len(tasks), desc="write submission"):
            merged_by_file[file_name] = part
            file_metadata.append(metadata)
            instruments.update(metadata["instrument_row_counts"])
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp_context_for_platform(),
            initializer=init_sequence_worker,
            initargs=(sequence_model, worker_threads, support_model),
        ) as executor:
            futures = [executor.submit(score_test_parquet_worker, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="write submission"):
                file_name, metadata, part = future.result()
                merged_by_file[file_name] = part
                file_metadata.append(metadata)
                instruments.update(metadata["instrument_row_counts"])

    merged = pd.concat(
        [merged_by_file[path.name] for path in parquet_files],
        ignore_index=True,
    )
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
        "method": "top1_consensus_plus_sequence_ngram_plus_global_peptide_support",
        "score_col": score_col,
        "alpha": float(alpha),
        "beta": float(beta),
        "seq_weight": float(seq_weight),
        "support_weight": float(support_weight),
        "score_transform": score_transform,
        "support_model": support_model.metadata if support_model is not None else None,
        "rows": int(len(merged)),
        "unique_index": unique_index,
        "score_min": float(merged["score"].min()),
        "score_max": float(merged["score"].max()),
        "score_mean": float(merged["score"].mean()),
        "instrument_row_counts": dict(instruments),
        "workers": int(workers),
        "worker_threads": int(worker_threads),
        "file_metadata": file_metadata,
        "zip_path": str(zip_path),
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
    texts, labels = collect_unique_train_sequences(
        train_root=Path(args.train_root).expanduser().resolve(),
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
        valid_support_model = build_peptide_support_model(
            parquet_root=Path(args.pred_root).expanduser().resolve(),
            score_col="score",
            score_transform=args.score_transform,
            topk_per_file=int(args.support_topk_per_file),
            count_scale=float(args.support_count_scale),
            workers=int(args.workers),
        )
        leaderboard = evaluate_grid(
            pred_root=Path(args.pred_root).expanduser().resolve(),
            sequence_model=sequence_model,
            support_model=valid_support_model,
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
            "accepted_target_psm_at_1pct",
            "accepted_decoy_rows_at_1pct",
            "top1_target_rate",
        ]
        print(leaderboard[display_cols].head(30).to_string(index=False))
        print(
            "\nBest params: "
            f"alpha={selected_alpha:g}, beta={selected_beta:g}, "
            f"seq_weight={selected_seq_weight:g}, "
            f"support_weight={selected_support_weight:g}"
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
        test_support_model = build_peptide_support_model(
            parquet_root=Path(args.test_parquet_dir).expanduser().resolve(),
            score_col=args.test_score_col,
            score_transform=args.score_transform,
            topk_per_file=int(args.support_topk_per_file),
            count_scale=float(args.support_count_scale),
            workers=int(args.workers),
        )
        build_test_submission(
            parquet_dir=Path(args.test_parquet_dir).expanduser().resolve(),
            score_col=args.test_score_col,
            sequence_model=sequence_model,
            support_model=test_support_model,
            alpha=float(selected_alpha),
            beta=float(selected_beta),
            seq_weight=float(selected_seq_weight),
            support_weight=float(selected_support_weight),
            score_transform=args.score_transform,
            out_dir=Path(args.submission_out_dir).expanduser().resolve(),
            expected_rows=expected_rows,
            workers=int(args.workers),
        )

    print(f"\nWrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
