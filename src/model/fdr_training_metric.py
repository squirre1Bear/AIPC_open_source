from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from lightgbm.callback import EarlyStopException
except Exception:  # pragma: no cover - only used when importing metric helpers without LightGBM installed.
    EarlyStopException = None


PRIMARY_FDR_METRIC_NAME = "unique_peptide_at_1pct_fdr"


@dataclass
class FdrEvalMetadata:
    labels: np.ndarray
    peptide_codes: np.ndarray
    has_peptide_key: bool
    fdr_threshold: float = 0.01
    conservative_tdc: bool = True

    @classmethod
    def from_frame(
        cls,
        df,
        fdr_threshold: float = 0.01,
        conservative_tdc: bool = True,
        label_col: str = "label",
        peptide_col: str = "peptide_key",
    ) -> "FdrEvalMetadata":
        if label_col not in df.columns:
            raise ValueError(f"FDR eval metadata requires column: {label_col}")

        labels = df[label_col].to_numpy().astype(np.int8, copy=False)

        if peptide_col in df.columns:
            values = df[peptide_col].to_numpy()
            try:
                peptide_codes, _ = pd.factorize(values, sort=False, use_na_sentinel=False)
            except TypeError:
                peptide_codes, _ = pd.factorize(values, sort=False)
            peptide_codes = peptide_codes.astype(np.int64, copy=False)
            has_peptide_key = True
        else:
            peptide_codes = np.arange(labels.shape[0], dtype=np.int64)
            has_peptide_key = False

        return cls(
            labels=labels,
            peptide_codes=peptide_codes,
            has_peptide_key=has_peptide_key,
            fdr_threshold=float(fdr_threshold),
            conservative_tdc=bool(conservative_tdc),
        )


def unique_peptide_fdr_metric(scores: np.ndarray, meta: FdrEvalMetadata) -> Dict:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape[0] != meta.labels.shape[0]:
        raise ValueError(
            f"score length mismatch: scores={scores.shape[0]}, labels={meta.labels.shape[0]}"
        )

    safe_scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
    order = np.argsort(-safe_scores, kind="mergesort")

    labels = meta.labels[order]
    is_target = labels == 1
    is_decoy = labels == 0

    cum_target = np.cumsum(is_target, dtype=np.int64)
    cum_decoy = np.cumsum(is_decoy, dtype=np.int64)
    decoy_for_fdr = np.maximum(cum_decoy, 1) if meta.conservative_tdc else cum_decoy
    fdr = decoy_for_fdr / np.maximum(cum_target, 1)
    raw_fdr = cum_decoy / np.maximum(cum_target, 1)
    q_value = np.minimum.accumulate(fdr[::-1])[::-1]

    accepted_all_mask = q_value <= meta.fdr_threshold
    accepted_target_mask = accepted_all_mask & is_target
    accepted_decoy_mask = accepted_all_mask & is_decoy

    accepted_target_psm = int(accepted_target_mask.sum())
    if accepted_target_psm > 0:
        accepted_codes = meta.peptide_codes[order][accepted_target_mask]
        accepted_unique = int(np.unique(accepted_codes).size)
    else:
        accepted_unique = 0

    if bool(accepted_all_mask.any()):
        cutoff_idx = int(np.flatnonzero(accepted_all_mask)[-1])
        score_threshold = float(safe_scores[order][cutoff_idx])
        rank_cutoff = cutoff_idx + 1
        estimated_fdr = float(fdr[cutoff_idx])
        raw_cutoff_fdr = float(raw_fdr[cutoff_idx])
        cutoff_target_rows = int(cum_target[cutoff_idx])
        cutoff_decoy_rows = int(cum_decoy[cutoff_idx])
    else:
        score_threshold = None
        rank_cutoff = 0
        estimated_fdr = None
        raw_cutoff_fdr = None
        cutoff_target_rows = 0
        cutoff_decoy_rows = 0

    return {
        "rows": int(labels.shape[0]),
        "target_rows": int(is_target.sum()),
        "decoy_rows": int(is_decoy.sum()),
        "accepted_target_psm_at_1pct": accepted_target_psm,
        "accepted_unique_peptide_at_1pct": accepted_unique,
        "accepted_decoy_rows_at_1pct": int(accepted_decoy_mask.sum()),
        "accepted_total_rows_at_1pct": int(accepted_all_mask.sum()),
        "score_threshold_at_1pct": score_threshold,
        "rank_cutoff_at_1pct": rank_cutoff,
        "estimated_fdr_at_cutoff": estimated_fdr,
        "raw_decoy_over_target_at_cutoff": raw_cutoff_fdr,
        "cutoff_target_rows": cutoff_target_rows,
        "cutoff_decoy_rows": cutoff_decoy_rows,
        "conservative_tdc": bool(meta.conservative_tdc),
        "has_peptide_key": bool(meta.has_peptide_key),
    }


@dataclass(eq=False)
class UniquePeptideFdrEarlyStopping:
    valid_features: object
    valid_meta: FdrEvalMetadata
    stopping_rounds: int
    eval_period: int = 50
    metric_name: str = PRIMARY_FDR_METRIC_NAME
    min_delta: float = 0.0
    best_iteration: int = 0
    best_score: float = float("-inf")
    best_metric: Optional[Dict] = None
    history: List[Dict] = field(default_factory=list)

    before_iteration: bool = False
    order: int = 30

    def __post_init__(self) -> None:
        self.eval_period = max(1, int(self.eval_period))
        self.stopping_rounds = int(self.stopping_rounds)
        self.min_delta = float(self.min_delta)

    def __call__(self, env) -> None:
        iteration = int(env.iteration) + 1
        end_iteration = int(env.end_iteration)
        should_eval = (
            iteration == 1
            or iteration % self.eval_period == 0
            or iteration == end_iteration
        )
        if not should_eval:
            return

        pred = env.model.predict(self.valid_features, num_iteration=iteration)
        metric = unique_peptide_fdr_metric(pred, self.valid_meta)
        score = float(metric["accepted_unique_peptide_at_1pct"])
        improved = score > self.best_score + self.min_delta

        if improved:
            self.best_iteration = iteration
            self.best_score = score
            self.best_metric = dict(metric)

        row = {
            "iteration": iteration,
            "metric": self.metric_name,
            "value": int(score),
            "improved": bool(improved),
            **metric,
        }
        self.history.append(row)

        best_text = (
            f"best={int(self.best_score)}@{self.best_iteration}"
            if self.best_iteration > 0
            else "best=NA"
        )
        print(
            f"[{iteration}] valid {self.metric_name}={int(score)} "
            f"target_psm={metric['accepted_target_psm_at_1pct']} "
            f"decoy={metric['accepted_decoy_rows_at_1pct']} "
            f"rank_cutoff={metric['rank_cutoff_at_1pct']} {best_text}"
        )

        if (
            self.stopping_rounds > 0
            and self.best_iteration > 0
            and iteration - self.best_iteration >= self.stopping_rounds
        ):
            print(
                f"Early stopping on valid {self.metric_name}. "
                f"Best iteration: {self.best_iteration}, best score: {int(self.best_score)}"
            )
            if EarlyStopException is None:
                raise RuntimeError("LightGBM EarlyStopException is unavailable.")
            raise EarlyStopException(
                self.best_iteration - 1,
                [("valid", self.metric_name, self.best_score, True)],
            )


def resolve_best_iteration(model, stopper: UniquePeptideFdrEarlyStopping, fallback_rounds: int) -> int:
    model_best = int(getattr(model, "best_iteration", 0) or 0)
    if model_best > 0:
        return model_best

    if stopper.best_iteration > 0:
        try:
            model.best_iteration = int(stopper.best_iteration)
        except Exception:
            pass
        return int(stopper.best_iteration)

    try:
        current = int(model.current_iteration())
        if current > 0:
            return current
    except Exception:
        pass

    return int(fallback_rounds)
