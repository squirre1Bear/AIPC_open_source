from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd
import torch

PROTON_MASS_AMU = 1.007276
VOCAB = [
    "<pad>",
    "<mask>",
    "A",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "Y",
    "C[57.02]",
    "M[15.99]",
    "N[.98]",
    "Q[.98]",
    "X",
    "<unk>",
]
S2I = {token: idx for idx, token in enumerate(VOCAB)}
MOD_PATTERN = re.compile(r"\[[^\]]+\]")
AA_SPLIT_PATTERN = re.compile(r"(?<=.)(?=[A-Z])")


def safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    return float(value)


def normalize_mass(precursor_mass: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return precursor_mass / 4000.0


def normalize_mz(precursor_mz: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return precursor_mz / 2000.0


def normalize_charge(charge: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return charge / 10.0


def count_modifications(sequence: str) -> int:
    return len(MOD_PATTERN.findall(sequence))


def clean_sequence(sequence: str) -> str:
    return (
        sequence.replace("n[42]", "")
        .replace("N[.98]", "N")
        .replace("Q[.98]", "Q")
        .replace("M[15.99]", "M")
        .replace("C[57.02]", "C")
    )


def to_unimod_sequence(sequence: str) -> str:
    return (
        sequence.replace("n[42]", "n(UniMod:1)")
        .replace("N[.98]", "N(UniMod:7)")
        .replace("Q[.98]", "Q(UniMod:7)")
        .replace("M[15.99]", "M(UniMod:35)")
        .replace("C[57.02]", "C(UniMod:4)")
    )


def tokenize_sequence(sequence: str, max_token_len: int) -> torch.Tensor:
    sequence = sequence.replace("I", "L").replace("n[42]", "X")
    sequence = (
        sequence.replace("cC", "C[57.02]")
        .replace("oxM", "M[15.99]")
        .replace("M(ox)", "M[15.99]")
        .replace("deamN", "N[.98]")
        .replace("deamQ", "Q[.98]")
        .replace("a", "X")
    )
    parts = AA_SPLIT_PATTERN.split(sequence) if sequence else []
    token_ids = [S2I.get(part, S2I["<unk>"]) for part in parts[:max_token_len]]
    tokens = torch.zeros(max_token_len, dtype=torch.long)
    if token_ids:
        tokens[: len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    return tokens


def masked_mean(features: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = (~pad_mask).unsqueeze(-1)
    total = valid_mask.sum(dim=1).clamp_min(1)
    masked = features * valid_mask
    return masked.sum(dim=1) / total


def compute_batch_aux_features(
    spectra: torch.Tensor,
    spectra_mask: torch.Tensor,
    precursor_masses: torch.Tensor,
    precursor_charges: torch.Tensor,
    predicted_rt: torch.Tensor,
    delta_rt: torch.Tensor,
    sequences: Iterable[str],
) -> torch.Tensor:
    valid_mask = (~spectra_mask).float()
    valid_peak_ratio = valid_mask.mean(dim=1)
    mz_sum = (spectra[:, :, 0] * valid_mask).sum(dim=1)
    mz_mean = mz_sum / valid_mask.sum(dim=1).clamp_min(1.0)
    precursor_mz = precursor_masses / precursor_charges.clamp_min(1.0) + PROTON_MASS_AMU
    estimated_rt = torch.clamp(predicted_rt - delta_rt, 0.0, 1.0)
    peptide_length = torch.tensor(
        [len(clean_sequence(seq)) for seq in sequences],
        dtype=torch.float32,
        device=spectra.device,
    )
    mod_count = torch.tensor(
        [count_modifications(seq) for seq in sequences],
        dtype=torch.float32,
        device=spectra.device,
    )
    aux = torch.stack(
        [
            normalize_mz(precursor_mz),
            normalize_charge(precursor_charges),
            estimated_rt,
            predicted_rt,
            delta_rt,
            delta_rt.abs(),
            valid_peak_ratio,
            normalize_mz(mz_mean),
            peptide_length / 50.0,
            mod_count / 5.0,
        ],
        dim=1,
    )
    return aux


def get_fdr_result(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decoy"] = np.where(df["label"] == 1, 0, 1)
    target_num = (df["decoy"] == 0).cumsum().replace(0, 1)
    decoy_num = (df["decoy"] == 1).cumsum().replace(0, 1)
    df["q_value"] = decoy_num / target_num
    df["q_value"] = df["q_value"][::-1].cummin()[::-1]
    return df


def count_targets_and_unique_peptides(
    scores: np.ndarray,
    labels: np.ndarray,
    sequences: list[str],
    fdr_threshold: float = 0.01,
) -> tuple[int, int]:
    eval_df = pd.DataFrame(
        {
            "score": scores,
            "label": labels,
            "sequence": [clean_sequence(seq) for seq in sequences],
        }
    ).sort_values("score", ascending=False)
    eval_df = get_fdr_result(eval_df)
    passed = eval_df[(eval_df["label"] == 1) & (eval_df["q_value"] <= fdr_threshold)]
    return int(len(passed)), int(passed["sequence"].nunique())


def estimate_rt_from_row(row: pd.Series) -> float:
    predicted_rt = safe_float(row.get("predicted_rt"), 0.0)
    delta_rt = safe_float(row.get("delta_rt_model", row.get("delta_rt")), 0.0)
    return max(0.0, min(1.0, predicted_rt - delta_rt))


def build_prediction_frame(
    parquet_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    q_value_threshold: float = 1.0,
) -> pd.DataFrame:
    merged = parquet_df.merge(pred_df[["index", "score"]], on="index", how="left")
    merged["score"] = pd.to_numeric(merged["score"], errors="coerce").fillna(0.0)
    merged = merged.sort_values("score", ascending=False)
    merged = get_fdr_result(merged)
    merged = merged[merged["q_value"] < q_value_threshold]
    merged["cleaned_sequence"] = merged["precursor_sequence"].map(clean_sequence)
    merged_target = merged[merged["label"] == 1].copy()
    merged_target = merged_target.rename(
        columns={
            "precursor_sequence": "modified_sequence",
            "charge": "precursor_charge",
        }
    )
    merged_target["modified_sequence"] = merged_target["modified_sequence"].map(to_unimod_sequence)
    merged_target = merged_target.sort_values("score", ascending=False)
    merged_target = merged_target.drop_duplicates(subset="scan_number", keep="first")
    return merged_target[
        [
            "cleaned_sequence",
            "precursor_mz",
            "precursor_charge",
            "modified_sequence",
            "label",
            "score",
            "q_value",
            "scan_number",
        ]
    ]
