# 整理parquet表格里的辅助列，变成固定长度的数值向量

import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

# 神经网络输入的尺度不要差太大，所以要对每个量进行归一化
AUX_FEATURE_COLUMNS = [
    "precursor_mz_norm",
    "charge_norm",
    "rt_norm",
    "predicted_rt",
    "delta_rt",
    "abs_delta_rt",
    "sage_score_norm",
    "neg_log_spectrum_q",
    "peptide_length_norm",
    "mod_count_norm",
]

# 把pandas数据变成numpy张量数组
def _safe_numeric(series: Optional[pd.Series], length: int) -> np.ndarray:
    if series is None:
        return np.zeros(length, dtype=np.float32)
    arr = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32, copy=True)
    return arr


# 获取肽段长度（这里先去除了[+n]修饰，再求长度）
def _sequence_lengths(sequences: Iterable[str], default_len: int) -> np.ndarray:
    lengths = []
    for seq in sequences:
        text = seq if isinstance(seq, str) else ""
        cleaned = re.sub(r"\[[^\]]+\]", "", text)
        cleaned = cleaned.replace("n", "")
        lengths.append(len(cleaned) if cleaned else default_len)
    return np.asarray(lengths, dtype=np.float32)


# 统计修饰数
def _modification_counts(sequences: Iterable[str]) -> np.ndarray:
    counts = []
    for seq in sequences:
        text = seq if isinstance(seq, str) else ""
        counts.append(float(text.count("[")))
    return np.asarray(counts, dtype=np.float32)


def build_aux_features_from_df(df: pd.DataFrame, default_seq_len: int = 0) -> np.ndarray:
    length = len(df)
    sequence_col = (
        "precursor_sequence"
        if "precursor_sequence" in df.columns
        else "modified_sequence"
        if "modified_sequence" in df.columns
        else None
    )
    sequences = df[sequence_col].tolist() if sequence_col else [""] * length

    # 读取10维特征，并转为张量
    precursor_mz = _safe_numeric(df.get("precursor_mz"), length)
    charge = _safe_numeric(df.get("charge", df.get("precursor_charge")), length)
    rt = _safe_numeric(df.get("rt"), length)
    predicted_rt = _safe_numeric(df.get("predicted_rt"), length)
    delta_rt = _safe_numeric(df.get("delta_rt", df.get("delta_rt_model")), length)
    sage_score = _safe_numeric(df.get("sage_discriminant_score"), length)
    spectrum_q = _safe_numeric(df.get("spectrum_q"), length)
    peptide_length = _sequence_lengths(sequences, default_seq_len)
    mod_count = _modification_counts(sequences)

    # 归一化，并将生成（n, 10）形状的数组
    # np.clip(x, a, b)  x>a输出a,x<b输出b,此外输出x
    # np.tanh(x) x映射到(-1, 1)范围
    features = np.stack(
        [
            np.clip(precursor_mz / 2000.0, 0.0, 3.0),
            np.clip(charge / 10.0, 0.0, 2.0),
            np.clip(rt / 100.0, 0.0, 2.0),
            np.clip(predicted_rt, -2.0, 2.0),
            np.clip(delta_rt, -2.0, 2.0),
            np.clip(np.abs(delta_rt), 0.0, 2.0),
            np.tanh(sage_score / 10.0),
            np.clip(-np.log10(np.clip(spectrum_q, 1e-8, 1.0)) / 8.0, 0.0, 2.0),
            np.clip(peptide_length / 50.0, 0.0, 2.0),
            np.clip(mod_count / 5.0, 0.0, 2.0),
        ],
        axis=1,
    )
    return features.astype(np.float32, copy=False)
