  # cd ~/aipc
  # source ~/miniconda3/etc/profile.d/conda.sh
  # conda activate aipc
  # export DATA=/root/autodl-tmp/datasets/aipc
  #
  # python src/preprocess/add_feature_1.py --dirs \
  #   $DATA/processed_split/train/mzml $DATA/processed_split/train/tims $DATA/processed_split/train/wiff \
  #   $DATA/processed_split/valid/mzml $DATA/processed_split/valid/tims $DATA/processed_split/valid/wiff \
  #   $DATA/processed/bas_merged

# 添加前体质量相关特征、RT 归一化，以及组内顺序特征等
from pathlib import Path

import os

# 必须放在 import polars / numpy 前面
# 每个子进程只允许底层库使用 1 个线程，20 个子进程 ≈ 20 CPU 并行
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
import subprocess
import argparse
import numpy as np
from tqdm import tqdm
import math
import polars as pl
import re
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed


# 路径
# PROCESSED_DIRS = [
#     Path(r"/root/autodl-tmp/datasets/aipc/processed/mzml_merged"),
#     Path(r"/root/autodl-tmp/datasets/aipc/processed/tims_merged"),
#     Path(r"/root/autodl-tmp/datasets/aipc/processed/wiff_merged"),
# ]
PROCESSED_DIRS = [
    Path(r"/root/autodl-tmp/datasets/aipc/processed/bas_merged")
]

# 并行处理文件数
# 每个 parquet 文件放到一个独立子进程中处理
# 每个子进程内部 Polars / BLAS 限制为 1 线程
N_WORKERS = 12

# 选择是否要备份
MAKE_BACKUP = False

# 临时文件后缀，先写入临时文件，再替换原文件
TMP_SUFFIX = ".tmp_feature.parquet"

# 每张谱图保留 TOPK_PEAKS 个峰
TOPK_PEAKS = 500

# 每个 parquet 文件放到独立子进程处理。
# 这样即使 Polars / Arrow 底层因为内存压力崩溃，也不会拖死整个任务，
# 并且每个文件处理完后操作系统会彻底回收内存。
RUN_EACH_FILE_IN_SUBPROCESS = True

# 修复了肽段解析逻辑，需要设为 True，强制重新计算并覆盖。
FORCE_REBUILD_AUX_FEATURES = False

ORDER_AWARE_FEATURES = [
    "scan_candidate_count",
    "scan_candidate_count_log1p",
    "candidate_order_in_scan",
    "candidate_order_pct_in_scan",
    "candidate_order_reciprocal",
    "candidate_order_log1p",
    "is_first_candidate_in_scan",
    "is_top3_candidate_in_scan",
    "abs_delta_rt_rank_in_scan",
    "abs_delta_rt_rank_pct_in_scan",
    "is_best_abs_delta_rt_in_scan",
    "abs_delta_rt_min_in_scan",
    "abs_delta_rt_mean_in_scan",
    "abs_delta_rt_std_in_scan",
    "abs_delta_rt_gap_to_best_in_scan",
    "abs_delta_rt_z_in_scan",
    "candidate_order_rt_rank_gap",
    "abs_candidate_order_rt_rank_gap",
    "candidate_order_matches_abs_delta_rank",
    "scan_unique_charge_count",
    "scan_has_multiple_charges",
    "same_charge_candidate_count_in_scan",
    "same_charge_candidate_fraction_in_scan",
    "candidate_order_in_scan_charge",
    "candidate_order_pct_in_scan_charge",
    "is_first_candidate_for_charge_in_scan",
]

# 只读取本脚本需要的列，避免把 parquet 里的无关列全部读进内存
NEEDED_COLS = [
    "file_id",
    "instrument",
    "index",
    "scan_number",
    "group_key",
    "precursor_sequence",
    "peptide_key",

    "mz_array",
    "intensity_array",
    "precursor_mz",

    "rt",
    "predicted_rt",
    "delta_rt",
    "charge",
    "ion_mobility",
    "has_ion_mobility",

    "label",
    "sage_discriminant_score",
    "spectrum_q",
    "fp_q_value",
    "in_fp",
]

# 定义化学常量
PROTON = 1.007276466812
H2O = 18.01056468403
C13_DIFF = 1.0033548378

AA_MASS = {
    "A": 71.037113805,
    "R": 156.101111050,
    "N": 114.042927470,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.058577540,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "L": 113.084064015,
    "K": 128.094963050,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.079312980,
    "Y": 163.063328575,
    "V": 99.068413945,
}

MOD_MASS = {
    "oxidation": 15.99491461957,
    "ox": 15.99491461957,
    "unimod:35": 15.99491461957,

    "carbamidomethyl": 57.021463735,
    "cam": 57.021463735,
    "unimod:4": 57.021463735,

    "acetyl": 42.010564684,
    "unimod:1": 42.010564684,

    "phospho": 79.966330410,
    "phosphorylation": 79.966330410,
    "unimod:21": 79.966330410,

    "deamidated": 0.984015585,
    "deamidation": 0.984015585,
    "unimod:7": 0.984015585,

    "gln->pyro-glu": -17.026549101,
    "glu->pyro-glu": -18.010564684,
    "pyro-glu": -17.026549101,

    "tmt6plex": 229.162932,
    "tmtpro": 304.207146,
    "itraq4plex": 144.102063,
    "itraq8plex": 304.205360,

    "label:13c(6)15n(2)": 8.014199,
    "label:13c(6)15n(4)": 10.008269,
}


def safe_float(x, default=None):
    try:
        if x is None:
            return default

        v = float(x)

        if math.isnan(v) or math.isinf(v):
            return default

        return v

    except Exception:
        return default


def keep_topk_peaks(mz_array, intensity_array, topk=TOPK_PEAKS):
    if mz_array is None or intensity_array is None:
        return [], []

    try:
        mz = np.asarray(mz_array, dtype=np.float32)
        intensity = np.asarray(intensity_array, dtype=np.float32)

    except Exception:
        return [], []

    if len(mz) == 0 or len(intensity) == 0 or len(mz) != len(intensity):
        return [], []

    mask = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz = mz[mask]
    intensity = intensity[mask]

    if len(mz) == 0:
        return [], []

    if len(mz) > topk:
        idx = np.argpartition(-intensity, topk - 1)[:topk]
        mz = mz[idx]
        intensity = intensity[idx]

    order = np.argsort(mz)
    mz = mz[order]
    intensity = intensity[order]

    return mz.tolist(), intensity.tolist()


def trim_spectrum_struct(s):
    new_mz, new_intensity = keep_topk_peaks(
        s["mz_array"],
        s["intensity_array"],
        TOPK_PEAKS,
    )

    return {
        "mz_array": new_mz,
        "intensity_array": new_intensity,
    }


def parse_numeric_mass(text: str):
    text = str(text).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)

    if m is None:
        return None

    return safe_float(m.group(0), default=None)


def normalize_mod_name(name: str) -> str:
    name = str(name).strip()
    name = name.lower()
    name = name.replace("_", "")
    name = name.replace(" ", "")
    return name


COMMON_NUMERIC_MODS = [
    (57.02, 57.021463735, 0.05),
    (57.0, 57.021463735, 0.05),

    (42.0, 42.010564684, 0.05),
    (42.01, 42.010564684, 0.05),

    (15.99, 15.99491461957, 0.05),
    (16.0, 15.99491461957, 0.05),

    (79.97, 79.966330410, 0.05),
    (80.0, 79.966330410, 0.05),

    (0.98, 0.984015585, 0.03),

    (-17.03, -17.026549101, 0.05),
    (-18.01, -18.010564684, 0.05),
]


def snap_numeric_mod_mass(x):
    if x is None:
        return None

    x = float(x)

    for approx, exact, tol in COMMON_NUMERIC_MODS:
        if abs(x - approx) <= tol:
            return exact

    return x


def get_mod_delta(mod_text: str):
    if mod_text is None:
        return None

    key = normalize_mod_name(mod_text)

    if key in MOD_MASS:
        return MOD_MASS[key]

    numeric = parse_numeric_mass(mod_text)

    if numeric is not None:
        return snap_numeric_mod_mass(numeric)

    return None


def strip_flanking_aa(seq: str) -> str:
    s = str(seq).strip()

    dot_positions = []
    depth = 0

    for i, ch in enumerate(s):
        if ch in ["[", "(", "{"]:
            depth += 1
        elif ch in ["]", ")", "}"]:
            depth = max(0, depth - 1)
        elif ch == "." and depth == 0:
            dot_positions.append(i)

    if len(dot_positions) != 2:
        return s

    left = s[:dot_positions[0]]
    middle = s[dot_positions[0] + 1:dot_positions[1]]
    right = s[dot_positions[1] + 1:]

    if (
        len(left) == 1
        and len(right) == 1
        and left in AA_MASS
        and right in AA_MASS
        and len(middle) > 0
    ):
        return middle

    return s


def parse_peptide_features(seq):
    if seq is None or str(seq).strip() == "":
        return {
            "peptide_length": 0,
            "mod_count": 0,
            "unknown_mod_count": 0,
            "has_mod": 0,
            "neutral_mass": None,
            "basic_aa_count": 0,
            "acidic_aa_count": 0,
            "hydrophobic_aa_count": 0,
            "missed_cleavage_like": 0,
            "parse_ok": 0,
        }

    s = strip_flanking_aa(seq)

    aa_list = []
    mod_count = 0
    unknown_mod_count = 0
    neutral_mass = H2O
    parse_ok = 1
    pending_nterm_delta = 0.0

    i = 0

    while i < len(s):
        ch = s[i]

        if (
            ch == "n"
            and i == 0
            and i + 1 < len(s)
            and s[i + 1] in ["[", "(", "{"]
        ):
            open_ch = s[i + 1]
            close_ch = {"[": "]", "(": ")", "{": "}"}[open_ch]
            j = s.find(close_ch, i + 2)

            if j == -1:
                parse_ok = 0
                i += 1
                continue

            mod_text = s[i + 2:j]
            delta = get_mod_delta(mod_text)

            mod_count += 1

            if delta is None:
                unknown_mod_count += 1
                parse_ok = 0
            else:
                pending_nterm_delta += delta

            i = j + 1
            continue

        if ch == "n" and i == 0:
            i += 1
            continue

        if (
            ch == "c"
            and len(aa_list) > 0
            and i + 1 < len(s)
            and s[i + 1] in ["[", "(", "{"]
        ):
            open_ch = s[i + 1]
            close_ch = {"[": "]", "(": ")", "{": "}"}[open_ch]
            j = s.find(close_ch, i + 2)

            if j == -1:
                parse_ok = 0
                i += 1
                continue

            rest = s[j + 1:].strip("-_. ")

            if rest == "":
                mod_text = s[i + 2:j]
                delta = get_mod_delta(mod_text)

                mod_count += 1

                if delta is None:
                    unknown_mod_count += 1
                    parse_ok = 0
                else:
                    neutral_mass += delta

                i = j + 1
                continue

        if ch in ["[", "(", "{"]:
            open_ch = ch
            close_ch = {"[": "]", "(": ")", "{": "}"}[open_ch]
            j = s.find(close_ch, i + 1)

            if j == -1:
                parse_ok = 0
                i += 1
                continue

            mod_text = s[i + 1:j]
            delta = get_mod_delta(mod_text)

            mod_count += 1

            if delta is None:
                unknown_mod_count += 1
                parse_ok = 0
            else:
                if len(aa_list) == 0:
                    pending_nterm_delta += delta
                else:
                    neutral_mass += delta

            i = j + 1
            continue

        aa_ch = ch.upper() if ch.isalpha() else ch

        if aa_ch in AA_MASS:
            aa = aa_ch
            aa_list.append(aa)

            mass = AA_MASS[aa]

            if len(aa_list) == 1 and pending_nterm_delta != 0.0:
                mass += pending_nterm_delta
                pending_nterm_delta = 0.0

            neutral_mass += mass
            i += 1

            while i < len(s) and s[i] in ["[", "(", "{"]:
                open_ch = s[i]
                close_ch = {"[": "]", "(": ")", "{": "}"}[open_ch]
                j = s.find(close_ch, i + 1)

                if j == -1:
                    parse_ok = 0
                    i += 1
                    break

                mod_text = s[i + 1:j]
                delta = get_mod_delta(mod_text)

                mod_count += 1

                if delta is None:
                    unknown_mod_count += 1
                    parse_ok = 0
                else:
                    neutral_mass += delta

                i = j + 1

            continue

        if ch in ["-", "_", ".", " "]:
            i += 1
            continue

        parse_ok = 0
        i += 1

    peptide_length = len(aa_list)

    if peptide_length == 0:
        parse_ok = 0
        neutral_mass = None

    if peptide_length > 0 and pending_nterm_delta != 0.0:
        neutral_mass += pending_nterm_delta
        pending_nterm_delta = 0.0

    basic_aa_count = sum(1 for aa in aa_list if aa in ["K", "R", "H"])
    acidic_aa_count = sum(1 for aa in aa_list if aa in ["D", "E"])

    hydrophobic_aa_count = sum(
        1 for aa in aa_list
        if aa in ["A", "I", "L", "M", "F", "W", "Y", "V"]
    )

    missed_cleavage_like = 0

    for idx in range(len(aa_list) - 1):
        if aa_list[idx] in ["K", "R"] and aa_list[idx + 1] != "P":
            missed_cleavage_like += 1

    return {
        "peptide_length": peptide_length,
        "mod_count": mod_count,
        "unknown_mod_count": unknown_mod_count,
        "has_mod": 1 if mod_count > 0 else 0,
        "neutral_mass": neutral_mass,
        "basic_aa_count": basic_aa_count,
        "acidic_aa_count": acidic_aa_count,
        "hydrophobic_aa_count": hydrophobic_aa_count,
        "missed_cleavage_like": missed_cleavage_like,
        "parse_ok": parse_ok,
    }


def parse_peptide_struct(seq):
    return parse_peptide_features(seq)


def cast_existing(df: pl.DataFrame, casts: dict) -> pl.DataFrame:
    exprs = []

    for col_name, dtype in casts.items():
        if col_name in df.columns:
            exprs.append(pl.col(col_name).cast(dtype))

    if exprs:
        return df.with_columns(exprs)

    return df


def is_valid_parquet_file(path: Path) -> bool:
    """快速检查 parquet 文件头尾，避免把半写入文件覆盖原文件。"""
    try:
        if not path.exists() or path.stat().st_size < 8:
            return False

        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            tail = f.read(4)

        return head == b"PAR1" and tail == b"PAR1"

    except Exception:
        return False


def cleanup_tmp_file(tmp_path: Path):
    """失败时尽量清理临时 parquet，避免下次把坏 tmp 文件也扫进去。"""
    try:
        if tmp_path.exists():
            tmp_path.unlink()
            print(f"已删除临时文件：{tmp_path}")
    except Exception as e:
        print(f"临时文件删除失败：{tmp_path}，错误：{e}")


def build_unique_spectrum_safe(df: pl.DataFrame) -> pl.DataFrame:
    """
    每个 group_key 只裁剪一次谱图，然后 join 回原 df。
    避免在大 List 列上重复 map，降低内存峰值。
    """
    unique_spectrum_raw = (
        df
        .select(["group_key", "mz_array", "intensity_array"])
        .unique(["group_key"], keep="first")
    )

    group_dtype = unique_spectrum_raw.schema["group_key"]

    group_keys = []
    mz_lists = []
    intensity_lists = []

    for row in unique_spectrum_raw.iter_rows(named=True):
        new_mz, new_intensity = keep_topk_peaks(
            row["mz_array"],
            row["intensity_array"],
            TOPK_PEAKS,
        )

        group_keys.append(row["group_key"])
        mz_lists.append(new_mz)
        intensity_lists.append(new_intensity)

    del unique_spectrum_raw
    gc.collect()

    unique_spectrum = pl.DataFrame(
        {
            "group_key": group_keys,
            "mz_array": mz_lists,
            "intensity_array": intensity_lists,
        },
        schema={
            "group_key": group_dtype,
            "mz_array": pl.List(pl.Float32),
            "intensity_array": pl.List(pl.Float32),
        },
    )

    del group_keys, mz_lists, intensity_lists
    gc.collect()

    return unique_spectrum


def add_feature_1(path: Path):
    print(f"正在处理 {path}", flush=True)

    schema_cols = pl.scan_parquet(path).collect_schema().names()

    missing_order_features = [c for c in ORDER_AWARE_FEATURES if c not in schema_cols]
    if "aux_feature_done" in schema_cols and not FORCE_REBUILD_AUX_FEATURES and not missing_order_features:
        print(f"已处理过，跳过：{path}", flush=True)
        return

    if "aux_feature_done" in schema_cols and missing_order_features:
        print(f"检测到旧版 aux 特征缺少候选顺序特征，将重算：{missing_order_features[:5]}", flush=True)

    read_cols = [c for c in NEEDED_COLS if c in schema_cols]
    df = pl.read_parquet(path, columns=read_cols)
    df = df.with_row_index("__input_row_order")

    if "index" in df.columns:
        df = df.with_columns(
            pl.when(pl.col("index").is_not_null())
            .then(pl.col("index").cast(pl.Int64))
            .otherwise(pl.col("__input_row_order").cast(pl.Int64))
            .alias("__candidate_source_order")
        )
    else:
        df = df.with_columns(
            pl.col("__input_row_order").cast(pl.Int64).alias("__candidate_source_order")
        )

    df = cast_existing(df, {
        "precursor_mz": pl.Float32,
        "rt": pl.Float32,
        "predicted_rt": pl.Float32,
        "delta_rt": pl.Float32,
        "charge": pl.Int8,
        "ion_mobility": pl.Float32,
        "has_ion_mobility": pl.Int8,
        "fp_q_value": pl.Float32,
        "in_fp": pl.Int8,
        "sage_discriminant_score": pl.Float32,
        "spectrum_q": pl.Float32,
        "label": pl.Int8,
    })

    unique_spectrum = build_unique_spectrum_safe(df)

    df = (
        df
        .drop(["mz_array", "intensity_array"])
        .join(unique_spectrum, on="group_key", how="left")
    )

    del unique_spectrum
    gc.collect()

    peptide_feature = (
        df
        .select("precursor_sequence")
        .unique()
        .with_columns(
            pl.col("precursor_sequence")
            .map_elements(
                parse_peptide_struct,
                return_dtype=pl.Struct({
                    "peptide_length": pl.Int64,
                    "mod_count": pl.Int64,
                    "unknown_mod_count": pl.Int64,
                    "has_mod": pl.Int64,
                    "neutral_mass": pl.Float64,
                    "basic_aa_count": pl.Int64,
                    "acidic_aa_count": pl.Int64,
                    "hydrophobic_aa_count": pl.Int64,
                    "missed_cleavage_like": pl.Int64,
                    "parse_ok": pl.Int64,
                }),
            )
            .alias("pep")
        )
        .select([
            "precursor_sequence",

            pl.col("pep")
            .struct.field("peptide_length")
            .cast(pl.Int32)
            .alias("peptide_length"),

            pl.col("pep")
            .struct.field("mod_count")
            .cast(pl.Int32)
            .alias("mod_count"),

            pl.col("pep")
            .struct.field("unknown_mod_count")
            .cast(pl.Int32)
            .alias("unknown_mod_count"),

            pl.col("pep")
            .struct.field("has_mod")
            .cast(pl.Int8)
            .alias("has_mod"),

            pl.col("pep")
            .struct.field("neutral_mass")
            .cast(pl.Float32)
            .alias("neutral_mass"),

            pl.col("pep")
            .struct.field("basic_aa_count")
            .cast(pl.Int32)
            .alias("basic_aa_count"),

            pl.col("pep")
            .struct.field("acidic_aa_count")
            .cast(pl.Int32)
            .alias("acidic_aa_count"),

            pl.col("pep")
            .struct.field("hydrophobic_aa_count")
            .cast(pl.Int32)
            .alias("hydrophobic_aa_count"),

            pl.col("pep")
            .struct.field("missed_cleavage_like")
            .cast(pl.Int32)
            .alias("missed_cleavage_like"),

            pl.col("pep")
            .struct.field("parse_ok")
            .cast(pl.Int8)
            .alias("parse_ok"),
        ])
    )

    if peptide_feature.height == 0:
        print("peptide_feature长度为0，有误！", flush=True)
        del df
        del peptide_feature
        gc.collect()
        return

    df = df.join(peptide_feature, on="precursor_sequence", how="left")

    del peptide_feature
    gc.collect()

    df = df.with_columns([
        (
            (pl.col("neutral_mass") + pl.col("charge") * PROTON) / pl.col("charge")
        )
        .cast(pl.Float32)
        .alias("theoretical_precursor_mz")
    ])

    df = df.with_columns([
        (
            (pl.col("precursor_mz") - pl.col("theoretical_precursor_mz"))
            / pl.col("theoretical_precursor_mz")
            * 1_000_000.0
        )
        .cast(pl.Float32)
        .alias("precursor_ppm_error")
    ])

    df = df.with_columns([
        pl.col("precursor_ppm_error")
        .abs()
        .cast(pl.Float32)
        .alias("abs_precursor_ppm_error")
    ])

    df = df.with_columns([
        (
            (
                pl.col("precursor_mz")
                - (pl.col("theoretical_precursor_mz") + (-1) * C13_DIFF / pl.col("charge"))
            )
            / (pl.col("theoretical_precursor_mz") + (-1) * C13_DIFF / pl.col("charge"))
            * 1_000_000.0
        )
        .cast(pl.Float32)
        .alias("ppm_iso_-1"),

        (
            (
                pl.col("precursor_mz")
                - (pl.col("theoretical_precursor_mz") + 0 * C13_DIFF / pl.col("charge"))
            )
            / (pl.col("theoretical_precursor_mz") + 0 * C13_DIFF / pl.col("charge"))
            * 1_000_000.0
        )
        .cast(pl.Float32)
        .alias("ppm_iso_0"),

        (
            (
                pl.col("precursor_mz")
                - (pl.col("theoretical_precursor_mz") + 1 * C13_DIFF / pl.col("charge"))
            )
            / (pl.col("theoretical_precursor_mz") + 1 * C13_DIFF / pl.col("charge"))
            * 1_000_000.0
        )
        .cast(pl.Float32)
        .alias("ppm_iso_1"),

        (
            (
                pl.col("precursor_mz")
                - (pl.col("theoretical_precursor_mz") + 2 * C13_DIFF / pl.col("charge"))
            )
            / (pl.col("theoretical_precursor_mz") + 2 * C13_DIFF / pl.col("charge"))
            * 1_000_000.0
        )
        .cast(pl.Float32)
        .alias("ppm_iso_2"),
    ])

    df = df.with_columns([
        pl.col("ppm_iso_-1").abs().cast(pl.Float32).alias("abs_ppm_iso_-1"),
        pl.col("ppm_iso_0").abs().cast(pl.Float32).alias("abs_ppm_iso_0"),
        pl.col("ppm_iso_1").abs().cast(pl.Float32).alias("abs_ppm_iso_1"),
        pl.col("ppm_iso_2").abs().cast(pl.Float32).alias("abs_ppm_iso_2"),
    ])

    df = df.with_columns([
        pl.min_horizontal([
            pl.col("abs_ppm_iso_-1"),
            pl.col("abs_ppm_iso_0"),
            pl.col("abs_ppm_iso_1"),
            pl.col("abs_ppm_iso_2"),
        ])
        .cast(pl.Float32)
        .alias("min_abs_precursor_ppm"),
    ])

    df = df.with_columns([
        pl.when(pl.col("min_abs_precursor_ppm") == pl.col("abs_ppm_iso_-1"))
        .then(pl.lit(-1))
        .when(pl.col("min_abs_precursor_ppm") == pl.col("abs_ppm_iso_0"))
        .then(pl.lit(0))
        .when(pl.col("min_abs_precursor_ppm") == pl.col("abs_ppm_iso_1"))
        .then(pl.lit(1))
        .otherwise(pl.lit(2))
        .cast(pl.Int8)
        .alias("best_isotope_offset")
    ])

    stats = df.select([
        pl.col("rt").mean().alias("rt_mean"),
        pl.col("rt").std().alias("rt_std"),
        pl.col("rt").quantile(0.01).alias("rt_p01"),
        pl.col("rt").quantile(0.99).alias("rt_p99"),

        pl.col("delta_rt").mean().alias("delta_rt_mean"),
        pl.col("delta_rt").std().alias("delta_rt_std"),

        pl.col("predicted_rt").mean().alias("predicted_rt_mean"),
        pl.col("predicted_rt").std().alias("predicted_rt_std"),

        pl.col("ion_mobility").mean().alias("im_mean"),
        pl.col("ion_mobility").std().alias("im_std"),
        pl.col("ion_mobility").quantile(0.01).alias("im_p01"),
        pl.col("ion_mobility").quantile(0.99).alias("im_p99"),
    ]).row(0, named=True)

    rt_mean = safe_float(stats["rt_mean"], 0.0)
    rt_std = safe_float(stats["rt_std"], 0.0)
    rt_p01 = safe_float(stats["rt_p01"], 0.0)
    rt_p99 = safe_float(stats["rt_p99"], 0.0)

    delta_rt_mean = safe_float(stats["delta_rt_mean"], 0.0)
    delta_rt_std = safe_float(stats["delta_rt_std"], 0.0)

    predicted_rt_mean = safe_float(stats["predicted_rt_mean"], 0.0)
    predicted_rt_std = safe_float(stats["predicted_rt_std"], 0.0)

    im_mean = safe_float(stats["im_mean"], 0.0)
    im_std = safe_float(stats["im_std"], 0.0)
    im_p01 = safe_float(stats["im_p01"], 0.0)
    im_p99 = safe_float(stats["im_p99"], 0.0)

    rt_range = rt_p99 - rt_p01
    im_range = im_p99 - im_p01

    if rt_std == 0:
        rt_std = 1.0
    if delta_rt_std == 0:
        delta_rt_std = 1.0
    if predicted_rt_std == 0:
        predicted_rt_std = 1.0
    if im_std == 0:
        im_std = 1.0
    if rt_range == 0:
        rt_range = 1.0
    if im_range == 0:
        im_range = 1.0

    df = df.with_columns([
        pl.col("delta_rt").abs().cast(pl.Float32).alias("abs_delta_rt"),

        ((pl.col("rt") - rt_mean) / rt_std)
        .cast(pl.Float32)
        .alias("rt_z_in_file"),

        ((pl.col("rt") - rt_p01) / rt_range)
        .cast(pl.Float32)
        .alias("rt_norm_in_file"),

        ((pl.col("delta_rt") - delta_rt_mean) / delta_rt_std)
        .cast(pl.Float32)
        .alias("delta_rt_z_in_file"),

        ((pl.col("predicted_rt") - predicted_rt_mean) / predicted_rt_std)
        .cast(pl.Float32)
        .alias("predicted_rt_z_in_file"),

        (
            pl.when(pl.col("has_ion_mobility") == 1)
            .then((pl.col("ion_mobility") - im_mean) / im_std)
            .otherwise(0.0)
        )
        .cast(pl.Float32)
        .alias("ion_mobility_z_in_file"),

        (
            pl.when(pl.col("has_ion_mobility") == 1)
            .then((pl.col("ion_mobility") - im_p01) / im_range)
            .otherwise(0.0)
        )
        .cast(pl.Float32)
        .alias("ion_mobility_norm_in_file"),
    ])

    df = df.with_columns([
        pl.len().over("group_key").cast(pl.Int32).alias("scan_candidate_count"),
        pl.col("__candidate_source_order")
        .rank(method="ordinal", descending=False)
        .over("group_key")
        .cast(pl.Int32)
        .alias("candidate_order_in_scan"),
        pl.col("abs_delta_rt")
        .rank(method="ordinal", descending=False)
        .over("group_key")
        .cast(pl.Int32)
        .alias("abs_delta_rt_rank_in_scan"),
        pl.col("abs_delta_rt").min().over("group_key").cast(pl.Float32).alias("abs_delta_rt_min_in_scan"),
        pl.col("abs_delta_rt").mean().over("group_key").cast(pl.Float32).alias("abs_delta_rt_mean_in_scan"),
        pl.col("abs_delta_rt").std().over("group_key").fill_null(0.0).cast(pl.Float32).alias("abs_delta_rt_std_in_scan"),
        pl.col("charge").n_unique().over("group_key").cast(pl.Int16).alias("scan_unique_charge_count"),
        pl.len().over(["group_key", "charge"]).cast(pl.Int32).alias("same_charge_candidate_count_in_scan"),
        pl.col("__candidate_source_order")
        .rank(method="ordinal", descending=False)
        .over(["group_key", "charge"])
        .cast(pl.Int32)
        .alias("candidate_order_in_scan_charge"),
    ])

    df = df.with_columns([
        pl.col("scan_candidate_count").log1p().cast(pl.Float32).alias("scan_candidate_count_log1p"),
        pl.col("candidate_order_in_scan").log1p().cast(pl.Float32).alias("candidate_order_log1p"),
        (1.0 / pl.col("candidate_order_in_scan").cast(pl.Float32)).cast(pl.Float32).alias("candidate_order_reciprocal"),
        (
            pl.when(pl.col("scan_candidate_count") > 1)
            .then(
                (pl.col("candidate_order_in_scan") - 1).cast(pl.Float32)
                / (pl.col("scan_candidate_count") - 1).cast(pl.Float32)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("candidate_order_pct_in_scan")
        ),
        (
            pl.when(pl.col("scan_candidate_count") > 1)
            .then(
                (pl.col("abs_delta_rt_rank_in_scan") - 1).cast(pl.Float32)
                / (pl.col("scan_candidate_count") - 1).cast(pl.Float32)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("abs_delta_rt_rank_pct_in_scan")
        ),
        (pl.col("candidate_order_in_scan") == 1).cast(pl.Int8).alias("is_first_candidate_in_scan"),
        (pl.col("candidate_order_in_scan") <= 3).cast(pl.Int8).alias("is_top3_candidate_in_scan"),
        (pl.col("abs_delta_rt_rank_in_scan") == 1).cast(pl.Int8).alias("is_best_abs_delta_rt_in_scan"),
        (pl.col("abs_delta_rt") - pl.col("abs_delta_rt_min_in_scan")).cast(pl.Float32).alias("abs_delta_rt_gap_to_best_in_scan"),
        (
            pl.when(pl.col("abs_delta_rt_std_in_scan") > 1e-12)
            .then((pl.col("abs_delta_rt") - pl.col("abs_delta_rt_mean_in_scan")) / pl.col("abs_delta_rt_std_in_scan"))
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("abs_delta_rt_z_in_scan")
        ),
        (pl.col("candidate_order_in_scan") - pl.col("abs_delta_rt_rank_in_scan"))
        .cast(pl.Int32)
        .alias("candidate_order_rt_rank_gap"),
        (pl.col("candidate_order_in_scan") - pl.col("abs_delta_rt_rank_in_scan"))
        .abs()
        .cast(pl.Int32)
        .alias("abs_candidate_order_rt_rank_gap"),
        (pl.col("candidate_order_in_scan") == pl.col("abs_delta_rt_rank_in_scan"))
        .cast(pl.Int8)
        .alias("candidate_order_matches_abs_delta_rank"),
        (pl.col("scan_unique_charge_count") > 1).cast(pl.Int8).alias("scan_has_multiple_charges"),
        (
            pl.col("same_charge_candidate_count_in_scan").cast(pl.Float32)
            / pl.col("scan_candidate_count").cast(pl.Float32)
        )
        .cast(pl.Float32)
        .alias("same_charge_candidate_fraction_in_scan"),
        (
            pl.when(pl.col("same_charge_candidate_count_in_scan") > 1)
            .then(
                (pl.col("candidate_order_in_scan_charge") - 1).cast(pl.Float32)
                / (pl.col("same_charge_candidate_count_in_scan") - 1).cast(pl.Float32)
            )
            .otherwise(0.0)
            .cast(pl.Float32)
            .alias("candidate_order_pct_in_scan_charge")
        ),
        (pl.col("candidate_order_in_scan_charge") == 1)
        .cast(pl.Int8)
        .alias("is_first_candidate_for_charge_in_scan"),
    ])

    df = df.with_columns([
        pl.col("fp_q_value").is_not_null().cast(pl.Int8).alias("has_fp_q_value"),

        pl.col("fp_q_value")
        .fill_null(999.0)
        .cast(pl.Float32)
        .alias("fp_q_value_filled"),

        pl.col("sage_discriminant_score")
        .fill_null(0.0)
        .cast(pl.Float32)
        .alias("sage_score_filled"),

        pl.col("spectrum_q")
        .fill_null(999.0)
        .cast(pl.Float32)
        .alias("spectrum_q_filled"),

        pl.col("label").cast(pl.Int8),
    ])

    all_cols = [
        "file_id",
        "instrument",
        "index",
        "scan_number",
        "group_key",
        "precursor_sequence",
        "peptide_key",

        "mz_array",
        "intensity_array",
        "precursor_mz",

        "rt",
        "predicted_rt",
        "delta_rt",
        "charge",
        "ion_mobility",
        "has_ion_mobility",

        "label",
        "sage_discriminant_score",
        "spectrum_q",
        "fp_q_value",
        "in_fp",

        "peptide_length",
        "mod_count",
        "unknown_mod_count",
        "has_mod",
        "neutral_mass",
        "parse_ok",
        "basic_aa_count",
        "acidic_aa_count",
        "hydrophobic_aa_count",
        "missed_cleavage_like",

        "theoretical_precursor_mz",
        "precursor_ppm_error",
        "abs_precursor_ppm_error",
        "ppm_iso_-1",
        "ppm_iso_0",
        "ppm_iso_1",
        "ppm_iso_2",
        "abs_ppm_iso_-1",
        "abs_ppm_iso_0",
        "abs_ppm_iso_1",
        "abs_ppm_iso_2",
        "min_abs_precursor_ppm",
        "best_isotope_offset",

        "abs_delta_rt",
        "rt_z_in_file",
        "rt_norm_in_file",
        "delta_rt_z_in_file",
        "predicted_rt_z_in_file",
        "ion_mobility_z_in_file",
        "ion_mobility_norm_in_file",

        "scan_candidate_count",
        "scan_candidate_count_log1p",
        "candidate_order_in_scan",
        "candidate_order_pct_in_scan",
        "candidate_order_reciprocal",
        "candidate_order_log1p",
        "is_first_candidate_in_scan",
        "is_top3_candidate_in_scan",
        "abs_delta_rt_rank_in_scan",
        "abs_delta_rt_rank_pct_in_scan",
        "is_best_abs_delta_rt_in_scan",
        "abs_delta_rt_min_in_scan",
        "abs_delta_rt_mean_in_scan",
        "abs_delta_rt_std_in_scan",
        "abs_delta_rt_gap_to_best_in_scan",
        "abs_delta_rt_z_in_scan",
        "candidate_order_rt_rank_gap",
        "abs_candidate_order_rt_rank_gap",
        "candidate_order_matches_abs_delta_rank",
        "scan_unique_charge_count",
        "scan_has_multiple_charges",
        "same_charge_candidate_count_in_scan",
        "same_charge_candidate_fraction_in_scan",
        "candidate_order_in_scan_charge",
        "candidate_order_pct_in_scan_charge",
        "is_first_candidate_for_charge_in_scan",

        "has_fp_q_value",
        "fp_q_value_filled",
        "sage_score_filled",
        "spectrum_q_filled",
    ]

    tmp_order_cols = [
        col_name for col_name in ["__input_row_order", "__candidate_source_order"]
        if col_name in df.columns
    ]
    if tmp_order_cols:
        df = df.drop(tmp_order_cols)

    remaining_cols = [c for c in df.columns if c not in all_cols]
    all_cols = [c for c in all_cols if c in df.columns] + remaining_cols
    df = df.select(all_cols)

    recomputed_cols = set(df.columns) | {"aux_feature_done"}
    preserve_cols = [
        col_name for col_name in schema_cols
        if col_name not in recomputed_cols
    ]
    if preserve_cols:
        preserved_df = pl.read_parquet(path, columns=preserve_cols)
        if preserved_df.height != df.height:
            raise RuntimeError(
                f"保留旧特征列时行数不一致：preserved={preserved_df.height}, current={df.height}, path={path}"
            )
        df = pl.concat([df, preserved_df], how="horizontal")
        del preserved_df
        gc.collect()

    tmp_path = path.with_name(path.name + TMP_SUFFIX)

    cleanup_tmp_file(tmp_path)

    df = df.with_columns([
        pl.lit(1).cast(pl.Int8).alias("aux_feature_done")
    ])

    row_count = df.height
    col_count = len(df.columns)

    df_deleted = False

    try:
        df.write_parquet(tmp_path)

        if not is_valid_parquet_file(tmp_path):
            raise RuntimeError(f"临时 parquet 写入不完整：{tmp_path}")

        del df
        df_deleted = True
        gc.collect()

        if MAKE_BACKUP:
            backup_path = path.with_name(path.name + ".bak")

            if not backup_path.exists():
                os.replace(path, backup_path)
            else:
                path.unlink()

            os.replace(tmp_path, path)

        else:
            os.replace(tmp_path, path)

    except BaseException:
        if not df_deleted:
            try:
                del df
            except Exception:
                pass

            gc.collect()

        cleanup_tmp_file(tmp_path)
        raise

    print(f"特征构建完成：{path}", flush=True)
    print(f"行数：{row_count}", flush=True)
    print(f"列数：{col_count}", flush=True)


def run_one_file_task(file: Path):
    """
    主进程中的任务函数。

    默认使用 --one-file 模式启动独立子进程处理单个 parquet 文件。
    这样可以保留崩溃隔离能力：
    某个文件导致 Polars / Arrow 底层崩溃时，不会拖死整个主任务。
    """
    try:
        if RUN_EACH_FILE_IN_SUBPROCESS:
            env = os.environ.copy()
            env["POLARS_MAX_THREADS"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["PYTHONUNBUFFERED"] = "1"

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--one-file",
                    str(file),
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                cleanup_tmp_file(file.with_name(file.name + TMP_SUFFIX))

                return {
                    "status": "failed",
                    "file": str(file),
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "error": f"子进程退出码：{result.returncode}",
                }

            return {
                "status": "ok",
                "file": str(file),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": "",
            }

        else:
            add_feature_1(file)

            return {
                "status": "ok",
                "file": str(file),
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "error": "",
            }

    except BaseException as e:
        cleanup_tmp_file(file.with_name(file.name + TMP_SUFFIX))

        return {
            "status": "failed",
            "file": str(file),
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "error": repr(e),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[str(path) for path in PROCESSED_DIRS],
        help="要处理的 parquet 目录。默认使用脚本顶部 PROCESSED_DIRS。",
    )
    parser.add_argument(
        "--one-file",
        type=str,
        default=None,
        help="内部/调试参数：只处理单个 parquet 文件。",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="调试用，只处理前 N 个文件。",
    )
    args = parser.parse_args()

    if args.one_file is not None:
        add_feature_1(Path(args.one_file))
        return

    all_files = []

    for processed_dir in [Path(item) for item in args.dirs]:
        files = [
            p for p in sorted(processed_dir.glob("*.parquet"))
            if ".tmp" not in p.name and not p.name.endswith(".bak")
        ]

        print(f"{processed_dir} 找到 {len(files)} 个 parquet 文件")
        all_files.extend(files)

    if args.max_files is not None:
        all_files = all_files[: args.max_files]

    print(f"共找到 {len(all_files)} 个 parquet 文件")

    if len(all_files) == 0:
        print("没有需要处理的文件")
        return

    max_workers = min(N_WORKERS, len(all_files))

    print(f"并行子进程数：{max_workers}")
    print("开始并行构建辅助特征...")

    failed = []
    failed_logs = []
    ok_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(run_one_file_task, file): file
            for file in all_files
        }

        for future in tqdm(
            as_completed(future_to_file),
            total=len(future_to_file),
            desc="Adding aux features",
        ):
            file = future_to_file[future]

            try:
                result = future.result()
            except BaseException as e:
                failed.append(str(file))
                cleanup_tmp_file(file.with_name(file.name + TMP_SUFFIX))

                msg = (
                    f"处理失败：{file}\n"
                    f"主进程捕获异常：{repr(e)}"
                )
                failed_logs.append(msg)
                tqdm.write(msg)
                continue

            if result["status"] == "ok":
                ok_count += 1
                tqdm.write(f"完成：{result['file']}")

            else:
                failed.append(result["file"])

                msg = (
                    f"处理失败：{result['file']}\n"
                    f"{result['error']}\n"
                    f"stdout:\n{result['stdout']}\n"
                    f"stderr:\n{result['stderr']}"
                )

                failed_logs.append(msg)
                tqdm.write(msg)

            gc.collect()

    print("=" * 80)
    print("文件处理完成")
    print(f"成功：{ok_count}")
    print(f"失败：{len(failed)}")

    if len(failed) > 0:
        print("【以下文件处理失败】：")
        for f in failed:
            print(f)

        log_path = Path.cwd() / "add_aux_features_failed.log"
        log_text = ("\n\n" + "=" * 80 + "\n\n").join(failed_logs)
        log_path.write_text(log_text, encoding="utf-8")

        print(f"失败日志已保存到：{log_path}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one-file":
        add_feature_1(Path(sys.argv[2]))
    else:
        main()
