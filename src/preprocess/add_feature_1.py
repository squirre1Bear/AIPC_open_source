# 添加前体质量相关特征、RT 归一化等
from pathlib import Path

import numpy as np
from tqdm import tqdm
import math
import polars as pl
import os
import re
import gc

# 路径
PRODESSED_DIRS = [
    Path(r"E:\AIPC_dataset\processed\mzml_merged"),
    Path(r"E:\AIPC_dataset\processed\tims_merged"),
    Path(r"E:\AIPC_dataset\processed\wiff_merged")
]

# 选择是否要备份
MAKE_BACKUP = False

# 临时文件后缀，先写入临时文件，再替换原文件
TMP_SUFFIX = ".tmp_feature.parquet"

# 每张谱图保留 TOPK_PEAKS 个峰
TOPK_PEAKS = 500

# 只读取本脚本需要的列，避免把 parquet 里的无关列全部读进内存
NEEDED_COLS = [
    "file_id",
    "instrument",
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
    "V": 99.068413945
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
    "label:13c(6)15n(4)": 10.008269
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
        # 这里仍然先用 float32 计算，减少 numpy 临时数组内存
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

    # 注意：tolist() 后会变成 Python float。
    # Polars 会把 Python float 推断成 Float64。
    # 所以后面 map_elements 的 return_dtype 要先声明 Float64，再 cast 成 Float32。
    return mz.tolist(), intensity.tolist()


def trim_spectrum_struct(s):
    new_mz, new_intensity = keep_topk_peaks(
        s["mz_array"],
        s["intensity_array"],
        TOPK_PEAKS
    )

    return {
        "mz_array": new_mz,
        "intensity_array": new_intensity
    }


def parse_numeric_mass(text: str):
    text = str(text).strip()
    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)

    if m is None:
        return None

    return safe_float(m.group(0), default=None)


def normalize_mod_name(name: str) -> str:
    name = str(name).strip()
    name = name.replace("_", "")
    name = name.replace(" ", "")
    return name.lower()


def get_mod_delta(mod_text: str):
    if mod_text is None:
        return None

    key = normalize_mod_name(mod_text)

    if key in MOD_MASS:
        return MOD_MASS[key]

    numeric = parse_numeric_mass(mod_text)

    if numeric is not None:
        return numeric

    return None


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

    s = str(seq).strip()

    if s.count(".") == 2:
        parts = s.split(".")
        s = parts[1]

    aa_list = []
    mod_count = 0
    unknown_mod_count = 0
    neutral_mass = H2O
    parse_ok = 1
    pending_nterm_delta = 0.0

    i = 0

    while i < len(s):
        ch = s[i]

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
                pending_nterm_delta += delta

            i = j + 1
            continue

        if ch in AA_MASS:
            aa_list.append(ch)

            mass = AA_MASS[ch]

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

        i += 1

    peptide_length = len(aa_list)

    if peptide_length == 0:
        parse_ok = 0
        neutral_mass = None

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

    # 注意：
    # 这里返回的是 Python int / Python float。
    # Polars 会把 Python int 推断为 Int64，把 Python float 推断为 Float64。
    # 所以后面 map_elements 的 return_dtype 必须先写 Int64 / Float64，
    # 再在展开 struct 后 cast 成 Int32 / Int8 / Float32。
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


def add_feature_1(path: Path):
    print(f"正在处理 {path}")

    # 先只读 schema，避免为了检查 aux_feature_done 就把整个 parquet 读进内存
    schema_cols = pl.scan_parquet(path).collect_schema().names()

    if "aux_feature_done" in schema_cols:
        print(f"已处理过，跳过：{path}")
        return

    read_cols = [c for c in NEEDED_COLS if c in schema_cols]
    df = pl.read_parquet(path, columns=read_cols)

    # 统一字段类型，尽量用 Float32 / Int8 降低内存
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

    # -------------------------------------------------------
    # 构建谱图相关特征，保留强度前 TOPK_PEAKS 个峰
    # 每一个 group_key 只保留一次，避免对同一张谱图的多条 PSM 重复裁剪
    # 这里避免手动 rows.append(...) 生成巨大 Python 列表

    unique_spectrum = (
        df
        .select(["group_key", "mz_array", "intensity_array"])
        .unique(["group_key"], keep="first")
        .with_columns(
            pl.struct(["mz_array", "intensity_array"])
            .map_elements(
                trim_spectrum_struct,
                return_dtype=pl.Struct({
                    # trim_spectrum_struct 返回的是 Python list[float]，
                    # Polars 会识别为 List(Float64)，因此这里先写 Float64。
                    "mz_array": pl.List(pl.Float64),
                    "intensity_array": pl.List(pl.Float64),
                })
            )
            .alias("trimmed")
        )
        .select([
            "group_key",

            pl.col("trimmed")
            .struct.field("mz_array")
            .cast(pl.List(pl.Float32))
            .alias("mz_array"),

            pl.col("trimmed")
            .struct.field("intensity_array")
            .cast(pl.List(pl.Float32))
            .alias("intensity_array"),
        ])
    )

    df = (
        df
        .drop(["mz_array", "intensity_array"])
        .join(unique_spectrum, on="group_key", how="left")
    )

    del unique_spectrum
    gc.collect()

    # -------------------------------------------------------
    # 解析肽段特征
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
                })
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
        print("peptide_feature长度为0，有误！")
        del df
        del peptide_feature
        gc.collect()
        return

    df = df.join(peptide_feature, on="precursor_sequence", how="left")

    del peptide_feature
    gc.collect()

    # -------------------------------------------------------
    # 前体质量与同位素误差特征
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
        .alias("ppm_iso_2")
    ])

    df = df.with_columns([
        pl.col("ppm_iso_-1").abs().cast(pl.Float32).alias("abs_ppm_iso_-1"),
        pl.col("ppm_iso_0").abs().cast(pl.Float32).alias("abs_ppm_iso_0"),
        pl.col("ppm_iso_1").abs().cast(pl.Float32).alias("abs_ppm_iso_1"),
        pl.col("ppm_iso_2").abs().cast(pl.Float32).alias("abs_ppm_iso_2")
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

    # -------------------------------------------------------
    # 归一化 RT。不同仪器 rt 范围可能不同，因此在文件内进行归一化
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
    ]).to_dicts()[0]

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

    # --------------------------------------------------------
    # 填充训练辅助字段
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

    # --------------------------------------------------------
    # 所有字段按序输出

    all_cols = [
        "file_id",
        "instrument",
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

        "has_fp_q_value",
        "fp_q_value_filled",
        "sage_score_filled",
        "spectrum_q_filled",
    ]

    remaining_cols = [c for c in df.columns if c not in all_cols]
    all_cols = [c for c in all_cols if c in df.columns] + remaining_cols
    df = df.select(all_cols)

    # --------------------------------------------------------
    # 先写入临时文件，之后覆盖掉原文件

    tmp_path = path.with_name(path.name + TMP_SUFFIX)

    df = df.with_columns([
        pl.lit(1).cast(pl.Int8).alias("aux_feature_done")
    ])

    row_count = df.height
    col_count = len(df.columns)

    df.write_parquet(tmp_path)

    # 写完后尽早释放 df
    del df
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

    print(f"特征构建完成：{path}")
    print(f"行数：{row_count}")
    print(f"列数：{col_count}")


def main():
    all_files = []

    for dir in PRODESSED_DIRS:
        files = sorted(dir.glob("*.parquet"))
        print(f"{dir} 找到 {len(files)} 个parquet文件")
        all_files.extend(files)

    print(f"共找到 {len(all_files)} 个parquet文件")

    failed = []

    for file in tqdm(all_files):
        try:
            add_feature_1(file)
            gc.collect()

        except Exception as e:
            failed.append(str(file))
            print(f"\n处理失败：{file}")
            print(f"错误信息：{e}")
            gc.collect()

    print("文件处理完成")

    if len(failed) > 0:
        print("【以下文件处理失败】：")
        for f in failed:
            print(f)


if __name__ == "__main__":
    main()