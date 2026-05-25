# 统计 b/y 离子的匹配特征
from pathlib import Path
import os

# Keep native libraries conservative before importing polars/numpy.
os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import sys
import subprocess
import gc
from tqdm import tqdm
import polars as pl
import numpy as np
import re
from collections import OrderedDict
import math

PROCESSED_DIRS = [
    Path(r"E:\AIPC_dataset\processed\mzml_merged"),
    Path(r"E:\AIPC_dataset\processed\tims_merged"),
    Path(r"E:\AIPC_dataset\processed\wiff_merged"),
]

TMP_SUFFIX = ".tmp_fragment.parquet"
MAKE_BACKUP = False

# Process each parquet in a subprocess so the OS fully releases Polars/Arrow/Python heap memory.
RUN_EACH_FILE_IN_SUBPROCESS = True

# Batch size for Python feature lists. Lower value reduces peak RAM; higher value is faster.
FEATURE_BATCH_SIZE = int(os.environ.get("FRAGMENT_FEATURE_BATCH_SIZE", "20000"))

# Cache sizes affect only speed, not results. The old theoretical-ion cache was very large.
SPEC_CACHE_SIZE = int(os.environ.get("FRAGMENT_SPEC_CACHE_SIZE", "512"))
THEO_CACHE_SIZE = int(os.environ.get("FRAGMENT_THEO_CACHE_SIZE", "8192"))

# ppm 容差
PPM_TOLERANCES = [10.0, 20.0, 50.0]
MAIN_TOLERANCE = 20.0
MAX_FRAGMENT_CHARGE = 2

# ----------------------------------------
# 化学常量
# ----------------------------------------
PROTON = 1.007276466812
H2O = 18.01056468403
NH3 = 17.026549101

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

COMMON_NUMERIC_MODS = [
    (42.0, 42.010564684, 0.05),        # Acetyl
    (42.01, 42.010564684, 0.05),       # Acetyl
    (57.02, 57.021463735, 0.05),       # Carbamidomethyl
    (15.99, 15.99491461957, 0.05),     # Oxidation
    (79.97, 79.966330410, 0.05),       # Phospho
    (0.98, 0.984015585, 0.03),         # Deamidated
    (-17.03, -17.026549101, 0.05),     # pyro-Glu
    (-18.01, -18.010564684, 0.05),     # pyro-Glu / H2O loss
]

def snap_numeric_mod_mass(x):
    """
    把 [42]、[57.02] 这类近似数字修饰修正到常见精确质量。
    如果不是常见修饰，就保留原始数字。
    """
    if x is None:
        return None

    x = float(x)

    for approx, exact, tol in COMMON_NUMERIC_MODS:
        if abs(x - approx) <= tol:
            return exact

    return x

# ----------------------------------------
# 新增的列名
# ----------------------------------------
FRAGMENT_FEATURE_NAMES = [
# 解析状态
    "fragment_parse_ok",
    "fragment_position_count",
    "main_ion_count",
    "b_ion_count",
    "y_ion_count",
    "loss_ion_count",

    # 10 ppm 命中特征
    "matched_b_count_10ppm",
    "matched_y_count_10ppm",    # 10 ppm 下命中的b y离子个数
    "matched_total_count_10ppm",
    "matched_b_fraction_10ppm",
    "matched_y_fraction_10ppm",   # 命中的b y离子比例
    "matched_total_fraction_10ppm",

    # 20 ppm 命中特征
    "matched_b_count_20ppm",
    "matched_y_count_20ppm",
    "matched_total_count_20ppm",
    "matched_b_fraction_20ppm",
    "matched_y_fraction_20ppm",
    "matched_total_fraction_20ppm",

    # 50 ppm 命中特征
    "matched_b_count_50ppm",
    "matched_y_count_50ppm",
    "matched_total_count_50ppm",
    "matched_b_fraction_50ppm",
    "matched_y_fraction_50ppm",
    "matched_total_fraction_50ppm",

    # 解释峰强
    "matched_peak_count_20ppm",  # 20 ppm 下谱图中能被b y离子解释的谱峰数
    "explained_intensity_fraction_20ppm",
    "top50_explained_intensity_fraction_20ppm",

    # ppm 误差统计
    "mean_abs_fragment_ppm_20ppm",
    "median_abs_fragment_ppm_20ppm",
    "std_abs_fragment_ppm_20ppm",  # 20 ppm 下匹配成功的ppm最小值、中位数、标准差
    "mean_signed_fragment_ppm_20ppm",
    "intensity_weighted_abs_fragment_ppm_20ppm",  # 按峰强度加权后的绝对 ppm 平均误差

    # 匹配峰排名
    "matched_peak_rank_mean_20ppm",
    "matched_peak_rank_min_20ppm",   # 按峰强度排名，1为最强峰
    "matched_top10_peak_count_20ppm",
    "matched_top50_peak_count_20ppm",

    # ladder / coverage 特征
    "longest_b_ladder_20ppm",
    "longest_y_ladder_20ppm",
    "longest_combined_ladder_20ppm",    # 以上两值中的最大值
    "b_y_complement_pair_count_20ppm",   # 同一个肽段切分点位上，b y离子均在 20ppm 下均命中的数量
    "n_terminal_coverage_20ppm",
    "c_terminal_coverage_20ppm",   # +1 电荷的b y离子匹配结果

    # neutral loss 特征
    "matched_loss_count_20ppm",
    "matched_loss_fraction_20ppm",
    "loss_explained_intensity_fraction_20ppm",
]

def safe_float(x, default=0.0):
    try:
        if x is None:
            return default

        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v

    except Exception:
        return default

def safe_div(a, b, default=0.0):
    if b == 0 or b is None:
        return default

    return float(a) / float(b)

# 计算 flags 中，最长连续 Ture 的长度
def longest_true_run(flags):
    longest = 0
    cur = 0
    for x in flags:
        if x:
            cur += 1
            if cur > longest:
                longest = cur
        else:
            cur = 0

    return longest

# LRU 缓存，避免重复解析肽段或谱图
class LRUCache:
    def __init__(self, max_size=4096):
        self.max_size = max_size
        self.data = OrderedDict()

    def get(self, key):
        # 缓存没有则返回None
        if key not in self.data:
            return None
        value = self.data.pop(key)
        self.data[key] = value
        return value

    def put(self, key, value):
        if key in self.data:
            self.data.pop(key)  # 先删掉旧的，在新插入末尾
        elif len(self.data) >= self.max_size:
            self.data.popitem(last=False)  # last=False表示先删除最前面的，也就是最久没有使用的
        self.data[key] = value

def preprocess_spectrum(mz_array, intensity_array):

    # 谱图为空，或是强度、质荷比无法一一对应
    if mz_array is None or intensity_array is None:
        return {
            "mz": np.asarray([], dtype=np.float64),
            "intensity": np.asarray([], dtype=np.float64),
            "rank": np.asarray([], dtype=np.int32),
            "total_intensity": 0.0,
            "top50_total_intensity": 0.0,
        }

    # 转为 numpy 数组
    mz = np.asarray(mz_array, dtype=np.float64)
    intensity = np.asarray(intensity_array, dtype=np.float64)

    if len(mz) == 0 or len(intensity) == 0 or len(mz) != len(intensity):
        return {
            "mz": np.asarray([], dtype=np.float64),
            "intensity": np.asarray([], dtype=np.float64),
            "rank": np.asarray([], dtype=np.int32),
            "total_intensity": 0.0,
            "top50_total_intensity": 0.0,
        }


    # 过滤谱图中的非法值和0
    mask = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (mz > 0)
        & (intensity > 0)
    )

    mz = mz[mask]
    intensity = intensity[mask]

    # 过滤后谱图为空
    if len(mz) == 0:
        return {
            "mz": np.asarray([], dtype=np.float64),
            "intensity": np.asarray([], dtype=np.float64),
            "rank": np.asarray([], dtype=np.int32),
            "total_intensity": 0.0,
            "top50_total_intensity": 0.0,
        }

    # 按 mz 从小到大对 mz、intensity 重排序
    order_mz = np.argsort(mz)
    mz = mz[order_mz]
    intensity = intensity[order_mz]

    # 峰强度 从大到小 排序
    order_intensity = np.argsort(-intensity)
    rank = np.empty(len(intensity), dtype=np.int32)
    rank[order_intensity] = np.arange(1, len(order_intensity) + 1, dtype=np.int32)

    # 总强度
    total_intensity = float(np.sum(intensity))
    top50_total_intensity = float(np.sum(intensity[rank <= 50]))

    return {
        "mz": mz,
        "intensity": intensity,
        "rank": rank,
        "total_intensity": total_intensity,
        "top50_total_intensity": top50_total_intensity
    }

# 标准化修饰的名称
def normalize_mod_name(name):
    name = str(name).strip()    # 去掉首尾空格
    name = name.lower()
    name = name.replace("_", "")
    name = name.replace(" ", "")
    return name

# 根据修饰名中的数字提取质量
def parse_numeric_mass(text):
    text = str(text).strip()

    m = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    # 如果没有匹配到数字，m 会是 None。
    if m is None:
        return None
    return safe_float(m.group(0), default=None)

# 根据修饰文本获取质量偏移
def get_mod_delta(mod):
    if mod is None:
        return None

    # 标准化名称，统一大小写、下划线等
    key = normalize_mod_name(mod)

    if key in MOD_MASS:
        return MOD_MASS[key]

    # 如果修饰不在字典中，尝试从修饰名中提取质量
    # 如 [57.02]
    numeric = parse_numeric_mass(mod)

    # 获取常见修饰的精确质量
    if numeric is not None:
        return snap_numeric_mod_mass(numeric)

    return numeric

def strip_flanking_aa(seq: str) -> str:
    """
    只处理真正的 K.PEPTIDE.R 格式。
    注意：
    [57.02] 里的小数点不能当作分隔符。
    """
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

    # 只有类似 K.PEPTIDE.R 才剥掉两侧氨基酸
    if (
        len(left) == 1
        and len(right) == 1
        and left in AA_MASS
        and right in AA_MASS
        and len(middle) > 0
    ):
        return middle

    return s

# 肽段解析，返回氨基酸序列 + 每个残基质量
def parse_peptide_to_residue_masses(seq):
    if seq is None or str(seq).strip() == "":
        return [], [], 0

    # 处理 K.PEPTIDE.R 的情况
    # 之前直接找两个 . 的代码会把IGTHNGTFHC[57.02]DEALAC[57.02]ALLR识别错误
    s = strip_flanking_aa(seq)

    aa_list = []
    masses = []    # 记录残基质量
    parse_ok = 1    # 记录是否解析成功
    # 形如 “[修饰]PEPTIDE”，还没有读入残基便有修饰，因此先将修饰信息记录到 pending_nterm_delta 中
    pending_nterm_delta = 0.0  # 记录 n 端修饰的质量

    i = 0
    while i < len(s):
        ch = s[i]

        # ----------------------------------------------------------
        # 情况1: 处理 n[42]PEPTIDE
        # 小写 n 表示 N-terminal，不是氨基酸 N。
        if (ch == "n" and i == 0 and i + 1 < len(s) and s[i + 1] in ["{", "[", "("] ):
            open_ch = s[i + 1]
            close_ch = {"{": "}", "[": "]", "(": ")"}[open_ch]
            j = s.find(close_ch, i + 2)

            if j == -1:
                parse_ok = 0
                i += 1
                continue

            mod_text = s[i + 2:j]
            delta = get_mod_delta(mod_text)

            if delta is None:
                parse_ok = 0
            else:
                pending_nterm_delta += delta

            i = j + 1
            continue

        # 如果当前只有 n 没有修饰，则 n 只表示n端开头，跳过
        if ch == "n" and i == 0:
            i += 1
            continue

        # ----------------------------------------------------
        # 情况2：处理 C-terminal 标记，例如 PEPTIDEc[17]
        # 小写 c 只有在已经读到残基，并且 c[...] 后面没有其他真实字符时，才作为 C 端修饰处理。
        if (
            ch == "c"
            and len(masses) > 0
            and i + 1 < len(s)
            and s[i + 1] in ["{", "[", "("]
        ):
            open_ch = s[i + 1]
            close_ch = {"{": "}", "[": "]", "(": ")"}[open_ch]
            j = s.find(close_ch, i + 2)

            if j == -1:
                parse_ok = 0
                i += 1
                continue

            rest = s[j + 1:].strip("-_. ")

            # 只有 c[...] 在末尾时才认为是 C 端修饰
            if rest == "":
                mod_text = s[i + 2:j]
                delta = get_mod_delta(mod_text)

                if delta is None:
                    parse_ok = 0
                else:
                    masses[-1] += delta

                i = j + 1
                continue

        # ----------------------------------------------------------
        # 情况3：当前字符是修饰的左括号
        if ch in ["{", "[", "("]:
            open_ch = ch

            # 找到对应的右括号
            close_ch = {"{": "}", "[": "]", "(": ")"}[open_ch]
            j = s.find(close_ch, i+1)

            # 没找到右括号，格式有误。跳过该字符
            if j == -1:
                parse_ok = 0
                i += 1
                continue

            # 取出括号内的修饰
            mod_text = s[i+1: j]
            # 获取修饰质量偏移
            delta = get_mod_delta(mod_text)

            # 当前修饰解析失败
            if delta is None:
                parse_ok = 0
            else:
                if len(masses) == 0:
                    # 当前还没有残基质量，则当前为N端修饰
                    pending_nterm_delta += delta
                else:
                    masses[-1] += delta
            i = j + 1
            continue

        # 情况4：当前字符为正常氨基酸
        if ch in AA_MASS:
            aa = ch
            mass = AA_MASS[aa]

            # n端有修饰的情况下，将该修饰质量加到第一个残基上
            if len(aa_list) == 0 and pending_nterm_delta != 0:
                mass += pending_nterm_delta
                pending_nterm_delta = 0.0

            # 将氨基酸加入列表
            aa_list.append(aa)
            i += 1

            # 处理残基后可能存在的修饰
            while i < len(s) and s[i] in ["{", "[", "("]:
                open_ch = s[i]
                close_ch = {"{": "}", "[": "]", "(": ")"}[open_ch]
                j = s.find(close_ch, i+1)

                if j == -1:
                    parse_ok = 0
                    i += 1
                    break

                mod_text = s[i+1: j]
                delta = get_mod_delta(mod_text)

                if delta is None:
                    parse_ok = 0
                else:
                    mass += delta

                i = j + 1

            # 将当前残基质量加入 masses
            masses.append(mass)
            continue

        # 情况5：当前为分隔符，跳过
        if ch in ["-", "_", ".", " "]:
            i += 1
            continue

        # 如果遇到了未知字符，则跳过
        print(f"{seq}中遇到未知字符{ch}")
        i += 1

    # 肽段长度不够生成 b/y 离子
    if len(aa_list) < 2:
        return aa_list, masses, 0

    return aa_list, masses, parse_ok

# 生成理论 b/y 离子
def build_theoretical_b_y_ions(seq, precursor_chage):
    # 从肽段解析出 氨基酸序列、残基（含修饰）的质量
    aa_list, residue_masses, parse_ok = parse_peptide_to_residue_masses(seq)

    # 记录残基数量
    n = len(residue_masses)

    if n < 2:
        return{
            "parse_ok": 0,        # 肽段无法解析 / 长度小于2个氨基酸
            "position_count": 0,  # 可切割点位数量
            "main_ions": [],      # b/y 离子切割结果
            "loss_ions": []       # 计算丢失中性粒子（H20、NH3）后的理论碎片
        }

    try:
        precursor_chage = int(precursor_chage)
    except Exception:
        precursor_chage = 1

    if precursor_chage < 1:
        precursor_chage = 1

    # 限制碎片离子电荷数上限
    max_frag_z = min(MAX_FRAGMENT_CHARGE, max(1, precursor_chage))

    # 生成 list 类型的碎片离子电荷数
    fragment_charges = list(range(1, max_frag_z + 1))

    # 统计残基质量前缀和
    prefix = np.cumsum(np.asarray(residue_masses, dtype=np.float64))

    # 残基总质量（也等于肽段质量-mass(H2O)）
    total_residue_mass = prefix[-1]

    # b/y 离子
    main_ions = []
    # 丢失H2O/NH3的离子
    loss_ions = []

    # 遍历 n-1 个可切割点
    for cleavage_idx in range(1, n):
        # 默认 b 离子不加质量， y 离子加 H2O 质量
        b_neutral = float(prefix[cleavage_idx - 1])
        y_neutral = total_residue_mass - b_neutral + H2O
        position = cleavage_idx

        # 遍历允许的电荷数，生成 b/y 离子
        for z in fragment_charges:
            b_mz = (b_neutral + z * PROTON) / z
            y_mz = (y_neutral + z * PROTON) / z

            main_ions.append({
                "series": "b",         # 离子系列
                "position": position,  # 切割位置（1~n-1）
                "charge": z,
                "mz": b_mz
            })

            main_ions.append({
                "series": "y",
                "position": position,
                "charge": z,
                "mz": y_mz
            })

        # 处理中性粒子丢失的质量
        # neutral loss只计算 +1 电荷，防止引入过多噪声
        b1_mz = b_neutral + PROTON
        y1_mz = y_neutral + PROTON

        # 仅考虑 H2O/NH3
        for loss_name, loss_mass in [("H2O", H2O), ("NH3", NH3)]:
            b_loss = b1_mz - loss_mass
            y_loss = y1_mz - loss_mass
            if b_loss > 0:
                loss_ions.append({
                    "series": "b_loss_" + loss_name,
                    "position": position,
                    "charge": 1,
                    "mz": b_loss,
                })

            if y_loss > 0:
                loss_ions.append({
                    "series": "y_loss_" + loss_name,
                    "position": position,
                    "charge": 1,
                    "mz": y_loss,
                })

    return {
        "parse_ok": parse_ok,
        "position_count": n-1,
        "main_ions": main_ions,
        "loss_ions": loss_ions
    }

# 匹配理论离子和最近峰
def nearest_peak_match(spec, theo_mz):
    """
    输入：
        spec:谱图字典
        theo_mz:某个理论离子的 m/z
    返回：
        best_idx:最接近实测峰在数组中的位置
        signed_ppm:带正负号的 ppm 误差
        abs_ppm:ppm 误差绝对值
        intensity:最接近实测峰的强度
        rank: 最接近实测峰的强度排名
    """
    mz = spec["mz"]
    if len(mz) == 0:
        return -1, 999999.0, 999999.0, 0.0, 999999

    # np.searchsorted 寻找在有序数组中的插入位置
    # 最近峰只会在插入位置 左1/右1 的峰里
    pos = np.searchsorted(mz, theo_mz)

    candidates = []

    if pos > 0:
        candidates.append(pos - 1)

    if pos < len(mz):
        candidates.append(pos)

    if len(candidates) == 0:
        return -1, 999999.0, 999999.0, 0.0, 999999

    best_idx = min(candidates, key=lambda idx: abs(mz[idx] - theo_mz))

    # ppm = （实测mz - 理论mz）/ 理论mz * 100w
    signed_ppm = (mz[best_idx] - theo_mz) / theo_mz * 1_000_000.0
    abs_ppm = abs(signed_ppm)

    return (
        int(best_idx),
        float(signed_ppm),
        float(abs_ppm),
        float(spec["intensity"][best_idx]),
        float(spec["rank"][best_idx])
    )

# 计算 PSM 的片段特征
def compute_fragment_features(seq, charge, spec, theo_cache):

    key = (str(seq), int(charge) if charge is not None else 1)

    # 尝试从缓存取出理论离子
    theo = theo_cache.get(key)
    if theo is None:
        theo = build_theoretical_b_y_ions(seq, charge)
        theo_cache.put(key, theo)

    main_ions = theo["main_ions"]
    loss_ions = theo["loss_ions"]
    position_count = int(theo["position_count"])
    main_ion_count = len(main_ions)
    loss_ion_count = len(loss_ions)

    b_ion_count = sum(1 for ion in main_ions if ion["series"]=="b")
    y_ion_count = sum(1 for ion in main_ions if ion["series"]=="y")

    # 初始化特征为 0.0
    feat = {name: 0.0 for name in FRAGMENT_FEATURE_NAMES}

    # 写入上面求出的基础特征
    feat["fragment_parse_ok"] = int(theo["parse_ok"])
    feat["fragment_position_count"] = int(position_count)
    feat["main_ion_count"] = int(main_ion_count)
    feat["b_ion_count"] = int(b_ion_count)
    feat["y_ion_count"] = int(y_ion_count)
    feat["loss_ion_count"] = int(loss_ion_count)

    # 如果没有理论主离子，则无法做匹配
    if main_ion_count == 0:
        return feat

    # 记录不同容差下的命中数量
    matched_counts = {
        10.0:{"b": 0, "y": 0, "total": 0},
        20.0: {"b": 0, "y": 0, "total": 0},
        50.0: {"b": 0, "y": 0, "total": 0}
    }

    # 记录 20ppm 命中的峰下标
    matched_peak_indices_20 = set()
    matched_top50_peak_indices_20 = set()
    loss_matched_peak_indices_20 = set()

    # 记录 20ppm 命中离子的 ppm 误差、强度、排名
    abs_ppms_20 = []
    signed_ppms_20 = []
    intensities_20 = []
    ranks_20 = []

    # 记录所有可切点点位上，是否有 1+ b/y 离子命中
    b1_hits = [False] * position_count
    y1_hits = [False] * position_count

    # -------------------------------------
    # 主 b/y 离子匹配
    for ion in main_ions:
        best_idx, signed_ppm, abs_ppm, intensity, rank = nearest_peak_match(spec, ion["mz"])

        # 判断是否进入 10/20/50 ppm
        for tolerance in PPM_TOLERANCES:
            if abs_ppm <= tolerance:
                matched_counts[tolerance][ion["series"]] += 1
                matched_counts[tolerance]["total"] += 1

        # 计算 20ppm 其他特征
        if abs_ppm <= MAIN_TOLERANCE:
            matched_peak_indices_20.add(best_idx)
            abs_ppms_20.append(abs_ppm)
            signed_ppms_20.append(signed_ppm)
            intensities_20.append(intensity)
            ranks_20.append(rank)

            if rank <= 50:
                matched_top50_peak_indices_20.add(best_idx)
                feat["matched_top50_peak_count_20ppm"] += 1

            if rank <= 10:
                feat["matched_top10_peak_count_20ppm"] += 1

            # ion["position"] 下标从 1 开始，记录时需要 -1
            pos_idx = int(ion["position"]) - 1

            # 记录 1+ b/y 离子匹配结果，用于统计 ladder 特征
            if 0 <= pos_idx < position_count and int(ion["charge"]) == 1:
                if ion["series"] == "b":  b1_hits[pos_idx] = True
                elif ion["series"] == "y": y1_hits[pos_idx] = True


    # --------------------------------------------------------------------
    # neutral loss 离子匹配
    matched_loss_count_20 = 0

    for ion in loss_ions:
        best_idx, signed_ppm, abs_ppm, intensity, rank = nearest_peak_match(spec, ion["mz"])
        if abs_ppm <= MAIN_TOLERANCE:
            matched_loss_count_20 += 1
            loss_matched_peak_indices_20.add(best_idx)

    # --------------------------------------------------------------------
    # 计算 10/20/50 ppm 命中数量和比例
    for tolerance in PPM_TOLERANCES:
        suffix = f"{int(tolerance)}ppm"

        b_count = matched_counts[tolerance]["b"]
        y_count = matched_counts[tolerance]["y"]
        total_count = matched_counts[tolerance]["total"]

        feat[f"matched_b_count_{suffix}"] = int(b_count)
        feat[f"matched_y_count_{suffix}"] = int(y_count)
        feat[f"matched_total_count_{suffix}"] = int(total_count)

        # 计算命中比例： 命中离子数 / 理论离子数
        feat[f"matched_b_fraction_{suffix}"] = safe_div(b_count, b_ion_count)
        feat[f"matched_y_fraction_{suffix}"] = safe_div(y_count, y_ion_count)
        feat[f"matched_total_fraction_{suffix}"] = safe_div(total_count, main_ion_count)

    # --------------------------------------------------------
    # 解释峰强度比例
    total_intensity = safe_float(spec["total_intensity"], 0.0)
    top50_total_intensity = safe_float(spec["top50_total_intensity"], 0.0)

    # 20 ppm 命中的主离子对应的实测峰强度之和。
    explained_intensity = 0.0
    for idx in matched_peak_indices_20:
        if idx >= 0:
            explained_intensity += float(spec["intensity"][idx])

    # 20 ppm 命中的 top50 峰强度之和。
    top50_explained_intensity = 0.0
    for idx in matched_top50_peak_indices_20:
        if idx >= 0:
            top50_explained_intensity += float(spec["intensity"][idx])

    # neutral loss 命中峰强度之和。
    loss_explained_intensity = 0.0
    for idx in loss_matched_peak_indices_20:
        if idx >= 0:
            loss_explained_intensity += float(spec["intensity"][idx])

    feat["matched_peak_count_20ppm"] = int(len(matched_peak_indices_20))
    feat["explained_intensity_fraction_20ppm"] = safe_div(explained_intensity, total_intensity)
    feat["top50_explained_intensity_fraction_20ppm"] = safe_div(top50_explained_intensity, top50_total_intensity)

    feat["matched_loss_count_20ppm"] = int(matched_loss_count_20)
    feat["matched_loss_fraction_20ppm"] = safe_div(matched_loss_count_20, loss_ion_count)
    feat["loss_explained_intensity_fraction_20ppm"] = safe_div(loss_explained_intensity, total_intensity)

    # --------------------------------------------------------
    # ppm 误差统计
    if len(abs_ppms_20) > 0:
        abs_arr = np.asarray(abs_ppms_20, dtype=np.float64)
        signed_arr = np.asarray(signed_ppms_20, dtype=np.float64)
        int_arr = np.asarray(intensities_20, dtype=np.float64)
        rank_arr = np.asarray(ranks_20, dtype=np.float64)

        feat["mean_abs_fragment_ppm_20ppm"] = float(np.mean(abs_arr))
        feat["median_abs_fragment_ppm_20ppm"] = float(np.median(abs_arr))
        feat["std_abs_fragment_ppm_20ppm"] = float(np.std(abs_arr))
        feat["mean_signed_fragment_ppm_20ppm"] = float(np.mean(signed_arr))

        # 按强度加权的绝对 ppm 误差。
        # 强度越高的峰，对最终误差影响越大。
        if np.sum(int_arr) > 0:
            feat["intensity_weighted_abs_fragment_ppm_20ppm"] = float(
                np.sum(abs_arr * int_arr) / np.sum(int_arr)
            )
        else:
            feat["intensity_weighted_abs_fragment_ppm_20ppm"] = 0.0

        feat["matched_peak_rank_mean_20ppm"] = float(np.mean(rank_arr))
        feat["matched_peak_rank_min_20ppm"] = float(np.min(rank_arr))

    else:
        # 如果 20 ppm 内没有任何命中，就用一个很大的数表示这些统计不可用。
        feat["mean_abs_fragment_ppm_20ppm"] = 999999.0
        feat["median_abs_fragment_ppm_20ppm"] = 999999.0
        feat["std_abs_fragment_ppm_20ppm"] = 999999.0
        feat["mean_signed_fragment_ppm_20ppm"] = 999999.0
        feat["intensity_weighted_abs_fragment_ppm_20ppm"] = 999999.0
        feat["matched_peak_rank_mean_20ppm"] = 999999.0
        feat["matched_peak_rank_min_20ppm"] = 999999.0

    # --------------------------------------------------------
    # ladder / coverage 特征
    longest_b = longest_true_run(b1_hits)
    longest_y = longest_true_run(y1_hits)

    feat["longest_b_ladder_20ppm"] = int(longest_b)
    feat["longest_y_ladder_20ppm"] = int(longest_y)
    feat["longest_combined_ladder_20ppm"] = int(max(longest_b, longest_y))

    if position_count > 0:
        # b/y complement pair：
        # 同一个 cleavage 位置上，b 离子和 y 离子都命中。
        feat["b_y_complement_pair_count_20ppm"] = int(
            sum(1 for b_hit, y_hit in zip(b1_hits, y1_hits) if b_hit and y_hit)
        )

        # N 端覆盖率：b 离子命中的 cleavage 位置比例。
        feat["n_terminal_coverage_20ppm"] = safe_div(sum(b1_hits), position_count)

        # C 端覆盖率：y 离子命中的 cleavage 位置比例。
        feat["c_terminal_coverage_20ppm"] = safe_div(sum(y1_hits), position_count)

    return feat



INT_FRAGMENT_FEATURE_COLS = [
    "fragment_parse_ok",
    "fragment_position_count",
    "main_ion_count",
    "b_ion_count",
    "y_ion_count",
    "loss_ion_count",
    "matched_b_count_10ppm",
    "matched_y_count_10ppm",
    "matched_total_count_10ppm",
    "matched_b_count_20ppm",
    "matched_y_count_20ppm",
    "matched_total_count_20ppm",
    "matched_b_count_50ppm",
    "matched_y_count_50ppm",
    "matched_total_count_50ppm",
    "matched_peak_count_20ppm",
    "matched_top10_peak_count_20ppm",
    "matched_top50_peak_count_20ppm",
    "longest_b_ladder_20ppm",
    "longest_y_ladder_20ppm",
    "longest_combined_ladder_20ppm",
    "b_y_complement_pair_count_20ppm",
    "matched_loss_count_20ppm",
]


def empty_feature_data():
    return {name: [] for name in FRAGMENT_FEATURE_NAMES}


def empty_feature_df():
    return pl.DataFrame({
        name: pl.Series(
            name,
            [],
            dtype=pl.Int32 if name in INT_FRAGMENT_FEATURE_COLS else pl.Float32,
        )
        for name in FRAGMENT_FEATURE_NAMES
    })


def cast_fragment_feature_df(feature_df: pl.DataFrame) -> pl.DataFrame:
    if feature_df.height == 0:
        return empty_feature_df()

    feature_df = feature_df.with_columns([
        pl.col(c).cast(pl.Int32) for c in INT_FRAGMENT_FEATURE_COLS if c in feature_df.columns
    ])

    float_cols = [c for c in feature_df.columns if c not in INT_FRAGMENT_FEATURE_COLS]
    feature_df = feature_df.with_columns([
        pl.col(c).cast(pl.Float32) for c in float_cols
    ])

    return feature_df.select(FRAGMENT_FEATURE_NAMES)


def build_fragment_feature_df(work_df: pl.DataFrame, spec_cache: LRUCache, theo_cache: LRUCache) -> pl.DataFrame:
    feature_batches = []
    feature_data = empty_feature_data()
    batch_len = 0

    for row in tqdm(work_df.iter_rows(named=True), total=work_df.height):
        group_key = row["group_key"]

        spec = spec_cache.get(group_key)
        if spec is None:
            spec = preprocess_spectrum(row["mz_array"], row["intensity_array"])
            spec_cache.put(group_key, spec)

        feat = compute_fragment_features(
            seq=row["precursor_sequence"],
            charge=row["charge"],
            spec=spec,
            theo_cache=theo_cache,
        )

        for name in FRAGMENT_FEATURE_NAMES:
            feature_data[name].append(feat[name])

        batch_len += 1
        if batch_len >= FEATURE_BATCH_SIZE:
            feature_batches.append(cast_fragment_feature_df(pl.DataFrame(feature_data)))
            feature_data = empty_feature_data()
            batch_len = 0
            gc.collect()

    if batch_len > 0:
        feature_batches.append(cast_fragment_feature_df(pl.DataFrame(feature_data)))

    del feature_data
    gc.collect()

    if not feature_batches:
        return empty_feature_df()

    if len(feature_batches) == 1:
        return feature_batches[0]

    # Keep chunked columns to avoid an unnecessary full rechunk copy before the final parquet write.
    return pl.concat(feature_batches, how="vertical", rechunk=False)


def is_valid_parquet_file(path: Path) -> bool:
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
    try:
        if tmp_path.exists():
            tmp_path.unlink()
            print(f"已删除临时文件：{tmp_path}")
    except Exception as e:
        print(f"临时文件删除失败：{tmp_path}，错误：{e}")


def add_fragment_feature(path: Path):
    schema_cols = pl.scan_parquet(path).collect_schema().names()
    if "fragment_feature_done" in schema_cols:
        print(f"fragment 特征已处理过，跳过：{path}")
        return

    if "aux_feature_done" not in schema_cols:
        print(f"基础辅助特征尚未完成，跳过：{path}")
        return

    print(f"\n开始处理：{path}")

    required_cols = [
        "precursor_sequence",
        "charge",
        "mz_array",
        "intensity_array",
        "group_key",
    ]

    missed_cols = [c for c in required_cols if c not in schema_cols]
    if missed_cols:
        raise ValueError(f"缺少必要列：{missed_cols}")

    # Phase 1: read only columns needed for fragment matching and build features in small batches.
    # This avoids keeping the full parquet plus large Python feature lists in memory at the same time.
    work_df = pl.read_parquet(path, columns=required_cols)
    row_count = work_df.height

    spec_cache = LRUCache(max_size=SPEC_CACHE_SIZE)
    theo_cache = LRUCache(max_size=THEO_CACHE_SIZE)

    feature_df = build_fragment_feature_df(work_df, spec_cache, theo_cache)

    if feature_df.height != row_count:
        raise RuntimeError(
            f"fragment feature 行数不一致：feature_df={feature_df.height}, 原始行数={row_count}"
        )

    del work_df
    del spec_cache
    del theo_cache
    gc.collect()

    # Phase 2: read the full parquet only when we are ready to append columns.
    # Existing partial fragment columns are dropped before adding freshly computed columns.
    df = pl.read_parquet(path)
    existing_feature_cols = [
        c for c in FRAGMENT_FEATURE_NAMES + ["fragment_feature_done"] if c in df.columns
    ]
    if existing_feature_cols:
        df = df.drop(existing_feature_cols)

    df = df.hstack(feature_df)
    del feature_df
    gc.collect()

    df = df.with_columns([
        pl.lit(1).cast(pl.Int8).alias("fragment_feature_done")
    ])

    tmp_path = path.with_name(path.name + TMP_SUFFIX)
    cleanup_tmp_file(tmp_path)

    df_deleted = False
    try:
        df.write_parquet(tmp_path)

        if not is_valid_parquet_file(tmp_path):
            raise RuntimeError(f"临时 parquet 写入不完整：{tmp_path}")

        del df
        df_deleted = True
        gc.collect()

        if MAKE_BACKUP:
            backup_path = path.with_name(path.name + ".bak_fragment")

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

    print(f"b/y 特征添加完成：{path}")
    print(f"行数：{row_count}")

def main():
    all_files = []
    for dir in PROCESSED_DIRS:
        files = [
            p for p in sorted(dir.rglob("*.parquet"))
            if ".tmp" not in p.name
            and not p.name.endswith(".bak")
            and not p.name.endswith(".bak_fragment")
        ]
        print(f"{dir} 找到 {len(files)} 个 parquet")
        all_files.extend(files)

    print(f"共需要处理 {len(all_files)} 个文件")
    failed = []

    for path in tqdm(all_files):
        try:
            if RUN_EACH_FILE_IN_SUBPROCESS:
                env = os.environ.copy()
                env.setdefault("POLARS_MAX_THREADS", "1")
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("MKL_NUM_THREADS", "1")
                env.setdefault("OPENBLAS_NUM_THREADS", "1")
                env.setdefault("PYTHONUNBUFFERED", "1")

                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--one-file", str(path)],
                    env=env,
                )

                if result.returncode != 0:
                    failed.append(str(path))
                    print(f"处理失败：{path}")
                    print(f"子进程退出码：{result.returncode}")
                    cleanup_tmp_file(path.with_name(path.name + TMP_SUFFIX))
            else:
                add_fragment_feature(path)

            gc.collect()

        except Exception as e:
            failed.append(str(path))
            print(f"处理失败：{path}")
            print(f"错误信息：{e}")
            cleanup_tmp_file(path.with_name(path.name + TMP_SUFFIX))
            gc.collect()

    print("处理完成")

    if len(failed) > 0:
        print("以下文件处理失败")
        for f in failed:
            print(f)
    else:
        print("无失败文件")

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--one-file":
        add_fragment_feature(Path(sys.argv[2]))
    else:
        main()
