# 统计 b/y 离子的匹配特征
from pathlib import Path
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
    # 转为 numpy 数组
    mz = np.asarray(mz_array, dtype=np.float64)
    intensity = np.asarray(intensity_array, dtype=np.float64)

    # 谱图为空，或是强度、质荷比无法一一对应
    if (mz_array is None or intensity_array is None) or (len(mz) == 0 or len(intensity) == 0 or len(mz) != len(intensity)):
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
    # 如 [+15.9949] [mass=57.02]
    return

# 肽段解析，返回氨基酸序列 + 每个残基质量
def parse_peptide_to_residue_masses(seq):
    if seq is None or str(seq).strip() == "":
        return [], [], 0

    s = str(seq).strip()
    # K.PEPTIDE.R 的情况，中间部分是真正的肽段
    if s.count(".") == 2:
        s = s.split(".")[1]

    aa_list = []
    masses = []    # 记录残基质量
    parse_ok = 1    # 记录是否解析成功
    pending_nterm_delta = 0.0  # 记录 n 端修饰的质量

    i = 0
    while i < len(s):
        ch = s[i]

        # 情况1：当前字符是修饰的左括号
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





# 生成理论b y 离子
def build_theoretical_b_y_ions(seq, precursor_chage):


# 计算 PSM 的片段特征
def compute_fragment_features(seq, charge, spec, theo_cache):

    key = (str(seq), int(charge) if charge is not None else 1)

    # 尝试从缓存取出理论离子
    theo = theo_cache.get(key)
    if theo is None:
        theo = build_theoretical_ions(seq, charge)
        theo_cache.put(key, theo)


def add_fragment_feature(path: Path):
    schema_cols = pl.scan_parquet(path).collect_schema().names()
    if "fragment_feature_done" in schema_cols:
        print(f"fragment 特征已处理过，跳过：{path}")
        return

    if "aux_feature_done" not in schema_cols:
        print(f"基础辅助特征尚未完成，跳过：{path}")
        return

    print(f"\n开始处理：{path}")

    df = pl.read_parquet(path)
    required_cols = [
        "precursor_sequence",
        "charge",
        "mz_array",
        "intensity_array",
        "group_key"
    ]

    missed_cols = [c for c in required_cols if c not in df.columns]
    if missed_cols:
        raise ValueError(f"缺少必要列：{missed_cols}")

    # 记录下处理后的谱图 spec、理论 b/y 离子 theo
    spec_cache = LRUCache(max_size=2048)
    theo_cache = LRUCache(max_size=200000)

    # 存放新增特征的数据
    feature_data = {name: [] for name in FRAGMENT_FEATURE_NAMES}

    iter_df = df.select([
        "group_key",
        "precursor_sequence",
        "charge",
        "mz_array",
        "intensity_array",
    ])
    for row in tqdm(iter_df.iter_rows(named=True), total=df.height):
        group_key = row["group_key"]

        # 当前 PSM 对应的谱图可能已被处理过，先查缓存
        spec = spec_cache.get(group_key)
        if spec is None:
            # 获取处理后的谱图
            spec = preprocess_spectrum(row["mz_array"], row["intensity_array"])
            spec_cache.put(group_key, spec)



















    df = df.with_columns([
        pl.lit(1).cast(pl.Int8).alias("fragment_feature_done")
    ])



def main():
    all_files = []
    for dir in PROCESSED_DIRS:
        files = sorted(dir.rglob(f"{dir}/*.parquet"))
        print(f"{dir} 找到 {len(files)} 个 parquet")
        all_files.extend(files)

    print(f"共需要处理 {len(all_files)} 个文件")
    failed = []

    for path in tqdm(all_files):
        try:
            add_fragment_feature(path)