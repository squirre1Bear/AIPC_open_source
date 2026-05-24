# 统计 b/y 离子的匹配特征
from pathlib import Path
from tqdm import tqdm
import polars as pl
from collections import OrderedDict

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