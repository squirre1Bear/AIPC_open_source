from pathlib import Path
import polars as pl
from tqdm import tqdm

# 设置路径
MZML_DIR = Path(r"E:\AIPC_dataset\mzml")
OUT_DIR = Path(r"E:\AIPC_dataset\processed\mzml_merged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_mzml_folders(mzml_dir: Path):
    # 保存目录，每个目录下都还有 sage、rawspectrum 数据
    folders = []
    # rglob为递归搜索，不需要自己再打开子文件夹
    for folder in mzml_dir.rglob("*"):
        if not folder.is_dir():
            continue

        # 这里没有递归，得到的文件夹是raw、sage父目录
        raw_file = list(folder.glob("*_rawspectrum.parquet"))
        sage_file = list(folder.glob("*_sage.parquet"))
        if len(raw_file) > 0 and len(sage_file) > 0:
            folders.append(folder)

    return folders

def merge_one_folder(folder: Path):
    raw_files = list(folder.glob("*_rawspectrum.parquet"))
    sage_files = list(folder.glob("*_sage.parquet"))
    fp_files = list(folder.glob("*_fp.parquet"))
    if len(raw_files) == 0 or len(sage_files) == 0 or len(fp_files) == 0:
        print(f"跳过{folder}，缺少 raw 或 sage/fp 文件")
        return None

    print(f"正在处理{folder.name}")
    raw = pl.read_parquet(raw_files[0])
    raw = raw.select([
        "scan",              # 谱图编号，用于和 sage 表进行匹配
        "precursor_mz",      # 前体离子的质荷比
        "rt",                # retention time，保留时间
        "mz_array",          # 谱峰的 m/z 数组
        "intensity_array",   # 谱峰强度数组
    ])
    raw = raw.rename({
        "scan": "scan_number"
    })

    sage = pl.read_parquet(sage_files[0])
    sage = sage.select([
        "scan",                       # 谱图编号，用于和 raw 表合并
        "precursor_sequence",         # 肽段序列
        "label",                      # 标签，通常用于表示正负样本
        "charge",                     # 电荷数
        "predicted_rt",               # 模型预测的保留时间
        "delta_rt",                   # 预测 RT 和真实 RT 的差值
        "sage_discriminant_score",    # Sage 的判别分数
        "spectrum_q",                 # 谱图层面的 q-value
    ])
    sage = sage.rename({
        "scan": "scan_number"
    })

    fp = pl.read_parquet(fp_files[0])
    fp = fp.select([
        "scan",
        "detect_sequence",
        "q-value"
    ])
    fp = fp.rename({
        "scan": "scan_number",
        "detect_sequence": "precursor_sequence",
        "q-value": "fp_q_value"
    })

    # 先按 scan 合并 raw 和 sage
    merged = sage.join(raw, how="inner", on="scan_number")
    # 根据 scan、sequence 合并数据，并添加 in_fp 列
    merged = merged.join(fp, how="left", on=["scan_number", "precursor_sequence"])
    merged = merged.with_columns([
        pl.col("fp_q_value").is_not_null().cast(pl.Int8).alias("in_fp")
    ])

    merged = merged.with_columns([
        pl.lit("mzml").alias("instrument"),
        pl.lit(folder.name).alias("file_id"),
        pl.lit(0.0).alias("ion_mobility"),
        pl.lit(0).alias("has_ion_mobility"),

        (
            pl.lit("mzml_")
            + pl.lit(folder.name)
            + pl.lit("_")
            + pl.col("scan_number").cast(pl.Utf8)
        ).alias("group_key"),

        # 肽段key，可以添加修饰等信息
        pl.col("precursor_sequence").alias("peptide_key")
    ])

    merged = merged.select([
        # 名称和分组字段
        "file_id",
        "instrument",
        "scan_number",
        "group_key",
        "precursor_sequence",
        "peptide_key",

        # 谱图相关字段
        "mz_array",
        "intensity_array",
        "precursor_mz",

        # 肽段、rt字段
        "rt",
        "predicted_rt",
        "delta_rt",
        "charge",
        "ion_mobility",
        "has_ion_mobility",

        # 打分相关字段
        "label",
        "sage_discriminant_score",
        "spectrum_q",
        "fp_q_value",
        "in_fp"
    ])

    return merged

def main():
    folders = find_mzml_folders(MZML_DIR)
    print(f"发现{len(folders)}个mzml目录")
    for i, folder in enumerate(tqdm(folders)):
        out_path = OUT_DIR / f"mzml_merged_{i:04d}.parquet"
        if out_path.exists():
            continue

        try:
            merged = merge_one_folder(folder)
            if merged is None:
                continue
            merged.write_parquet(out_path)
            print(f"已保存{out_path}, 行数{len(merged)}")
        except Exception as e:
            print(f"处理失败{folder}，错误信息{e}")

if __name__ == "__main__":
    main()