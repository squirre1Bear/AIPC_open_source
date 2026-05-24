from pathlib import Path
import polars as pl
from tqdm import tqdm

INSTRUMENT = "wiff"
DATASET = Path(f"E:\\AIPC_dataset\\{INSTRUMENT}")
OUT_PATH = Path(f"E:\\AIPC_dataset\\processed\\{INSTRUMENT}_merged")
OUT_PATH.mkdir(exist_ok=True, parents=True)

# 返回所有parquet文件绝对路径
def find_parquets(dir: Path):
    parquets = []
    for parquet in dir.rglob("*.parquet"):
        parquets.append(parquet)
    print(f"找到{len(parquets)}个parquet文件")
    return parquets

def process_one_parquet(parquet: Path, i):
    print(f"正在处理 {parquet.name}")
    df = pl.read_parquet(parquet)
    df = df.rename({
        "scan": "scan_number",
        "peptide": "precursor_sequence"
    })

    # 把 list 展开
    df = df.explode(["charge", "precursor_sequence", "label", "predicted_rt", "delta_rt", "spectrum_q", "sage_discriminant_score", "ion_mobility"])


    # 添加其他信息
    df = df.with_columns([
        pl.lit(INSTRUMENT).alias("instrument"),
        # tims中有ion_mobility，wiff中没有
        pl.lit(0 if INSTRUMENT=='wiff' else 1).alias("has_ion_mobility"),
        pl.lit(parquet.name).alias("file_id"),
        (
            pl.lit(f"{INSTRUMENT}_")
            + pl.lit(parquet.name)
            + pl.lit("_")
            + pl.col("scan_number").cast(pl.Utf8)
        ).alias("group_key"),
        pl.col("precursor_sequence").alias("peptide_key"),
        pl.lit(None).cast(pl.Float64).alias("fp_q_value"),
        pl.lit(-1).cast(pl.Int8).alias("in_fp"),
    ])

    df = df.select([
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

        # 肽段、rt字段
        "rt",
        "predicted_rt",
        "delta_rt",
        "precursor_mz",
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
    return df


if __name__ == "__main__":
    parquets = find_parquets(DATASET)
    for i, parquet in enumerate(tqdm(parquets)):
        out_path = OUT_PATH / f"{INSTRUMENT}_merged_{i:04d}.parquet"
        if out_path.exists():
            continue
        try:
            df = process_one_parquet(parquet, i)
            if df is None:
                continue
            df.write_parquet(out_path)
            print(f"已保存{out_path}, 行数{len(df)}")
        except Exception as e:
            print(f"处理失败{parquet}, 错误信息{e}")