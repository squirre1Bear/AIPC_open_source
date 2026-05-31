# 添加仪器名、统一初始列名
from pathlib import Path
import polars as pl
from tqdm import tqdm

BAS_DIR = Path("/root/autodl-tmp/datasets/aipc/bas_data")
OUT_DIR = Path("/root/autodl-tmp/datasets/aipc/processed/bas_merged")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def infer_instrument(file_name: str) -> str:
    name = file_name.lower()
    if name.startswith("mzml"):
        return "mzml"
    if name.startswith("tims"):
        return "tims"
    if name.startswith("wiff"):
        return "wiff"
    raise ValueError(f"无法从文件名判断仪器类型: {file_name}")


def process_one_file(path: Path, out_path: Path):
    instrument = infer_instrument(path.name)

    df = pl.read_parquet(path)

    # 测试集里叫 delta_rt_model，这里统一成 delta_rt
    if "delta_rt_model" in df.columns and "delta_rt" not in df.columns:
        df = df.rename({"delta_rt_model": "delta_rt"})

    has_ion_mobility = 1 if instrument == "tims" else 0

    df = df.with_columns([
        pl.lit(path.name).alias("file_id"),
        pl.lit(instrument).alias("instrument"),
        pl.lit(has_ion_mobility).cast(pl.Int8).alias("has_ion_mobility"),

        (
            pl.lit(f"{instrument}_")
            + pl.lit(path.name)
            + pl.lit("_")
            + pl.col("scan_number").cast(pl.Utf8)
        ).alias("group_key"),

        pl.col("precursor_sequence").alias("peptide_key"),

        # 测试集没有这些训练辅助字段，补齐占位
        pl.lit(None).cast(pl.Int8).alias("label"),
        pl.lit(None).cast(pl.Float32).alias("sage_discriminant_score"),
        pl.lit(None).cast(pl.Float32).alias("spectrum_q"),
        pl.lit(None).cast(pl.Float32).alias("fp_q_value"),
        pl.lit(-1).cast(pl.Int8).alias("in_fp"),
    ])

    df = df.select([
        "index",

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
    ])

    df.write_parquet(out_path)
    print(f"保存: {out_path}, rows={df.height}")


def main():
    files = sorted(BAS_DIR.glob("*.parquet"))
    print(f"找到测试 parquet: {len(files)}")

    for i, path in enumerate(tqdm(files)):
        out_path = OUT_DIR / f"bas_merged_{i:04d}_{path.stem}.parquet"
        if out_path.exists():
            continue
        process_one_file(path, out_path)


if __name__ == "__main__":
    main()