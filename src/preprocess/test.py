from pathlib import Path
import os
import polars as pl
from tqdm import tqdm


RAW_BAS_DIR = Path("/root/autodl-tmp/datasets/aipc/bas_data")
PROC_BAS_DIR = Path("/root/autodl-tmp/datasets/aipc/processed/bas_merged")

TMP_SUFFIX = ".tmp_repair_index.parquet"


def processed_to_raw_name(processed_name: str) -> str:
    """
    把 processed 文件名映射回官方原始 bas_data 文件名。

    例：
      bas_merged_0029_wiff_bas_b_6.parquet
    -> wiff_bas_b_6.parquet
    """
    stem = Path(processed_name).stem

    prefix = "bas_merged_"

    if not stem.startswith(prefix):
        raise ValueError(f"无法识别 processed 文件名: {processed_name}")

    rest = stem[len(prefix):]

    # rest = 0029_wiff_bas_b_6
    parts = rest.split("_", 1)

    if len(parts) != 2:
        raise ValueError(f"无法解析 processed 文件名: {processed_name}")

    raw_stem = parts[1]

    return raw_stem + ".parquet"


def with_row_id(df: pl.DataFrame) -> pl.DataFrame:
    if hasattr(df, "with_row_index"):
        return df.with_row_index("__row_id")
    return df.with_row_count("__row_id")


def repair_one_file(proc_path: Path, force: bool = False):
    raw_name = processed_to_raw_name(proc_path.name)
    raw_path = RAW_BAS_DIR / raw_name

    if not raw_path.exists():
        raise FileNotFoundError(f"找不到对应原始测试文件: {raw_path}")

    proc_cols = pl.scan_parquet(proc_path).collect_schema().names()

    if "index" in proc_cols and not force:
        print(f"已有 index，跳过: {proc_path.name}")
        return

    print(f"\n修复: {proc_path.name}")
    print(f"对应原始文件: {raw_path.name}")

    # 只读取原始 index，不读取大数组
    raw_index = pl.read_parquet(raw_path, columns=["index"])
    raw_index = with_row_id(raw_index)

    # 读取已处理特征文件
    proc_df = pl.read_parquet(proc_path)

    if "index" in proc_df.columns:
        proc_df = proc_df.drop("index")

    proc_df = with_row_id(proc_df)

    if raw_index.height != proc_df.height:
        raise RuntimeError(
            f"行数不一致: processed={proc_df.height}, raw={raw_index.height}, "
            f"proc={proc_path}, raw={raw_path}"
        )

    # 按行号把官方 index 加回来
    proc_df = proc_df.join(
        raw_index,
        on="__row_id",
        how="left"
    )

    if proc_df["index"].null_count() > 0:
        raise RuntimeError(f"index join 后出现空值: {proc_path}")

    proc_df = proc_df.drop("__row_id")

    # 把 index 放到第一列
    cols = proc_df.columns
    cols = ["index"] + [c for c in cols if c != "index"]
    proc_df = proc_df.select(cols)

    tmp_path = proc_path.with_name(proc_path.name + TMP_SUFFIX)

    if tmp_path.exists():
        tmp_path.unlink()

    proc_df.write_parquet(tmp_path)

    # 简单校验 parquet 头尾
    with open(tmp_path, "rb") as f:
        head = f.read(4)
        f.seek(-4, os.SEEK_END)
        tail = f.read(4)

    if head != b"PAR1" or tail != b"PAR1":
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"临时 parquet 写入不完整: {tmp_path}")

    os.replace(tmp_path, proc_path)

    print(f"完成: {proc_path.name}, rows={proc_df.height}, unique_index={proc_df['index'].n_unique()}")


def main():
    files = sorted(PROC_BAS_DIR.glob("*.parquet"))

    print(f"processed bas files: {len(files)}")
    print(f"RAW_BAS_DIR: {RAW_BAS_DIR}")
    print(f"PROC_BAS_DIR: {PROC_BAS_DIR}")

    if len(files) == 0:
        raise RuntimeError(f"没有找到 processed bas parquet: {PROC_BAS_DIR}")

    for f in tqdm(files):
        repair_one_file(f, force=True)

    print("\n修复完成，开始全局检查")

    total_rows = 0
    total_unique_sum = 0

    all_index_parts = []

    for f in tqdm(files):
        df = pl.read_parquet(f, columns=["index"])
        total_rows += df.height
        total_unique_sum += df["index"].n_unique()
        all_index_parts.append(df)

    all_index = pl.concat(all_index_parts, how="vertical")

    print("total_rows:", total_rows)
    print("sum per-file unique:", total_unique_sum)
    print("global unique index:", all_index["index"].n_unique())

    assert total_rows == 10_768_114
    assert all_index["index"].n_unique() == 10_768_114

    print("OK: processed bas_merged 中的 index 已恢复正确")


if __name__ == "__main__":
    main()