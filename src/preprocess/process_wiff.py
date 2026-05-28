from pathlib import Path
import os

# 必须放在 import polars 之前
# 20 个进程，每个进程 Polars 只用 1 个线程，避免 CPU 过度竞争
os.environ["POLARS_MAX_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl
from tqdm import tqdm


INSTRUMENT = "wiff"

DATASET = Path(f"/root/autodl-tmp/datasets/aipc/original/{INSTRUMENT}")
OUT_PATH = Path(f"/root/autodl-tmp/datasets/aipc/processed/{INSTRUMENT}_merged")

OUT_PATH.mkdir(exist_ok=True, parents=True)

# 并行 CPU 数
N_WORKERS = 20

# 临时文件后缀，先写 tmp，再原子替换，避免半写入文件
TMP_SUFFIX = ".tmp.parquet"


def find_parquets(dir_path: Path):
    """
    返回所有 parquet 文件绝对路径。
    排序后处理，保证输出编号稳定。
    """
    parquets = sorted(dir_path.rglob("*.parquet"))
    print(f"找到 {len(parquets)} 个 parquet 文件")
    return parquets


def is_list_dtype(dtype) -> bool:
    """
    判断 Polars dtype 是否为 List 类型。
    """
    return isinstance(dtype, pl.List)


def process_one_parquet(parquet: Path) -> pl.DataFrame:
    """
    处理单个 parquet 文件，返回处理后的 DataFrame。
    """
    df = pl.read_parquet(parquet)

    rename_map = {}

    if "scan" in df.columns:
        rename_map["scan"] = "scan_number"

    if "peptide" in df.columns:
        rename_map["peptide"] = "precursor_sequence"

    if rename_map:
        df = df.rename(rename_map)

    # wiff 通常没有 ion_mobility。
    # 如果原始文件中没有该列，则补 0.0。
    if "ion_mobility" not in df.columns:
        df = df.with_columns(
            pl.lit(0.0).cast(pl.Float32).alias("ion_mobility")
        )

    # 需要展开的列。
    # 只展开真实存在且是 List 类型的列，避免 scalar 列被错误 explode。
    candidate_explode_cols = [
        "charge",
        "precursor_sequence",
        "label",
        "predicted_rt",
        "delta_rt",
        "spectrum_q",
        "sage_discriminant_score",
        "ion_mobility",
    ]

    schema = df.schema

    explode_cols = [
        col for col in candidate_explode_cols
        if col in df.columns and is_list_dtype(schema[col])
    ]

    if len(explode_cols) > 0:
        df = df.explode(explode_cols)

    # 添加其他信息
    df = df.with_columns([
        pl.lit(INSTRUMENT).alias("instrument"),

        # tims 中有 ion_mobility，wiff 中没有
        pl.lit(0 if INSTRUMENT == "wiff" else 1)
        .cast(pl.Int8)
        .alias("has_ion_mobility"),

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

    # 确保常用数值字段类型更紧凑
    cast_exprs = []

    cast_map = {
        "scan_number": pl.Int64,
        "rt": pl.Float32,
        "predicted_rt": pl.Float32,
        "delta_rt": pl.Float32,
        "precursor_mz": pl.Float32,
        "charge": pl.Int8,
        "ion_mobility": pl.Float32,
        "has_ion_mobility": pl.Int8,
        "label": pl.Int8,
        "sage_discriminant_score": pl.Float32,
        "spectrum_q": pl.Float32,
        "fp_q_value": pl.Float32,
        "in_fp": pl.Int8,
    }

    for col_name, dtype in cast_map.items():
        if col_name in df.columns:
            cast_exprs.append(pl.col(col_name).cast(dtype, strict=False))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

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

        # 肽段、RT 字段
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
        "in_fp",
    ])

    return df


def cleanup_tmp_file(tmp_path: Path):
    """
    清理临时文件。
    """
    try:
        if tmp_path.exists():
            tmp_path.unlink()
    except Exception as e:
        print(f"临时文件删除失败：{tmp_path}，错误：{e}")


def is_valid_parquet_file(path: Path) -> bool:
    """
    简单检查 parquet 文件头尾，避免半写入文件被当作正常输出。
    """
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


def process_one_task(task):
    """
    子进程执行函数。

    重要：
    子进程里直接写 parquet，不把 DataFrame 返回主进程。
    这样可以避免大 DataFrame 在进程间传输，节省内存和时间。
    """
    i, parquet, out_path = task

    if out_path.exists():
        return {
            "status": "exists",
            "input": str(parquet),
            "output": str(out_path),
            "rows": 0,
            "error": "",
        }

    tmp_path = out_path.with_name(out_path.name + TMP_SUFFIX)

    try:
        cleanup_tmp_file(tmp_path)

        df = process_one_parquet(parquet)

        if df is None:
            return {
                "status": "skipped",
                "input": str(parquet),
                "output": str(out_path),
                "rows": 0,
                "error": "process_one_parquet 返回 None",
            }

        row_count = df.height

        df.write_parquet(tmp_path)

        if not is_valid_parquet_file(tmp_path):
            raise RuntimeError(f"临时 parquet 写入不完整：{tmp_path}")

        os.replace(tmp_path, out_path)

        return {
            "status": "ok",
            "input": str(parquet),
            "output": str(out_path),
            "rows": row_count,
            "error": "",
        }

    except Exception as e:
        cleanup_tmp_file(tmp_path)

        return {
            "status": "failed",
            "input": str(parquet),
            "output": str(out_path),
            "rows": 0,
            "error": repr(e) + "\n" + traceback.format_exc(limit=8),
        }


def main():
    parquets = find_parquets(DATASET)

    if len(parquets) == 0:
        print("没有找到需要处理的 parquet 文件")
        return

    tasks = []
    exists_count = 0

    for i, parquet in enumerate(parquets):
        out_path = OUT_PATH / f"{INSTRUMENT}_merged_{i:04d}.parquet"

        if out_path.exists():
            exists_count += 1
            continue

        tasks.append((i, parquet, out_path))

    print(f"已存在并跳过：{exists_count}")
    print(f"待处理任务数：{len(tasks)}")

    if len(tasks) == 0:
        print("没有需要处理的新任务")
        return

    max_workers = min(N_WORKERS, len(tasks))
    print(f"并行进程数：{max_workers}")

    ok_count = 0
    skipped_count = 0
    failed_count = 0
    failed_logs = []

    # 使用 spawn 对 Polars / Arrow 更安全
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
    ) as executor:
        futures = [
            executor.submit(process_one_task, task)
            for task in tasks
        ]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Merging {INSTRUMENT}",
        ):
            try:
                result = future.result()
            except Exception as e:
                failed_count += 1
                msg = f"子进程异常退出：{repr(e)}"
                failed_logs.append(msg)
                tqdm.write(msg)
                continue

            status = result["status"]

            if status == "ok":
                ok_count += 1
                tqdm.write(
                    f"已保存 {result['output']}，行数 {result['rows']}"
                )

            elif status == "exists":
                pass

            elif status == "skipped":
                skipped_count += 1
                tqdm.write(
                    f"跳过 {result['input']}，原因：{result['error']}"
                )

            else:
                failed_count += 1
                msg = (
                    f"处理失败：{result['input']}\n"
                    f"输出路径：{result['output']}\n"
                    f"错误信息：\n{result['error']}"
                )
                failed_logs.append(msg)
                tqdm.write(msg)

    print("=" * 80)
    print("处理完成")
    print(f"成功：{ok_count}")
    print(f"跳过：{skipped_count}")
    print(f"失败：{failed_count}")
    print(f"原本已存在：{exists_count}")

    if failed_logs:
        log_path = OUT_PATH / f"{INSTRUMENT}_merge_failed.log"
        log_path.write_text(
            "\n\n" + "=" * 80 + "\n\n".join(failed_logs),
            encoding="utf-8",
        )
        print(f"失败日志已保存到：{log_path}")


if __name__ == "__main__":
    main()