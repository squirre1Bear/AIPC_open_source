from __future__ import annotations

import os

# 20 个进程 * 每个 Polars 1 个线程 ≈ 20 CPU
os.environ.setdefault("POLARS_MAX_THREADS", "1")

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import traceback

import polars as pl
from tqdm import tqdm


# 设置路径
MZML_DIR = Path(r"/root/autodl-tmp/datasets/aipc/original/mzml")
OUT_DIR = Path(r"/root/autodl-tmp/datasets/aipc/processed/mzml_merged")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 并行 CPU 数
N_WORKERS = 20

def find_mzml_folders(mzml_dir: Path):
    folders = []

    for folder in mzml_dir.rglob("*"):
        if not folder.is_dir():
            continue

        raw_file = list(folder.glob("*_rawspectrum.parquet"))
        sage_file = list(folder.glob("*_sage.parquet"))

        if len(raw_file) > 0 and len(sage_file) > 0:
            folders.append(folder)

    # 排序保证每次输出编号稳定
    folders = sorted(folders, key=lambda p: str(p))
    return folders


def merge_one_folder(folder: Path):
    """
    处理单个 mzML 文件夹：
    读取 raw/sage/fp 三类 parquet，合并后返回 DataFrame。
    """
    raw_files = sorted(folder.glob("*_rawspectrum.parquet"))
    sage_files = sorted(folder.glob("*_sage.parquet"))
    fp_files = sorted(folder.glob("*_fp.parquet"))

    if len(raw_files) == 0 or len(sage_files) == 0 or len(fp_files) == 0:
        return None, f"缺少 raw/sage/fp 文件: {folder}"

    raw = pl.read_parquet(
        raw_files[0],
        columns=[
            "scan",
            "precursor_mz",
            "rt",
            "mz_array",
            "intensity_array",
        ],
    ).rename({
        "scan": "scan_number",
    })

    sage = pl.read_parquet(
        sage_files[0],
        columns=[
            "scan",
            "precursor_sequence",
            "label",
            "charge",
            "predicted_rt",
            "delta_rt",
            "sage_discriminant_score",
            "spectrum_q",
        ],
    ).rename({
        "scan": "scan_number",
    })

    fp = pl.read_parquet(
        fp_files[0],
        columns=[
            "scan",
            "detect_sequence",
            "q-value",
        ],
    ).rename({
        "scan": "scan_number",
        "detect_sequence": "precursor_sequence",
        "q-value": "fp_q_value",
    })

    # 先按 scan 合并 sage 和 raw
    merged = sage.join(
        raw,
        how="inner",
        on="scan_number",
    )

    # 再根据 scan + sequence 合并 fp
    merged = merged.join(
        fp,
        how="left",
        on=["scan_number", "precursor_sequence"],
    )

    merged = merged.with_columns([
        pl.col("fp_q_value").is_not_null().cast(pl.Int8).alias("in_fp"),

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

        pl.col("precursor_sequence").alias("peptide_key"),
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

        # 肽段、RT 字段
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
        "in_fp",
    ])

    return merged, None


def process_one_task(task):
    """
    子进程执行的函数。

    注意：
    不把 merged DataFrame 返回主进程，而是在子进程里直接写 parquet。
    这样可以避免大对象在进程间传输，速度和内存都会更好。
    """
    i, folder, out_path = task

    if out_path.exists():
        return {
            "status": "exists",
            "folder": str(folder),
            "out_path": str(out_path),
            "rows": 0,
            "error": "",
        }

    try:
        merged, error = merge_one_folder(folder)

        if merged is None:
            return {
                "status": "skipped",
                "folder": str(folder),
                "out_path": str(out_path),
                "rows": 0,
                "error": error,
            }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.write_parquet(out_path)

        return {
            "status": "ok",
            "folder": str(folder),
            "out_path": str(out_path),
            "rows": merged.height,
            "error": "",
        }

    except Exception as e:
        return {
            "status": "failed",
            "folder": str(folder),
            "out_path": str(out_path),
            "rows": 0,
            "error": repr(e) + "\n" + traceback.format_exc(limit=5),
        }


def main():
    folders = find_mzml_folders(MZML_DIR)
    print(f"发现 {len(folders)} 个 mzml 目录")

    tasks = []
    existing_count = 0

    for i, folder in enumerate(folders):
        out_path = OUT_DIR / f"mzml_merged_{i:04d}.parquet"

        if out_path.exists():
            existing_count += 1
            continue

        tasks.append((i, folder, out_path))

    print(f"已存在并跳过: {existing_count}")
    print(f"待处理任务数: {len(tasks)}")
    print(f"并行进程数: {N_WORKERS}")

    if len(tasks) == 0:
        print("没有需要处理的新任务")
        return

    ok_count = 0
    skipped_count = 0
    failed_count = 0
    failed_logs = []

    max_workers = min(N_WORKERS, len(tasks))

    # 对 Polars 更安全，避免 fork 复制多线程状态
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
            desc="Merging mzML",
        ):
            try:
                result = future.result()
            except Exception as e:
                failed_count += 1
                failed_logs.append(f"子进程异常退出: {repr(e)}")
                continue

            status = result["status"]

            if status == "ok":
                ok_count += 1
                tqdm.write(
                    f"已保存 {result['out_path']}，行数 {result['rows']}"
                )

            elif status == "skipped":
                skipped_count += 1
                tqdm.write(
                    f"跳过 {result['folder']}，原因: {result['error']}"
                )

            elif status == "exists":
                # 理论上主进程已经跳过了，这里只是双重保险
                pass

            else:
                failed_count += 1
                msg = (
                    f"处理失败: {result['folder']}\n"
                    f"输出路径: {result['out_path']}\n"
                    f"错误信息:\n{result['error']}"
                )
                failed_logs.append(msg)
                tqdm.write(msg)

    print("=" * 80)
    print("处理完成")
    print(f"成功: {ok_count}")
    print(f"跳过: {skipped_count}")
    print(f"失败: {failed_count}")
    print(f"原本已存在: {existing_count}")

    if failed_logs:
        log_path = OUT_DIR / "mzml_merge_failed.log"
        log_path.write_text(
            "\n\n" + "=" * 80 + "\n\n".join(failed_logs),
            encoding="utf-8",
        )
        print(f"失败日志已保存到: {log_path}")


if __name__ == "__main__":
    main()