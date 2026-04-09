import os
import shutil
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def ensure_clean_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def process_and_split_parquet_streaming_fast(
    input_directory: str,
    output_directory: str,
    rows_per_file: int = 256,
    batch_size: int = 50000,
    random_state: int = 42,
):
    """
    更快的极简版 parquet 处理脚本

    方案：
    1. 只随机打乱输入 parquet 文件顺序
    2. 按文件流式读取
    3. 每个 batch 内局部 shuffle
    4. 严格按 rows_per_file 切分输出
    5. 丢弃最后不足 rows_per_file 的尾部

    优点：
    - 速度明显快于全局 shuffle / 分桶 shuffle
    - 内存占用低
    - 适合先跑通项目全流程

    适用场景：
    - 当前目标是尽快得到可训练数据
    - 后续仍会做模型训练、验证、FDR 控制
    """

    logging.info("--- 开始快速流式切分 Parquet 文件 ---")
    logging.info(f"输入目录: {input_directory}")
    logging.info(f"输出目录: {output_directory}")
    logging.info(f"每个文件目标行数: {rows_per_file}")
    logging.info(f"扫描 batch_size: {batch_size}")

    if not os.path.exists(input_directory):
        logging.error(f"输入目录不存在: {input_directory}")
        return

    ensure_clean_dir(output_directory)

    parquet_files = [
        os.path.join(input_directory, f)
        for f in os.listdir(input_directory)
        if f.endswith(".parquet")
    ]

    if not parquet_files:
        logging.error("输入目录下没有找到 parquet 文件。")
        return

    rng = np.random.default_rng(random_state)
    rng.shuffle(parquet_files)

    logging.info(f"共找到 {len(parquet_files)} 个 parquet 文件，已随机打乱文件顺序。")

    start_time = datetime.now()

    carry_buffer = None
    output_index = 0
    total_rows_read = 0
    total_rows_written = 0

    for file_idx, parquet_file in enumerate(parquet_files, start=1):
        logging.info(f"[{file_idx}/{len(parquet_files)}] 正在处理文件: {os.path.basename(parquet_file)}")

        try:
            dataset = ds.dataset(parquet_file, format="parquet")
            scanner = dataset.scanner(batch_size=batch_size, use_threads=True)
        except Exception as e:
            logging.error(f"创建 scanner 失败 {parquet_file}: {e}")
            continue

        batch_idx = 0

        try:
            for record_batch in scanner.to_batches():
                batch_idx += 1

                batch_df = record_batch.to_pandas()
                n = len(batch_df)
                if n == 0:
                    continue

                total_rows_read += n

                # batch 内局部随机
                batch_df = batch_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

                # 与前面残留拼接
                if carry_buffer is not None and len(carry_buffer) > 0:
                    batch_df = pd.concat([carry_buffer, batch_df], ignore_index=True)
                    carry_buffer = None

                full_chunks = len(batch_df) // rows_per_file

                for i in range(full_chunks):
                    start_row = i * rows_per_file
                    end_row = (i + 1) * rows_per_file
                    out_df = batch_df.iloc[start_row:end_row]

                    output_file = os.path.join(output_directory, f"part.{output_index:05d}.parquet")
                    out_df.to_parquet(output_file, index=False, engine="pyarrow")

                    output_index += 1
                    total_rows_written += len(out_df)

                remain_start = full_chunks * rows_per_file
                carry_buffer = batch_df.iloc[remain_start:].reset_index(drop=True)

                if batch_idx % 10 == 0:
                    logging.info(
                        f"文件 {os.path.basename(parquet_file)} 已处理 {batch_idx} 个 batch，"
                        f"累计读取 {total_rows_read} 行，已写出 {output_index} 个文件，"
                        f"buffer 剩余 {len(carry_buffer)} 行"
                    )

        except Exception as e:
            logging.error(f"处理文件失败 {parquet_file}: {e}")
            continue

    # 丢弃最后不足 rows_per_file 的残留
    dropped_rows = 0
    if carry_buffer is not None and len(carry_buffer) > 0:
        dropped_rows = len(carry_buffer)
        logging.info(f"丢弃最后不足 {rows_per_file} 行的残留数据，共 {dropped_rows} 行")

    end_time = datetime.now()

    logging.info("--- 处理完成 ---")
    logging.info(f"累计读取总行数: {total_rows_read}")
    logging.info(f"累计写出总行数: {total_rows_written}")
    logging.info(f"累计输出文件数: {output_index}")
    logging.info(f"最终丢弃行数: {dropped_rows}")
    logging.info(f"总耗时: {end_time - start_time}")


if __name__ == "__main__":
    input_data_directory = "./data/mzml_parquet"
    output_data_directory = "./data/mzml_parquet_split"

    target_rows_per_file = 256

    process_and_split_parquet_streaming_fast(
        input_directory=input_data_directory,
        output_directory=output_data_directory,
        rows_per_file=target_rows_per_file,
        batch_size=50000,   # 内存紧张可调到 10000~20000
        random_state=42,
    )

# python3 ./src/2_shuffle_dataset_dask.py