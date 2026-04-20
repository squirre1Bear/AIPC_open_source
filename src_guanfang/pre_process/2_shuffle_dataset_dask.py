import os
import shutil
import logging
from datetime import datetime
import dask.dataframe as dd
import pyarrow as pa
import pandas as pd

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_and_split_parquet_with_dask(input_directory, output_directory, rows_per_file=256):
    """
    使用 Dask 读取、打乱和切分 Parquet 文件，严格保证每个文件的行数，
    并舍弃任何行数不足rows_per_file的批次。
    
    Args:
        input_directory (str): 包含Parquet文件的输入目录。
        output_directory (str): 输出小Parquet文件的目录。
        rows_per_file (int): 每个输出Parquet文件包含的行数。
    """
    logging.info(f"--- 开始使用 Dask 处理 Parquet 文件 ---")
    logging.info(f"输入目录: {input_directory}")
    logging.info(f"输出目录: {output_directory}")
    logging.info(f"每个文件目标行数: {rows_per_file}")

    if not os.path.exists(input_directory):
        logging.error(f"错误: 输入目录 '{input_directory}' 不存在。")
        return

    # 1. 创建并清空输出目录
    if os.path.exists(output_directory):
        logging.info(f"正在清空现有输出目录: {output_directory}")
        try:
            shutil.rmtree(output_directory)
            logging.info("输出目录清空成功。")
        except Exception as e:
            logging.error(f"清空输出目录失败: {e}")
            return
    try:
        os.makedirs(output_directory)
        logging.info(f"输出目录已创建: {output_directory}")
    except Exception as e:
        logging.error(f"创建输出目录失败: {e}")
        return

    # 2. 使用 Dask 惰性读取所有 Parquet 文件
    start_read_time = datetime.now()
    logging.info("正在使用 Dask 惰性读取所有 Parquet 文件...")
    try:
        ddf = dd.read_parquet(os.path.join(input_directory, '*.parquet'), engine='pyarrow')
        logging.info(f"Dask DataFrame 创建完成，包含 {ddf.npartitions} 个分区。")
    except Exception as e:
        logging.error(f"使用 Dask 读取文件时发生错误: {e}")
        return
        
    # 3. 随机打乱整个数据集（会触发计算）
    logging.info("正在对 Dask DataFrame 进行随机打乱...")
    start_shuffle_time = datetime.now()
    try:
        shuffled_df = ddf.compute().sample(frac=1, random_state=42)
    except MemoryError:
        logging.error("内存不足，无法将整个数据集加载到内存中进行打乱。")
        logging.info("如果内存充足，请尝试增加计算资源。")
        return
        
    end_shuffle_time = datetime.now()
    logging.info(f"打乱操作完成。耗时: {end_shuffle_time - start_shuffle_time}")
    
    total_rows = len(shuffled_df)
    logging.info(f"总行数: {total_rows}")
    
    # 调整计算，只考虑能形成完整批次的行
    num_complete_batches = total_rows // rows_per_file
    logging.info(f"总行数: {total_rows}，目标每个文件 {rows_per_file} 行，预计生成 {num_complete_batches} 个文件。")
    
    # 4. 严格按照行数切分并写入
    logging.info("正在将数据严格切分并写入新的 Parquet 文件...")
    start_write_time = datetime.now()
    
    for i in range(num_complete_batches):
        start_row = i * rows_per_file
        end_row = (i + 1) * rows_per_file
        
        # 从打乱后的Pandas DataFrame中提取批次
        batch_df = shuffled_df.iloc[start_row:end_row]
        
        # 定义输出文件名
        output_file = os.path.join(output_directory, f"part.{i:05d}.parquet")
        
        try:
            # 将批次写入新的Parquet文件
            batch_df.to_parquet(output_file, engine='pyarrow')
        except Exception as e:
            logging.error(f"写入文件 {output_file} 时发生错误: {e}")
            # 可以在这里选择是否继续
            continue

    end_write_time = datetime.now()
    logging.info(f"所有小 Parquet 文件写入完成。耗时: {end_write_time - start_write_time}")
    logging.info(f"--- 处理完成 ---")

# --- 示例用法 ---
if __name__ == "__main__":
    input_data_directory = "/zhangxiaofan/DDA_BERT_deltaRT/test_data/test_mzml_parquet"
    output_data_directory = "/zhangxiaofan/DDA_BERT_deltaRT/test_data/test_mzml_parquet_split"
    
    target_rows_per_file = 256
    
    process_and_split_parquet_with_dask(input_data_directory, output_data_directory, target_rows_per_file)