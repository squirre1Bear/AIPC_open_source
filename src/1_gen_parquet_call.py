import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed


# 存储源数据在 ./data/mzml
raw_data_dir = glob.glob("./data/mzml/*")  # 每个batch文件夹下还有AIPC_data_001 010 ...

sage_list = glob.glob('./data/mzml/*/*/*_sage.parquet')
fp_list = [i[:-len('sage.parquet')]+'fp.parquet' for i in sage_list]
rawspectrum_list = [i[:-len('sage.parquet')]+'rawspectrum.parquet' for i in sage_list]

# 输出到mzml文件夹下，直接输出
parquet_list = [
    os.path.join(
        "./data/mzml_parquet",
        os.path.basename(i).replace("_sage.parquet","_mzml.parquet")
    )
    for i in sage_list
]

def run_task(i):
    cmd = (
        f"python3 ./src/1_gen_parquet.py "
        f"-raw {rawspectrum_list[i]} "
        f"-sage_sr {sage_list[i]} "
        f"-fp_sr {fp_list[i]} "
        f"-parquet_path {parquet_list[i]}"
    )
    os.system(cmd)
    return sage_list[i]

if __name__ == "__main__":
    max_workers = 10  # 你可以改成CPU核心数，比如 5
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_task, i) for i in range(len(sage_list))]
        for future in as_completed(futures):
            done_file = future.result()
            print(f"{done_file} has been done")

# python3 ./src/1_gen_parquet_call.py

# import os
# import glob
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# # 存储E:\AIPC_dataset\mzml\batch_1~50
# raw_data_dir = glob.glob("E:/AIPC_dataset/mzml/*")  # 每个batch文件夹下还有AIPC_data_001 010 ...
#
# sage_list = glob.glob('E:/AIPC_dataset/mzml/*/*/*_sage.parquet')
# fp_list = [i[:-len('sage.parquet')]+'fp.parquet' for i in sage_list]
# rawspectrum_list = [i[:-len('sage.parquet')]+'rawspectrum.parquet' for i in sage_list]
#
# # 输出到mzml文件夹下，直接输出
# parquet_list = [
#     os.path.join(
#         "E:/AIPC_dataset/mzml",
#         os.path.basename(i).replace("_sage.parquet","_mzml.parquet")
#     )
#     for i in sage_list
# ]
#
# def run_task(i):
#     cmd = (
#         f"python 1_gen_parquet.py "
#         f"-raw {rawspectrum_list[i]} "
#         f"-sage_sr {sage_list[i]} "
#         f"-fp_sr {fp_list[i]} "
#         f"-parquet_path {parquet_list[i]}"
#     )
#     os.system(cmd)
#     return sage_list[i]
#
# if __name__ == "__main__":
#     max_workers = 10  # 你可以改成CPU核心数，比如 5
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(run_task, i) for i in range(len(sage_list))]
#         for future in as_completed(futures):
#             done_file = future.result()
#             print(f"{done_file} has been done")

