import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

raw_data_dir = glob.glob("/zhangxiaofan/DDA_BERT_deltaRT/test_data/test_raw_mzml_dataset/*")

sage_list = glob.glob('/zhangxiaofan/DDA_BERT_deltaRT/test_data/test_raw_mzml_dataset/*/*/*_sage.parquet')
fp_list = [i[:-len('sage.parquet')]+'fp.parquet' for i in sage_list]
rawspectrum_list = [i[:-len('sage.parquet')]+'rawspectrum.parquet' for i in sage_list]
parquet_list = [f"/zhangxiaofan/DDA_BERT_deltaRT/test_data/test_mzml_parquet/{i.split('/')[-1][:-len('sage.parquet')]}mzml.parquet" for i in sage_list]

def run_task(i):
    cmd = (
        f"python /zhangxiaofan/DDA_BERT_deltaRT/test_data/1_gen_parquet.py "
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
