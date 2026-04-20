import zipfile
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

def unzip_file(zip_file, output_dir):
    """解压 ZIP 文件到指定目录"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"解压完成: {zip_file} -> {output_dir}")
    except Exception as e:
        print(f"{zip_file} 解压失败: {e}")

def process_zip(zip_file, output_zip):
    """处理单个 zip 文件"""
    fn = os.path.basename(zip_file)[:-4]  # 去掉 .zip 后缀
    out_dir = os.path.join(output_zip, fn)
    unzip_file(zip_file, out_dir)
    return zip_file

if __name__ == "__main__":
    file_path = '/zhangxiaofan/DDA_BERT_deltaRT/test_data/test_raw_mzml_dataset'
    output_zip = '/zhangxiaofan/DDA_BERT_deltaRT/test_data/test_mzml_parquet'

    zip_file_list = glob.glob(f'{file_path}/*.zip')
    print("找到 ZIP 文件数量:", len(zip_file_list))

    # 并行解压
    max_workers = 10   # 可根据 CPU 数量调整
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_zip, zf, output_zip): zf for zf in zip_file_list}
        
        for future in as_completed(futures):
            zf = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"{zf} 出错: {exc}")
