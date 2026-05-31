# 划分训练90% + 测试集10%
# 由于空间不够，会直接移动文件，而不是额外生成一份数据
# 结构为：
# /root/autodl-tmp/datasets/aipc/processed_split/train/mzml
#                                               |     |tims
#                                               |     |wiff
#                                               /valid/mzml
#                                               |     /tims
#                                               |     /wiff

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import errno
import polars as pl
import argparse
import hashlib
import os


# 服务器运行
# AIPC_DATA_ROOT=/root/autodl-tmp/datasets/aipc python src/preprocess/split_train_valid.py
DEFAULT_DATA_ROOT = Path(os.environ.get("AIPC_DATA_ROOT", r"E:\AIPC_dataset"))

INSTRUMENTS = ["mzml", "tims", "wiff"]

SOURCE_DIR_NAMES = {
    "mzml": "mzml_merged",
    "tims": "tims_merged",
    "wiff": "wiff_merged"
}

REQUIRED_BASE_COLS = [
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
]

# 检查 featrue1 feature2 是否处理完成
REQUIRED_FEATURE_FLAGS = [
    "aux_feature_done",
    "fragment_feature_done"
]



# 返回存在的仪器的根目录、(所有 parquet 文件的路径, 仪器名)
def find_processed_files(processed_dir):
    # 3个仪器处理后的代码根目录
    source_dir = []
    all_files = []

    for instrument in INSTRUMENTS:
        dir = processed_dir / SOURCE_DIR_NAMES[instrument]
        source_dir.append(dir)

        if not dir.exists():
            print(f"{instrument} 目录不存在，跳过 {dir}")
            continue

        files = sorted(dir.glob("*.parquet"))
        print(f"找到 {len(files)} 个 parquet 文件")

        for f in files:
            all_files.append((f, instrument))

    return source_dir, all_files

# 统计单个 parquet 行数、target / decoy 数量
def get_one_file_info(path: Path, instrument: str):
    schema_cols = pl.scan_parquet(path).collect_schema().names()

    missing_cols = [c for c in REQUIRED_BASE_COLS if c not in schema_cols]
    missing_flags = [c for c in REQUIRED_FEATURE_FLAGS if c not in schema_cols]

    if missing_cols:
        return {
            "path": str(path),
            "instrument": instrument,
            "ok": False,
            "skip_reason": f"缺少基础列: {missing_cols}",
        }

    if missing_flags:
        return {
            "path": str(path),
            "instrument": instrument,
            "ok": False,
            "skip_reason": f"缺少特征完成标记: {missing_flags}",
        }

    status_df = (
        pl.scan_parquet(path)
        .select([
            # 统计当前 parquet 总行数
            pl.len().alias("total_rows"),

            # 统计 target(label=1) 数量
            pl.col("label").cast(pl.Int64).sum().alias("target_rows")
        ])
        .collect()
    )

    row = status_df.row(0, named=True)
    total_rows = row["total_rows"]
    target_rows = row["target_rows"]
    decoy_rows = total_rows - target_rows

    return {
        "path": str(path),
        "file_name": path.name,
        "instrument": instrument,
        "total_rows": total_rows,
        "target_rows": target_rows,
        "decoy_rows": decoy_rows,
        "size_bytes": int(path.stat().st_size),
        # 是否已完成第一二步特征填写
        "aux_feature_done": int("aux_feature_done" in schema_cols),
        "fragment_feature_done": int("fragment_feature_done" in schema_cols),
        "ok": True,
        # 正常文件没有跳过
        "skip_reason": ""
    }

# 生成稳定的 hash 值
def stable_hash(text, seed):
    s = f"{seed}|{text}"
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

# 按仪器分组，划分 train/valid。整个文件划分到一组
def choose_valid_files(file_infos, valid_ratio, seed):
    by_instrument = defaultdict(list)

    # 按仪器进行分组
    for info in file_infos:
        by_instrument[info["instrument"]].append(info)

    assigned = []

    for instrument, items in by_instrument.items():
        # 形如 {
        #    "mzml": [{"file_name": ... , "total_rows":...} , {}, ...]
        #    "tims": ....
        # }

        # 计算当前仪器总行数、目标 valid 行数
        total_rows = sum(i["total_rows"] for i in items)
        target_valid_rows = int(total_rows * valid_ratio)

        # 生成稳定的 hash 对文件排序
        items = sorted(
            items,
            key=lambda x: stable_hash(
                f"{x['instrument']}|{x['file_name']}|{x['total_rows']}",
                seed
            )
        )

        valid_rows = 0
        valid_items = []
        train_items = []

        # 先把文件装入 valid，满了再装入 train
        for x in items:
            if valid_rows < target_valid_rows:
                x["split"] = "valid"
                valid_items.append(x)
                valid_rows += x["total_rows"]
            else:
                x["split"] = "train"
                train_items.append(x)

        assigned.extend(train_items + valid_items)
    return assigned

def move_one_file(src: Path, dst: Path):
    # 防止覆盖已有文件
    if dst.exists():
        raise FileExistsError(f"目标文件已存在，停止覆盖: {dst}")

    dst.parent.mkdir(exist_ok=True, parents=True)

    try:
        os.replace(src, dst)

    except OSError as e:
        # EXDEV 表示源目录和目标目录不在同一个磁盘 / 分区
        # 这种情况下 os.replace 不能零拷贝移动
        if e.errno == errno.EXDEV:
            raise RuntimeError("源目录和目标目录不在同一磁盘，无法零拷贝移动。") from e
        raise



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help="AIPC 数据根目录"
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="验证集比例"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="划分随机种子"
    )

    arg = parser.parse_args()

    data_root = Path(arg.data_root)
    processed_root = data_root / "processed"
    split_root = data_root / "processed_split"

    source_dir, all_files = find_processed_files(processed_root)

    if len(all_files) == 0:
        print("没有找到待划分 parquet")
        return

    print("========== 扫描 parquet 信息 ==========")

    ok_info = []
    bad_info = []

    for path, instrument in tqdm(all_files):
        try:
            info = get_one_file_info(path, instrument)

            if info["ok"]:
                ok_info.append(info)
            else:
                bad_info.append(info)

        except Exception as e:
            bad_info.append({
                "path": str(path),
                "instrument": instrument,
                "ok": False,
                "skip_reason": str(e),
            })

    if bad_info:
        print(f"跳过 {len(bad_info)} 个异常文件")
        for x in bad_info[:10]:
            print(f"- {x['path']}: {x['skip_reason']}")

    if not ok_info:
        print("没有可划分文件")
        return

    assigned = choose_valid_files(ok_info, arg.valid_ratio, seed=arg.seed)

    print("划分结果如下：")
    for split in ["train", "valid"]:
        items = [x for x in assigned if x["split"] == split]
        print(f"{split}:"
              f"{len(items)} 个文件，"
              f"{sum(x['total_rows'] for x in items)} 行")

    # 根据划分结果移动文件
    for info in tqdm(assigned):
        source = Path(info["path"])
        destination = split_root / info["split"] / info["instrument"] / info["file_name"]
        move_one_file(source, destination)

    # 删除移动后的空目录
    for d in source_dir:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()

    print(f"\n划分完成: {split_root}")


if __name__ == "__main__":
    main()