# 给训练集生成 oof v1 得分，防止给见过的数据预测分数

# python src/preprocess/add_group_feature_oof.py \
#   --data-root /root/autodl-tmp/datasets/aipc \
#   --fold-model-dir ~/aipc/models/lgbm_v1_oof \
#   --splits train valid \
#   --workers=12
#   --force

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


PREPROCESS_DIR = Path(__file__).resolve().parent
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from add_group_feature import (  # noqa: E402
    TMP_SUFFIX,
    add_group_features_to_file,
    add_group_features_to_file_ensemble,
    cleanup_tmp_file,
    list_split_files,
)


MAX_PARALLEL_FILES = 12


def load_fold_manifest(fold_model_dir: Path) -> Dict:
    manifest_path = fold_model_dir / "folds.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"找不到 OOF folds.json: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if "files" not in manifest or "n_folds" not in manifest:
        raise RuntimeError(f"folds.json 格式异常: {manifest_path}")

    return manifest


def build_fold_map(manifest: Dict) -> Dict[str, int]:
    fold_map = {}
    for item in manifest["files"]:
        key = item["key"].replace("\\", "/")
        fold_map[key] = int(item["fold"])
    return fold_map


def path_key(path: Path, data_root: Path, split: str) -> str:
    split_root = data_root / "processed_split" / split
    try:
        return path.relative_to(split_root).as_posix()
    except ValueError:
        parts = [p.lower() for p in path.parts]
        for inst in ["mzml", "tims", "wiff"]:
            if inst in parts:
                return f"{inst}/{path.name}"
        return path.name


def fold_model_dirs_from_manifest(fold_model_dir: Path, manifest: Dict) -> List[Path]:
    n_folds = int(manifest["n_folds"])
    model_dirs = [fold_model_dir / f"fold_{fold_id}" for fold_id in range(n_folds)]

    missing = []
    for model_dir in model_dirs:
        if not (model_dir / "model.txt").exists():
            missing.append(str(model_dir / "model.txt"))
        if not (model_dir / "feature_columns.json").exists():
            missing.append(str(model_dir / "feature_columns.json"))

    if missing:
        raise FileNotFoundError(
            "OOF fold 模型不完整，无法进行 valid/test 5-fold ensemble。缺少："
            + "; ".join(missing[:20])
        )

    return model_dirs


def model_dirs_for_file(
    path: Path,
    split: str,
    data_root: Path,
    fold_model_dir: Path,
    full_model_dir: Path,
    ensemble_model_dirs: List[Path],
    fold_map: Dict[str, int],
) -> Tuple[List[Path], str]:
    if split != "train":
        if full_model_dir is not None:
            return [full_model_dir], "full"
        return ensemble_model_dirs, f"fold_ensemble_{len(ensemble_model_dirs)}"

    key = path_key(path, data_root, split="train")
    if key not in fold_map:
        raise KeyError(f"folds.json 中找不到 train 文件: key={key}, path={path}")

    fold_id = fold_map[key]
    return [fold_model_dir / f"fold_{fold_id}"], f"oof_fold_{fold_id}"


def process_one(
    path: Path,
    split: str,
    data_root: Path,
    fold_model_dir: Path,
    full_model_dir: Optional[Path],
    ensemble_model_dirs: List[Path],
    fold_map: Dict[str, int],
    force: bool,
) -> Tuple[str, bool, str]:
    try:
        model_dirs, source = model_dirs_for_file(
            path=path,
            split=split,
            data_root=data_root,
            fold_model_dir=fold_model_dir,
            full_model_dir=full_model_dir,
            ensemble_model_dirs=ensemble_model_dirs,
            fold_map=fold_map,
        )

        for model_dir in model_dirs:
            if not (model_dir / "model.txt").exists():
                raise FileNotFoundError(f"找不到模型文件: {model_dir / 'model.txt'}")

            if not (model_dir / "feature_columns.json").exists():
                raise FileNotFoundError(f"找不到 feature_columns.json: {model_dir / 'feature_columns.json'}")

        print(f"\n[{source}] {path}")
        if len(model_dirs) == 1:
            add_group_features_to_file(path=path, model_dir=model_dirs[0], force=force)
        else:
            add_group_features_to_file_ensemble(path=path, model_dirs=model_dirs, force=force)
        return str(path), True, source

    except BaseException as e:
        cleanup_tmp_file(path.with_name(path.name + TMP_SUFFIX))
        return str(path), False, repr(e)

    finally:
        gc.collect()


def collect_files(data_root: Path, splits, max_files):
    split_to_files = {}
    for split in splits:
        files = list_split_files(data_root, [split])
        if max_files is not None:
            files = files[:max_files]
        split_to_files[split] = files
    return split_to_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/datasets/aipc")
    parser.add_argument("--fold-model-dir", type=str, default="~/aipc/models/lgbm_v1_oof")
    parser.add_argument(
        "--full-model-dir",
        type=str,
        default="",
        help="可选：如果提供，则 valid 使用该 full v1；默认 valid 使用 fold-model-dir 下所有 fold 做 ensemble。",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=["train", "valid"],
        help="train 使用对应 OOF fold 模型；valid 默认使用所有 fold 模型平均。",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=MAX_PARALLEL_FILES)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    fold_model_dir = Path(args.fold_model_dir).expanduser()
    full_model_dir = Path(args.full_model_dir).expanduser() if args.full_model_dir else None

    manifest = load_fold_manifest(fold_model_dir)
    fold_map = build_fold_map(manifest)
    ensemble_model_dirs = fold_model_dirs_from_manifest(fold_model_dir, manifest)

    split_to_files = collect_files(data_root, args.splits, args.max_files)
    jobs = []
    for split, files in split_to_files.items():
        for path in files:
            jobs.append((split, path))

    print("========== OOF group feature ==========")
    print("data_root:", data_root)
    print("fold_model_dir:", fold_model_dir)
    print("valid/test score mode:", "full" if full_model_dir is not None else f"fold_ensemble_{len(ensemble_model_dirs)}")
    print("full_model_dir:", full_model_dir)
    print("ensemble_model_dirs:")
    for model_dir in ensemble_model_dirs:
        print("  ", model_dir)
    print("splits:", args.splits)
    print("force:", args.force)
    print("workers:", args.workers)
    print("files:", len(jobs))

    if not jobs:
        print("没有文件需要处理")
        return

    failed = []
    workers = max(1, min(args.workers, len(jobs)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(
                process_one,
                path,
                split,
                data_root,
                fold_model_dir,
                full_model_dir,
                ensemble_model_dirs,
                fold_map,
                args.force,
            ): (split, path)
            for split, path in jobs
        }

        for future in tqdm(as_completed(future_to_job), total=len(future_to_job)):
            split, path = future_to_job[future]
            try:
                path_str, ok, msg = future.result()
            except BaseException as e:
                path_str, ok, msg = str(path), False, repr(e)

            if not ok:
                failed.append((split, path_str, msg))
                print(f"处理失败: split={split}, path={path_str}")
                print(msg)

    print()
    print("OOF group 特征处理完成")
    if failed:
        print(f"失败文件数: {len(failed)}")
        for split, path_str, msg in failed[:50]:
            print(f"[{split}] {path_str}: {msg}")
        raise RuntimeError("存在 OOF group 特征处理失败文件，请先修复再训练 v2")

    print("无失败文件")


if __name__ == "__main__":
    main()
