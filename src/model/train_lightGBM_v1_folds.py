# 训练 K 折 LightGBM v1，用于后续生成 OOF v1 分数

# python src/model/train_lightGBM_v1_folds.py \
#   --data-root /root/autodl-tmp/datasets/aipc \
#   --out-dir ~/aipc/models/lgbm_v1_oof \
#   --n-folds 5 \
#   --balance-by-rows \
#   --reuse-folds \
#   --train-max-rows 3000000 \
#   --valid-max-rows 1000000 \
#   --max-rows-per-file 80000 \
#   --num-boost-round 3000 \
#   --early-stopping-rounds 150 \
#   --num-threads 16

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import gc
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

from train_lightGBM_v1 import (
    CATEGORICAL_FEATURES,
    ID_COLUMNS_FOR_VALID,
    compute_basic_metrics,
    compute_sampled_fdr_metric,
    list_split_files,
    polars_to_pandas_xy,
    resolve_feature_columns,
    sample_one_file,
    check_file_ready,
)


INSTRUMENTS = ["mzml", "tims", "wiff"]


# ============================================================
# 1. fold 划分工具
# ============================================================

def stable_hash(text: str, seed: int) -> int:
    raw = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.md5(raw).hexdigest(), 16)


def infer_instrument(path: Path) -> str:
    lower_parts = [p.lower() for p in path.parts]

    for inst in INSTRUMENTS:
        if inst in lower_parts:
            return inst

    lower = str(path).lower()

    for inst in INSTRUMENTS:
        if inst in lower:
            return inst

    return "unknown"


def file_key(path: Path, train_root: Path) -> str:
    try:
        return path.relative_to(train_root).as_posix()
    except ValueError:
        inst = infer_instrument(path)
        return f"{inst}/{path.name}"


def get_row_count(path: Path) -> int:
    try:
        return int(
            pl.scan_parquet(path)
            .select(pl.len())
            .collect()
            .item()
        )
    except Exception:
        return 1


def build_fold_manifest(
    files: List[Path],
    train_root: Path,
    n_folds: int,
    seed: int,
    balance_by_rows: bool,
) -> Dict:
    rows = []

    for path in files:
        inst = infer_instrument(path)
        row_count = get_row_count(path) if balance_by_rows else 1

        rows.append({
            "key": file_key(path, train_root),
            "path": str(path),
            "file_name": path.name,
            "instrument": inst,
            "rows": int(row_count),
            "fold": None,
        })

    by_inst: Dict[str, List[Dict]] = {inst: [] for inst in INSTRUMENTS}
    by_inst["unknown"] = []

    for item in rows:
        by_inst.setdefault(item["instrument"], []).append(item)

    for inst, items in by_inst.items():
        fold_load = [0 for _ in range(n_folds)]

        items = sorted(
            items,
            key=lambda x: (-x["rows"], stable_hash(x["key"], seed)),
        )

        for item in items:
            fold_id = min(range(n_folds), key=lambda i: (fold_load[i], i))
            item["fold"] = int(fold_id)
            fold_load[fold_id] += int(item["rows"])

        print(f"{inst}: files={len(items)}, fold_load={fold_load}")

    rows = sorted(rows, key=lambda x: (x["fold"], x["instrument"], x["key"]))

    return {
        "schema_version": 1,
        "description": (
            "OOF v1 folds for generating out-of-fold "
            "lgbm_v1_score/group features on processed_split/train."
        ),
        "n_folds": int(n_folds),
        "seed": int(seed),
        "balance_by_rows": bool(balance_by_rows),
        "train_root": str(train_root),
        "files": rows,
    }


def load_or_create_manifest(
    args,
    train_root: Path,
    train_files: List[Path],
    out_dir: Path,
) -> Dict:
    manifest_path = Path(args.folds_json).expanduser() if args.folds_json else out_dir / "folds.json"

    if manifest_path.exists() and args.reuse_folds:
        print(f"读取已有 folds: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("生成新的 OOF folds")

    manifest = build_fold_manifest(
        files=train_files,
        train_root=train_root,
        n_folds=args.n_folds,
        seed=args.seed,
        balance_by_rows=args.balance_by_rows,
    )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"folds 已保存: {manifest_path}")

    return manifest


# ============================================================
# 2. v1 专用三仪器均衡采样
#    注意：这里不要求 group_feature_done
# ============================================================

def split_files_by_instrument(files: List[Path]) -> Dict[str, List[Path]]:
    by_inst = {
        "mzml": [],
        "tims": [],
        "wiff": [],
    }

    for p in files:
        inst = infer_instrument(p)

        if inst in by_inst:
            by_inst[inst].append(p)
        else:
            print(f"警告：无法判断仪器类型，跳过: {p}")

    return by_inst


def normalize_part_schema_v1(
    df: pl.DataFrame,
    feature_cols: List[str],
    keep_id_cols: bool,
) -> pl.DataFrame:
    """
    统一每个采样 part 的 schema。
    这个函数是 v1 专用版，不依赖 group 特征。
    """

    cast_exprs = []

    if "label" in df.columns:
        cast_exprs.append(pl.col("label").cast(pl.Int8))

    if "instrument" in df.columns:
        cast_exprs.append(pl.col("instrument").cast(pl.Utf8))

    id_cast_map = {
        "file_id": pl.Utf8,
        "scan_number": pl.Int64,
        "group_key": pl.Utf8,
        "peptide_key": pl.Utf8,
        "precursor_sequence": pl.Utf8,
    }

    for c, dtype in id_cast_map.items():
        if c in df.columns:
            cast_exprs.append(pl.col(c).cast(dtype))

    for c in feature_cols:
        if c not in df.columns:
            continue

        if c in CATEGORICAL_FEATURES:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    final_cols = []

    for c in feature_cols:
        if c in df.columns:
            final_cols.append(c)

    if "label" in df.columns:
        final_cols.append("label")

    if "instrument" in df.columns:
        final_cols.append("instrument")

    if keep_id_cols:
        for c in ID_COLUMNS_FOR_VALID:
            if c in df.columns:
                final_cols.append(c)

    final_cols = list(dict.fromkeys(final_cols))

    return df.select(final_cols)


def collect_sample_dataset_balanced_v1(
    files: List[Path],
    feature_cols: List[str],
    max_total_rows: int,
    max_rows_per_file: int,
    neg_pos_ratio: float,
    seed: int,
    keep_id_cols: bool = False,
) -> pl.DataFrame:
    """
    v1 专用三仪器均衡采样。
    不从 files 前面开始顺序采到 max_total_rows，而是 mzML / TIMS / WIFF 分别采样。
    """

    by_inst = split_files_by_instrument(files)
    instruments = ["mzml", "tims", "wiff"]

    target_per_inst = max_total_rows // len(instruments)

    parts = []

    for inst_idx, inst in enumerate(instruments):
        inst_files = list(by_inst.get(inst, []))

        print()
        print(f"========== 采样 {inst} ==========")
        print(f"{inst} files: {len(inst_files)}")
        print(f"{inst} target rows: {target_per_inst}")

        if len(inst_files) == 0:
            print(f"{inst} 没有文件，跳过")
            continue

        rng = np.random.default_rng(seed + inst_idx * 1000)
        rng.shuffle(inst_files)

        inst_parts = []
        inst_rows = 0

        for i, path in enumerate(inst_files):
            if inst_rows >= target_per_inst:
                break

            # v1 的 check_file_ready 只应要求：
            # label, instrument, aux_feature_done, fragment_feature_done
            # 不应要求 group_feature_done
            if not check_file_ready(path):
                continue

            remain = target_per_inst - inst_rows
            per_file = min(max_rows_per_file, remain)

            try:
                part = sample_one_file(
                    path=path,
                    feature_cols=feature_cols,
                    max_rows_per_file=per_file,
                    neg_pos_ratio=neg_pos_ratio,
                    seed=seed + inst_idx * 100000 + i,
                    keep_id_cols=keep_id_cols,
                )

                if part.height == 0:
                    continue

                if "instrument" not in part.columns:
                    part = part.with_columns(pl.lit(inst).alias("instrument"))

                part = normalize_part_schema_v1(
                    part,
                    feature_cols=feature_cols,
                    keep_id_cols=keep_id_cols,
                )

                inst_parts.append(part)
                inst_rows += part.height

                del part
                gc.collect()

            except Exception as e:
                print(f"采样失败: {path}")
                print(f"错误: {e}")

        if len(inst_parts) == 0:
            print(f"{inst} 没有采样到数据")
            continue

        inst_df = pl.concat(inst_parts, how="vertical", rechunk=False)

        print(f"{inst} sampled rows: {inst_df.height}")
        print(inst_df.group_by("label").len().sort("label"))

        parts.append(inst_df)

        del inst_parts
        gc.collect()

    if len(parts) == 0:
        raise RuntimeError("没有采样到任何数据")

    df = pl.concat(parts, how="vertical", rechunk=False)

    if df.height > max_total_rows:
        df = df.sample(n=max_total_rows, seed=seed + 999, shuffle=True)

    print()
    print("========== 总采样结果 ==========")
    print("sampled rows:", df.height)

    if "instrument" in df.columns:
        print("instrument counts:")
        print(df.group_by("instrument").len().sort("instrument"))

    print("label counts:")
    print(df.group_by("label").len().sort("label"))

    return df


def value_counts_dict(df: pl.DataFrame, col: str) -> Dict[str, int]:
    if col not in df.columns:
        return {}

    rows = (
        df.group_by(col)
        .len()
        .sort(col)
        .to_dicts()
    )

    return {
        str(row[col]): int(row["len"])
        for row in rows
    }


# ============================================================
# 3. LightGBM 参数
# ============================================================

def params_from_args(args, fold_seed: int) -> Dict:
    return {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",

        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,

        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,

        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,

        "max_bin": args.max_bin,
        "verbosity": -1,
        "seed": fold_seed,
        "num_threads": args.num_threads,
    }


# ============================================================
# 4. 单 fold 训练
# ============================================================

def train_one_fold(
    fold_id: int,
    manifest: Dict,
    feature_cols: List[str],
    args,
    out_dir: Path,
) -> None:
    fold_dir = out_dir / f"fold_{fold_id}"
    model_path = fold_dir / "model.txt"

    if model_path.exists() and args.skip_existing:
        print(f"fold_{fold_id} 已存在，跳过: {model_path}")
        return

    fold_dir.mkdir(parents=True, exist_ok=True)

    valid_items = [
        x for x in manifest["files"]
        if int(x["fold"]) == fold_id
    ]

    train_items = [
        x for x in manifest["files"]
        if int(x["fold"]) != fold_id
    ]

    train_files = [Path(x["path"]) for x in train_items]
    valid_files = [Path(x["path"]) for x in valid_items]

    if not train_files or not valid_files:
        raise RuntimeError(f"fold_{fold_id} train/valid 文件为空")

    fold_seed = args.seed + fold_id * 1000

    with open(fold_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    with open(fold_dir / "train_files.json", "w", encoding="utf-8") as f:
        json.dump([str(p) for p in train_files], f, indent=2, ensure_ascii=False)

    with open(fold_dir / "valid_files.json", "w", encoding="utf-8") as f:
        json.dump([str(p) for p in valid_files], f, indent=2, ensure_ascii=False)

    print()
    print("=" * 80)
    print(f"开始训练 fold_{fold_id}")
    print(f"train files: {len(train_files)}")
    print(f"valid files: {len(valid_files)}")
    print(f"fold_dir: {fold_dir}")

    # --------------------------------------------------------
    # 采样训练集
    # --------------------------------------------------------
    print()
    print("========== 采样 fold train ==========")

    train_df = collect_sample_dataset_balanced_v1(
        files=train_files,
        feature_cols=feature_cols,
        max_total_rows=args.train_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=fold_seed,
        keep_id_cols=False,
    )

    # --------------------------------------------------------
    # 采样验证集
    # --------------------------------------------------------
    print()
    print("========== 采样 fold valid ==========")

    valid_df = collect_sample_dataset_balanced_v1(
        files=valid_files,
        feature_cols=feature_cols,
        max_total_rows=args.valid_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=fold_seed + 10000,
        keep_id_cols=True,
    )

    print()
    print("========== fold 采样分布检查 ==========")

    if "instrument" in train_df.columns:
        print("train instrument counts:")
        print(train_df.group_by("instrument").len().sort("instrument"))
    else:
        print("警告：train_df 中没有 instrument 列")

    if "instrument" in valid_df.columns:
        print("valid instrument counts:")
        print(valid_df.group_by("instrument").len().sort("instrument"))
    else:
        print("警告：valid_df 中没有 instrument 列")

    print("train label counts:")
    print(train_df.group_by("label").len().sort("label"))

    print("valid label counts:")
    print(valid_df.group_by("label").len().sort("label"))

    # --------------------------------------------------------
    # 转 pandas
    # --------------------------------------------------------
    print()
    print("========== 转换为 LightGBM 输入 ==========")

    X_train, y_train = polars_to_pandas_xy(train_df, feature_cols)
    X_valid, y_valid = polars_to_pandas_xy(valid_df, feature_cols)

    categorical_features = [
        c for c in CATEGORICAL_FEATURES
        if c in feature_cols
    ]

    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("categorical_features:", categorical_features)

    # --------------------------------------------------------
    # LightGBM Dataset
    # --------------------------------------------------------
    lgb_train = lgb.Dataset(
        X_train,
        label=y_train,
        feature_name=feature_cols,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    lgb_valid = lgb.Dataset(
        X_valid,
        label=y_valid,
        feature_name=feature_cols,
        categorical_feature=categorical_features,
        reference=lgb_train,
        free_raw_data=False,
    )

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------
    params = params_from_args(args, fold_seed)

    with open(fold_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print()
    print("========== 开始训练 LightGBM v1 fold ==========")

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=args.num_boost_round,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(period=50),
        ],
    )

    model.save_model(str(model_path))

    print()
    print(f"fold_{fold_id} 模型已保存: {model_path}")
    print("best_iteration:", model.best_iteration)

    # --------------------------------------------------------
    # 验证
    # --------------------------------------------------------
    print()
    print("========== 验证集评估 ==========")

    valid_pred = model.predict(
        X_valid,
        num_iteration=model.best_iteration,
    )

    basic_metrics = compute_basic_metrics(y_valid, valid_pred)
    sampled_fdr_metrics = compute_sampled_fdr_metric(valid_df, valid_pred)

    metrics = {
        "fold": int(fold_id),
        "basic_metrics": basic_metrics,
        "sampled_fdr_metrics": sampled_fdr_metrics,
        "best_iteration": int(model.best_iteration),
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_valid": float(np.mean(y_valid)),

        "train_instrument_counts": value_counts_dict(train_df, "instrument"),
        "valid_instrument_counts": value_counts_dict(valid_df, "instrument"),
        "train_label_counts": value_counts_dict(train_df, "label"),
        "valid_label_counts": value_counts_dict(valid_df, "label"),
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    with open(fold_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 特征重要性
    # --------------------------------------------------------
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    importance.to_csv(fold_dir / "feature_importance.csv", index=False)

    # --------------------------------------------------------
    # 保存采样验证集预测
    # --------------------------------------------------------
    pred_df = valid_df.select([
        c for c in ID_COLUMNS_FOR_VALID + ["label"]
        if c in valid_df.columns
    ]).with_columns([
        pl.Series("score", valid_pred).cast(pl.Float32)
    ])

    pred_df.write_parquet(fold_dir / "valid_sample_pred.parquet")

    print()
    print(f"metrics 已保存: {fold_dir / 'metrics.json'}")
    print(f"特征重要性已保存: {fold_dir / 'feature_importance.csv'}")
    print(f"验证集采样预测已保存: {fold_dir / 'valid_sample_pred.parquet'}")
    print(f"fold_{fold_id} 训练完成")

    # --------------------------------------------------------
    # 清理内存
    # --------------------------------------------------------
    del train_df
    del valid_df
    del X_train
    del X_valid
    del y_train
    del y_valid
    del lgb_train
    del lgb_valid
    del model
    gc.collect()


# ============================================================
# 5. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default="/root/autodl-tmp/datasets/aipc",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="~/aipc/models/lgbm_v1_oof",
    )

    parser.add_argument(
        "--folds-json",
        type=str,
        default="",
        help="可选：已有 folds.json 路径。如果为空，默认使用 out_dir/folds.json。",
    )

    parser.add_argument(
        "--reuse-folds",
        action="store_true",
        help="如果 folds.json 已存在，是否复用。",
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--balance-by-rows",
        action="store_true",
        help="fold 划分时是否按文件行数均衡。强烈建议开启。",
    )

    parser.add_argument(
        "--only-fold",
        type=int,
        default=None,
        help="只训练某一个 fold，调试时推荐先用 --only-fold 0。",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果 fold/model.txt 已存在，是否跳过该 fold。",
    )

    parser.add_argument(
        "--train-max-rows",
        type=int,
        default=3_000_000,
    )

    parser.add_argument(
        "--valid-max-rows",
        type=int,
        default=1_000_000,
    )

    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=80_000,
    )

    parser.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260519,
    )

    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--num-threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--num-leaves",
        type=int,
        default=127,
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--min-data-in-leaf",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--feature-fraction",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--bagging-fraction",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--lambda-l1",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--lambda-l2",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--max-bin",
        type=int,
        default=255,
    )

    args = parser.parse_args()

    if args.n_folds < 2:
        raise ValueError("--n-folds 必须 >= 2")

    data_root = Path(args.data_root)
    split_root = data_root / "processed_split"
    train_root = split_root / "train"

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== 路径 ==========")
    print("data_root:", data_root)
    print("train_root:", train_root)
    print("out_dir:", out_dir)

    train_files = list_split_files(train_root)

    print()
    print(f"train files: {len(train_files)}")

    if len(train_files) == 0:
        raise RuntimeError(f"未找到 train parquet: {train_root}")

    feature_cols = resolve_feature_columns(train_files)

    print()
    print("========== v1 使用特征 ==========")
    print(f"feature count: {len(feature_cols)}")
    for c in feature_cols:
        print(c)

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    manifest = load_or_create_manifest(
        args=args,
        train_root=train_root,
        train_files=train_files,
        out_dir=out_dir,
    )

    fold_ids = list(range(int(manifest["n_folds"])))

    if args.only_fold is not None:
        if args.only_fold not in fold_ids:
            raise ValueError(f"--only-fold 必须在 {fold_ids} 内")

        fold_ids = [args.only_fold]

    for fold_id in fold_ids:
        train_one_fold(
            fold_id=fold_id,
            manifest=manifest,
            feature_cols=feature_cols,
            args=args,
            out_dir=out_dir,
        )

    print()
    print("全部 OOF v1 fold 模型训练完成")
    print(f"folds.json: {out_dir / 'folds.json'}")
    print(f"feature_columns.json: {out_dir / 'feature_columns.json'}")


if __name__ == "__main__":
    main()