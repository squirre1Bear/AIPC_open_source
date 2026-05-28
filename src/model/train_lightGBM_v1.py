# python src/model/train_lightGBM_v1.py \
#   --data-root /root/autodl-tmp/datasets/aipc \
#   --out-dir /root/aipc/models/lgbm_v1 \
#   --train-max-rows 3000000 \
#   --valid-max-rows 1000000 \
#   --max-rows-per-file 80000 \
#   --neg-pos-ratio 2.0

from pathlib import Path
import argparse
import json
import os
import gc
from typing import List, Dict

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb


# ============================================================
# 1. 候选特征列
# ============================================================
# 注意：
# 这里只放“测试集也能构造出来”的字段。
# 不放 sage_discriminant_score / spectrum_q / fp_q_value / in_fp。
# 不放 mz_array / intensity_array / precursor_sequence / group_key。

FEATURE_CANDIDATES = [
    # -------------------------
    # 原始数值字段
    # -------------------------
    "precursor_mz",
    "rt",
    "predicted_rt",
    "delta_rt",
    "charge",
    "ion_mobility",
    "has_ion_mobility",

    # -------------------------
    # 肽段基础特征
    # -------------------------
    "peptide_length",
    "mod_count",
    "unknown_mod_count",
    "has_mod",
    "neutral_mass",
    "parse_ok",
    "basic_aa_count",
    "acidic_aa_count",
    "hydrophobic_aa_count",
    "missed_cleavage_like",

    # -------------------------
    # 前体质量 / 同位素特征
    # -------------------------
    "theoretical_precursor_mz",
    "precursor_ppm_error",
    "abs_precursor_ppm_error",
    "ppm_iso_-1",
    "ppm_iso_0",
    "ppm_iso_1",
    "ppm_iso_2",
    "abs_ppm_iso_-1",
    "abs_ppm_iso_0",
    "abs_ppm_iso_1",
    "abs_ppm_iso_2",
    "min_abs_precursor_ppm",
    "best_isotope_offset",

    # -------------------------
    # RT / IM 特征
    # -------------------------
    "abs_delta_rt",
    "rt_z_in_file",
    "rt_norm_in_file",
    "delta_rt_z_in_file",
    "predicted_rt_z_in_file",
    "ion_mobility_z_in_file",
    "ion_mobility_norm_in_file",

    # -------------------------
    # fragment 解析状态
    # -------------------------
    "fragment_parse_ok",
    "fragment_position_count",
    "main_ion_count",
    "b_ion_count",
    "y_ion_count",
    "loss_ion_count",

    # -------------------------
    # 10 ppm fragment 特征
    # -------------------------
    "matched_b_count_10ppm",
    "matched_y_count_10ppm",
    "matched_total_count_10ppm",
    "matched_b_fraction_10ppm",
    "matched_y_fraction_10ppm",
    "matched_total_fraction_10ppm",

    # -------------------------
    # 20 ppm fragment 特征，最重要
    # -------------------------
    "matched_b_count_20ppm",
    "matched_y_count_20ppm",
    "matched_total_count_20ppm",
    "matched_b_fraction_20ppm",
    "matched_y_fraction_20ppm",
    "matched_total_fraction_20ppm",

    "matched_peak_count_20ppm",
    "explained_intensity_fraction_20ppm",
    "top50_explained_intensity_fraction_20ppm",

    "mean_abs_fragment_ppm_20ppm",
    "median_abs_fragment_ppm_20ppm",
    "std_abs_fragment_ppm_20ppm",
    "mean_signed_fragment_ppm_20ppm",
    "intensity_weighted_abs_fragment_ppm_20ppm",

    "matched_peak_rank_mean_20ppm",
    "matched_peak_rank_min_20ppm",
    "matched_top10_peak_count_20ppm",
    "matched_top50_peak_count_20ppm",

    "longest_b_ladder_20ppm",
    "longest_y_ladder_20ppm",
    "longest_combined_ladder_20ppm",
    "b_y_complement_pair_count_20ppm",
    "n_terminal_coverage_20ppm",
    "c_terminal_coverage_20ppm",

    "matched_loss_count_20ppm",
    "matched_loss_fraction_20ppm",
    "loss_explained_intensity_fraction_20ppm",

    # -------------------------
    # 50 ppm fragment 特征
    # -------------------------
    "matched_b_count_50ppm",
    "matched_y_count_50ppm",
    "matched_total_count_50ppm",
    "matched_b_fraction_50ppm",
    "matched_y_fraction_50ppm",
    "matched_total_fraction_50ppm",

    # -------------------------
    # 我们额外生成的仪器类别特征
    # -------------------------
    "instrument_id",
]


CATEGORICAL_FEATURES = [
    "instrument_id",
    "charge",
    "has_ion_mobility",
    "has_mod",
    "parse_ok",
    "fragment_parse_ok",
    "best_isotope_offset",
]


ID_COLUMNS_FOR_VALID = [
    "file_id",
    "instrument",
    "scan_number",
    "group_key",
    "peptide_key",
    "precursor_sequence",
]


# ============================================================
# 2. 工具函数
# ============================================================

def instrument_to_id_expr():
    return (
        pl.when(pl.col("instrument") == "mzml")
        .then(pl.lit(0))
        .when(pl.col("instrument") == "tims")
        .then(pl.lit(1))
        .when(pl.col("instrument") == "wiff")
        .then(pl.lit(2))
        .otherwise(pl.lit(-1))
        .cast(pl.Int8)
        .alias("instrument_id")
    )


def list_parquet_files(root: Path) -> List[Path]:
    files = [
        p for p in sorted(root.rglob("*.parquet"))
        if ".tmp" not in p.name
        and not p.name.endswith(".bak")
        and not p.name.endswith(".bak_fragment")
    ]
    return files


def list_split_files(split_root: Path) -> List[Path]:
    files = []
    for instrument in ["mzml", "tims", "wiff"]:
        d = split_root / instrument
        if d.exists():
            inst_files = list_parquet_files(d)
            print(f"{split_root.name}/{instrument}: {len(inst_files)} files")
            files.extend(inst_files)
        else:
            print(f"目录不存在: {d}")
    return files


def get_schema_cols(path: Path) -> List[str]:
    return pl.scan_parquet(path).collect_schema().names()


def check_file_ready(path: Path) -> bool:
    cols = set(get_schema_cols(path))

    required = {
        "label",
        "instrument",
        "aux_feature_done",
        "fragment_feature_done",
    }

    missing = required - cols
    if missing:
        print(f"跳过未完成文件: {path}, 缺少: {missing}")
        return False

    return True


def resolve_feature_columns(files: List[Path]) -> List[str]:
    """
    根据第一个可用文件确定实际存在的特征列。
    """
    for p in files:
        cols = set(get_schema_cols(p))
        if "label" not in cols:
            continue

        feature_cols = [c for c in FEATURE_CANDIDATES if c in cols or c == "instrument_id"]

        missing_core = [
            "min_abs_precursor_ppm",
            "matched_total_fraction_20ppm",
            "explained_intensity_fraction_20ppm",
            "longest_y_ladder_20ppm",
        ]

        for c in missing_core:
            if c not in feature_cols:
                print(f"警告：核心特征 {c} 不在文件 {p.name} 中")

        return feature_cols

    raise RuntimeError("没有找到可用于确定 feature columns 的 parquet 文件")


def clean_polars_df(df: pl.DataFrame, feature_cols: List[str], keep_id_cols: bool) -> pl.DataFrame:
    """
    统一 instrument_id，并只保留需要的列。
    """
    df = df.with_columns([
        instrument_to_id_expr()
    ])

    required_cols = feature_cols + ["label"]

    if keep_id_cols:
        for c in ID_COLUMNS_FOR_VALID:
            if c in df.columns:
                required_cols.append(c)

    required_cols = [c for c in required_cols if c in df.columns]

    df = df.select(required_cols)

    # label 强制转为 0/1
    df = df.with_columns([
        pl.col("label").cast(pl.Int8)
    ])

    # 特征强制转成数值
    cast_exprs = []
    for c in feature_cols:
        if c in df.columns:
            if c in CATEGORICAL_FEATURES:
                cast_exprs.append(pl.col(c).cast(pl.Int16))
            else:
                cast_exprs.append(pl.col(c).cast(pl.Float32))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df


def sample_one_file(
    path: Path,
    feature_cols: List[str],
    max_rows_per_file: int,
    neg_pos_ratio: float,
    seed: int,
    keep_id_cols: bool = False,
) -> pl.DataFrame:
    """
    从单个 parquet 中做 target/decoy 分层采样。
    避免直接把全量 2 亿行读入内存。
    """

    cols = set(get_schema_cols(path))

    read_cols = [
        c for c in feature_cols
        if c in cols and c != "instrument_id"
    ]

    read_cols += ["label", "instrument"]

    if keep_id_cols:
        read_cols += [c for c in ID_COLUMNS_FOR_VALID if c in cols]

    read_cols = list(dict.fromkeys(read_cols))

    df = pl.read_parquet(path, columns=read_cols)
    df = clean_polars_df(df, feature_cols, keep_id_cols=keep_id_cols)

    if df.height == 0:
        return df

    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)

    pos_n = pos.height
    neg_n = neg.height

    if pos_n == 0 or neg_n == 0:
        # 极端情况：直接随机采样
        n = min(df.height, max_rows_per_file)
        return df.sample(n=n, seed=seed, shuffle=True)

    # 让训练集大致保持 1 : neg_pos_ratio
    max_pos = int(max_rows_per_file / (1.0 + neg_pos_ratio))
    take_pos = min(pos_n, max_pos)
    take_neg = min(neg_n, int(take_pos * neg_pos_ratio))

    # 如果某个文件 target 很少，至少让 decoy 也进来
    if take_pos <= 0:
        take_pos = min(pos_n, max(1, max_rows_per_file // 10))
        take_neg = min(neg_n, max_rows_per_file - take_pos)

    pos_s = pos.sample(n=take_pos, seed=seed, shuffle=True)
    neg_s = neg.sample(n=take_neg, seed=seed + 17, shuffle=True)

    out = pl.concat([pos_s, neg_s], how="vertical")
    out = out.sample(n=out.height, seed=seed + 33, shuffle=True)

    return out


def collect_sample_dataset(
    files: List[Path],
    feature_cols: List[str],
    max_total_rows: int,
    max_rows_per_file: int,
    neg_pos_ratio: float,
    seed: int,
    keep_id_cols: bool = False,
) -> pl.DataFrame:
    """
    从很多 parquet 中采样，得到训练/验证用的小表。
    """
    parts = []
    total = 0

    for i, path in enumerate(tqdm(files)):
        if total >= max_total_rows:
            break

        if not check_file_ready(path):
            continue

        remain = max_total_rows - total
        per_file = min(max_rows_per_file, remain)

        try:
            part = sample_one_file(
                path=path,
                feature_cols=feature_cols,
                max_rows_per_file=per_file,
                neg_pos_ratio=neg_pos_ratio,
                seed=seed + i,
                keep_id_cols=keep_id_cols,
            )

            if part.height == 0:
                continue

            parts.append(part)
            total += part.height

            del part
            gc.collect()

        except Exception as e:
            print(f"采样失败: {path}")
            print(f"错误: {e}")

    if not parts:
        raise RuntimeError("没有采样到任何数据")

    df = pl.concat(parts, how="vertical", rechunk=False)
    return df


def polars_to_pandas_xy(df: pl.DataFrame, feature_cols: List[str]):
    """
    转为 LightGBM 可用的 pandas。
    """
    y = df["label"].to_numpy()

    X = df.select(feature_cols).to_pandas()

    # 防止 inf 进入模型
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


def compute_basic_metrics(y_true, pred) -> Dict:
    out = {}

    try:
        out["auc"] = float(roc_auc_score(y_true, pred))
    except Exception:
        out["auc"] = None

    try:
        out["average_precision"] = float(average_precision_score(y_true, pred))
    except Exception:
        out["average_precision"] = None

    out["pred_mean"] = float(np.mean(pred))
    out["pred_std"] = float(np.std(pred))
    out["label_mean"] = float(np.mean(y_true))

    return out


def compute_sampled_fdr_metric(valid_df: pl.DataFrame, pred: np.ndarray) -> Dict:
    """
    在采样验证集上计算一个近似 FDR 指标。
    这个不是官方完整指标，但可以快速判断 score 方向是否正确。
    """
    df = valid_df.select([
        "label",
        "peptide_key",
        "instrument",
    ]).with_columns([
        pl.Series("score", pred).cast(pl.Float32)
    ])

    df = df.sort("score", descending=True)

    labels = df["label"].to_numpy()
    is_target = labels == 1
    is_decoy = labels == 0

    cum_target = np.cumsum(is_target)
    cum_decoy = np.cumsum(is_decoy)

    fdr = cum_decoy / np.maximum(cum_target, 1)

    # q-value: 从低分到高分反向 cummin
    q = np.minimum.accumulate(fdr[::-1])[::-1]
    keep = q <= 0.01

    accepted = df.with_columns([
        pl.Series("q_value", q).cast(pl.Float32),
        pl.Series("accepted_1pct", keep).cast(pl.Int8),
    ]).filter(
        (pl.col("accepted_1pct") == 1) & (pl.col("label") == 1)
    )

    metric = {
        "sample_valid_rows": int(df.height),
        "accepted_target_psm_at_1pct": int(accepted.height),
        "accepted_unique_peptide_at_1pct": int(accepted["peptide_key"].n_unique()) if accepted.height > 0 else 0,
    }

    by_inst = {}
    for inst in ["mzml", "tims", "wiff"]:
        sub = accepted.filter(pl.col("instrument") == inst)
        by_inst[inst] = {
            "accepted_target_psm_at_1pct": int(sub.height),
            "accepted_unique_peptide_at_1pct": int(sub["peptide_key"].n_unique()) if sub.height > 0 else 0,
        }

    metric["by_instrument"] = by_inst

    return metric


# ============================================================
# 3. 主训练流程
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
        default="/root/autodl-tmp/datasets/aipc/models/lgbm_table_v1",
    )

    parser.add_argument(
        "--train-max-rows",
        type=int,
        default=3_000_000,
        help="训练最多采样多少行。空间/内存不够时可调小。"
    )

    parser.add_argument(
        "--valid-max-rows",
        type=int,
        default=1_000_000,
        help="验证最多采样多少行。"
    )

    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=80_000,
        help="每个 parquet 最多采样多少行。"
    )

    parser.add_argument(
        "--neg-pos-ratio",
        type=float,
        default=2.0,
        help="训练采样时 decoy:target 比例。"
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

    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_root = data_root / "processed_split"

    train_root = split_root / "train"
    valid_root = split_root / "valid"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== 路径 ==========")
    print("data_root:", data_root)
    print("train_root:", train_root)
    print("valid_root:", valid_root)
    print("out_dir:", out_dir)

    train_files = list_split_files(train_root)
    valid_files = list_split_files(valid_root)

    print()
    print(f"train files: {len(train_files)}")
    print(f"valid files: {len(valid_files)}")

    if len(train_files) == 0 or len(valid_files) == 0:
        raise RuntimeError("train 或 valid 文件为空，请检查 processed_split 目录")

    feature_cols = resolve_feature_columns(train_files)

    print()
    print("========== 使用特征 ==========")
    for c in feature_cols:
        print(c)

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 采样训练集
    # --------------------------------------------------------
    print()
    print("========== 采样训练集 ==========")

    train_df = collect_sample_dataset(
        files=train_files,
        feature_cols=feature_cols,
        max_total_rows=args.train_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=args.seed,
        keep_id_cols=False,
    )

    print("train sampled rows:", train_df.height)
    print(train_df.group_by("label").len())

    # --------------------------------------------------------
    # 采样验证集
    # --------------------------------------------------------
    print()
    print("========== 采样验证集 ==========")

    valid_df = collect_sample_dataset(
        files=valid_files,
        feature_cols=feature_cols,
        max_total_rows=args.valid_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        neg_pos_ratio=args.neg_pos_ratio,
        seed=args.seed + 10000,
        keep_id_cols=True,
    )

    print("valid sampled rows:", valid_df.height)
    print(valid_df.group_by("label").len())

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

    # 释放 train_df，valid_df 先保留用于 sampled FDR
    del train_df
    gc.collect()

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
    # 模型参数
    # --------------------------------------------------------
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",

        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,

        "lambda_l1": 0.0,
        "lambda_l2": 5.0,

        "max_bin": 255,
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": max(1, os.cpu_count() or 1),
    }

    with open(out_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------
    print()
    print("========== 开始训练 LightGBM ==========")

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

    # --------------------------------------------------------
    # 保存模型
    # --------------------------------------------------------
    model_path = out_dir / "model.txt"
    model.save_model(str(model_path))

    print()
    print(f"模型已保存: {model_path}")
    print("best_iteration:", model.best_iteration)

    # --------------------------------------------------------
    # 验证集评估
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
        "basic_metrics": basic_metrics,
        "sampled_fdr_metrics": sampled_fdr_metrics,
        "best_iteration": int(model.best_iteration),
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_valid": float(np.mean(y_valid)),
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 保存特征重要性
    # --------------------------------------------------------
    imp_gain = model.feature_importance(importance_type="gain")
    imp_split = model.feature_importance(importance_type="split")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": imp_gain,
        "importance_split": imp_split,
    }).sort_values("importance_gain", ascending=False)

    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    # --------------------------------------------------------
    # 保存少量验证集预测，方便你观察
    # --------------------------------------------------------
    pred_df = valid_df.select([
        c for c in ID_COLUMNS_FOR_VALID + ["label"]
        if c in valid_df.columns
    ]).with_columns([
        pl.Series("score", valid_pred).cast(pl.Float32)
    ])

    pred_df.write_parquet(out_dir / "valid_sample_pred.parquet")

    print()
    print(f"特征重要性已保存: {out_dir / 'feature_importance.csv'}")
    print(f"验证集采样预测已保存: {out_dir / 'valid_sample_pred.parquet'}")
    print("训练完成")


if __name__ == "__main__":
    main()