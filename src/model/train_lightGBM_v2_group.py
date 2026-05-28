# python src/model/train_lightGBM_v2_group.py \
#   --data-root /root/autodl-tmp/datasets/aipc \
#   --v1-model-dir ~/aipc/models/lgbm_v1 \
#   --out-dir ~/aipc/models/lgbm_v2 \
#   --train-max-rows 3000000 \
#   --valid-max-rows 1000000 \
#   --max-rows-per-file 80000

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
# 1. group 特征
# ============================================================

GROUP_FEATURE_COLS = [
    "lgbm_v1_score",

    "lgbm_v1_group_size",
    "lgbm_v1_group_rank",
    "lgbm_v1_group_rank_pct",

    "lgbm_v1_group_max",
    "lgbm_v1_group_min",
    "lgbm_v1_group_mean",
    "lgbm_v1_group_std",
    "lgbm_v1_group_top2",

    "lgbm_v1_gap_to_top1",
    "lgbm_v1_gap_to_top2",
    "lgbm_v1_top1_margin",
    "lgbm_v1_minus_group_mean",
    "lgbm_v1_z_in_group",

    "lgbm_v1_softmax_in_group",

    "is_lgbm_v1_group_top1",
    "is_lgbm_v1_group_top3",
]


CATEGORICAL_FEATURES = [
    "instrument_id",
    "charge",
    "has_ion_mobility",
    "has_mod",
    "parse_ok",
    "fragment_parse_ok",
    "best_isotope_offset",
    "is_lgbm_v1_group_top1",
    "is_lgbm_v1_group_top3",
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

def split_files_by_instrument(files):
    by_inst = {
        "mzml": [],
        "tims": [],
        "wiff": [],
    }

    for p in files:
        s = str(p).lower()

        if "/mzml/" in s or "\\mzml\\" in s:
            by_inst["mzml"].append(p)
        elif "/tims/" in s or "\\tims\\" in s:
            by_inst["tims"].append(p)
        elif "/wiff/" in s or "\\wiff\\" in s:
            by_inst["wiff"].append(p)
        else:
            print(f"警告：无法判断仪器类型，跳过: {p}")

    return by_inst

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


def get_schema_cols(path: Path):
    return pl.scan_parquet(path).collect_schema().names()


def check_file_ready(path: Path):
    cols = set(get_schema_cols(path))

    required = {
        "label",
        "instrument",
        "aux_feature_done",
        "fragment_feature_done",
        "group_feature_done",
    }

    missing = required - cols

    if missing:
        print(f"跳过未完成文件: {path}, 缺少: {missing}")
        return False

    return True


def load_v1_feature_columns(v1_model_dir: Path):
    path = v1_model_dir / "feature_columns.json"

    if not path.exists():
        raise FileNotFoundError(f"找不到 v1 feature_columns.json: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)

    if not isinstance(cols, list) or len(cols) == 0:
        raise RuntimeError(f"v1 feature_columns.json 内容异常: {path}")

    return cols


def resolve_v2_feature_columns(files: List[Path], v1_feature_cols: List[str]):
    """
    v2 特征 = v1 特征 + group 特征。
    只保留实际存在的 group 特征。
    """
    for p in files:
        cols = set(get_schema_cols(p))

        if "label" not in cols:
            continue

        feature_cols = []

        for c in v1_feature_cols:
            if c == "instrument_id":
                feature_cols.append(c)
            elif c in cols:
                feature_cols.append(c)
            else:
                print(f"警告: v1 特征 {c} 不存在于 {p.name}")

        missing_group = []

        for c in GROUP_FEATURE_COLS:
            if c in cols:
                feature_cols.append(c)
            else:
                missing_group.append(c)

        if missing_group:
            raise RuntimeError(f"{p} 缺少 group 特征: {missing_group}")

        # 去重并保持顺序
        feature_cols = list(dict.fromkeys(feature_cols))

        return feature_cols

    raise RuntimeError("没有找到可用于确定 v2 特征列的文件")


def clean_polars_df(df: pl.DataFrame, feature_cols: List[str], keep_id_cols: bool) -> pl.DataFrame:
    # 如果模型特征中需要 instrument_id，就由 instrument 生成
    if "instrument_id" in feature_cols:
        df = df.with_columns([instrument_to_id_expr()])

    # 模型特征 + label
    required_cols = list(feature_cols) + ["label"]

    # 保留 instrument，方便后面检查三仪器采样比例
    if "instrument" in df.columns:
        required_cols.append("instrument")

    # valid 需要额外保留 id 字段，用于 FDR 评估和保存 valid_sample_pred
    if keep_id_cols:
        for c in ID_COLUMNS_FOR_VALID:
            if c in df.columns:
                required_cols.append(c)

    # 关键修复：去重并保持原顺序
    required_cols = list(dict.fromkeys(required_cols))

    # 只保留当前 df 中真实存在的列
    required_cols = [c for c in required_cols if c in df.columns]

    df = df.select(required_cols)

    # label 转成 0/1
    df = df.with_columns([
        pl.col("label").cast(pl.Int8)
    ])

    # 特征列类型转换
    cast_exprs = []

    for c in feature_cols:
        if c not in df.columns:
            continue

        if c in CATEGORICAL_FEATURES:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    return df

def normalize_part_schema(df: pl.DataFrame, feature_cols: List[str], keep_id_cols: bool) -> pl.DataFrame:
    """
    统一每个采样 part 的 schema，避免 mzML/tims/wiff concat 时类型不一致。
    """

    cast_exprs = []

    # label 固定
    if "label" in df.columns:
        cast_exprs.append(pl.col("label").cast(pl.Int8))

    # instrument 固定为字符串
    if "instrument" in df.columns:
        cast_exprs.append(pl.col("instrument").cast(pl.Utf8))

    # ID 字段统一类型
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

    # 模型特征统一类型
    for c in feature_cols:
        if c not in df.columns:
            continue

        if c in CATEGORICAL_FEATURES:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    # 固定列顺序，避免不同 part 列顺序不同
    final_cols = []

    # 模型特征在前
    for c in feature_cols:
        if c in df.columns:
            final_cols.append(c)

    # label 必须保留
    if "label" in df.columns:
        final_cols.append("label")

    # instrument 保留用于检查采样是否覆盖三仪器
    if "instrument" in df.columns:
        final_cols.append("instrument")

    # valid 额外保留 id 字段
    if keep_id_cols:
        for c in ID_COLUMNS_FOR_VALID:
            if c in df.columns:
                final_cols.append(c)

    # 去重并保持顺序
    final_cols = list(dict.fromkeys(final_cols))

    # 其他列不保留，避免无关列 schema 不一致
    df = df.select(final_cols)

    return df

def sample_one_file(
    path: Path,
    feature_cols: List[str],
    max_rows_per_file: int,
    neg_pos_ratio: float,
    seed: int,
    keep_id_cols: bool,
):
    cols = set(get_schema_cols(path))

    missing = [
        c for c in feature_cols
        if c != "instrument_id" and c not in cols
    ]

    if missing:
        raise RuntimeError(f"{path} 缺少特征列: {missing[:20]}")

    read_cols = []

    for c in feature_cols:
        if c == "instrument_id":
            read_cols.append("instrument")
        else:
            read_cols.append(c)

    read_cols += ["label"]

    if keep_id_cols:
        read_cols += [c for c in ID_COLUMNS_FOR_VALID if c in cols]

    read_cols = list(dict.fromkeys(read_cols))

    df = pl.read_parquet(path, columns=read_cols)
    df = clean_polars_df(df, feature_cols, keep_id_cols=keep_id_cols)

    if df.height == 0:
        return normalize_part_schema(df, feature_cols, keep_id_cols)

    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)

    pos_n = pos.height
    neg_n = neg.height

    if pos_n == 0 or neg_n == 0:
        n = min(df.height, max_rows_per_file)
        out = df.sample(n=n, seed=seed, shuffle=True)
        return normalize_part_schema(out, feature_cols, keep_id_cols)

    max_pos = int(max_rows_per_file / (1.0 + neg_pos_ratio))
    take_pos = min(pos_n, max_pos)
    take_neg = min(neg_n, int(take_pos * neg_pos_ratio))

    if take_pos <= 0:
        take_pos = min(pos_n, max(1, max_rows_per_file // 10))
        take_neg = min(neg_n, max_rows_per_file - take_pos)

    pos_s = pos.sample(n=take_pos, seed=seed, shuffle=True)
    neg_s = neg.sample(n=take_neg, seed=seed + 17, shuffle=True)

    out = pl.concat([pos_s, neg_s], how="vertical", rechunk=False)
    out = out.sample(n=out.height, seed=seed + 33, shuffle=True)

    return normalize_part_schema(out, feature_cols, keep_id_cols)


def collect_sample_dataset(
    files,
    feature_cols,
    max_total_rows,
    max_rows_per_file,
    neg_pos_ratio,
    seed,
    keep_id_cols=False,
):
    """
    按仪器均衡采样。
    不再从 files 前面开始扫到 max_total_rows 就停，
    而是 mzML / tims / wiff 分别采样。
    """
    by_inst = split_files_by_instrument(files)

    instruments = ["mzml", "tims", "wiff"]

    # 每个仪器目标采样行数
    target_per_inst = max_total_rows // len(instruments)

    parts = []
    total_rows = 0

    for inst_idx, inst in enumerate(instruments):
        inst_files = by_inst.get(inst, [])

        print()
        print(f"========== 采样 {inst} ==========")
        print(f"{inst} files: {len(inst_files)}")
        print(f"{inst} target rows: {target_per_inst}")

        if len(inst_files) == 0:
            print(f"{inst} 没有文件，跳过")
            continue

        inst_parts = []
        inst_rows = 0

        # 打乱文件顺序，避免总是取前几个文件
        rng = np.random.default_rng(seed + inst_idx * 1000)
        inst_files = list(inst_files)
        rng.shuffle(inst_files)

        for i, path in enumerate(tqdm(inst_files)):
            if inst_rows >= target_per_inst:
                break

            remain = target_per_inst - inst_rows
            per_file = min(max_rows_per_file, remain)

            if not check_file_ready(path):
                continue

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

                # 统一 schema，防止补 instrument 后类型不一致
                part = normalize_part_schema(part, feature_cols, keep_id_cols)

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
        print(inst_df.group_by("label").len())

        parts.append(inst_df)
        total_rows += inst_df.height

        del inst_parts
        gc.collect()

    if len(parts) == 0:
        raise RuntimeError("没有采样到任何数据")

    df = pl.concat(parts, how="vertical", rechunk=False)

    # 如果超过 max_total_rows，随机截断到 max_total_rows
    if df.height > max_total_rows:
        df = df.sample(n=max_total_rows, seed=seed + 999, shuffle=True)

    print()
    print("========== 总采样结果 ==========")
    print("sampled rows:", df.height)

    if "instrument" in df.columns:
        print(df.group_by("instrument").len())

    print(df.group_by("label").len())

    return df

def polars_to_pandas_xy(df: pl.DataFrame, feature_cols: List[str]):
    y = df["label"].to_numpy()

    X = df.select(feature_cols).to_pandas()
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
        "--v1-model-dir",
        type=str,
        default="/root/autodl-tmp/datasets/aipc/models/lgbm_table_v1",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="/root/autodl-tmp/datasets/aipc/models/lgbm_table_v2_group",
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
        default=42,
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
    v1_model_dir = Path(args.v1_model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_root = data_root / "processed_split"

    train_root = split_root / "train"
    valid_root = split_root / "valid"

    print("========== 路径 ==========")
    print("data_root:", data_root)
    print("v1_model_dir:", v1_model_dir)
    print("train_root:", train_root)
    print("valid_root:", valid_root)
    print("out_dir:", out_dir)

    train_files = list_split_files(train_root)
    valid_files = list_split_files(valid_root)

    if len(train_files) == 0 or len(valid_files) == 0:
        raise RuntimeError("train 或 valid 文件为空")

    v1_feature_cols = load_v1_feature_columns(v1_model_dir)
    feature_cols = resolve_v2_feature_columns(train_files, v1_feature_cols)

    print()
    print("========== v2 使用特征 ==========")
    for c in feature_cols:
        print(c)

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    with open(out_dir / "group_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(GROUP_FEATURE_COLS, f, indent=2, ensure_ascii=False)

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

        "learning_rate": 0.025,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 250,

        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,

        "lambda_l1": 0.0,
        "lambda_l2": 8.0,

        "max_bin": 255,
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": 8,
    }

    with open(out_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    # --------------------------------------------------------
    # 训练
    # --------------------------------------------------------
    print()
    print("========== 开始训练 LightGBM v2 group ==========")

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=args.num_boost_round,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds),
            lgb.log_evaluation(period=10),
        ],
    )

    model_path = out_dir / "model.txt"
    model.save_model(str(model_path))

    print()
    print(f"模型已保存: {model_path}")
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
    # 特征重要性
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
    # 保存验证集采样预测
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
    print("v2 group 模型训练完成")


if __name__ == "__main__":
    main()