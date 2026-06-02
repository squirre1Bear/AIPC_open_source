# 根据 lgbm_v1 的训练结果，给 bas_test 文件添加 group 特征

# python src/preprocess/add_group_feature_test.py \
#   --parquet-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
#   --fold-model-dir ~/aipc/models/lgbm_v1_oof \
#   --workers=12
#   --force

from pathlib import Path
import argparse
import json
import os
import gc

import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm


TMP_SUFFIX = ".tmp_group.parquet"
INSTRUMENTS = ["mzml", "tims", "wiff"]
RT_QVALUE_ANOMALY_INSTRUMENT_IDS = {1, 2}
RT_QVALUE_ANOMALY_FEATURES = [
    "predicted_rt",
    "delta_rt",
    "spectrum_q",
    "spectrum_q_filled",
    "abs_delta_rt",
    "delta_rt_z_in_file",
    "predicted_rt_z_in_file",
    "abs_delta_rt_rank_in_scan",
    "abs_delta_rt_rank_pct_in_scan",
    "is_best_abs_delta_rt_in_scan",
    "abs_delta_rt_min_in_scan",
    "abs_delta_rt_mean_in_scan",
    "abs_delta_rt_std_in_scan",
    "abs_delta_rt_gap_to_best_in_scan",
    "abs_delta_rt_z_in_scan",
    "candidate_order_rt_rank_gap",
    "abs_candidate_order_rt_rank_gap",
    "candidate_order_matches_abs_delta_rank",
]
RT_QVALUE_ANOMALY_CATEGORICAL_FEATURES = {
    "is_best_abs_delta_rt_in_scan",
    "candidate_order_matches_abs_delta_rank",
}

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


def with_row_id(df: pl.DataFrame) -> pl.DataFrame:
    if hasattr(df, "with_row_index"):
        return df.with_row_index("__row_id")
    return df.with_row_count("__row_id")


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


def infer_instrument_from_filename(file_name: str) -> str:
    lower_name = file_name.lower()
    for instrument in INSTRUMENTS:
        if instrument in lower_name:
            return instrument
    return "unknown"


def mask_rt_qvalue_anomaly_features(feature_df: pl.DataFrame, feature_cols):
    columns_to_mask = [
        col for col in RT_QVALUE_ANOMALY_FEATURES
        if col in feature_cols and col in feature_df.columns
    ]
    if not columns_to_mask or "instrument_id" not in feature_df.columns:
        return feature_df

    numeric_to_mask = [
        col for col in columns_to_mask
        if col not in RT_QVALUE_ANOMALY_CATEGORICAL_FEATURES
    ]
    categorical_to_mask = [
        col for col in columns_to_mask
        if col in RT_QVALUE_ANOMALY_CATEGORICAL_FEATURES
    ]
    anomaly_expr = pl.col("instrument_id").is_in(list(RT_QVALUE_ANOMALY_INSTRUMENT_IDS))
    exprs = []
    exprs.extend(
        pl.when(anomaly_expr)
        .then(pl.lit(None, dtype=pl.Float32))
        .otherwise(pl.col(col))
        .cast(pl.Float32)
        .alias(col)
        for col in numeric_to_mask
    )
    exprs.extend(
        pl.when(anomaly_expr)
        .then(pl.lit(-1))
        .otherwise(pl.col(col))
        .cast(pl.Int16)
        .alias(col)
        for col in categorical_to_mask
    )
    if exprs:
        feature_df = feature_df.with_columns(exprs)
    return feature_df


def is_valid_parquet_file(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 8:
            return False
        with open(path, "rb") as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            tail = f.read(4)
        return head == b"PAR1" and tail == b"PAR1"
    except Exception:
        return False


def cleanup_tmp_file(tmp_path: Path):
    if tmp_path.exists():
        tmp_path.unlink()


def load_model_and_features(model_dir: Path):
    model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    with open(model_dir / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return model, feature_cols


def load_models_and_features(model_dirs):
    model_dirs = [Path(p).expanduser() for p in model_dirs]
    if not model_dirs:
        raise ValueError("model_dirs 不能为空")

    models = []
    feature_cols = None

    for model_dir in model_dirs:
        model, current_feature_cols = load_model_and_features(model_dir)

        if feature_cols is None:
            feature_cols = current_feature_cols
        elif current_feature_cols != feature_cols:
            raise RuntimeError(
                "fold 模型 feature_columns.json 不一致："
                f"base={model_dirs[0]}, current={model_dir}"
            )

        models.append(model)

    return models, feature_cols


def fold_model_dirs_from_root(fold_model_dir: Path):
    fold_model_dir = fold_model_dir.expanduser()
    manifest_path = fold_model_dir / "folds.json"

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        n_folds = int(manifest["n_folds"])
        model_dirs = [fold_model_dir / f"fold_{fold_id}" for fold_id in range(n_folds)]
    else:
        model_dirs = sorted(
            [p for p in fold_model_dir.glob("fold_*") if p.is_dir()],
            key=lambda p: (
                0,
                int(p.name.split("_")[-1]),
            ) if p.name.split("_")[-1].isdigit() else (1, p.name),
        )

    if not model_dirs:
        raise FileNotFoundError(f"未找到 fold 模型目录: {fold_model_dir}/fold_*")

    missing = []
    for model_dir in model_dirs:
        if not (model_dir / "model.txt").exists():
            missing.append(str(model_dir / "model.txt"))
        if not (model_dir / "feature_columns.json").exists():
            missing.append(str(model_dir / "feature_columns.json"))

    if missing:
        raise FileNotFoundError(
            "fold 模型不完整，无法进行 test 5-fold ensemble。缺少："
            + "; ".join(missing[:20])
        )

    return model_dirs


def prepare_feature_df(path: Path, feature_cols):
    schema = set(pl.scan_parquet(path).collect_schema().names())

    required = ["group_key", "aux_feature_done", "fragment_feature_done"]
    missing = [c for c in required if c not in schema]
    if missing:
        raise RuntimeError(f"{path} 缺少: {missing}")

    read_cols = ["group_key"]

    for c in feature_cols:
        if c == "instrument_id":
            read_cols.append("instrument")
        else:
            if c not in schema:
                raise RuntimeError(f"{path} 缺少模型特征列: {c}")
            read_cols.append(c)

    read_cols = list(dict.fromkeys(read_cols))

    df = pl.read_parquet(path, columns=read_cols)
    df = with_row_id(df)

    if "instrument_id" in feature_cols:
        df = df.with_columns([instrument_to_id_expr()])

    df = df.select(["__row_id", "group_key"] + feature_cols)

    cast_exprs = []
    cat_cols = {
        "instrument_id",
        "charge",
        "has_ion_mobility",
        "has_mod",
        "parse_ok",
        "fragment_parse_ok",
        "best_isotope_offset",
        "is_first_candidate_in_scan",
        "is_top3_candidate_in_scan",
        "is_best_abs_delta_rt_in_scan",
        "candidate_order_matches_abs_delta_rank",
        "scan_has_multiple_charges",
        "is_first_candidate_for_charge_in_scan",
    }

    for c in feature_cols:
        if c in cat_cols:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))

    df = df.with_columns(cast_exprs)
    return df


def predict(model, df: pl.DataFrame, feature_cols):
    X = df.select(feature_cols).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and best_iter > 0:
        pred = model.predict(X, num_iteration=best_iter)
    else:
        pred = model.predict(X)

    return np.asarray(pred, dtype=np.float32)


def predict_ensemble(models, df: pl.DataFrame, feature_cols):
    X = df.select(feature_cols).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)

    pred_sum = np.zeros(df.height, dtype=np.float64)

    for model in models:
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None and best_iter > 0:
            pred = model.predict(X, num_iteration=best_iter)
        else:
            pred = model.predict(X)
        pred_sum += np.asarray(pred, dtype=np.float64)

    return (pred_sum / max(1, len(models))).astype(np.float32)


def build_group_features(feature_df: pl.DataFrame, pred):
    score_df = feature_df.select(["__row_id", "group_key"]).with_columns([
        pl.Series("lgbm_v1_score", pred).cast(pl.Float32)
    ])

    score_df = score_df.with_columns([
        pl.col("lgbm_v1_score")
        .rank(method="ordinal", descending=True)
        .over("group_key")
        .cast(pl.Int32)
        .alias("lgbm_v1_group_rank")
    ])

    stats = score_df.group_by("group_key").agg([
        pl.len().cast(pl.Int32).alias("lgbm_v1_group_size"),
        pl.col("lgbm_v1_score").max().cast(pl.Float32).alias("lgbm_v1_group_max"),
        pl.col("lgbm_v1_score").min().cast(pl.Float32).alias("lgbm_v1_group_min"),
        pl.col("lgbm_v1_score").mean().cast(pl.Float32).alias("lgbm_v1_group_mean"),
        pl.col("lgbm_v1_score").std().fill_null(0).cast(pl.Float32).alias("lgbm_v1_group_std"),
    ])

    top2 = score_df.filter(pl.col("lgbm_v1_group_rank") == 2).select([
        "group_key",
        pl.col("lgbm_v1_score").cast(pl.Float32).alias("lgbm_v1_group_top2"),
    ])

    score_df = score_df.join(stats, on="group_key", how="left")
    score_df = score_df.join(top2, on="group_key", how="left")

    score_df = score_df.with_columns([
        pl.col("lgbm_v1_group_top2")
        .fill_null(pl.col("lgbm_v1_group_max"))
        .cast(pl.Float32)
        .alias("lgbm_v1_group_top2")
    ])

    score_df = score_df.with_columns([
        pl.when(pl.col("lgbm_v1_group_size") > 1)
        .then((pl.col("lgbm_v1_group_rank") - 1) / (pl.col("lgbm_v1_group_size") - 1))
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("lgbm_v1_group_rank_pct"),

        (pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_max"))
        .cast(pl.Float32)
        .alias("lgbm_v1_gap_to_top1"),

        (pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_top2"))
        .cast(pl.Float32)
        .alias("lgbm_v1_gap_to_top2"),

        (pl.col("lgbm_v1_group_max") - pl.col("lgbm_v1_group_top2"))
        .cast(pl.Float32)
        .alias("lgbm_v1_top1_margin"),

        (pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_mean"))
        .cast(pl.Float32)
        .alias("lgbm_v1_minus_group_mean"),

        pl.when(pl.col("lgbm_v1_group_std") > 1e-12)
        .then((pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_mean")) / pl.col("lgbm_v1_group_std"))
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("lgbm_v1_z_in_group"),

        (pl.col("lgbm_v1_group_rank") == 1)
        .cast(pl.Int8)
        .alias("is_lgbm_v1_group_top1"),

        (pl.col("lgbm_v1_group_rank") <= 3)
        .cast(pl.Int8)
        .alias("is_lgbm_v1_group_top3"),
    ])

    score_df = score_df.with_columns([
        (pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_max"))
        .exp()
        .cast(pl.Float32)
        .alias("__exp_score")
    ])

    score_df = score_df.with_columns([
        pl.col("__exp_score").sum().over("group_key").cast(pl.Float32).alias("__exp_sum")
    ])

    score_df = score_df.with_columns([
        pl.when(pl.col("__exp_sum") > 0)
        .then(pl.col("__exp_score") / pl.col("__exp_sum"))
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias("lgbm_v1_softmax_in_group")
    ])

    return score_df.select(["__row_id"] + GROUP_FEATURE_COLS)


def process_one_file(
    path: Path,
    model_dirs,
    force: bool,
    mask_rt_qvalue_anomaly: bool = False,
):
    schema = set(pl.scan_parquet(path).collect_schema().names())
    if "group_feature_done" in schema and not force:
        print(f"已处理，跳过: {path}")
        return

    models, feature_cols = load_models_and_features(model_dirs)

    if len(model_dirs) == 1:
        print(f"v1 score model: {model_dirs[0]}")
    else:
        print(f"v1 score fold ensemble: {len(model_dirs)} models")

    feature_df = prepare_feature_df(path, feature_cols)
    if mask_rt_qvalue_anomaly:
        feature_df = mask_rt_qvalue_anomaly_features(feature_df, feature_cols)
    pred = predict_ensemble(models, feature_df, feature_cols)
    group_df = build_group_features(feature_df, pred)

    df = pl.read_parquet(path)
    df = with_row_id(df)

    old_cols = [c for c in GROUP_FEATURE_COLS + ["group_feature_done"] if c in df.columns]
    if old_cols:
        df = df.drop(old_cols)

    df = df.join(group_df, on="__row_id", how="left").drop("__row_id")
    df = df.with_columns([pl.lit(1).cast(pl.Int8).alias("group_feature_done")])

    tmp_path = path.with_name(path.name + TMP_SUFFIX)
    cleanup_tmp_file(tmp_path)
    df.write_parquet(tmp_path)

    if not is_valid_parquet_file(tmp_path):
        raise RuntimeError(f"临时 parquet 写入不完整: {tmp_path}")

    os.replace(tmp_path, path)
    print(f"group 特征完成: {path}, rows={df.height}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=str, default="/root/autodl-tmp/datasets/aipc/test_processed/bas_merged")
    parser.add_argument(
        "--fold-model-dir",
        type=str,
        default="",
        help="OOF v1 fold 总目录。提供后，测试集使用 fold_0..fold_N 平均打分。",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="/root/autodl-tmp/datasets/aipc/models/lgbm_table_v1",
        help="兼容旧流程：单个 full v1 模型目录。未提供 --fold-model-dir 时使用。",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only-instrument",
        choices=INSTRUMENTS,
        default=None,
        help="Only process one instrument. Default keeps the existing all-instrument flow.",
    )
    parser.add_argument(
        "--mask-rt-qvalue-anomaly",
        action="store_true",
        help="Mask tims/wiff RT/q-value anomaly features before v1 inference.",
    )
    args = parser.parse_args()

    d = Path(args.parquet_dir)

    if args.fold_model_dir:
        model_dirs = fold_model_dirs_from_root(Path(args.fold_model_dir))
        print("score mode: fold ensemble")
    else:
        model_dirs = [Path(args.model_dir).expanduser()]
        print("score mode: single model")

    print("model dirs:")
    for model_dir in model_dirs:
        print("  ", model_dir)

    files = sorted(d.glob("*.parquet"))
    if args.only_instrument is not None:
        files = [
            path for path in files
            if infer_instrument_from_filename(path.name) == args.only_instrument
        ]
    print("files:", len(files))

    for path in tqdm(files):
        process_one_file(
            path,
            model_dirs,
            force=args.force,
            mask_rt_qvalue_anomaly=args.mask_rt_qvalue_anomaly,
        )
        gc.collect()


if __name__ == "__main__":
    main()
