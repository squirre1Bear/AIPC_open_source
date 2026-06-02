# AIPC_GROUP_WORKERS=25 python src/preprocess/add_group_feature.py \
#   --data-root /root/autodl-tmp/datasets/aipc \
#   --model-dir /root/aipc/models/lgbm_v1 \
#   --splits train valid

from pathlib import Path
import argparse
import json
import os
import sys
import subprocess
import gc
import errno
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from tqdm import tqdm


# ============================================================
# 1. 默认配置
# ============================================================

DEFAULT_DATA_ROOT = "/root/autodl-tmp/datasets/aipc"
DEFAULT_MODEL_DIR = "/root/autodl-tmp/datasets/aipc/models/lgbm_table_v1"

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

TMP_SUFFIX = ".tmp_group.parquet"

RUN_EACH_FILE_IN_SUBPROCESS = True
MAX_PARALLEL_FILES = int(os.environ.get("AIPC_GROUP_WORKERS", "4"))

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


INT_GROUP_FEATURE_COLS = [
    "lgbm_v1_group_size",
    "lgbm_v1_group_rank",
    "is_lgbm_v1_group_top1",
    "is_lgbm_v1_group_top3",
]


# ============================================================
# 2. 工具函数
# ============================================================

def with_row_id(df: pl.DataFrame) -> pl.DataFrame:
    """
    兼容不同 Polars 版本。
    """
    if hasattr(df, "with_row_index"):
        return df.with_row_index("__row_id")
    return df.with_row_count("__row_id")


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
    try:
        if tmp_path.exists():
            tmp_path.unlink()
            print(f"已删除临时文件: {tmp_path}")
    except Exception as e:
        print(f"删除临时文件失败: {tmp_path}, 错误: {e}")


def get_schema_cols(path: Path):
    return pl.scan_parquet(path).collect_schema().names()


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


def list_split_files(data_root: Path, splits):
    files = []

    for split in splits:
        split_root = data_root / "processed_split" / split

        for instrument in INSTRUMENTS:
            d = split_root / instrument

            if not d.exists():
                print(f"目录不存在，跳过: {d}")
                continue

            inst_files = [
                p for p in sorted(d.glob("*.parquet"))
                if ".tmp" not in p.name
                and not p.name.endswith(".bak")
                and not p.name.endswith(".bak_fragment")
            ]

            print(f"{split}/{instrument}: {len(inst_files)} files")

            files.extend(inst_files)

    return files


def load_feature_columns(model_dir: Path):
    path = model_dir / "feature_columns.json"

    if not path.exists():
        raise FileNotFoundError(f"找不到 feature_columns.json: {path}")

    with open(path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise RuntimeError(f"feature_columns.json 内容异常: {path}")

    return feature_cols


def load_lgbm_model(model_dir: Path):
    model_path = model_dir / "model.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"找不到 LightGBM 模型: {model_path}")

    model = lgb.Booster(model_file=str(model_path))
    return model


def load_lgbm_models_and_feature_columns(model_dirs):
    """
    加载一个或多个 v1 模型。
    多模型模式用于 valid/test 的 5-fold ensemble 打分。
    """
    model_dirs = [Path(p).expanduser() for p in model_dirs]

    if not model_dirs:
        raise ValueError("model_dirs 不能为空")

    feature_cols = load_feature_columns(model_dirs[0])
    models = []

    for model_dir in model_dirs:
        current_feature_cols = load_feature_columns(model_dir)
        if current_feature_cols != feature_cols:
            raise RuntimeError(
                "fold 模型 feature_columns.json 不一致："
                f"base={model_dirs[0]}, current={model_dir}"
            )

        models.append(load_lgbm_model(model_dir))

    return models, feature_cols


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


def prepare_feature_frame(path: Path, feature_cols):
    """
    只读取模型预测需要的列，以及 group_key。
    不读取 mz_array / intensity_array，避免内存爆炸。
    """
    schema_cols = set(get_schema_cols(path))

    required_flags = ["aux_feature_done", "fragment_feature_done"]

    missing_flags = [c for c in required_flags if c not in schema_cols]
    if missing_flags:
        raise RuntimeError(f"{path} 缺少特征完成标记: {missing_flags}")

    if "group_feature_done" in schema_cols:
        print(f"group 特征已存在，将覆盖重算: {path}")

    if "group_key" not in schema_cols:
        raise RuntimeError(f"{path} 缺少 group_key")

    if "instrument_id" in feature_cols and "instrument" not in schema_cols:
        raise RuntimeError(f"{path} 需要 instrument 生成 instrument_id，但缺少 instrument 列")

    missing_features = [
        c for c in feature_cols
        if c != "instrument_id" and c not in schema_cols
    ]

    if missing_features:
        raise RuntimeError(f"{path} 缺少模型特征列: {missing_features[:20]}")

    read_cols = ["group_key"]

    for c in feature_cols:
        if c == "instrument_id":
            read_cols.append("instrument")
        else:
            read_cols.append(c)

    read_cols = list(dict.fromkeys(read_cols))

    df = pl.read_parquet(path, columns=read_cols)
    df = with_row_id(df)

    if "instrument_id" in feature_cols:
        df = df.with_columns([instrument_to_id_expr()])

    select_cols = ["__row_id", "group_key"] + feature_cols
    df = df.select(select_cols)

    cast_exprs = []

    for c in feature_cols:
        if c == "instrument_id":
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        elif c in [
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
        ]:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))

    df = df.with_columns(cast_exprs)

    return df


def predict_lgbm_scores(model, feature_df: pl.DataFrame, feature_cols):
    X = feature_df.select(feature_cols).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and best_iter > 0:
        pred = model.predict(X, num_iteration=best_iter)
    else:
        pred = model.predict(X)

    pred = np.asarray(pred, dtype=np.float32)

    del X
    gc.collect()

    return pred


def predict_lgbm_scores_ensemble(models, feature_df: pl.DataFrame, feature_cols):
    X = feature_df.select(feature_cols).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)

    pred_sum = np.zeros(feature_df.height, dtype=np.float64)

    for model in models:
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is not None and best_iter > 0:
            pred = model.predict(X, num_iteration=best_iter)
        else:
            pred = model.predict(X)

        pred_sum += np.asarray(pred, dtype=np.float64)

    pred = (pred_sum / max(1, len(models))).astype(np.float32)

    del X
    gc.collect()

    return pred


def build_group_feature_df(feature_df: pl.DataFrame, pred: np.ndarray) -> pl.DataFrame:
    """
    根据 lgbm_v1_score 在 group_key 内生成组内竞争特征。
    """
    score_df = feature_df.select(["__row_id", "group_key"]).with_columns([
        pl.Series("lgbm_v1_score", pred).cast(pl.Float32)
    ])

    score_df = score_df.with_columns([
        pl.col("lgbm_v1_score")
        .rank(method="ordinal", descending=True)
        .over("group_key")
        .cast(pl.Int32)
        .alias("lgbm_v1_group_rank"),
    ])

    group_stats = (
        score_df
        .group_by("group_key")
        .agg([
            pl.len().cast(pl.Int32).alias("lgbm_v1_group_size"),
            pl.col("lgbm_v1_score").max().cast(pl.Float32).alias("lgbm_v1_group_max"),
            pl.col("lgbm_v1_score").min().cast(pl.Float32).alias("lgbm_v1_group_min"),
            pl.col("lgbm_v1_score").mean().cast(pl.Float32).alias("lgbm_v1_group_mean"),
            pl.col("lgbm_v1_score").std().fill_null(0.0).cast(pl.Float32).alias("lgbm_v1_group_std"),
        ])
    )

    top2 = (
        score_df
        .filter(pl.col("lgbm_v1_group_rank") == 2)
        .select([
            "group_key",
            pl.col("lgbm_v1_score").cast(pl.Float32).alias("lgbm_v1_group_top2"),
        ])
    )

    score_df = score_df.join(group_stats, on="group_key", how="left")
    score_df = score_df.join(top2, on="group_key", how="left")

    score_df = score_df.with_columns([
        pl.col("lgbm_v1_group_top2")
        .fill_null(pl.col("lgbm_v1_group_max"))
        .cast(pl.Float32)
        .alias("lgbm_v1_group_top2")
    ])

    score_df = score_df.with_columns([
        (
            pl.when(pl.col("lgbm_v1_group_size") > 1)
            .then(
                (pl.col("lgbm_v1_group_rank") - 1)
                / (pl.col("lgbm_v1_group_size") - 1)
            )
            .otherwise(0.0)
        )
        .cast(pl.Float32)
        .alias("lgbm_v1_group_rank_pct"),

        (
            pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_max")
        )
        .cast(pl.Float32)
        .alias("lgbm_v1_gap_to_top1"),

        (
            pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_top2")
        )
        .cast(pl.Float32)
        .alias("lgbm_v1_gap_to_top2"),

        (
            pl.col("lgbm_v1_group_max") - pl.col("lgbm_v1_group_top2")
        )
        .cast(pl.Float32)
        .alias("lgbm_v1_top1_margin"),

        (
            pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_mean")
        )
        .cast(pl.Float32)
        .alias("lgbm_v1_minus_group_mean"),

        (
            pl.when(pl.col("lgbm_v1_group_std") > 1e-12)
            .then(
                (pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_mean"))
                / pl.col("lgbm_v1_group_std")
            )
            .otherwise(0.0)
        )
        .cast(pl.Float32)
        .alias("lgbm_v1_z_in_group"),

        (pl.col("lgbm_v1_group_rank") == 1)
        .cast(pl.Int8)
        .alias("is_lgbm_v1_group_top1"),

        (pl.col("lgbm_v1_group_rank") <= 3)
        .cast(pl.Int8)
        .alias("is_lgbm_v1_group_top3"),
    ])

    # softmax in group
    score_df = score_df.with_columns([
        (
            pl.col("lgbm_v1_score") - pl.col("lgbm_v1_group_max")
        )
        .exp()
        .cast(pl.Float32)
        .alias("__exp_score")
    ])

    score_df = score_df.with_columns([
        pl.col("__exp_score")
        .sum()
        .over("group_key")
        .cast(pl.Float32)
        .alias("__exp_sum")
    ])

    score_df = score_df.with_columns([
        (
            pl.when(pl.col("__exp_sum") > 0)
            .then(pl.col("__exp_score") / pl.col("__exp_sum"))
            .otherwise(0.0)
        )
        .cast(pl.Float32)
        .alias("lgbm_v1_softmax_in_group")
    ])

    out = score_df.select(["__row_id"] + GROUP_FEATURE_COLS)

    out = out.with_columns([
        pl.col(c).cast(pl.Int32) for c in INT_GROUP_FEATURE_COLS if c in out.columns
    ])

    float_cols = [c for c in GROUP_FEATURE_COLS if c not in INT_GROUP_FEATURE_COLS]

    out = out.with_columns([
        pl.col(c).cast(pl.Float32) for c in float_cols if c in out.columns
    ])

    return out


def add_group_features_to_file_with_model_dirs(
    path: Path,
    model_dirs,
    force: bool,
    mask_rt_qvalue_anomaly: bool = False,
):
    print(f"\n开始处理: {path}")

    schema_cols = set(get_schema_cols(path))

    if "group_feature_done" in schema_cols and not force:
        print(f"已存在 group_feature_done，跳过: {path}")
        return

    model_dirs = [Path(p).expanduser() for p in model_dirs]
    models, feature_cols = load_lgbm_models_and_feature_columns(model_dirs)

    if len(model_dirs) == 1:
        print(f"v1 score model: {model_dirs[0]}")
    else:
        print(f"v1 score ensemble models: {len(model_dirs)}")
        for model_dir in model_dirs:
            print(f"  - {model_dir}")

    feature_df = prepare_feature_frame(path, feature_cols)
    if mask_rt_qvalue_anomaly:
        feature_df = mask_rt_qvalue_anomaly_features(feature_df, feature_cols)
    pred = predict_lgbm_scores_ensemble(models, feature_df, feature_cols)

    group_feature_df = build_group_feature_df(feature_df, pred)

    if group_feature_df.height != feature_df.height:
        raise RuntimeError(
            f"group_feature_df 行数不一致: {group_feature_df.height} vs {feature_df.height}"
        )

    del pred
    del models
    del feature_df
    gc.collect()

    # 读全量 parquet，追加 group 特征
    df = pl.read_parquet(path)
    df = with_row_id(df)

    existing_group_cols = [
        c for c in GROUP_FEATURE_COLS + ["group_feature_done"]
        if c in df.columns
    ]

    if existing_group_cols:
        df = df.drop(existing_group_cols)

    df = df.join(group_feature_df, on="__row_id", how="left")
    df = df.drop("__row_id")

    del group_feature_df
    gc.collect()

    df = df.with_columns([
        pl.lit(1).cast(pl.Int8).alias("group_feature_done")
    ])

    tmp_path = path.with_name(path.name + TMP_SUFFIX)
    cleanup_tmp_file(tmp_path)

    row_count = df.height
    col_count = len(df.columns)

    df_deleted = False

    try:
        df.write_parquet(tmp_path)

        if not is_valid_parquet_file(tmp_path):
            raise RuntimeError(f"临时 parquet 写入不完整: {tmp_path}")

        del df
        df_deleted = True
        gc.collect()

        os.replace(tmp_path, path)

    except BaseException:
        if not df_deleted:
            try:
                del df
            except Exception:
                pass
            gc.collect()

        cleanup_tmp_file(tmp_path)
        raise

    print(f"group 特征写入完成: {path}")
    print(f"行数: {row_count}")
    print(f"列数: {col_count}")


def add_group_features_to_file(
    path: Path,
    model_dir: Path,
    force: bool,
    mask_rt_qvalue_anomaly: bool = False,
):
    add_group_features_to_file_with_model_dirs(
        path=path,
        model_dirs=[model_dir],
        force=force,
        mask_rt_qvalue_anomaly=mask_rt_qvalue_anomaly,
    )


def add_group_features_to_file_ensemble(
    path: Path,
    model_dirs,
    force: bool,
    mask_rt_qvalue_anomaly: bool = False,
):
    add_group_features_to_file_with_model_dirs(
        path=path,
        model_dirs=model_dirs,
        force=force,
        mask_rt_qvalue_anomaly=mask_rt_qvalue_anomaly,
    )


def run_one_file_subprocess(
    path: Path,
    model_dir: Path,
    force: bool,
    mask_rt_qvalue_anomaly: bool = False,
):
    env = os.environ.copy()
    env["POLARS_MAX_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--one-file",
        str(path),
        "--model-dir",
        str(model_dir),
    ]

    if force:
        cmd.append("--force")
    if mask_rt_qvalue_anomaly:
        cmd.append("--mask-rt-qvalue-anomaly")

    result = subprocess.run(cmd, env=env)

    if result.returncode != 0:
        cleanup_tmp_file(path.with_name(path.name + TMP_SUFFIX))
        return str(path), False, f"子进程退出码: {result.returncode}"

    return str(path), True, ""


# ============================================================
# 3. 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Step 6 训练出的 lgbm_table_v1 目录，里面应包含 model.txt 和 feature_columns.json"
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid"],
        choices=["train", "valid"],
        help="要处理 train、valid，还是两者都处理"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="如果文件已有 group_feature_done，是否强制重算覆盖"
    )

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

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="调试用，只处理前 N 个文件"
    )

    parser.add_argument(
        "--one-file",
        type=str,
        default=None,
        help="内部参数：只处理单个文件"
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    model_dir = Path(args.model_dir)

    if args.one_file is not None:
        add_group_features_to_file(
            path=Path(args.one_file),
            model_dir=model_dir,
            force=args.force,
            mask_rt_qvalue_anomaly=args.mask_rt_qvalue_anomaly,
        )
        return

    files = list_split_files(data_root, args.splits)
    if args.only_instrument is not None:
        files = [p for p in files if args.only_instrument in [part.lower() for part in p.parts]]

    if args.max_files is not None:
        files = files[: args.max_files]

    print()
    print(f"总共需要处理 {len(files)} 个 parquet")
    print(f"model_dir: {model_dir}")
    print(f"splits: {args.splits}")
    print(f"only_instrument: {args.only_instrument or 'all'}")
    print(f"force: {args.force}")
    print(f"mask_rt_qvalue_anomaly: {args.mask_rt_qvalue_anomaly}")

    if len(files) == 0:
        print("没有文件需要处理")
        return

    failed = []

    if RUN_EACH_FILE_IN_SUBPROCESS:
        workers = max(1, min(MAX_PARALLEL_FILES, len(files)))

        print(f"并行文件数: {workers}")
        print("可用环境变量 AIPC_GROUP_WORKERS=2/4/8 调整并行数")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(
                    run_one_file_subprocess,
                    path,
                    model_dir,
                    args.force,
                    args.mask_rt_qvalue_anomaly,
                ): path
                for path in files
            }

            for future in tqdm(as_completed(future_to_path), total=len(future_to_path)):
                path = future_to_path[future]

                try:
                    path_str, ok, msg = future.result()

                    if not ok:
                        failed.append(path_str)
                        print(f"处理失败: {path_str}")
                        print(msg)

                except Exception as e:
                    failed.append(str(path))
                    print(f"处理失败: {path}")
                    print(f"错误信息: {e}")
                    cleanup_tmp_file(path.with_name(path.name + TMP_SUFFIX))

                gc.collect()

    else:
        for path in tqdm(files):
            try:
                add_group_features_to_file(
                    path=path,
                    model_dir=model_dir,
                    force=args.force,
                    mask_rt_qvalue_anomaly=args.mask_rt_qvalue_anomaly,
                )
                gc.collect()
            except Exception as e:
                failed.append(str(path))
                print(f"处理失败: {path}")
                print(f"错误信息: {e}")
                cleanup_tmp_file(path.with_name(path.name + TMP_SUFFIX))
                gc.collect()

    print()
    print("group 特征处理完成")

    if failed:
        print("以下文件失败:")
        for f in failed:
            print(f)
    else:
        print("无失败文件")


if __name__ == "__main__":
    main()
