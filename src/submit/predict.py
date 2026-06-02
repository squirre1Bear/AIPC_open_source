# python src/submit/predict.py \
#   ~/aipc/models/lgbm_v1_oof \
#   --parquet_dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
#   --out_path /root/autodl-tmp/datasets/aipc/submissions/lgbm_v1_oof

  # python src/submit/predict.py \
  #   ~/aipc/models/lgbm_v2_binary \
  #   --parquet_dir $DATA/processed/bas_merged \
  #   --out_path /root/autodl-tmp/datasets/aipc/submissions/lgbm_v2_binary
from __future__ import annotations

import os
import argparse
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from tqdm import tqdm


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)


def mkdir_p(dirs: str):
    """
    保持官方脚本风格：
    如果输出目录不存在，则创建。
    """
    if not os.path.exists(dirs):
        os.makedirs(dirs, exist_ok=True)
    return True, "OK"


def load_lgbm_model_and_features(model_path: str):
    """
    加载 LightGBM 模型和 feature_columns.json。

    兼容两种传参：
    1. 传模型目录：
       ~/aipc/models/lgbm_v1
    2. 直接传 model.txt：
       ~/aipc/models/lgbm_v1/model.txt
    """
    model_path = Path(model_path).expanduser().resolve()

    if model_path.is_dir():
        model_file = model_path / "model.txt"
        feature_file = model_path / "feature_columns.json"
    else:
        model_file = model_path
        feature_file = model_path.parent / "feature_columns.json"

    if not model_file.exists():
        raise FileNotFoundError(f"找不到 LightGBM 模型文件: {model_file}")

    if not feature_file.exists():
        raise FileNotFoundError(f"找不到 feature_columns.json: {feature_file}")

    logger.info(f"Loading LightGBM model: {model_file}")
    model = lgb.Booster(model_file=str(model_file))

    logger.info(f"Loading feature columns: {feature_file}")
    with open(feature_file, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    if not isinstance(feature_cols, list) or len(feature_cols) == 0:
        raise RuntimeError(f"feature_columns.json 内容异常: {feature_file}")

    logger.info(f"Loaded {len(feature_cols)} feature columns")

    return model, feature_cols


def instrument_to_id_expr():
    """
    与训练脚本保持一致：
    mzml -> 0
    tims -> 1
    wiff -> 2
    unknown -> -1
    """
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
    """
    如果 parquet 中缺少 instrument，则从文件名推断。
    测试文件名一般类似：
      mzml_bas_a_7.parquet
      tims_bas_b_0.parquet
      wiff_bas_a_4.parquet
    """
    name = file_name.lower()

    if name.startswith("mzml"):
        return "mzml"
    if name.startswith("tims"):
        return "tims"
    if name.startswith("wiff"):
        return "wiff"

    return "unknown"


def normalize_test_columns(df: pl.DataFrame, file_name: str) -> pl.DataFrame:
    """
    对测试集字段做最小必要统一。

    官方原脚本会：
    - 若无 modified_sequence，则 precursor_sequence -> modified_sequence
    - 若无 index，则生成临时 index
    - 若无 weight，则补 1.0

    这里保留类似逻辑，但 LightGBM 不需要 modified_sequence / weight。
    为了兼容你的特征工程，额外处理：
    - delta_rt_model -> delta_rt
    - 补 instrument
    - 补 has_ion_mobility
    - 补 group_key / peptide_key
    """

    # 测试原始字段可能叫 delta_rt_model，这里统一为 delta_rt
    if "delta_rt_model" in df.columns and "delta_rt" not in df.columns:
        df = df.rename({"delta_rt_model": "delta_rt"})

    if "index" not in df.columns:
        raise RuntimeError(
            f"{file_name} 缺少官方 index，不能生成提交文件。"
            f"请先从原始 bas_data 中恢复 index。"
        )

    # 如果缺少 instrument，则从文件名推断
    if "instrument" not in df.columns:
        inst = infer_instrument_from_filename(file_name)
        df = df.with_columns(
            pl.lit(inst).alias("instrument")
        )

    # 如果缺少 has_ion_mobility，则根据 instrument 补齐
    if "has_ion_mobility" not in df.columns:
        df = df.with_columns(
            pl.when(pl.col("instrument") == "tims")
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int8)
            .alias("has_ion_mobility")
        )

    # 如果缺少 peptide_key，则使用 precursor_sequence
    if "peptide_key" not in df.columns and "precursor_sequence" in df.columns:
        df = df.with_columns(
            pl.col("precursor_sequence").alias("peptide_key")
        )

    # 如果缺少 group_key，则尝试构造
    if "group_key" not in df.columns:
        if "scan_number" not in df.columns:
            raise RuntimeError(f"{file_name} 缺少 group_key 且缺少 scan_number，无法构造 group_key")

        df = df.with_columns(
            (
                pl.col("instrument").cast(pl.Utf8)
                + pl.lit("_")
                + pl.lit(file_name)
                + pl.lit("_")
                + pl.col("scan_number").cast(pl.Utf8)
            ).alias("group_key")
        )

    return df


def prepare_feature_df(df: pl.DataFrame, feature_cols: list[str], file_name: str) -> pd.DataFrame:
    """
    从 parquet DataFrame 中取出模型需要的特征列。
    若缺少训练时需要的特征，直接报错，避免静默生成错误提交。
    """

    df = normalize_test_columns(df, file_name)

    # 如果训练特征里有 instrument_id，则根据 instrument 生成
    if "instrument_id" in feature_cols:
        if "instrument" not in df.columns:
            raise RuntimeError(f"{file_name} 缺少 instrument，无法生成 instrument_id")
        df = df.with_columns([instrument_to_id_expr()])

    missing = [
        c for c in feature_cols
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{file_name} 缺少模型需要的特征列，共 {len(missing)} 个，"
            f"前 30 个为: {missing[:30]}\n"
            f"说明：请先对测试集运行 process_bas_test.py、add_feature_1.py、add_feature_2.py。"
        )

    # 只取模型需要的列
    df_feat = df.select(feature_cols)

    # 类型统一
    categorical_like = {
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

    cast_exprs = []
    for c in feature_cols:
        if c in categorical_like:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))

    df_feat = df_feat.with_columns(cast_exprs)

    X = df_feat.to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def predict_one_file(
    model: lgb.Booster,
    feature_cols: list[str],
    file_path: str,
    out_path: str,
):
    """
    对单个 parquet 文件推理，生成无表头 csv：
      index,score
    与官方脚本的单文件输出格式保持一致。
    """
    file_path = Path(file_path)
    file_name = file_path.stem

    logger.info(f"parse: {file_path}, file_name: {file_name}")

    out_file = Path(out_path) / f"{file_name}_pred.csv"

    # 保持官方逻辑：如果已存在，则跳过
    if out_file.exists():
        logger.info(f"out_file: {out_file} exist!!!!!!!!!!!!")
        return

    df = pl.read_parquet(file_path)

    if "index" not in df.columns:
        logger.warning(f"{file_name} 缺少 index，将生成临时 index。")
        df = df.with_columns(
            pl.arange(0, pl.len()).cast(pl.Int64).alias("index")
        )

    index = df["index"].to_numpy()

    X = prepare_feature_df(df, feature_cols, file_path.name)

    best_iter = getattr(model, "best_iteration", None)

    if best_iter is not None and best_iter > 0:
        pred = model.predict(X, num_iteration=best_iter)
    else:
        pred = model.predict(X)

    pred = np.asarray(pred, dtype=np.float32)

    out_df = pd.DataFrame({
        "index": index.astype(np.int64),
        "score": pred,
    })

    # 官方单文件输出没有表头，这里保持一致
    out_df.to_csv(out_file, mode="w", header=False, index=False)

    logger.info(f"saved: {out_file}, rows={len(out_df):,}")


def merge_pred_files(out_path: str):
    """
    合并 out_path 下所有 *_pred.csv 为 all_pred.tsv。
    与官方脚本保持一致：
      - 读取无表头 pred.csv
      - 合并
      - 输出 all_pred.tsv，带表头 index score
    """
    out_path = Path(out_path)

    pred_files = sorted(out_path.glob("*_pred.csv"))

    if len(pred_files) == 0:
        logger.info("No pred files found. Skip merging.")
        return

    df_list = []

    for f in pred_files:
        try:
            tmp_df = pd.read_csv(f, header=None, names=["index", "score"])
            df_list.append(tmp_df)
        except Exception as e:
            logger.info(f"Error reading {f}: {e}")

    if len(df_list) == 0:
        logger.info("No valid pred files to merge.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)

    merged_path = out_path / "all_pred.tsv"

    merged_df.to_csv(
        merged_path,
        sep="\t",
        index=False,
    )

    logger.info(f"Merged TSV saved to: {merged_path}")
    logger.info(f"Rows: {len(merged_df):,}")
    logger.info(f"Unique index: {merged_df['index'].nunique():,}")


def check_submission_file(out_path: str, expected_rows: int | None):
    """
    可选检查 all_pred.tsv。
    """
    if expected_rows is None:
        return

    all_pred_path = Path(out_path) / "all_pred.tsv"

    if not all_pred_path.exists():
        logger.warning(f"all_pred.tsv 不存在，跳过检查: {all_pred_path}")
        return

    df = pd.read_csv(all_pred_path, sep="\t")

    logger.info("========== submission check ==========")
    logger.info(f"columns: {list(df.columns)}")
    logger.info(f"rows: {len(df):,}")
    logger.info(f"unique index: {df['index'].nunique():,}")
    logger.info(f"score null: {df['score'].isna().sum():,}")
    logger.info(f"score min: {df['score'].min()}")
    logger.info(f"score max: {df['score'].max()}")
    logger.info(f"score mean: {df['score'].mean()}")

    assert list(df.columns) == ["index", "score"]
    assert len(df) == expected_rows
    assert df["index"].nunique() == expected_rows
    assert df["score"].notna().all()

    finite_mask = np.isfinite(df["score"].to_numpy())
    assert finite_mask.all()

    logger.info("OK: all_pred.tsv format looks valid")


def main():
    parser = argparse.ArgumentParser()

    # 保持官方风格：第一个位置参数是模型路径
    parser.add_argument(
        "model_path",
        help="LightGBM v1 模型目录或 model.txt 路径，例如 ~/aipc/models/lgbm_v1"
    )

    parser.add_argument(
        "--parquet_dir",
        required=True,
        help="已经完成特征处理的测试 parquet 目录"
    )

    # 保留官方参数，但 LightGBM 不使用 config
    parser.add_argument(
        "--config",
        default="",
        help="保留官方接口；LightGBM 版本不使用"
    )

    parser.add_argument(
        "--out_path",
        default="",
        help="输出目录；若为空，则输出到 parquet_dir"
    )

    parser.add_argument(
        "--expected_rows",
        type=int,
        default=10_768_114,
        help="Basic 测试集期望行数；不想检查可设为 -1"
    )

    args = parser.parse_args()

    logger.info("Initializing LightGBM inference.")
    logger.info(f"inference use model path: {args.model_path}")
    logger.info(f"parquet_dir: {args.parquet_dir}")

    model, feature_cols = load_lgbm_model_and_features(args.model_path)

    if args.out_path == "":
        out_path = args.parquet_dir
    else:
        out_path = args.out_path

    mkdir_p(out_path)

    logger.info(f"**************out_path: {out_path}**************************")

    parquet_dir = Path(args.parquet_dir)

    data_path_list = [
        parquet_dir / f
        for f in os.listdir(parquet_dir)
        if f.endswith(".parquet")
    ]

    data_path_list = sorted(data_path_list)

    logger.info(f"found parquet files: {len(data_path_list)}")

    for file_path in tqdm(data_path_list):
        try:
            predict_one_file(
                model=model,
                feature_cols=feature_cols,
                file_path=str(file_path),
                out_path=out_path,
            )
        except Exception as e:
            logger.exception(f"load {file_path} error: {e}!!!")
            continue

    merge_pred_files(out_path)

    if args.expected_rows is not None and args.expected_rows > 0:
        check_submission_file(out_path, args.expected_rows)


if __name__ == "__main__":
    main()
