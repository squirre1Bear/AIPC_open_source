#python src/eval/predict_bas_lgbm_v2.py \
#   --parquet-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
#   --model-dir ~aipc/models/lgbm_v1 \
#   --out-dir /root/autodl-tmp/datasets/aipc/submissions/first_submit

#python src/eval/predict_bas_lgbm_v2.py \
#   --parquet-dir /root/autodl-tmp/datasets/aipc/processed/bas_merged \
#   --model-dir ~aipc/models/lgbm_table_v2_group \
#   --out-dir /root/autodl-tmp/datasets/aipc/submissions/first_submit

from pathlib import Path
import argparse
import json
import os
import gc

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from tqdm import tqdm


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


def load_model_and_features(model_dir: Path):
    model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    with open(model_dir / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return model, feature_cols


def prepare_df(path: Path, feature_cols):
    schema = set(pl.scan_parquet(path).collect_schema().names())

    required = ["index", "group_feature_done"]
    missing = [c for c in required if c not in schema]
    if missing:
        raise RuntimeError(f"{path} 缺少: {missing}")

    read_cols = ["index"]

    for c in feature_cols:
        if c == "instrument_id":
            read_cols.append("instrument")
        else:
            if c not in schema:
                raise RuntimeError(f"{path} 缺少模型特征列: {c}")
            read_cols.append(c)

    read_cols = list(dict.fromkeys(read_cols))

    df = pl.read_parquet(path, columns=read_cols)

    if "instrument_id" in feature_cols:
        df = df.with_columns([instrument_to_id_expr()])

    cat_cols = {
        "instrument_id",
        "charge",
        "has_ion_mobility",
        "has_mod",
        "parse_ok",
        "fragment_parse_ok",
        "best_isotope_offset",
        "is_lgbm_v1_group_top1",
        "is_lgbm_v1_group_top3",
    }

    cast_exprs = []
    for c in feature_cols:
        if c in cat_cols:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))

    df = df.with_columns(cast_exprs)
    return df


def predict_one_file(path: Path, model, feature_cols, out_dir: Path):
    df = prepare_df(path, feature_cols)

    X = df.select(feature_cols).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and best_iter > 0:
        pred = model.predict(X, num_iteration=best_iter)
    else:
        pred = model.predict(X)

    out = pl.DataFrame({
        "index": df["index"],
        "score": np.asarray(pred, dtype=np.float32),
    })

    out_path = out_dir / f"{path.stem}_pred.tsv"
    out.write_csv(out_path, separator="\t")

    print(f"保存预测: {out_path}, rows={out.height}")

    del df, X, out
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", type=str, default="/root/autodl-tmp/datasets/aipc/test_processed/bas_merged")
    parser.add_argument("--model-dir", type=str, default="/root/autodl-tmp/datasets/aipc/models/lgbm_table_v2_group")
    parser.add_argument("--out-dir", type=str, default="/root/autodl-tmp/datasets/aipc/submissions/first_submit")
    args = parser.parse_args()

    parquet_dir = Path(args.parquet_dir)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, feature_cols = load_model_and_features(model_dir)

    files = sorted(parquet_dir.glob("*.parquet"))
    print("test files:", len(files))

    for path in tqdm(files):
        predict_one_file(path, model, feature_cols, out_dir)

    pred_files = sorted(out_dir.glob("*_pred.tsv"))
    print("合并预测文件:", len(pred_files))

    dfs = []
    for f in pred_files:
        dfs.append(pl.read_csv(f, separator="\t"))

    all_pred = pl.concat(dfs, how="vertical")
    all_pred = all_pred.select([
        pl.col("index").cast(pl.Int64),
        pl.col("score").cast(pl.Float64),
    ])

    all_path = out_dir / "all_pred.tsv"
    all_pred.write_csv(all_path, separator="\t")

    print("保存 all_pred:", all_path)
    print("rows:", all_pred.height)
    print("unique index:", all_pred["index"].n_unique())


if __name__ == "__main__":
    main()