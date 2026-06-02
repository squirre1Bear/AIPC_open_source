# python src/model/train_lightGBM_v3.py \
#   --data-root /root/autodl-tmp/datasets/aipc \
#   --feature-json ~/aipc/models/lgbm_v2/feature_columns.json \
#   --out-dir ~/aipc/models/lgbm_v3_ranker \
#   --train-max-rows 3000000 \
#   --valid-max-rows 1000000 \
#   --max-rows-per-file 80000

from pathlib import Path
import argparse
import json
import os
import gc
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

from fdr_training_metric import (
    FdrEvalMetadata,
    UniquePeptideFdrEarlyStopping,
    resolve_best_iteration,
)


INSTRUMENTS = ["mzml", "tims", "wiff"]

CATEGORICAL_FEATURES = {
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
    "is_lgbm_v1_group_top1",
    "is_lgbm_v1_group_top3",
}

ID_COLS = [
    "file_id",
    "instrument",
    "scan_number",
    "group_key",
    "peptide_key",
    "precursor_sequence",
]


def get_schema_cols(path: Path) -> List[str]:
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


def list_split_files(split_root: Path) -> List[Path]:
    files = []

    for inst in INSTRUMENTS:
        d = split_root / inst

        if not d.exists():
            print(f"目录不存在，跳过: {d}")
            continue

        inst_files = [
            p for p in sorted(d.glob("*.parquet"))
            if ".tmp" not in p.name
            and not p.name.endswith(".bak")
            and not p.name.endswith(".bak_fragment")
        ]

        print(f"{split_root.name}/{inst}: {len(inst_files)} files")
        files.extend(inst_files)

    return files


def load_feature_columns(args) -> List[str]:
    if args.feature_json:
        path = Path(args.feature_json).expanduser()
    else:
        path = Path(args.base_model_dir).expanduser() / "feature_columns.json"

    if not path.exists():
        raise FileNotFoundError(f"找不到 feature_columns.json: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cols = json.load(f)

    if not isinstance(cols, list) or len(cols) == 0:
        raise RuntimeError(f"feature_columns.json 内容异常: {path}")

    print(f"读取特征列: {path}")
    print(f"特征数: {len(cols)}")

    return cols


def check_file_ready(path: Path, feature_cols: List[str]) -> bool:
    cols = set(get_schema_cols(path))

    required = {
        "label",
        "group_key",
        "instrument",
        "aux_feature_done",
        "fragment_feature_done",
        "group_feature_done",
    }

    missing = required - cols

    if missing:
        print(f"跳过未完成文件: {path}, 缺少: {missing}")
        return False

    missing_features = [
        c for c in feature_cols
        if c != "instrument_id" and c not in cols
    ]

    if missing_features:
        print(f"跳过: {path}, 缺少特征: {missing_features[:20]}")
        return False

    return True


def pick_informative_groups(
    path: Path,
    max_rows_per_file: int,
    seed: int,
) -> List[str]:
    """
    只选择同一 group 内同时有 target 和 decoy 的组。
    这些组对 lambdarank 最有训练价值。
    """
    meta = pl.read_parquet(path, columns=["group_key", "label"])

    meta = meta.with_columns([
        pl.col("label").cast(pl.Int8)
    ])

    grp = (
        meta
        .group_by("group_key")
        .agg([
            pl.len().alias("n"),
            pl.col("label").sum().alias("pos"),
        ])
        .with_columns([
            (pl.col("n") - pl.col("pos")).alias("neg")
        ])
        .filter(
            (pl.col("n") >= 2)
            & (pl.col("pos") > 0)
            & (pl.col("neg") > 0)
        )
    )

    if grp.height == 0:
        return []

    grp = grp.sample(fraction=1.0, shuffle=True, seed=seed)

    grp = grp.with_columns([
        pl.col("n").cum_sum().alias("cum_n")
    ])

    selected = grp.filter(pl.col("cum_n") <= max_rows_per_file)

    if selected.height == 0:
        selected = grp.head(1)

    return selected["group_key"].to_list()


def build_relevance_label(df: pl.DataFrame) -> pl.DataFrame:
    """
    构造 ranking label：
      decoy = 0
      普通 target = 1
      高置信 target = 2

    注意：
      in_fp / fp_q_value / spectrum_q 只用于构造训练标签，
      不会作为模型输入特征。
    """
    df = df.with_columns([
        pl.col("label").cast(pl.Int8)
    ])

    high_terms = []

    if "in_fp" in df.columns:
        high_terms.append(pl.col("in_fp").fill_null(-1) == 1)

    if "fp_q_value" in df.columns:
        high_terms.append(pl.col("fp_q_value").fill_null(999.0) <= 0.01)

    if "spectrum_q" in df.columns:
        high_terms.append(pl.col("spectrum_q").fill_null(999.0) <= 0.01)

    if len(high_terms) == 0:
        high_expr = pl.lit(False)
    else:
        high_expr = high_terms[0]
        for e in high_terms[1:]:
            high_expr = high_expr | e

    df = df.with_columns([
        (
            pl.when((pl.col("label") == 1) & high_expr)
            .then(pl.lit(2))
            .when(pl.col("label") == 1)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int32)
            .alias("__rel_label")
        )
    ])

    return df


def trim_by_group(df: pl.DataFrame, max_rows: int) -> pl.DataFrame:
    if df.height <= max_rows:
        return df

    grp = (
        df
        .group_by("group_key", maintain_order=True)
        .agg(pl.len().alias("n"))
        .with_columns(pl.col("n").cum_sum().alias("cum_n"))
    )

    selected = grp.filter(pl.col("cum_n") <= max_rows)

    if selected.height == 0:
        selected = grp.head(1)

    keys = selected["group_key"].to_list()

    return df.filter(pl.col("group_key").is_in(keys))


def prepare_file_dataframe(
    path: Path,
    feature_cols: List[str],
    max_rows_per_file: int,
    seed: int,
    keep_id_cols: bool,
) -> Tuple[pl.DataFrame, List[int]]:
    """
    读取一个 parquet，按完整 group 采样，并返回 group_sizes。
    """
    if not check_file_ready(path, feature_cols):
        return None, []

    selected_groups = pick_informative_groups(
        path=path,
        max_rows_per_file=max_rows_per_file,
        seed=seed,
    )

    if len(selected_groups) == 0:
        return None, []

    schema_cols = set(get_schema_cols(path))

    read_cols = ["group_key", "label"]

    for c in feature_cols:
        if c == "instrument_id":
            read_cols.append("instrument")
        else:
            read_cols.append(c)

    aux_cols = [
        "in_fp",
        "fp_q_value",
        "spectrum_q",
        "sage_discriminant_score",
    ]

    for c in aux_cols:
        if c in schema_cols:
            read_cols.append(c)

    if keep_id_cols:
        for c in ID_COLS:
            if c in schema_cols:
                read_cols.append(c)

    read_cols = list(dict.fromkeys(read_cols))

    df = pl.read_parquet(path, columns=read_cols)
    df = df.filter(pl.col("group_key").is_in(selected_groups))

    if df.height == 0:
        return None, []

    if "instrument_id" in feature_cols:
        df = df.with_columns([instrument_to_id_expr()])

    df = build_relevance_label(df)

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

    df = df.sort("group_key")

    group_sizes = (
        df
        .group_by("group_key", maintain_order=True)
        .agg(pl.len().alias("n"))
        ["n"]
        .to_list()
    )

    return df, [int(x) for x in group_sizes]


def collect_rank_dataset(
    files: List[Path],
    feature_cols: List[str],
    max_total_rows: int,
    max_rows_per_file: int,
    seed: int,
    keep_id_cols: bool,
) -> Tuple[pl.DataFrame, List[int]]:
    parts = []
    all_groups = []
    total_rows = 0

    for i, path in enumerate(tqdm(files)):
        if total_rows >= max_total_rows:
            break

        remain = max_total_rows - total_rows
        per_file = min(max_rows_per_file, remain)

        try:
            part, group_sizes = prepare_file_dataframe(
                path=path,
                feature_cols=feature_cols,
                max_rows_per_file=per_file,
                seed=seed + i,
                keep_id_cols=keep_id_cols,
            )

            if part is None or part.height == 0:
                continue

            if part.height > remain:
                part = trim_by_group(part, remain)
                group_sizes = (
                    part
                    .group_by("group_key", maintain_order=True)
                    .agg(pl.len().alias("n"))
                    ["n"]
                    .to_list()
                )
                group_sizes = [int(x) for x in group_sizes]

            parts.append(part)
            all_groups.extend(group_sizes)
            total_rows += part.height

            del part
            gc.collect()

        except Exception as e:
            print(f"文件采样失败: {path}")
            print(f"错误: {e}")

    if not parts:
        raise RuntimeError("没有采样到任何可训练 group")

    df = pl.concat(parts, how="vertical", rechunk=False)

    if sum(all_groups) != df.height:
        raise RuntimeError(
            f"group size 总和与 df 行数不一致: sum(group)={sum(all_groups)}, rows={df.height}"
        )

    return df, all_groups


def polars_to_lgb_xy(df: pl.DataFrame, feature_cols: List[str]):
    y = df["__rel_label"].to_numpy().astype(np.int32)

    X = df.select(feature_cols).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)

    return X, y


def compute_sampled_fdr(valid_df: pl.DataFrame, pred: np.ndarray, conservative_tdc: bool = True) -> Dict:
    df = valid_df.select([
        "label",
        "instrument",
        "group_key",
        "peptide_key",
    ]).with_columns([
        pl.Series("score", pred).cast(pl.Float32)
    ])

    df = df.sort("score", descending=True)

    labels = df["label"].to_numpy()

    is_target = labels == 1
    is_decoy = labels == 0

    cum_target = np.cumsum(is_target)
    cum_decoy = np.cumsum(is_decoy)

    decoy_for_fdr = np.maximum(cum_decoy, 1) if conservative_tdc else cum_decoy
    fdr = decoy_for_fdr / np.maximum(cum_target, 1)
    q = np.minimum.accumulate(fdr[::-1])[::-1]

    keep = q <= 0.01

    accepted = df.with_columns([
        pl.Series("q_value", q).cast(pl.Float32),
        pl.Series("accepted_1pct", keep).cast(pl.Int8),
    ]).filter(
        (pl.col("accepted_1pct") == 1)
        & (pl.col("label") == 1)
    )

    top1 = (
        df
        .sort(["group_key", "score"], descending=[False, True])
        .group_by("group_key", maintain_order=True)
        .first()
    )

    metric = {
        "sample_rows": int(df.height),
        "sample_groups": int(df["group_key"].n_unique()),
        "top1_target_rate": float(top1["label"].mean()) if top1.height > 0 else None,
        "accepted_target_psm_at_1pct": int(accepted.height),
        "accepted_unique_peptide_at_1pct": int(accepted["peptide_key"].n_unique()) if accepted.height > 0 else 0,
        "conservative_tdc": bool(conservative_tdc),
        "by_instrument": {},
    }

    for inst in INSTRUMENTS:
        sub = accepted.filter(pl.col("instrument") == inst)
        metric["by_instrument"][inst] = {
            "accepted_target_psm_at_1pct": int(sub.height),
            "accepted_unique_peptide_at_1pct": int(sub["peptide_key"].n_unique()) if sub.height > 0 else 0,
        }

    return metric


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/datasets/aipc")

    parser.add_argument(
        "--base-model-dir",
        type=str,
        default="~/aipc/models/lgbm_v2",
        help="用于读取 feature_columns.json。也可以用 --feature-json 显式指定。"
    )

    parser.add_argument(
        "--feature-json",
        type=str,
        default="",
        help="显式指定 feature_columns.json。如果为空，则使用 base-model-dir/feature_columns.json"
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default="~/aipc/models/lgbm_v3_ranker",
    )

    parser.add_argument("--train-max-rows", type=int, default=3_000_000)
    parser.add_argument("--valid-max-rows", type=int, default=1_000_000)
    parser.add_argument("--max-rows-per-file", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--num-boost-round", type=int, default=3000)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--fdr-threshold", type=float, default=0.01)
    parser.add_argument("--fdr-eval-period", type=int, default=50)
    parser.add_argument("--fdr-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--non-conservative-zero-decoy",
        action="store_true",
        help="使用 cum_decoy/cum_target；默认使用 max(cum_decoy, 1)/cum_target，更贴近官方工具。",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_root = data_root / "processed_split"

    train_root = split_root / "train"
    valid_root = split_root / "valid"

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("========== 路径 ==========")
    print("train_root:", train_root)
    print("valid_root:", valid_root)
    print("out_dir:", out_dir)

    feature_cols = load_feature_columns(args)

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)

    train_files = list_split_files(train_root)
    valid_files = list_split_files(valid_root)

    print(f"train files: {len(train_files)}")
    print(f"valid files: {len(valid_files)}")

    print("\n========== 采样完整 group 训练集 ==========")

    train_df, train_groups = collect_rank_dataset(
        files=train_files,
        feature_cols=feature_cols,
        max_total_rows=args.train_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        seed=args.seed,
        keep_id_cols=False,
    )

    print("train rows:", train_df.height)
    print("train groups:", len(train_groups))
    print(train_df.group_by("__rel_label").len())

    print("\n========== 采样完整 group 验证集 ==========")

    valid_df, valid_groups = collect_rank_dataset(
        files=valid_files,
        feature_cols=feature_cols,
        max_total_rows=args.valid_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        seed=args.seed + 10000,
        keep_id_cols=True,
    )

    print("valid rows:", valid_df.height)
    print("valid groups:", len(valid_groups))
    print(valid_df.group_by("__rel_label").len())

    print("\n========== 转换 LightGBM 输入 ==========")

    X_train, y_train = polars_to_lgb_xy(train_df, feature_cols)
    X_valid, y_valid = polars_to_lgb_xy(valid_df, feature_cols)

    categorical_features = [
        c for c in feature_cols
        if c in CATEGORICAL_FEATURES
    ]

    print("X_train:", X_train.shape)
    print("X_valid:", X_valid.shape)
    print("categorical_features:", categorical_features)

    lgb_train = lgb.Dataset(
        X_train,
        label=y_train,
        group=train_groups,
        feature_name=feature_cols,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )

    lgb_valid = lgb.Dataset(
        X_valid,
        label=y_valid,
        group=valid_groups,
        feature_name=feature_cols,
        categorical_feature=categorical_features,
        reference=lgb_train,
        free_raw_data=False,
    )

    params = {
        "objective": "lambdarank",
        "metric": "None",
        "eval_at": [1, 3, 5],

        "label_gain": [0, 1, 2],

        "learning_rate": 0.03,
        "num_leaves": 63,
        "max_depth": -1,
        "min_data_in_leaf": 100,

        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,

        "lambda_l1": 0.0,
        "lambda_l2": 10.0,

        "lambdarank_truncation_level": 10,

        "max_bin": 255,
        "verbosity": -1,
        "seed": args.seed,
        "num_threads": max(1, os.cpu_count() or 1),
    }

    with open(out_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)

    print("\n========== 开始训练 LightGBM LambdaRank ==========")

    fdr_stopper = UniquePeptideFdrEarlyStopping(
        valid_features=X_valid,
        valid_meta=FdrEvalMetadata.from_frame(
            valid_df,
            fdr_threshold=args.fdr_threshold,
            conservative_tdc=not args.non_conservative_zero_decoy,
        ),
        stopping_rounds=args.early_stopping_rounds,
        eval_period=args.fdr_eval_period,
        min_delta=args.fdr_min_delta,
    )

    model = lgb.train(
        params=params,
        train_set=lgb_train,
        num_boost_round=args.num_boost_round,
        valid_sets=[lgb_valid],
        valid_names=["valid"],
        callbacks=[fdr_stopper],
    )

    model_path = out_dir / "model.txt"
    best_iteration = resolve_best_iteration(model, fdr_stopper, args.num_boost_round)
    model.save_model(str(model_path), num_iteration=best_iteration)

    print("模型已保存:", model_path)
    print("best_iteration:", best_iteration)

    print("\n========== 采样验证集评估 ==========")

    valid_pred = model.predict(
        X_valid,
        num_iteration=best_iteration,
    )

    sampled_metric = compute_sampled_fdr(
        valid_df,
        valid_pred,
        conservative_tdc=not args.non_conservative_zero_decoy,
    )

    metrics = {
        "best_iteration": int(best_iteration),
        "early_stopping_metric": fdr_stopper.metric_name,
        "early_stopping_best_score": int(fdr_stopper.best_score) if np.isfinite(fdr_stopper.best_score) else None,
        "early_stopping_best_metric": fdr_stopper.best_metric,
        "fdr_threshold": float(args.fdr_threshold),
        "fdr_eval_period": int(args.fdr_eval_period),
        "fdr_conservative_tdc": bool(not args.non_conservative_zero_decoy),
        "train_rows": int(len(y_train)),
        "valid_rows": int(len(y_valid)),
        "train_groups": int(len(train_groups)),
        "valid_groups": int(len(valid_groups)),
        "sampled_metric": sampled_metric,
    }

    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(out_dir / "fdr_training_history.json", "w", encoding="utf-8") as f:
        json.dump(fdr_stopper.history, f, indent=2, ensure_ascii=False)

    imp_gain = model.feature_importance(importance_type="gain")
    imp_split = model.feature_importance(importance_type="split")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": imp_gain,
        "importance_split": imp_split,
    }).sort_values("importance_gain", ascending=False)

    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    valid_pred_df = valid_df.select([
        c for c in ID_COLS + ["label", "__rel_label"]
        if c in valid_df.columns
    ]).with_columns([
        pl.Series("score", valid_pred).cast(pl.Float32)
    ])

    valid_pred_df.write_parquet(out_dir / "valid_sample_pred.parquet")

    print("特征重要性:", out_dir / "feature_importance.csv")
    print("验证采样预测:", out_dir / "valid_sample_pred.parquet")
    print("v3 ranker 训练完成")


if __name__ == "__main__":
    main()
