# python src/model/train_lightGBM_v2_groupaware.py \
#   --data-root /root/autodl-tmp/datasets/aipc \
#   --v1-model-dir ~/aipc/models/lgbm_v1_oof \
#   --out-dir ~/aipc/models/lgbm_v2_ranker \
#   --rank-objective lambdarank \
#   --label-gain 0,1,4 \
#   --eval-at 1,3,5,10

from __future__ import annotations

from pathlib import Path
import argparse
import gc
import json
import os
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score
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

ID_COLUMNS_FOR_VALID = [
    "file_id",
    "instrument",
    "scan_number",
    "group_key",
    "peptide_key",
    "precursor_sequence",
]

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

HIGH_CONF_COLUMNS = ["in_fp", "fp_q_value", "spectrum_q"]
GROUP_META_COLUMNS = [
    "group_key",
    "label",
    "lgbm_v1_score",
    "in_fp",
    "fp_q_value",
    "spectrum_q",
]


def instrument_to_id_expr() -> pl.Expr:
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
    for instrument in INSTRUMENTS:
        instrument_dir = split_root / instrument
        if not instrument_dir.exists():
            print(f"目录不存在，跳过: {instrument_dir}")
            continue
        files.extend(
            p for p in sorted(instrument_dir.glob("*.parquet"))
            if ".tmp" not in p.name and not p.name.endswith(".bak")
        )
    return files


def split_files_by_instrument(files: List[Path]) -> Dict[str, List[Path]]:
    grouped = {instrument: [] for instrument in INSTRUMENTS}
    for path in files:
        parts = {part.lower() for part in path.parts}
        matched = None
        for instrument in INSTRUMENTS:
            if path.parent.name.lower() == instrument or instrument in parts or path.name.lower().startswith(instrument):
                matched = instrument
                break
        if matched is None:
            raise RuntimeError(f"无法从路径判断仪器类型: {path}")
        grouped[matched].append(path)
    return grouped


def load_v1_feature_columns(v1_model_dir: Path) -> List[str]:
    feature_file = v1_model_dir / "feature_columns.json"
    if not feature_file.exists():
        raise FileNotFoundError(f"找不到 v1 feature_columns.json: {feature_file}")
    with open(feature_file, "r", encoding="utf-8") as handle:
        feature_cols = json.load(handle)
    if not isinstance(feature_cols, list) or not feature_cols:
        raise RuntimeError(f"v1 feature_columns.json 内容异常: {feature_file}")
    return [str(col) for col in feature_cols]


def resolve_v2_feature_columns(files: List[Path], v1_feature_cols: List[str]) -> List[str]:
    available_cols = set()
    for path in files[: min(len(files), 30)]:
        available_cols.update(get_schema_cols(path))

    missing_group_cols = [col for col in GROUP_FEATURE_COLS if col not in available_cols]
    if missing_group_cols:
        raise RuntimeError(
            "训练 v2 ranker 前必须先生成 v1 group 特征，缺少列: "
            f"{missing_group_cols[:30]}"
        )

    feature_cols = []
    for col in v1_feature_cols:
        if col == "instrument_id" or col in available_cols:
            feature_cols.append(col)
    feature_cols.extend(GROUP_FEATURE_COLS)

    deduped = list(dict.fromkeys(feature_cols))
    if "lgbm_v1_score" not in deduped:
        raise RuntimeError("v2 特征缺少 lgbm_v1_score，请先运行 add_group_feature_oof/test")
    return deduped


def parse_int_list(text: str) -> List[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError(f"empty integer list: {text}")
    return values


def parse_eval_at(text: str) -> List[int]:
    values = parse_int_list(text)
    return sorted(set(values))


def get_schema_cols(path: Path) -> List[str]:
    return pl.scan_parquet(path).collect_schema().names()


def ensure_aux_columns(df: pl.DataFrame) -> pl.DataFrame:
    exprs = []
    if "in_fp" not in df.columns:
        exprs.append(pl.lit(-1).cast(pl.Int8).alias("in_fp"))
    if "fp_q_value" not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Float32).alias("fp_q_value"))
    if "spectrum_q" not in df.columns:
        exprs.append(pl.lit(None).cast(pl.Float32).alias("spectrum_q"))
    if exprs:
        df = df.with_columns(exprs)
    return df


def high_conf_expr() -> pl.Expr:
    return (
        (pl.col("label") == 1)
        & (
            (pl.col("in_fp").fill_null(-1) == 1)
            | (pl.col("fp_q_value").fill_null(999.0) <= 0.01)
            | (pl.col("spectrum_q").fill_null(999.0) <= 0.01)
        )
    )


def read_group_meta(path: Path) -> pl.DataFrame:
    schema = set(get_schema_cols(path))
    required = {"group_key", "label", "lgbm_v1_score"}
    missing = sorted(required - schema)
    if missing:
        raise RuntimeError(f"{path} 缺少 group-aware 采样所需列: {missing}")

    read_cols = [col for col in GROUP_META_COLUMNS if col in schema]
    meta = pl.read_parquet(path, columns=read_cols)
    meta = ensure_aux_columns(meta)
    meta = meta.with_columns([
        pl.col("label").cast(pl.Int8),
        pl.col("lgbm_v1_score").cast(pl.Float32),
        pl.col("in_fp").cast(pl.Int8, strict=False),
        pl.col("fp_q_value").cast(pl.Float32, strict=False),
        pl.col("spectrum_q").cast(pl.Float32, strict=False),
        high_conf_expr().cast(pl.Int8).alias("__high_conf_target"),
    ])
    return meta


def build_group_table(meta: pl.DataFrame, hard_margin: float) -> pl.DataFrame:
    stats = (
        meta.group_by("group_key")
        .agg([
            pl.len().alias("n"),
            pl.col("label").sum().alias("pos"),
            pl.col("__high_conf_target").max().alias("has_high_conf_target"),
            pl.when(pl.col("label") == 0)
            .then(pl.col("lgbm_v1_score"))
            .otherwise(None)
            .max()
            .fill_null(-999.0)
            .alias("max_decoy_score"),
            pl.when(pl.col("label") == 1)
            .then(pl.col("lgbm_v1_score"))
            .otherwise(None)
            .max()
            .fill_null(-999.0)
            .alias("max_target_score"),
        ])
        .with_columns([
            (pl.col("n") - pl.col("pos")).alias("neg"),
        ])
    )

    top1 = (
        meta.sort(["group_key", "lgbm_v1_score"], descending=[False, True])
        .group_by("group_key", maintain_order=True)
        .first()
        .select([
            "group_key",
            pl.col("label").alias("top1_label"),
            pl.col("lgbm_v1_score").alias("top1_score"),
        ])
    )

    group_table = stats.join(top1, on="group_key", how="left")
    group_table = group_table.with_columns([
        (pl.col("max_target_score") - pl.col("max_decoy_score")).alias("target_decoy_margin"),
    ])
    group_table = group_table.with_columns([
        (
            (pl.col("pos") > 0)
            & (pl.col("neg") > 0)
            & ((pl.col("top1_label") == 0) | (pl.col("target_decoy_margin") <= hard_margin))
        ).cast(pl.Int8).alias("is_hard_group"),
        (pl.col("has_high_conf_target") == 1).cast(pl.Int8).alias("is_high_conf_group"),
        (
            pl.col("max_decoy_score")
            + 0.5 * (pl.col("max_decoy_score") - pl.col("max_target_score"))
            + 0.2 * pl.col("n").log1p()
        ).alias("hardness"),
    ])
    return group_table


def take_groups(pool: pl.DataFrame, row_budget: int, seed: int, sort_by_hardness: bool) -> List[str]:
    if row_budget <= 0 or pool.height == 0:
        return []

    if sort_by_hardness and "hardness" in pool.columns:
        pool = pool.sort("hardness", descending=True)
    else:
        pool = pool.sample(fraction=1.0, shuffle=True, seed=seed)

    pool = pool.with_columns(pl.col("n").cum_sum().alias("__cum_rows"))
    selected = pool.filter(pl.col("__cum_rows") <= row_budget)
    if selected.height == 0:
        selected = pool.head(1)
    return selected["group_key"].to_list()


def select_groups_from_file(
    path: Path,
    max_rows_per_file: int,
    seed: int,
    hard_group_frac: float,
    high_conf_group_frac: float,
    hard_margin: float,
    prefer_informative_groups: bool,
) -> Tuple[List[str], pl.DataFrame]:
    meta = read_group_meta(path)
    group_table = build_group_table(meta, hard_margin=hard_margin)

    if prefer_informative_groups:
        informative = group_table.filter((pl.col("pos") > 0) & (pl.col("neg") > 0))
        if informative.height > 0:
            group_table = informative

    hard_budget = int(max_rows_per_file * hard_group_frac)
    high_conf_budget = int(max_rows_per_file * high_conf_group_frac)

    selected: List[str] = []
    selected_set = set()

    hard_pool = group_table.filter(pl.col("is_hard_group") == 1)
    hard_keys = take_groups(hard_pool, hard_budget, seed=seed + 1, sort_by_hardness=True)
    selected.extend(hard_keys)
    selected_set.update(hard_keys)

    high_pool = group_table.filter(
        (pl.col("is_high_conf_group") == 1) & (~pl.col("group_key").is_in(list(selected_set)))
    )
    high_keys = take_groups(high_pool, high_conf_budget, seed=seed + 2, sort_by_hardness=False)
    selected.extend(high_keys)
    selected_set.update(high_keys)

    selected_rows = int(
        group_table.filter(pl.col("group_key").is_in(list(selected_set)))["n"].sum()
    ) if selected_set else 0

    remaining_budget = max_rows_per_file - selected_rows
    normal_pool = group_table.filter(~pl.col("group_key").is_in(list(selected_set)))
    normal_keys = take_groups(normal_pool, remaining_budget, seed=seed + 3, sort_by_hardness=False)
    selected.extend(normal_keys)
    selected_set.update(normal_keys)

    selected_info = group_table.filter(pl.col("group_key").is_in(list(selected_set))).select([
        "group_key",
        "is_hard_group",
        "is_high_conf_group",
    ])
    return selected, selected_info


def read_selected_file_rows(
    path: Path,
    selected_groups: List[str],
    selected_info: pl.DataFrame,
    feature_cols: List[str],
    keep_id_cols: bool,
    high_rel_label: int,
    hard_decoy_weight: float,
    high_target_weight: float,
    hard_target_weight: float,
) -> Optional[pl.DataFrame]:
    if not selected_groups:
        return None

    schema = set(get_schema_cols(path))
    missing_features = [col for col in feature_cols if col != "instrument_id" and col not in schema]
    if missing_features:
        raise RuntimeError(f"{path} 缺少模型特征列: {missing_features[:30]}")

    read_cols = ["group_key", "label"]
    for col in feature_cols:
        read_cols.append("instrument" if col == "instrument_id" else col)
    for col in HIGH_CONF_COLUMNS:
        if col in schema:
            read_cols.append(col)
    if keep_id_cols:
        read_cols.extend([col for col in ID_COLUMNS_FOR_VALID if col in schema])
    read_cols = list(dict.fromkeys(read_cols))

    data_frame = pl.read_parquet(path, columns=read_cols)
    data_frame = data_frame.filter(pl.col("group_key").is_in(selected_groups))
    if data_frame.height == 0:
        return None

    data_frame = ensure_aux_columns(data_frame)
    if "instrument_id" in feature_cols:
        data_frame = data_frame.with_columns([instrument_to_id_expr()])

    data_frame = data_frame.join(selected_info, on="group_key", how="left")

    id_cast_map = {
        "file_id": pl.Utf8,
        "scan_number": pl.Int64,
        "group_key": pl.Utf8,
        "peptide_key": pl.Utf8,
        "precursor_sequence": pl.Utf8,
    }
    cast_id_exprs = [
        pl.col(col_name).cast(dtype)
        for col_name, dtype in id_cast_map.items()
        if col_name in data_frame.columns
    ]
    if cast_id_exprs:
        data_frame = data_frame.with_columns(cast_id_exprs)

    data_frame = data_frame.with_columns([
        pl.col("label").cast(pl.Int8),
        pl.col("is_hard_group").fill_null(0).cast(pl.Int8),
        pl.col("is_high_conf_group").fill_null(0).cast(pl.Int8),
        high_conf_expr().cast(pl.Int8).alias("__high_conf_target"),
    ])

    cast_exprs = []
    for col in feature_cols:
        if col not in data_frame.columns:
            continue
        if col in CATEGORICAL_FEATURES:
            cast_exprs.append(pl.col(col).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(col).cast(pl.Float32))
    if cast_exprs:
        data_frame = data_frame.with_columns(cast_exprs)

    data_frame = data_frame.with_columns([
        (
            pl.when((pl.col("label") == 1) & (pl.col("__high_conf_target") == 1))
            .then(pl.lit(high_rel_label))
            .when(pl.col("label") == 1)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int32)
            .alias("__rel_label")
        ),
        (
            pl.when((pl.col("label") == 0) & (pl.col("is_hard_group") == 1))
            .then(pl.lit(hard_decoy_weight))
            .when((pl.col("label") == 1) & (pl.col("__high_conf_target") == 1))
            .then(pl.lit(high_target_weight))
            .when((pl.col("label") == 1) & (pl.col("is_hard_group") == 1))
            .then(pl.lit(hard_target_weight))
            .when(pl.col("label") == 1)
            .then(pl.lit(1.2))
            .otherwise(pl.lit(1.0))
            .cast(pl.Float32)
            .alias("__weight")
        ),
    ])

    return data_frame


def collect_group_dataset(
    files: List[Path],
    feature_cols: List[str],
    max_total_rows: int,
    max_rows_per_file: int,
    seed: int,
    keep_id_cols: bool,
    args,
) -> Tuple[pl.DataFrame, List[int], Dict]:
    by_instrument = split_files_by_instrument(files)
    target_per_instrument = max_total_rows // len(INSTRUMENTS)
    parts = []
    summary = {
        "target_per_instrument": int(target_per_instrument),
        "by_instrument": {},
    }

    for instrument_index, instrument in enumerate(INSTRUMENTS):
        instrument_files = list(by_instrument.get(instrument, []))
        rng = np.random.default_rng(seed + instrument_index * 1000)
        rng.shuffle(instrument_files)

        instrument_parts = []
        instrument_rows = 0
        attempted_files = 0
        used_files = 0

        print()
        print(f"========== group-aware 采样 {instrument} ==========")
        print(f"files: {len(instrument_files)}")
        print(f"target rows: {target_per_instrument}")

        for file_index, path in enumerate(tqdm(instrument_files)):
            if instrument_rows >= target_per_instrument:
                break

            attempted_files += 1
            remain_rows = target_per_instrument - instrument_rows
            per_file_budget = min(max_rows_per_file, remain_rows)

            try:
                selected_groups, selected_info = select_groups_from_file(
                    path=path,
                    max_rows_per_file=per_file_budget,
                    seed=seed + instrument_index * 100000 + file_index,
                    hard_group_frac=args.hard_group_frac,
                    high_conf_group_frac=args.high_conf_group_frac,
                    hard_margin=args.hard_margin,
                    prefer_informative_groups=args.prefer_informative_groups,
                )
                part = read_selected_file_rows(
                    path=path,
                    selected_groups=selected_groups,
                    selected_info=selected_info,
                    feature_cols=feature_cols,
                    keep_id_cols=keep_id_cols,
                    high_rel_label=args.high_rel_label,
                    hard_decoy_weight=args.hard_decoy_weight,
                    high_target_weight=args.high_target_weight,
                    hard_target_weight=args.hard_target_weight,
                )
                if part is None or part.height == 0:
                    continue

                instrument_parts.append(part)
                instrument_rows += part.height
                used_files += 1
                del part
                gc.collect()

            except Exception as exc:
                print(f"采样失败: {path}")
                print(f"错误: {exc}")

        if instrument_parts:
            instrument_df = pl.concat(instrument_parts, how="vertical", rechunk=False)
            print(f"{instrument} sampled rows: {instrument_df.height}")
            print(instrument_df.group_by("label").len().sort("label"))
            print(instrument_df.group_by(["is_hard_group", "is_high_conf_group"]).len())
            summary["by_instrument"][instrument] = {
                "attempted_files": int(attempted_files),
                "used_files": int(used_files),
                "rows": int(instrument_df.height),
            }
            parts.append(instrument_df)
        else:
            summary["by_instrument"][instrument] = {
                "attempted_files": int(attempted_files),
                "used_files": int(used_files),
                "rows": 0,
            }

        del instrument_parts
        gc.collect()

    if not parts:
        raise RuntimeError("没有采样到任何 group-aware 数据")

    dataset = pl.concat(parts, how="vertical", rechunk=False)
    if dataset.height > max_total_rows:
        dataset = dataset.sample(n=max_total_rows, seed=seed + 999, shuffle=True)

    dataset = dataset.sort("group_key")
    group_sizes = (
        dataset.group_by("group_key", maintain_order=True)
        .agg(pl.len().alias("n"))["n"]
        .to_list()
    )
    group_sizes = [int(value) for value in group_sizes]

    if sum(group_sizes) != dataset.height:
        raise RuntimeError("group_sizes 与 dataset 行数不一致")

    summary["rows"] = int(dataset.height)
    summary["groups"] = int(len(group_sizes))
    summary["label_counts"] = {
        str(row["label"]): int(row["len"])
        for row in dataset.group_by("label").len().iter_rows(named=True)
    }
    summary["rel_label_counts"] = {
        str(row["__rel_label"]): int(row["len"])
        for row in dataset.group_by("__rel_label").len().iter_rows(named=True)
    }

    return dataset, group_sizes, summary


def to_pandas_features(df: pl.DataFrame, feature_cols: List[str]):
    features = df.select(feature_cols).to_pandas()
    features = features.replace([np.inf, -np.inf], np.nan)
    return features


def fdr_metric(
    df: pl.DataFrame,
    pred: np.ndarray,
    fdr_threshold: float = 0.01,
    conservative_tdc: bool = True,
) -> Dict:
    metric_df = df.select([
        col for col in ["label", "peptide_key", "instrument", "group_key"]
        if col in df.columns
    ]).with_columns(pl.Series("score", pred).cast(pl.Float32))
    metric_df = metric_df.sort("score", descending=True)

    labels = metric_df["label"].to_numpy()
    is_target = labels == 1
    is_decoy = labels == 0
    cum_target = np.cumsum(is_target)
    cum_decoy = np.cumsum(is_decoy)
    decoy_for_fdr = np.maximum(cum_decoy, 1) if conservative_tdc else cum_decoy
    fdr = decoy_for_fdr / np.maximum(cum_target, 1)
    q_value = np.minimum.accumulate(fdr[::-1])[::-1]
    accepted = metric_df.with_columns([
        pl.Series("q_value", q_value).cast(pl.Float32),
        pl.Series("accepted", q_value <= fdr_threshold).cast(pl.Boolean),
    ]).filter((pl.col("accepted")) & (pl.col("label") == 1))

    out = {
        "rows": int(metric_df.height),
        "target_rows": int(is_target.sum()),
        "decoy_rows": int(is_decoy.sum()),
        "accepted_target_psm_at_1pct": int(accepted.height),
        "accepted_unique_peptide_at_1pct": int(accepted["peptide_key"].n_unique()) if "peptide_key" in accepted.columns and accepted.height else 0,
        "conservative_tdc": bool(conservative_tdc),
    }

    if "group_key" in metric_df.columns:
        top1 = (
            metric_df.sort(["group_key", "score"], descending=[False, True])
            .group_by("group_key", maintain_order=True)
            .first()
        )
        out["group_count"] = int(top1.height)
        out["top1_target_rate"] = float(top1["label"].mean()) if top1.height else None

    by_instrument = {}
    if "instrument" in accepted.columns:
        for instrument in INSTRUMENTS:
            sub = accepted.filter(pl.col("instrument") == instrument)
            by_instrument[instrument] = {
                "accepted_target_psm_at_1pct": int(sub.height),
                "accepted_unique_peptide_at_1pct": int(sub["peptide_key"].n_unique()) if "peptide_key" in sub.columns and sub.height else 0,
            }
    out["accepted_by_instrument"] = by_instrument
    return out


def train_model(args, feature_cols, train_df, valid_df, train_groups, valid_groups):
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in feature_cols]
    train_features = to_pandas_features(train_df, feature_cols)
    valid_features = to_pandas_features(valid_df, feature_cols)

    if args.mode == "binary":
        train_label = train_df["label"].to_numpy().astype(np.int8)
        valid_label = valid_df["label"].to_numpy().astype(np.int8)
        train_weight = train_df["__weight"].to_numpy().astype(np.float32)
        valid_weight = valid_df["__weight"].to_numpy().astype(np.float32)

        train_set = lgb.Dataset(
            train_features,
            label=train_label,
            weight=train_weight,
            feature_name=feature_cols,
            categorical_feature=categorical_features,
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            valid_features,
            label=valid_label,
            weight=valid_weight,
            feature_name=feature_cols,
            categorical_feature=categorical_features,
            reference=train_set,
            free_raw_data=False,
        )
        params = {
            "objective": "binary",
            "metric": "None",
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
            "seed": args.seed,
            "num_threads": args.num_threads,
        }
    else:
        train_label = train_df["__rel_label"].to_numpy().astype(np.int32)
        valid_label = valid_df["__rel_label"].to_numpy().astype(np.int32)
        label_gain = parse_int_list(args.label_gain)
        max_rel_label = max(int(train_label.max()), int(valid_label.max()))
        if max_rel_label >= len(label_gain):
            raise RuntimeError(
                f"label_gain 长度不足: max label={max_rel_label}, label_gain={label_gain}"
            )

        params = {
            "objective": args.rank_objective,
            "metric": "None",
            "eval_at": parse_eval_at(args.eval_at),
            "label_gain": label_gain,
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
            "lambdarank_truncation_level": args.lambdarank_truncation_level,
            "max_bin": args.max_bin,
            "verbosity": -1,
            "seed": args.seed,
            "num_threads": args.num_threads,
        }

        fdr_stopper = UniquePeptideFdrEarlyStopping(
            valid_features=valid_features,
            valid_meta=FdrEvalMetadata.from_frame(
                valid_df,
                fdr_threshold=args.fdr_threshold,
                conservative_tdc=not args.non_conservative_zero_decoy,
            ),
            stopping_rounds=args.early_stopping_rounds,
            eval_period=args.fdr_eval_period,
            min_delta=args.fdr_min_delta,
        )

        ranker = lgb.LGBMRanker(**params, n_estimators=args.num_boost_round)
        ranker.fit(
            train_features,
            train_label,
            group=train_groups,
            eval_set=[(valid_features, valid_label)],
            eval_group=[valid_groups],
            feature_name=feature_cols,
            categorical_feature=categorical_features,
            callbacks=[fdr_stopper],
        )

        booster = ranker.booster_
        best_iteration = resolve_best_iteration(booster, fdr_stopper, args.num_boost_round)
        valid_pred = booster.predict(valid_features, num_iteration=best_iteration)
        return booster, params, valid_pred, fdr_stopper, best_iteration

    fdr_stopper = UniquePeptideFdrEarlyStopping(
        valid_features=valid_features,
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
        train_set=train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[fdr_stopper],
    )
    best_iteration = resolve_best_iteration(model, fdr_stopper, args.num_boost_round)
    valid_pred = model.predict(valid_features, num_iteration=best_iteration)
    return model, params, valid_pred, fdr_stopper, best_iteration


def safe_classification_metrics(valid_df: pl.DataFrame, pred: np.ndarray) -> Dict:
    labels = valid_df["label"].to_numpy().astype(np.int8)
    out = {}
    try:
        out["auc"] = float(roc_auc_score(labels, pred))
    except Exception:
        out["auc"] = None
    try:
        out["average_precision"] = float(average_precision_score(labels, pred))
    except Exception:
        out["average_precision"] = None
    out["label_mean"] = float(np.mean(labels)) if len(labels) else None
    out["pred_mean"] = float(np.mean(pred)) if len(pred) else None
    out["pred_std"] = float(np.std(pred)) if len(pred) else None
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/datasets/aipc")
    parser.add_argument("--v1-model-dir", type=str, default="~/aipc/models/lgbm_v1_oof")
    parser.add_argument("--out-dir", type=str, default="~/aipc/models/lgbm_v2_groupaware")
    parser.add_argument(
        "--only-instrument",
        choices=INSTRUMENTS,
        default=None,
        help="只使用某一种仪器的数据训练/验证；用于分别训练 mzml/tims/wiff 三个专用模型。",
    )
    parser.add_argument("--mode", choices=["binary", "rank"], default="rank")
    parser.add_argument("--rank-objective", choices=["lambdarank", "rank_xendcg"], default="lambdarank")
    parser.add_argument("--label-gain", type=str, default="0,1,4")
    parser.add_argument("--eval-at", type=str, default="1,3,5,10")

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
    parser.add_argument("--num-threads", type=int, default=max(1, os.cpu_count() or 1))

    parser.add_argument("--hard-group-frac", type=float, default=0.45)
    parser.add_argument("--high-conf-group-frac", type=float, default=0.25)
    parser.add_argument("--hard-margin", type=float, default=0.03)
    parser.add_argument(
        "--prefer-informative-groups",
        action="store_true",
        default=True,
        help="默认开启：ranker 训练优先采样同一 scan 内同时含 target 和 decoy 的 group。",
    )
    parser.add_argument(
        "--allow-uninformative-groups",
        action="store_true",
        help="关闭 informative-only 偏好，允许采样全 target 或全 decoy group。",
    )
    parser.add_argument("--high-rel-label", type=int, default=2)
    parser.add_argument("--hard-decoy-weight", type=float, default=2.5)
    parser.add_argument("--high-target-weight", type=float, default=2.5)
    parser.add_argument("--hard-target-weight", type=float, default=1.5)

    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-data-in-leaf", type=int, default=200)
    parser.add_argument("--feature-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=10.0)
    parser.add_argument("--max-bin", type=int, default=255)
    parser.add_argument("--lambdarank-truncation-level", type=int, default=10)
    args = parser.parse_args()

    if args.allow_uninformative_groups:
        args.prefer_informative_groups = False

    if args.hard_group_frac < 0 or args.high_conf_group_frac < 0 or args.hard_group_frac + args.high_conf_group_frac > 0.95:
        raise ValueError("hard/high-conf group fraction 配置不合理")

    data_root = Path(args.data_root)
    split_root = data_root / "processed_split"
    train_root = split_root / "train"
    valid_root = split_root / "valid"
    v1_model_dir = Path(args.v1_model_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_files = list_split_files(train_root)
    valid_files = list_split_files(valid_root)
    if args.only_instrument is not None:
        train_files = split_files_by_instrument(train_files)[args.only_instrument]
        valid_files = split_files_by_instrument(valid_files)[args.only_instrument]
    if not train_files or not valid_files:
        raise RuntimeError(
            f"train 或 valid 文件为空: only_instrument={args.only_instrument}"
        )

    v1_feature_cols = load_v1_feature_columns(v1_model_dir)
    feature_cols = resolve_v2_feature_columns(train_files, v1_feature_cols)

    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as feature_file:
        json.dump(feature_cols, feature_file, indent=2, ensure_ascii=False)
    with open(out_dir / "group_feature_columns.json", "w", encoding="utf-8") as group_file:
        json.dump(GROUP_FEATURE_COLS, group_file, indent=2, ensure_ascii=False)

    print("========== group-aware v2 ==========")
    print("mode:", args.mode)
    print("only_instrument:", args.only_instrument or "all")
    print("data_root:", data_root)
    print("v1_model_dir:", v1_model_dir)
    print("out_dir:", out_dir)
    print("feature count:", len(feature_cols))

    print("\n========== 采样 train 完整 group ==========")
    train_df, train_groups, train_summary = collect_group_dataset(
        files=train_files,
        feature_cols=feature_cols,
        max_total_rows=args.train_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        seed=args.seed,
        keep_id_cols=False,
        args=args,
    )

    print("\n========== 采样 valid 完整 group ==========")
    valid_df, valid_groups, valid_summary = collect_group_dataset(
        files=valid_files,
        feature_cols=feature_cols,
        max_total_rows=args.valid_max_rows,
        max_rows_per_file=args.max_rows_per_file,
        seed=args.seed + 10000,
        keep_id_cols=True,
        args=args,
    )

    sampling_summary = {"train": train_summary, "valid": valid_summary}
    with open(out_dir / "sampling_summary.json", "w", encoding="utf-8") as summary_file:
        json.dump(sampling_summary, summary_file, indent=2, ensure_ascii=False)

    model, params, valid_pred, fdr_stopper, best_iteration = train_model(
        args=args,
        feature_cols=feature_cols,
        train_df=train_df,
        valid_df=valid_df,
        train_groups=train_groups,
        valid_groups=valid_groups,
    )

    with open(out_dir / "params.json", "w", encoding="utf-8") as params_file:
        json.dump(params, params_file, indent=2, ensure_ascii=False)

    model_path = out_dir / "model.txt"
    model.save_model(str(model_path), num_iteration=best_iteration)

    metrics = {
        "mode": args.mode,
        "best_iteration": int(best_iteration),
        "early_stopping_metric": fdr_stopper.metric_name,
        "early_stopping_best_score": int(fdr_stopper.best_score) if np.isfinite(fdr_stopper.best_score) else None,
        "early_stopping_best_metric": fdr_stopper.best_metric,
        "fdr_threshold": float(args.fdr_threshold),
        "fdr_eval_period": int(args.fdr_eval_period),
        "fdr_conservative_tdc": bool(not args.non_conservative_zero_decoy),
        "train_rows": int(train_df.height),
        "valid_rows": int(valid_df.height),
        "train_groups": int(len(train_groups)),
        "valid_groups": int(len(valid_groups)),
        "classification_metrics_on_binary_label": safe_classification_metrics(valid_df, valid_pred),
        "official_like_sampled_fdr": fdr_metric(
            valid_df,
            valid_pred,
            fdr_threshold=args.fdr_threshold,
            conservative_tdc=not args.non_conservative_zero_decoy,
        ),
        "primary_tuning_objective": "official_like_sampled_fdr.accepted_unique_peptide_at_1pct",
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2, ensure_ascii=False)

    with open(out_dir / "fdr_training_history.json", "w", encoding="utf-8") as history_file:
        json.dump(fdr_stopper.history, history_file, indent=2, ensure_ascii=False)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    valid_pred_df = valid_df.select([
        col for col in ID_COLUMNS_FOR_VALID + ["label", "__rel_label", "__weight", "is_hard_group", "is_high_conf_group"]
        if col in valid_df.columns
    ]).with_columns([
        pl.Series("score", valid_pred).cast(pl.Float32)
    ])
    valid_pred_df.write_parquet(out_dir / "valid_sample_pred.parquet")

    print("\n========== 完成 ==========")
    print("model:", model_path)
    print("metrics:", out_dir / "metrics.json")
    print(json.dumps(metrics["official_like_sampled_fdr"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
