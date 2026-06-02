# python src/eval/compare_models.py \
#   --data-root /root/autodl-tmp/datasets/aipc \
#   --v1-model-dir ~/aipc/models/lgbm_v1 \
#   --v2-model-dir ~/aipc/models/lgbm_v2 \
#   --out-dir ~/aipc/eval/compare_v1_v2_full_valid

from pathlib import Path
import argparse
import json
import gc

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score


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


def load_model(model_dir: Path):
    model = lgb.Booster(model_file=str(model_dir / "model.txt"))
    with open(model_dir / "feature_columns.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return model, feature_cols


def list_valid_files(data_root: Path):
    valid_root = data_root / "processed_split" / "valid"
    files = []

    for inst in INSTRUMENTS:
        d = valid_root / inst
        if not d.exists():
            continue

        fs = [
            p for p in sorted(d.glob("*.parquet"))
            if ".tmp" not in p.name
            and not p.name.endswith(".bak")
            and not p.name.endswith(".bak_fragment")
        ]

        print(f"valid/{inst}: {len(fs)} files")
        files.extend(fs)

    return files


def get_schema(path: Path):
    return set(pl.scan_parquet(path).collect_schema().names())


def prepare_features(df: pl.DataFrame, feature_cols):
    if "instrument_id" in feature_cols:
        df = df.with_columns([instrument_to_id_expr()])

    cast_exprs = []

    for c in feature_cols:
        if c in CATEGORICAL_FEATURES:
            cast_exprs.append(pl.col(c).cast(pl.Int16))
        else:
            cast_exprs.append(pl.col(c).cast(pl.Float32))

    if cast_exprs:
        df = df.with_columns(cast_exprs)

    X = df.select(feature_cols).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def predict_model_on_file(path: Path, model, feature_cols):
    schema = get_schema(path)

    read_cols = [
        "label",
        "instrument",
        "group_key",
        "peptide_key",
    ]

    for c in feature_cols:
        if c == "instrument_id":
            read_cols.append("instrument")
        else:
            if c not in schema:
                raise RuntimeError(f"{path} 缺少特征列: {c}")
            read_cols.append(c)

    read_cols = list(dict.fromkeys(read_cols))

    df = pl.read_parquet(path, columns=read_cols)
    X = prepare_features(df, feature_cols)

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None and best_iter > 0:
        pred = model.predict(X, num_iteration=best_iter)
    else:
        pred = model.predict(X)

    out = df.select([
        "label",
        "instrument",
        "group_key",
        "peptide_key",
    ]).with_columns([
        pl.Series("score", np.asarray(pred, dtype=np.float32))
    ])

    del df, X
    gc.collect()

    return out


def compute_metrics(df: pl.DataFrame, score_col: str):
    df = df.select([
        "label",
        "instrument",
        "group_key",
        "peptide_key",
        score_col,
    ]).rename({
        score_col: "score"
    })

    labels = df["label"].to_numpy()
    scores = df["score"].to_numpy()

    metrics = {}

    try:
        metrics["auc"] = float(roc_auc_score(labels, scores))
    except Exception:
        metrics["auc"] = None

    try:
        metrics["average_precision"] = float(average_precision_score(labels, scores))
    except Exception:
        metrics["average_precision"] = None

    sorted_df = df.sort("score", descending=True)

    sorted_labels = sorted_df["label"].to_numpy()
    is_target = sorted_labels == 1
    is_decoy = sorted_labels == 0

    cum_target = np.cumsum(is_target)
    cum_decoy = np.cumsum(is_decoy)

    fdr = cum_decoy / np.maximum(cum_target, 1)
    q_value = np.minimum.accumulate(fdr[::-1])[::-1]

    keep = q_value <= 0.01

    accepted = sorted_df.with_columns([
        pl.Series("q_value", q_value).cast(pl.Float32),
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

    metrics.update({
        "rows": int(df.height),
        "target_rows": int((df["label"] == 1).sum()),
        "decoy_rows": int((df["label"] == 0).sum()),
        "group_count": int(df["group_key"].n_unique()),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "top1_target_rate": float(top1["label"].mean()) if top1.height > 0 else None,
        "accepted_target_psm_at_1pct": int(accepted.height),
        "accepted_unique_peptide_at_1pct": int(accepted["peptide_key"].n_unique()) if accepted.height > 0 else 0,
        "by_instrument": {},
    })

    for inst in INSTRUMENTS:
        sub_all = df.filter(pl.col("instrument") == inst)
        sub_acc = accepted.filter(pl.col("instrument") == inst)

        metrics["by_instrument"][inst] = {
            "rows": int(sub_all.height),
            "accepted_target_psm_at_1pct": int(sub_acc.height),
            "accepted_unique_peptide_at_1pct": int(sub_acc["peptide_key"].n_unique()) if sub_acc.height > 0 else 0,
        }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/root/autodl-tmp/datasets/aipc")
    parser.add_argument("--v1-model-dir", type=str, default="~/aipc/models/lgbm_v1")
    parser.add_argument("--v2-model-dir", type=str, default="~/aipc/models/lgbm_v2")
    parser.add_argument("--out-dir", type=str, default="~/aipc/eval/compare_v1_v2_full_valid")
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    v1_model_dir = Path(args.v1_model_dir).expanduser()
    v2_model_dir = Path(args.v2_model_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("加载模型")
    v1_model, v1_cols = load_model(v1_model_dir)
    v2_model, v2_cols = load_model(v2_model_dir)

    files = list_valid_files(data_root)
    if args.max_files is not None:
        files = files[:args.max_files]

    print("valid files:", len(files))

    parts = []

    for path in tqdm(files):
        try:
            pred_v1 = predict_model_on_file(path, v1_model, v1_cols)
            pred_v2 = predict_model_on_file(path, v2_model, v2_cols)

            part = pred_v1.rename({"score": "score_v1"}).with_columns([
                pl.Series("score_v2", pred_v2["score"]).cast(pl.Float32)
            ])

            parts.append(part)

            del pred_v1, pred_v2, part
            gc.collect()

        except Exception as e:
            print(f"预测失败: {path}")
            print(e)

    df = pl.concat(parts, how="vertical", rechunk=False)

    print("合并 valid 预测完成:", df.height)

    metrics = {}

    metrics["v1"] = compute_metrics(df, "score_v1")
    metrics["v2"] = compute_metrics(df, "score_v2")

    # 尝试不同 blend
    blend_results = {}

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # alpha 越大，越偏向 v2
        tmp = df.with_columns([
            (
                (1.0 - alpha) * pl.col("score_v1")
                + alpha * pl.col("score_v2")
            ).cast(pl.Float32).alias("score_blend")
        ])

        m = compute_metrics(tmp, "score_blend")
        blend_results[str(alpha)] = m

        print(
            f"alpha={alpha:.1f} | "
            f"AUC={m['auc']} | "
            f"unique@1%={m['accepted_unique_peptide_at_1pct']} | "
            f"top1={m['top1_target_rate']}"
        )

    metrics["blend"] = blend_results

    # 找 unique peptide 最大的 alpha
    best_alpha = max(
        blend_results.keys(),
        key=lambda a: blend_results[a]["accepted_unique_peptide_at_1pct"]
    )

    metrics["best_blend_alpha"] = best_alpha
    metrics["best_blend_metric"] = blend_results[best_alpha]

    print(json.dumps({
        "v1_unique": metrics["v1"]["accepted_unique_peptide_at_1pct"],
        "v2_unique": metrics["v2"]["accepted_unique_peptide_at_1pct"],
        "best_alpha": best_alpha,
        "best_blend_unique": metrics["best_blend_metric"]["accepted_unique_peptide_at_1pct"],
    }, indent=2, ensure_ascii=False))

    with open(out_dir / "compare_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    df.write_parquet(out_dir / "valid_v1_v2_scores.parquet")

    print("保存完成:")
    print(out_dir / "compare_metrics.json")
    print(out_dir / "valid_v1_v2_scores.parquet")


if __name__ == "__main__":
    main()
