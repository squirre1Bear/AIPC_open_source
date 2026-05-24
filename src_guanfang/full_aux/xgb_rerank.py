from __future__ import annotations

import argparse
import logging
import os
from typing import Iterable

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .common import build_prediction_frame, clean_sequence

LOGGER = logging.getLogger("xgb_rerank")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_score"] = pd.to_numeric(df["base_score"], errors="coerce").fillna(0.0)
    df["predicted_rt"] = pd.to_numeric(df["predicted_rt"], errors="coerce").fillna(0.0)
    df["delta_rt_model"] = pd.to_numeric(df["delta_rt_model"], errors="coerce").fillna(0.0)
    df["abs_delta"] = df["delta_rt_model"].abs()
    df["peptide_length"] = df["precursor_sequence"].map(lambda seq: len(clean_sequence(seq)))
    df["mod_count"] = df["precursor_sequence"].str.count(r"\[")
    df["charge_score"] = df["charge"] * df["base_score"]
    df["rt_score"] = df["predicted_rt"] * df["base_score"]
    df["scan_size"] = df.groupby("scan_number")["index"].transform("count")
    df["scan_rank"] = df.groupby("scan_number")["base_score"].rank(method="first", ascending=False)
    df["scan_top_score"] = df.groupby("scan_number")["base_score"].transform("max")
    df["score_gap"] = df["scan_top_score"] - df["base_score"]
    return df


def train_and_score(df: pd.DataFrame, n_jobs: int) -> np.ndarray:
    feature_df = build_features(df)
    x = feature_df[
        [
            "base_score",
            "precursor_mz",
            "charge",
            "predicted_rt",
            "delta_rt_model",
            "abs_delta",
            "peptide_length",
            "mod_count",
            "charge_score",
            "rt_score",
            "scan_size",
            "scan_rank",
            "score_gap",
        ]
    ].fillna(0.0)
    y = feature_df["label"].astype(int).to_numpy()

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=n_jobs,
    )
    model.fit(x, y)
    return model.predict_proba(x)[:, 1]


def iter_parquet_files(parquet_dir: str) -> Iterable[str]:
    for file_name in sorted(os.listdir(parquet_dir)):
        if file_name.endswith(".parquet"):
            yield os.path.join(parquet_dir, file_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--base_pred_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--q_value_threshold", type=float, default=1.0)
    parser.add_argument("--n_jobs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)

    for parquet_path in iter_parquet_files(args.parquet_dir):
        file_name = os.path.basename(parquet_path)[: -len(".parquet")]
        pred_path = os.path.join(args.out_path, f"{file_name}_pred.csv")
        result_path = os.path.join(args.out_path, f"{file_name}_result.tsv")
        LOGGER.info("processing %s", file_name)

        parquet_df = pd.read_parquet(
            parquet_path,
            columns=[
                "index",
                "scan_number",
                "precursor_mz",
                "charge",
                "precursor_sequence",
                "label",
                "delta_rt_model",
                "predicted_rt",
            ],
        )
        base_pred = pd.read_csv(os.path.join(args.base_pred_dir, f"{file_name}_pred.csv"))
        merged = parquet_df.merge(base_pred[["index", "score"]], on="index", how="left")
        merged = merged.rename(columns={"score": "base_score"})
        rescored = train_and_score(merged, n_jobs=args.n_jobs)

        pred_df = pd.DataFrame(
            {
                "index": merged["index"].astype(int),
                "score": rescored,
                "label": merged["label"].astype(float),
                "weight": np.ones(len(merged), dtype=np.float32),
            }
        )
        pred_df.to_csv(pred_path, index=False)

        result_df = build_prediction_frame(parquet_df, pred_df, q_value_threshold=args.q_value_threshold)
        result_df.to_csv(result_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
