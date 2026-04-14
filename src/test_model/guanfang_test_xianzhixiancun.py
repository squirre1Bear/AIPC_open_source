
from __future__ import annotations

import os
import sys
import yaml
import logging
import argparse
import warnings
from contextlib import nullcontext

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

import torch
from torch import Tensor
from torch.utils.data import DataLoader
import lightning.pytorch as ptl

from src.dataset import (
    PROTON_MASS_AMU,
    SpectrumDataset,
    mkdir_p,
    padding,
    collate_batch_weight_deltaRT_index,
)
from src.train_model.model_rerank import AIPCRerankNet


# -----------------------------
# Global config
# -----------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
warnings.filterwarnings("ignore")

# 提升 Tensor Core 利用率
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# -----------------------------
# Utility
# -----------------------------
def get_parquet_columns(file_path: str) -> list[str]:
    """Read parquet schema only, without loading the full table."""
    try:
        return pq.read_schema(file_path).names
    except Exception:
        # fallback
        pf = pq.ParquetFile(file_path)
        return pf.schema_arrow.names


def read_parquet_selected_pd(file_path: str, columns: list[str]) -> pd.DataFrame:
    """Read only selected columns into pandas."""
    existing_cols = get_parquet_columns(file_path)
    use_cols = [c for c in columns if c in existing_cols]
    if not use_cols:
        raise ValueError(f"No requested columns found in parquet: {file_path}")
    return pd.read_parquet(file_path, columns=use_cols)


def read_parquet_selected_pl(file_path: str, columns: list[str]) -> pl.DataFrame:
    """Read only selected columns into polars."""
    existing_cols = get_parquet_columns(file_path)
    use_cols = [c for c in columns if c in existing_cols]
    if not use_cols:
        raise ValueError(f"No requested columns found in parquet: {file_path}")
    return pl.read_parquet(file_path, columns=use_cols)


def ensure_index_pd(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a true 'index' column."""
    if "index" not in df.columns:
        df = df.reset_index(drop=True)
        df["index"] = np.arange(len(df), dtype=np.int32)
    return df


def ensure_index_pl(df: pl.DataFrame) -> pl.DataFrame:
    """Ensure Polars DataFrame has a true 'index' column."""
    if "index" not in df.columns:
        df = df.with_row_index("index")
    return df


def build_inference_df(file_path: str) -> pl.DataFrame:
    """
    Load only the columns needed for inference.
    Use polars to reduce memory pressure.
    Normalize column names to match SpectrumDataset(pl.DataFrame) expectations:
      - precursor_charge
      - modified_sequence
    """
    desired_cols = [
        "mz_array",
        "intensity_array",
        "precursor_mz",
        "charge",
        "precursor_charge",
        "precursor_sequence",
        "modified_sequence",
        "label",
        "weight",
        "delta_rt_model",
        "predicted_rt",
        "index",
    ]
    df = read_parquet_selected_pl(file_path, desired_cols)
    df = ensure_index_pl(df)

    # Normalize charge column
    if "precursor_charge" not in df.columns and "charge" in df.columns:
        df = df.rename({"charge": "precursor_charge"})

    # Normalize peptide sequence column
    if "modified_sequence" not in df.columns and "precursor_sequence" in df.columns:
        df = df.rename({"precursor_sequence": "modified_sequence"})

    # Fill optional columns if missing
    if "label" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("label"))
    if "weight" not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias("weight"))
    if "delta_rt_model" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("delta_rt_model"))
    if "predicted_rt" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("predicted_rt"))

    required = [
        "mz_array",
        "intensity_array",
        "precursor_mz",
        "precursor_charge",
        "modified_sequence",
        "label",
        "weight",
        "delta_rt_model",
        "predicted_rt",
        "index",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Inference parquet missing required columns: {missing}, file={file_path}"
        )
    return df


def build_postprocess_df(file_path: str) -> pd.DataFrame:
    """
    Load only the columns needed for postprocessing.
    Avoid loading mz_array / intensity_array again.
    """
    desired_cols = [
        "index",
        "charge",
        "precursor_charge",
        "precursor_sequence",
        "modified_sequence",
        "scan_number",
        "precursor_mz",
        "label",
    ]
    df = read_parquet_selected_pd(file_path, desired_cols)
    df = ensure_index_pd(df)

    # Normalize columns for internal processing
    if "charge" not in df.columns and "precursor_charge" in df.columns:
        df["charge"] = df["precursor_charge"]
    if "precursor_sequence" not in df.columns and "modified_sequence" in df.columns:
        df["precursor_sequence"] = df["modified_sequence"]

    if "label" not in df.columns:
        df["label"] = 1

    required = ["index", "charge", "precursor_sequence", "scan_number", "precursor_mz", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Postprocess parquet missing required columns: {missing}, file={file_path}"
        )
    return df


def get_fdr_result(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the q-value (False Discovery Rate) for PSMs."""
    df = df.copy()

    if "label" not in df.columns:
        df["q_value"] = 0.0
        return df

    df["decoy"] = np.where(df["label"] == 1, 0, 1)

    target_num = (df["decoy"] == 0).cumsum()
    decoy_num = (df["decoy"] == 1).cumsum()

    target_num = target_num.replace(0, 1)
    decoy_num = decoy_num.replace(0, 1)

    df["q_value"] = decoy_num / target_num
    df["q_value"] = df["q_value"][::-1].cummin()[::-1]
    return df


# -----------------------------
# Lightning module
# -----------------------------
class Evalute(ptl.LightningModule):
    """Lightning Module for model evaluation and result aggregation."""

    def __init__(
        self,
        out_path: str,
        file_name: str,
        model: AIPCRerankNet,
    ) -> None:
        super().__init__()
        self.out_path = out_path
        self.file_name = file_name
        self.model = model
        self._reset_metrics()

    def test_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, list, list],
    ) -> torch.Tensor:
        """Single test step to perform prediction and accumulate results."""
        spectra, spectra_mask, precursors, tokens, peptide, label, weight, index = batch

        spectra = spectra.to(self.device).to(torch.bfloat16)
        spectra_mask = spectra_mask.to(self.device).to(torch.bfloat16)
        precursors = precursors.to(self.device).to(torch.bfloat16)
        tokens = tokens.to(self.device).to(torch.long)

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )

        with amp_ctx:
            pred, _ = self.model.pred(spectra, spectra_mask, precursors, tokens)

        label = label.to(torch.int32).cpu().detach().numpy()
        pred = pred.to(torch.float32).cpu().detach().numpy()
        index = index.to(torch.int32).cpu().detach().numpy()
        weight = weight.to(torch.float32).cpu().detach().numpy()

        if pred is not None and len(label) > 0:
            self.pred_list.extend(np.asarray(pred).reshape(-1).tolist())
            self.label_list.extend(np.asarray(label).reshape(-1).tolist())
            self.index_list.extend(np.asarray(index).reshape(-1).tolist())
            self.weight_list.extend(np.asarray(weight).reshape(-1).tolist())

    def on_test_end(self) -> None:
        """Called when testing is complete. Save results and optionally compute AUC."""
        df = pd.DataFrame(
            {
                "index": self.index_list,
                "score": self.pred_list,
                "label": self.label_list,
                "weight": self.weight_list,
            }
        )

        pred_csv = os.path.join(self.out_path, f"{self.file_name}_pred.csv")
        df.to_csv(pred_csv, index=False)

        # AUC only valid when at least two classes exist
        unique_labels = set(self.label_list)
        if len(self.label_list) > 0 and len(unique_labels) > 1:
            auc = roc_auc_score(self.label_list, self.pred_list)
            logging.info(f"auc: {auc:.6f}")
        else:
            logging.info("auc skipped: labels are empty or only one class exists.")

    def _reset_metrics(self) -> None:
        self.pred_list = []
        self.label_list = []
        self.index_list = []
        self.weight_list = []


# -----------------------------
# DataLoader
# -----------------------------
def gen_dl(df: pl.DataFrame, config: dict) -> DataLoader:
    """Generate DataLoader from a Polars DataFrame and config."""
    s2i = {v: i for i, v in enumerate(config["vocab"])}
    logging.info(f"gen Vocab: {s2i}")

    ds = SpectrumDataset(
        df,
        s2i,
        config["n_peaks"],
        need_label=True,
        need_weight=True,
        need_deltaRT=True,
        need_index=True,
    )

    dl = DataLoader(
        ds,
        batch_size=config["predict_batch_size"],
        num_workers=0,
        shuffle=False,
        collate_fn=collate_batch_weight_deltaRT_index,
        pin_memory=torch.cuda.is_available(),
    )
    logging.info(f"Data: {len(ds):,} samples, DataLoader: {len(dl):,}")
    return dl


# -----------------------------
# Postprocess
# -----------------------------
def postprocess_file(file_path: str, score_dir: str) -> pd.DataFrame:
    """Integrate parquet metadata with predicted scores and generate final output."""
    file_name = os.path.basename(file_path)[:-len(".parquet")]
    score_path = os.path.join(score_dir, file_name + "_pred.csv")

    sage_parquet = build_postprocess_df(file_path)
    sage_score = pd.read_csv(score_path).drop_duplicates(subset="index")

    assert len(sage_parquet) == len(
        sage_score
    ), (
        f"{file_name}: parquet length ({len(sage_parquet)}) "
        f"and score length ({len(sage_score)}) are inconsistent"
    )

    sage_parquet = sage_parquet.merge(sage_score[["index", "score"]], on="index", how="left")

    def _post(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["precursor_id"] = df["charge"].astype(str) + "_" + df["precursor_sequence"]
        df["psm_id"] = (
            df["scan_number"].astype(str)
            + "_"
            + df["charge"].astype(str)
            + "_"
            + df["precursor_sequence"]
        )
        df = df.sort_values("score", ascending=False)
        df = get_fdr_result(df)
        df = df[df["q_value"] < 1.0]
        return df

    sage_parquet = _post(sage_parquet)

    sage_parquet["cleaned_sequence"] = (
        sage_parquet["precursor_sequence"]
        .str.replace(r"n\[42\]", "", regex=True)
        .str.replace(r"N\[\.98\]", "N", regex=True)
        .str.replace(r"Q\[\.98\]", "Q", regex=True)
        .str.replace(r"M\[15\.99\]", "M", regex=True)
        .str.replace(r"C\[57\.02\]", "C", regex=True)
    )

    sage_parquet_target = sage_parquet[sage_parquet["label"] == 1].copy()

    sage_parquet_target = sage_parquet_target.rename(
        columns={
            "precursor_sequence": "modified_sequence",
            "charge": "precursor_charge",
        }
    )

    required_columns = [
        "cleaned_sequence",
        "precursor_mz",
        "precursor_charge",
        "modified_sequence",
        "label",
        "score",
        "q_value",
        "scan_number",
    ]

    missing_columns = [col for col in required_columns if col not in sage_parquet_target.columns]
    if missing_columns:
        raise ValueError(
            f"File {file_name} is missing columns after postprocess: {missing_columns}"
        )

    sage_parquet_target["modified_sequence"] = (
        sage_parquet_target["modified_sequence"]
        .str.replace(r"n\[42\]", "n(UniMod:1)", regex=True)
        .str.replace(r"N\[\.98\]", "N(UniMod:7)", regex=True)
        .str.replace(r"Q\[\.98\]", "Q(UniMod:7)", regex=True)
        .str.replace(r"M\[15\.99\]", "M(UniMod:35)", regex=True)
        .str.replace(r"C\[57\.02\]", "C(UniMod:4)", regex=True)
    )

    sage_parquet_target = sage_parquet_target.sort_values(by="score", ascending=False)
    sage_parquet_target_unique = sage_parquet_target.drop_duplicates(
        subset="scan_number", keep="first"
    )

    return sage_parquet_target_unique[required_columns]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    """Main function to perform prediction and post-processing."""
    logging.info("Initializing inference.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--config", default="model.yaml")
    parser.add_argument("--out_path", default="")
    args = parser.parse_args()

    logging.info(f"Inference use model path: {args.model_path}")
    model_type = args.model_path.split(".")[-1].lower()

    if model_type in ("pth", "bin", "pt"):
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)

        vocab = [
            "<pad>", "<mask>", "A", "D", "E", "F", "G", "H", "I", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y",
            "C[57.02]", "M[15.99]", "N[.98]", "Q[.98]", "X", "<unk>"
        ]
        config["vocab"] = vocab
        s2i = {v: i for i, v in enumerate(vocab)}
        logging.info(f"Vocab: {s2i}, n_peaks: {config['n_peaks']}")

        if model_type == "pth":
            model = torch.load(args.model_path, map_location="cpu")

        elif model_type == "pt":
            # 兼容 state_dict / 完整模型 两种格式
            raw_obj = torch.load(args.model_path, map_location="cpu")
            model = AIPCRerankNet(
                vocab_size=len(vocab),
                token_embed_dim=128,
                precursor_dim=64,
                hidden_dim=256,
                n_heads=8,
                n_layers=2,
                dropout=0.1,
                max_token_len=64,
            )
            if isinstance(raw_obj, dict):
                # 常见情况: {"state_dict": ...} 或直接 state_dict
                state_dict = raw_obj.get("state_dict", raw_obj)
                # 兼容 lightning 保存的 "model.xxx"
                new_state_dict = {}
                for k, v in state_dict.items():
                    nk = k
                    if nk.startswith("model."):
                        nk = nk[len("model."):]
                    new_state_dict[nk] = v
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                logging.info(f"load_state_dict done, missing={missing}, unexpected={unexpected}")
            else:
                model = raw_obj

        else:
            model = torch.load_bin(args.model_path, config)

    elif model_type == "ckpt":
        model, config = AIPCRerankNet.load_ckpt(args.model_path)

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    logging.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    model.eval()
    model = model.to(torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    gpu_num = torch.cuda.device_count()
    logging.info(f"Evaluate model with {gpu_num} gpus")

    out_path = args.parquet_dir if args.out_path == "" else args.out_path
    mkdir_p(out_path)
    logging.info(f"**************out_path: {out_path}**************************")

    data_path_list = [
        os.path.join(args.parquet_dir, f)
        for f in os.listdir(args.parquet_dir)
        if f.endswith(".parquet")
    ]
    data_path_list.sort()

    # Windows / 单卡友好版，不使用 DDP
    if torch.cuda.is_available():
        trainer = ptl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
    else:
        trainer = ptl.Trainer(
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

    for file_path in data_path_list:
        file_name = os.path.basename(file_path)[:-len(".parquet")]
        logging.info(f"Parse: {file_path}, file_name: {file_name}")

        out_file = os.path.join(out_path, f"{file_name}_pred.csv")
        if os.path.exists(out_file):
            logging.info(f"Output score file: {out_file} exists. Skipping prediction.")
            continue

        try:
            # 只读推理必要列，避免 ArrowMemoryError
            df = build_inference_df(file_path)

            dl = gen_dl(df, config)

            logging.info(f"Total {gpu_num} GPU(s) available ......")
            evaluate = Evalute(out_path, file_name, model)
            trainer.test(evaluate, dataloaders=dl)

        except Exception as e:
            logger.exception(f"Processing inference failed for {file_path}: {e}")
            continue

        try:
            # 后处理同样只读必要列，避免再次 OOM
            result_data = postprocess_file(file_path, out_path)
            result_path = os.path.join(out_path, file_name + "_result.tsv")
            result_data.to_csv(result_path, sep="\t", index=False)
            logging.info(f"Saved result: {result_path}")

        except Exception as e:
            logger.exception(f"Postprocess failed for {file_path}: {e}")
            continue


if __name__ == "__main__":
    main()

# python -m src.test_model.guanfang_test_xianzhixiancun --model_path model/best_full_aux_20260407_v1/best.pt --parquet_dir "E:\AIPC_dataset\bas_test_dataset" --out_path data/score
