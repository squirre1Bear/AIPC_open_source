import argparse
import gc
import logging
import os
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.dataset import PROTON_MASS_AMU, SpectrumDataset, mkdir_p, padding
from src.train_model.feature_utils import build_aux_features_from_df
from src.train_model.model_rerank import AIPCRerankNet


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


REQUIRED_PREDICT_COLUMNS = [
    "mz_array",
    "intensity_array",
    "precursor_mz",
    "charge",
    "precursor_sequence",
    "label",
    "weight",
    "index",
    "rt",
    "predicted_rt",
    "delta_rt",
    "sage_discriminant_score",
    "spectrum_q",
]

REQUIRED_POST_COLUMNS = [
    "precursor_mz",
    "charge",
    "precursor_sequence",
    "label",
    "index",
    "scan_number",
    "scan",
]


def normalize_output_stem(file_path: str) -> str:
    stem = os.path.basename(file_path)[: -len(".parquet")]
    if stem.endswith("_benchmark"):
        stem = stem[: -len("_benchmark")]
    return stem


def load_input_file_list(parquet_dir: str, file_list_path: str = "") -> List[str]:
    if file_list_path:
        with open(file_list_path, "r", encoding="utf-8") as f:
            data_path_list = [line.strip() for line in f if line.strip()]
        missing = [path for path in data_path_list if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(f"Some parquet files listed in {file_list_path} do not exist: {missing[:5]}")
        return data_path_list

    return sorted(os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith(".parquet"))


class PredictDatasetWithIndexWeight(Dataset):
    def __init__(self, df: pd.DataFrame, s2i: dict, n_peaks: int):
        self.df = df.reset_index(drop=True).copy()
        self.aux_features = build_aux_features_from_df(self.df)
        self.base_ds = SpectrumDataset(
            self.df,
            s2i,
            n_peaks=n_peaks,
            need_label=True,
            need_weight=True,
            need_index=False,
            need_deltaRT=False,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spectrum, precursor_mz, precursor_charge, tokens, peptide, label, weight = self.base_ds[idx]
        row = self.df.iloc[idx]
        index = row["index"]
        aux_features = self.aux_features[idx]
        return spectrum, precursor_mz, precursor_charge, tokens, label, index, weight, aux_features


def collate_batch_index_weight_local(batch):
    spectra, precursor_mzs, precursor_charges, tokens, label, index, weight, aux_features = zip(*batch)

    spectra, spectra_mask = padding(spectra)
    tokens = torch.stack(tokens, dim=0)

    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges]).T.float()

    label = torch.tensor(label, dtype=torch.float32)
    index = torch.tensor(index, dtype=torch.int32)
    weight = torch.tensor(weight, dtype=torch.float32)
    aux_features = torch.tensor(np.asarray(aux_features), dtype=torch.float32)

    return spectra, spectra_mask, precursors, tokens, label, index, weight, aux_features


def get_fdr_result(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decoy"] = np.where(df["label"] == 1, 0, 1)
    target_num = (df["decoy"] == 0).cumsum().replace(0, 1)
    decoy_num = (df["decoy"] == 1).cumsum().replace(0, 1)
    df["q_value"] = decoy_num / target_num
    df["q_value"] = df["q_value"][::-1].cummin()[::-1]
    return df


def load_vocab_from_yaml(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    vocab = ["<pad>", "<mask>"] + list(config["residues"].keys()) + ["<unk>"]
    s2i = {v: i for i, v in enumerate(vocab)}
    return config, vocab, s2i


def load_model(model_path, vocab_size, device):
    ckpt = torch.load(model_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError(f"{model_path} is not a valid AIPCRerankNet checkpoint")

    cfg = ckpt["config"]
    model = AIPCRerankNet(
        vocab_size=vocab_size,
        token_embed_dim=cfg.get("token_embed_dim", 128),
        precursor_dim=cfg.get("precursor_dim", 64),
        hidden_dim=cfg.get("hidden_dim", 256),
        n_heads=cfg.get("n_heads", 8),
        n_layers=cfg.get("n_layers", 2),
        dropout=cfg.get("dropout", 0.1),
        max_token_len=cfg.get("max_token_len", 64),
    )

    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    model.to(device)
    return model


def get_available_columns(file_path):
    pf = pq.ParquetFile(file_path)
    return pf.schema_arrow.names


def pick_existing_columns(file_path, columns):
    available = set(get_available_columns(file_path))
    return [c for c in columns if c in available]


def gen_dl(df, s2i, n_peaks, batch_size, index_offset=0):
    df = df.reset_index(drop=True).copy()

    if "weight" not in df.columns:
        df["weight"] = 1.0
    if "index" not in df.columns:
        df["index"] = (df.index.to_numpy(dtype=np.int64) + int(index_offset)).astype(np.int32)
    else:
        df["index"] = df["index"].astype(np.int32)

    ds = PredictDatasetWithIndexWeight(df, s2i, n_peaks=n_peaks)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch_index_weight_local,
    )
    return dl


@torch.no_grad()
def predict_batch_df(dl, model, device):
    pred_rows = []

    for spectra, spectra_mask, precursors, tokens, label, index, weight, aux_features in dl:
        spectra = spectra.to(device, non_blocking=True)
        spectra_mask = spectra_mask.to(device, non_blocking=True)
        precursors = precursors.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        aux_features = aux_features.to(device, non_blocking=True)

        logits = model(spectra, spectra_mask, precursors, tokens, aux_features)
        score = logits.detach().cpu().numpy()

        batch_df = pd.DataFrame(
            {
                "index": index.cpu().numpy().astype(np.int32),
                "score": score.astype(np.float32),
                "label": label.cpu().numpy().astype(np.int32),
                "weight": weight.cpu().numpy().astype(np.float32),
            }
        )
        pred_rows.append(batch_df)

    if not pred_rows:
        return pd.DataFrame(columns=["index", "score", "label", "weight"])

    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df = pred_df.drop_duplicates(subset="index")
    return pred_df


@torch.no_grad()
def predict_one_file_streaming(file_path, s2i, n_peaks, predict_batch_size, parquet_batch_rows, model, device, out_pred_csv):
    predict_columns = pick_existing_columns(file_path, REQUIRED_PREDICT_COLUMNS)
    missing_predict = [c for c in ["mz_array", "intensity_array", "precursor_mz", "charge", "precursor_sequence", "label"] if c not in predict_columns]
    if missing_predict:
        raise KeyError(f"{os.path.basename(file_path)} missing required predict columns: {missing_predict}")

    pf = pq.ParquetFile(file_path)
    total_rows = pf.metadata.num_rows
    logger.info("Streaming parquet: rows=%d, row_groups=%d, parquet_batch_rows=%d", total_rows, pf.num_row_groups, parquet_batch_rows)

    wrote_header = False
    total_pred_rows = 0
    row_offset = 0

    for batch_idx, record_batch in enumerate(pf.iter_batches(batch_size=parquet_batch_rows, columns=predict_columns), start=1):
        batch_df = record_batch.to_pandas()
        dl = gen_dl(batch_df, s2i, n_peaks, predict_batch_size, index_offset=row_offset)
        pred_df = predict_batch_df(dl, model, device)

        pred_df.to_csv(out_pred_csv, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total_pred_rows += len(pred_df)
        row_offset += len(batch_df)

        logger.info("Predict batch %d: batch_rows=%d, cumulative_pred_rows=%d/%d", batch_idx, len(batch_df), total_pred_rows, total_rows)

        del batch_df, dl, pred_df, record_batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if total_pred_rows != total_rows:
        raise ValueError(f"Prediction row mismatch for {os.path.basename(file_path)}: pred_rows={total_pred_rows}, parquet_rows={total_rows}")


def postprocess_file_light(file_path, score_dir):
    file_name = normalize_output_stem(file_path)
    score_path = os.path.join(score_dir, file_name + "_pred.csv")

    post_columns = pick_existing_columns(file_path, REQUIRED_POST_COLUMNS)
    must_have = ["precursor_mz", "charge", "precursor_sequence", "label"]
    missing_post = [c for c in must_have if c not in post_columns]
    if missing_post:
        raise KeyError(f"{file_name} missing required postprocess columns: {missing_post}")

    df = pd.read_parquet(file_path, columns=post_columns).reset_index(drop=True)
    if "index" not in df.columns:
        df["index"] = df.index.astype(np.int32)
    else:
        df["index"] = df["index"].astype(np.int32)

    score_df = pd.read_csv(score_path).drop_duplicates(subset="index")

    if len(df) != len(score_df):
        raise ValueError(f"{file_name}: parquet length ({len(df)}) and score length ({len(score_df)}) are inconsistent")

    df = df.merge(score_df[["index", "score"]], on="index", how="left")

    scan_col = "scan_number" if "scan_number" in df.columns else "scan"
    if scan_col not in df.columns:
        raise KeyError(f"{file_name} has neither scan_number nor scan column")

    df["precursor_id"] = df["charge"].astype(str) + "_" + df["precursor_sequence"]
    df["psm_id"] = df[scan_col].astype(str) + "_" + df["charge"].astype(str) + "_" + df["precursor_sequence"]

    df = df.sort_values("score", ascending=False)
    df = get_fdr_result(df)
    df = df[df["q_value"] < 1.0]

    df["cleaned_sequence"] = (
        df["precursor_sequence"]
        .str.replace(r"n\[42\]", "", regex=True)
        .str.replace(r"N\[\.98\]", "N", regex=True)
        .str.replace(r"Q\[\.98\]", "Q", regex=True)
        .str.replace(r"M\[15\.99\]", "M", regex=True)
        .str.replace(r"C\[57\.02\]", "C", regex=True)
    )

    target_df = df[df["label"] == 1].copy()
    target_df = target_df.rename(
        columns={
            "precursor_sequence": "modified_sequence",
            "charge": "precursor_charge",
            scan_col: "scan_number",
        }
    )

    target_df["modified_sequence"] = (
        target_df["modified_sequence"]
        .str.replace(r"n\[42\]", "n(UniMod:1)", regex=True)
        .str.replace(r"N\[\.98\]", "N(UniMod:7)", regex=True)
        .str.replace(r"Q\[\.98\]", "Q(UniMod:7)", regex=True)
        .str.replace(r"M\[15\.99\]", "M(UniMod:35)", regex=True)
        .str.replace(r"C\[57\.02\]", "C(UniMod:4)", regex=True)
    )

    target_df = target_df.sort_values(by="score", ascending=False)
    target_df = target_df.drop_duplicates(subset="scan_number", keep="first")

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

    missing_columns = [c for c in required_columns if c not in target_df.columns]
    if missing_columns:
        raise KeyError(f"{file_name} missing columns: {missing_columns}")

    return target_df[required_columns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--config", required=True, help="same yaml used in training")
    parser.add_argument("--out_path", default="")
    parser.add_argument("--predict_batch_size", type=int, default=512)
    parser.add_argument("--parquet_batch_rows", type=int, default=4096, help="rows loaded from parquet per streaming chunk")
    parser.add_argument("--file_list", default="", help="Optional text file containing absolute parquet paths to process, one per line")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    config, vocab, s2i = load_vocab_from_yaml(args.config)
    logger.info("Loaded vocab size = %d", len(vocab))

    model = load_model(args.model_path, len(vocab), device)

    out_path = args.out_path if args.out_path else args.parquet_dir
    mkdir_p(out_path)
    logger.info("Output dir: %s", out_path)

    data_path_list = load_input_file_list(args.parquet_dir, args.file_list)
    logger.info("Total parquet files to process: %d", len(data_path_list))

    for file_path in data_path_list:
        file_name = normalize_output_stem(file_path)
        logger.info("Processing: %s", file_name)

        pred_csv = os.path.join(out_path, f"{file_name}_pred.csv")
        benchmark_tsv = os.path.join(out_path, f"{file_name}_benchmark_result.tsv")

        if os.path.exists(benchmark_tsv):
            logger.info("Skip existing result: %s", benchmark_tsv)
            continue

        if os.path.exists(pred_csv):
            os.remove(pred_csv)

        predict_one_file_streaming(
            file_path=file_path,
            s2i=s2i,
            n_peaks=config["n_peaks"],
            predict_batch_size=args.predict_batch_size,
            parquet_batch_rows=args.parquet_batch_rows,
            model=model,
            device=device,
            out_pred_csv=pred_csv,
        )
        result_df = postprocess_file_light(file_path, out_path)
        result_df.to_csv(benchmark_tsv, sep="\t", index=False)
        logger.info("Saved: %s", benchmark_tsv)


if __name__ == "__main__":
    main()
