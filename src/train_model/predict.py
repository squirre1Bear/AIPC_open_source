import argparse
import glob
import os
from typing import List, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.dataset import PROTON_MASS_AMU, SpectrumDataset, padding
from src.train_model.feature_utils import build_aux_features_from_df
from src.train_model.model_rerank import AIPCRerankNet


def build_vocab_from_yaml(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vocab = ["<pad>", "<mask>"] + list(cfg["residues"].keys()) + ["<unk>"]
    s2i = {v: i for i, v in enumerate(vocab)}
    return cfg, vocab, s2i


def collate_predict(batch: List[Tuple]):
    spectra, precursor_mzs, precursor_charges, tokens, peptides, aux_features = zip(*batch)
    spectra, spectra_mask = padding(list(spectra))
    tokens = torch.stack(tokens, dim=0)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges]).T.float()
    aux_features = torch.tensor(aux_features, dtype=torch.float32)
    return spectra, spectra_mask, precursors, tokens, aux_features, list(peptides)


class PredictDatasetWithAux(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, s2i: dict, cfg: dict):
        self.df = df.reset_index(drop=True).copy()
        self.aux_features = build_aux_features_from_df(self.df)
        self.base_ds = SpectrumDataset(
            self.df,
            s2i,
            n_peaks=cfg.get("n_peaks", 300),
            max_length=cfg.get("max_length", 50),
            min_mz=cfg.get("min_mz", 50.0),
            max_mz=cfg.get("max_mz", 2500.0),
            min_intensity=cfg.get("min_intensity", 0.01),
            remove_precursor_tol=cfg.get("remove_precursor_tol", 2.0),
            need_label=False,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        spectrum, precursor_mz, precursor_charge, tokens, peptide = self.base_ds[idx]
        return spectrum, precursor_mz, precursor_charge, tokens, peptide, self.aux_features[idx]


@torch.no_grad()
def predict_one_file(model, file_path: str, s2i: dict, cfg: dict, batch_size: int, num_workers: int, device):
    df = pd.read_parquet(file_path)
    ds = PredictDatasetWithAux(df, s2i, cfg)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_predict,
    )

    scores = []
    probs = []
    for spectra, spectra_mask, precursors, tokens, aux_features, _ in loader:
        spectra = spectra.to(device)
        spectra_mask = spectra_mask.to(device)
        precursors = precursors.to(device)
        tokens = tokens.to(device)
        aux_features = aux_features.to(device)
        logits = model(spectra, spectra_mask, precursors, tokens, aux_features)
        prob = torch.sigmoid(logits)
        scores.append(logits.cpu())
        probs.append(prob.cpu())

    score = torch.cat(scores).numpy()
    prob = torch.cat(probs).numpy()
    out_df = df.copy()
    out_df["score"] = score
    out_df["pred_score"] = score
    out_df["prob"] = prob
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Predict AIPC scores for parquet files.")
    parser.add_argument("--model_path", required=True, help="Checkpoint path produced by train.py")
    parser.add_argument("--parquet_dir", required=True, help="Directory containing test parquet files")
    parser.add_argument("--config", required=True, help="YAML used for peptide vocabulary and spectrum preprocessing")
    parser.add_argument("--out_path", required=True, help="Directory to save per-file scored parquet/tsv")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    cfg_yaml, vocab, s2i = build_vocab_from_yaml(args.config)

    ckpt = torch.load(args.model_path, map_location="cpu")
    model_cfg = ckpt["config"]
    model = AIPCRerankNet(
        vocab_size=model_cfg["vocab_size"],
        token_embed_dim=model_cfg["token_embed_dim"],
        precursor_dim=model_cfg["precursor_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        max_token_len=model_cfg["max_token_len"],
    )
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {args.parquet_dir}")

    for file_path in parquet_files:
        out_df = predict_one_file(model, file_path, s2i, cfg_yaml, args.batch_size, args.num_workers, device)

        stem = os.path.splitext(os.path.basename(file_path))[0]
        out_parquet = os.path.join(args.out_path, f"{stem}.parquet")
        out_tsv = os.path.join(args.out_path, f"{stem}.tsv")
        out_df.to_parquet(out_parquet, index=False)

        cols = []
        for c in ["psm_id", "scan", "precursor_sequence", "modified_sequence", "score", "pred_score", "prob"]:
            if c in out_df.columns:
                cols.append(c)
        if "score" not in cols:
            cols.append("score")
        out_df[cols].to_csv(out_tsv, sep="\t", index=False)


if __name__ == "__main__":
    main()
