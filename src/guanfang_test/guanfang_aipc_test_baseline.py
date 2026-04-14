from __future__ import annotations

import os
import sys
import yaml
import math
import argparse
import logging
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 复用你原来的 dataset / collate / mkdir_p
from ..transformer.dataset import (
    SpectrumDataset,
    mkdir_p,
    collate_batch_weight_deltaRT_index,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)


# =========================
# AIPCRerankNet
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class AIPCRerankNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        token_embed_dim: int = 128,
        precursor_dim: int = 64,
        hidden_dim: int = 256,
        aux_feature_dim: int = 10,
        n_heads: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_token_len: int = 64,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, token_embed_dim, padding_idx=0)
        self.token_pos = PositionalEncoding(token_embed_dim, max_len=max_token_len)

        token_layer = nn.TransformerEncoderLayer(
            d_model=token_embed_dim,
            nhead=max(1, min(n_heads, token_embed_dim // 16 if token_embed_dim >= 16 else 1)),
            dim_feedforward=token_embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=n_layers)

        self.spec_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        spec_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.spec_encoder = nn.TransformerEncoder(spec_layer, num_layers=n_layers)

        self.precursor_mlp = nn.Sequential(
            nn.Linear(2, precursor_dim),
            nn.GELU(),
            nn.LayerNorm(precursor_dim),
            nn.Dropout(dropout),
            nn.Linear(precursor_dim, precursor_dim),
            nn.GELU(),
        )

        self.aux_feature_mlp = nn.Sequential(
            nn.Linear(aux_feature_dim, precursor_dim),
            nn.GELU(),
            nn.LayerNorm(precursor_dim),
            nn.Dropout(dropout),
            nn.Linear(precursor_dim, precursor_dim),
            nn.GELU(),
        )

        fused_dim = hidden_dim + token_embed_dim + precursor_dim + precursor_dim + 2
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        valid = (~mask).unsqueeze(-1).float()
        denom = valid.sum(dim=1).clamp_min(eps)
        return (x * valid).sum(dim=1) / denom

    def forward(self, spectra, spectra_mask, precursors, tokens, aux_features=None):
        token_pad_mask = tokens.eq(0)
        tok = self.token_embed(tokens)
        tok = self.token_pos(tok)
        tok = self.token_encoder(tok, src_key_padding_mask=token_pad_mask)
        tok_pool = self.masked_mean(tok, token_pad_mask)

        spec = self.spec_proj(spectra)
        spec = self.spec_encoder(spec, src_key_padding_mask=spectra_mask)
        spec_pool = self.masked_mean(spec, spectra_mask)

        prec = self.precursor_mlp(precursors[:, :2])

        if aux_features is None:
            aux_features = torch.zeros(
                (precursors.size(0), self.aux_feature_mlp[0].in_features),
                dtype=precursors.dtype,
                device=precursors.device,
            )
        aux = self.aux_feature_mlp(aux_features)

        tok_for_cos = tok_pool
        if tok_for_cos.size(1) < spec_pool.size(1):
            tok_for_cos = F.pad(tok_for_cos, (0, spec_pool.size(1) - tok_for_cos.size(1)))
        else:
            tok_for_cos = tok_for_cos[:, :spec_pool.size(1)]

        cosine = F.cosine_similarity(
            F.normalize(spec_pool, dim=-1),
            F.normalize(tok_for_cos, dim=-1),
            dim=-1,
        ).unsqueeze(-1)

        length_feat = tokens.ne(0).sum(dim=1, keepdim=True).float() / max(1, tokens.size(1))

        fused = torch.cat([spec_pool, tok_pool, prec, aux, cosine, length_feat], dim=-1)
        logit = self.classifier(fused).squeeze(-1)
        return logit


# =========================
# Utilities
# =========================
def get_fdr_result(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["decoy"] = np.where(df["label"] == 1, 0, 1)

    target_num = (df["decoy"] == 0).cumsum()
    decoy_num = (df["decoy"] == 1).cumsum()

    target_num = target_num.replace(0, 1)
    decoy_num = decoy_num.replace(0, 1)

    df["q_value"] = decoy_num / target_num
    df["q_value"] = df["q_value"][::-1].cummin()[::-1]
    return df


def extract_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

    if "module" in ckpt and isinstance(ckpt["module"], dict):
        state_dict = ckpt["module"]
    elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise ValueError("Resolved state_dict is not a dict")

    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned[nk] = v
    return cleaned


def build_default_vocab():
    return [
        "<pad>", "<mask>", "A", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N",
        "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y",
        "C[57.02]", "M[15.99]", "N[.98]", "Q[.98]", "X", "<unk>"
    ]


def infer_n_layers_from_state_dict(state_dict: dict) -> int:
    token_layer_ids = []
    spec_layer_ids = []

    for k in state_dict.keys():
        if k.startswith("token_encoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                token_layer_ids.append(int(parts[2]))

        if k.startswith("spec_encoder.layers."):
            parts = k.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                spec_layer_ids.append(int(parts[2]))

    max_token_layer = max(token_layer_ids) + 1 if token_layer_ids else 0
    max_spec_layer = max(spec_layer_ids) + 1 if spec_layer_ids else 0

    inferred = max(max_token_layer, max_spec_layer, 0)
    if inferred <= 0:
        inferred = 2
    return inferred


def infer_n_heads_from_state_dict(state_dict: dict, config: dict, hidden_dim: int, token_embed_dim: int) -> int:
    # 优先从 config 读；没有就兜底 8
    n_heads = config.get("n_heads", 8)

    # 简单合法性修正，避免 hidden_dim / token_embed_dim 和 n_heads 不整除导致 transformer 初始化异常
    if hidden_dim % n_heads != 0:
        for cand in [8, 4, 2, 1]:
            if hidden_dim % cand == 0 and token_embed_dim % cand == 0:
                n_heads = cand
                break

    return n_heads


def infer_model_hparams_from_state_dict(state_dict: dict, vocab_size: int, config: dict) -> dict:
    hp = {}
    hp["vocab_size"] = vocab_size

    if "token_embed.weight" in state_dict:
        hp["token_embed_dim"] = state_dict["token_embed.weight"].shape[1]
    else:
        hp["token_embed_dim"] = config.get("token_embed_dim", 128)

    if "spec_proj.0.weight" in state_dict:
        hp["hidden_dim"] = state_dict["spec_proj.0.weight"].shape[0]
    else:
        hp["hidden_dim"] = config.get("hidden_dim", 256)

    if "precursor_mlp.0.weight" in state_dict:
        hp["precursor_dim"] = state_dict["precursor_mlp.0.weight"].shape[0]
    else:
        hp["precursor_dim"] = config.get("precursor_dim", 64)

    if "aux_feature_mlp.0.weight" in state_dict:
        hp["aux_feature_dim"] = state_dict["aux_feature_mlp.0.weight"].shape[1]
    else:
        hp["aux_feature_dim"] = config.get("aux_feature_dim", 10)

    hp["n_layers"] = infer_n_layers_from_state_dict(state_dict)
    hp["n_heads"] = infer_n_heads_from_state_dict(
        state_dict=state_dict,
        config=config,
        hidden_dim=hp["hidden_dim"],
        token_embed_dim=hp["token_embed_dim"],
    )
    hp["dropout"] = config.get("dropout", 0.1)
    hp["max_token_len"] = config.get("max_token_len", config.get("max_length", 64))

    return hp


def load_rerank_model(model_path: str, config: dict, device: torch.device) -> AIPCRerankNet:
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = extract_state_dict(ckpt)

    hparams = infer_model_hparams_from_state_dict(
        state_dict=state_dict,
        vocab_size=len(config["vocab"]),
        config=config,
    )
    logging.info(f"Inferred model hparams: {hparams}")

    model = AIPCRerankNet(**hparams)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"Missing keys: {missing}")
    if unexpected:
        logging.warning(f"Unexpected keys: {unexpected}")

    model.eval()
    model.to(device)
    return model


def ensure_required_runtime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "weight" not in df.columns:
        df["weight"] = 1.0

    if "index" not in df.columns:
        df = df.reset_index(drop=True)
        df["index"] = np.arange(len(df), dtype=np.int64)

    return df


def gen_dl(df, config):
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
        pin_memory=True,
    )
    logging.info(f"Data: {len(ds):,} samples, DataLoader: {len(dl):,}")
    return dl


def postprocess_file(file_path, score_dir):
    file_name = os.path.basename(file_path)[:-len(".parquet")]
    score_path = os.path.join(score_dir, file_name + "_pred.csv")

    sage_parquet = pd.read_parquet(file_path)
    sage_parquet = ensure_required_runtime_columns(sage_parquet)

    sage_score = pd.read_csv(score_path).drop_duplicates(subset="index")

    assert len(sage_parquet) == len(sage_score), (
        f"{file_name}: parquet length ({len(sage_parquet)}) and "
        f"score length ({len(sage_score)}) are inconsistent"
    )

    sage_parquet = sage_parquet.merge(sage_score[["index", "score"]], on="index", how="left")

    def _post(df):
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

    sage_parquet = sage_parquet.drop(columns=["mz_array", "intensity_array"], errors="ignore")
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
        print(f"Warning: File {file_name} is missing columns: {', '.join(missing_columns)}")
        sys.exit()

    sage_parquet_target["modified_sequence"] = (
        sage_parquet_target["modified_sequence"]
        .str.replace(r"n\[42\]", "n(UniMod:1)", regex=True)
        .str.replace(r"N\[\.98\]", "N(UniMod:7)", regex=True)
        .str.replace(r"Q\[\.98\]", "Q(UniMod:7)", regex=True)
        .str.replace(r"M\[15\.99\]", "M(UniMod:35)", regex=True)
        .str.replace(r"C\[57\.02\]", "C(UniMod:4)", regex=True)
    )

    sage_parquet_target = sage_parquet_target.sort_values(by="score", ascending=False)
    sage_parquet_target_unique = sage_parquet_target.drop_duplicates(subset="scan_number", keep="first")

    return sage_parquet_target_unique[required_columns]


@torch.no_grad()
def predict_one_file(model, dl, out_path: str, file_name: str, device: torch.device):
    pred_list = []
    label_list = []
    index_list = []
    weight_list = []

    use_amp = torch.cuda.is_available()

    for batch in dl:
        spectra, spectra_mask, precursors, tokens, peptide, label, weight, index = batch

        spectra = spectra.to(device)
        spectra_mask = spectra_mask.to(device).bool()
        precursors = precursors.to(device)
        tokens = tokens.to(device).long()

        # AIPCRerankNet 的 spec_proj 输入要求最后一维为 2
        if spectra.size(-1) > 2:
            spectra = spectra[:, :, :2]

        if spectra.dim() != 3 or spectra.size(-1) != 2:
            raise ValueError(f"Expected spectra shape [B, N, 2], got {tuple(spectra.shape)}")

        spectra = spectra.float()
        precursors = precursors.float()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = model(spectra, spectra_mask, precursors, tokens, aux_features=None)
            pred = torch.sigmoid(logits)

        pred = pred.detach().float().cpu().numpy()
        label = label.detach().cpu().numpy()
        index = index.detach().cpu().numpy()
        weight = weight.detach().cpu().numpy()

        pred_list.extend(np.atleast_1d(pred).tolist())
        label_list.extend(np.atleast_1d(label).tolist())
        index_list.extend(np.atleast_1d(index).tolist())
        weight_list.extend(np.atleast_1d(weight).tolist())

    df = pd.DataFrame(
        {
            "index": index_list,
            "score": pred_list,
            "label": label_list,
            "weight": weight_list,
        }
    )
    pred_csv_path = os.path.join(out_path, f"{file_name}_pred.csv")
    df.to_csv(pred_csv_path, index=False)

    try:
        auc = roc_auc_score(label_list, pred_list)
        logging.info(f"{file_name} auc: {auc}")
    except Exception as e:
        logging.warning(f"Failed to calculate AUC for {file_name}: {e}")

    return pred_csv_path


def main() -> None:
    logging.info("Initializing inference.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--config", default="aipc_test_mzml.yaml")
    parser.add_argument("--out_path", default="")
    args = parser.parse_args()

    logging.info(f"Inference use model path: {args.model_path}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)

    config["vocab"] = build_default_vocab()
    s2i = {v: i for i, v in enumerate(config["vocab"])}
    logging.info(f"Vocab: {s2i}, n_peaks: {config['n_peaks']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = load_rerank_model(args.model_path, config, device)
    n_params = int(np.sum([p.numel() for p in model.parameters()]))
    logging.info(f"Model loaded with {n_params:,d} parameters")

    out_path = args.parquet_dir if args.out_path == "" else args.out_path
    mkdir_p(out_path)
    logging.info(f"************** out_path: {out_path} **************************")

    data_path_list = [
        os.path.join(args.parquet_dir, f)
        for f in os.listdir(args.parquet_dir)
        if f.endswith(".parquet")
    ]
    data_path_list = sorted(data_path_list)

    if len(data_path_list) == 0:
        logging.warning(f"No parquet files found in {args.parquet_dir}")
        return

    for file_path in data_path_list:
        file_name = os.path.basename(file_path)[:-len(".parquet")]
        logging.info(f"Parse: {file_path}, file_name: {file_name}")

        out_file = os.path.join(out_path, f"{file_name}_pred.csv")
        if os.path.exists(out_file):
            logging.info(f"Output score file: {out_file} exists. Skipping prediction.")
        else:
            df = pd.read_parquet(file_path)

            try:
                df = ensure_required_runtime_columns(df)
                dl = gen_dl(df, config)
                predict_one_file(model, dl, out_path, file_name, device)

            except Exception as e:
                logger.exception(f"Loading {file_path} parquet error: {e} !!! Skipping file.")
                continue

        try:
            result_data = postprocess_file(file_path, out_path)
            result_path = os.path.join(out_path, file_name + "_result.tsv")
            result_data.to_csv(result_path, sep="\t", index=False)
            logging.info(f"Saved result: {result_path}")
        except Exception as e:
            logger.exception(f"Postprocess {file_path} error: {e}")

    logging.info("Inference finished.")


if __name__ == "__main__":
    main()

# python3 -m src.guanfang_test.guanfang_aipc_test_baseline --model_path model/best_full_aux_20260407_v1/best.pt --parquet_dir data/bas_test_dataset --out_path data/score
