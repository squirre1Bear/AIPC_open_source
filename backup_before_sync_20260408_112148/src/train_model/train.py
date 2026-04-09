import argparse
import glob
import json
import logging
import os
import pickle
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from model_rerank import AIPCRerankNet
from feature_utils import build_aux_features_from_df


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_sequence(sequence: str) -> str:
    text = sequence if isinstance(sequence, str) else ""
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = text.replace("n", "")
    return text


class PklChunkBatchDataset(Dataset):
    """
    Each PKL file stores one pre-collated chunk of samples. Avoid scanning the
    whole directory at startup because 70k+ small files make exact indexing
    disproportionately expensive.
    """

    def __init__(
        self,
        pkl_files: List[str],
        parquet_dir: Optional[str] = None,
        name: str = "dataset",
        inspect_rows: bool = True,
    ):
        self.pkl_files = pkl_files
        self.parquet_dir = parquet_dir
        self.name = name
        self.rows_per_chunk = self._infer_rows_per_chunk() if inspect_rows else 256
        self.total_rows = len(self.pkl_files) * self.rows_per_chunk
        logging.info(
            "[%s] Prepared %d PKL chunks | rows_per_chunk=%d | estimated_samples=%d | parquet_dir=%s",
            self.name,
            len(self.pkl_files),
            self.rows_per_chunk,
            self.total_rows,
            self.parquet_dir or "<none>",
        )

    def _infer_rows_per_chunk(self) -> int:
        if not self.pkl_files:
            return 0
        with open(self.pkl_files[0], "rb") as f:
            obj = pickle.load(f)
        return int(len(obj["label"]))

    def __len__(self) -> int:
        return len(self.pkl_files)

    def _match_parquet_file(self, pkl_path: str) -> Optional[str]:
        if not self.parquet_dir:
            return None
        stem = os.path.splitext(os.path.basename(pkl_path))[0]
        parquet_name = f"{stem.split('_')[0]}.parquet"
        parquet_path = os.path.join(self.parquet_dir, parquet_name)
        return parquet_path if os.path.exists(parquet_path) else None

    def __getitem__(self, idx: int):
        path = self.pkl_files[idx]
        with open(path, "rb") as f:
            obj = pickle.load(f)

        n = int(len(obj["label"]))
        weight_arr = obj.get("weight", np.ones(n, dtype=np.float32))
        peptides = obj.get("peptides", [""] * n)

        spectra = torch.from_numpy(np.asarray(obj["spectra"], dtype=np.float32).copy())
        spectra_mask = torch.from_numpy(np.asarray(obj["spectra_mask"], dtype=np.bool_).copy())
        precursors = torch.from_numpy(np.asarray(obj["precursors"], dtype=np.float32).copy())
        tokens = torch.from_numpy(np.asarray(obj["tokens"], dtype=np.int64).copy())
        label = torch.from_numpy(np.asarray(obj["label"], dtype=np.float32).copy())
        weight = torch.from_numpy(np.asarray(weight_arr, dtype=np.float32).copy())

        aux_features = obj.get("aux_features")
        if aux_features is None:
            parquet_path = self._match_parquet_file(path)
            if parquet_path is not None:
                df = pd.read_parquet(
                    parquet_path,
                    columns=[
                        "precursor_mz",
                        "charge",
                        "rt",
                        "predicted_rt",
                        "delta_rt",
                        "sage_discriminant_score",
                        "spectrum_q",
                        "precursor_sequence",
                    ],
                )
                aux_features = build_aux_features_from_df(df)
            else:
                precursor_np = precursors.numpy()
                dummy_df = pd.DataFrame(
                    {
                        "precursor_mz": (precursor_np[:, 0] / np.clip(precursor_np[:, 1], 1.0, None)) + 1.007276,
                        "charge": precursor_np[:, 1],
                        "precursor_sequence": peptides,
                    }
                )
                aux_features = build_aux_features_from_df(dummy_df)
        aux_features = torch.from_numpy(np.asarray(aux_features, dtype=np.float32).copy())

        return spectra, spectra_mask, precursors, tokens, aux_features, label, weight, list(peptides)


def collate_pkl_chunks(batch):
    (
        spectra_list,
        spectra_mask_list,
        precursors_list,
        tokens_list,
        aux_feature_list,
        label_list,
        weight_list,
        peptide_lists,
    ) = zip(*batch)

    max_spec_len = max(x.size(1) for x in spectra_list)
    max_tok_len = max(x.size(1) for x in tokens_list)

    padded_spectra = []
    padded_masks = []
    padded_tokens = []

    for spectra, spectra_mask, tokens in zip(spectra_list, spectra_mask_list, tokens_list):
        if spectra.size(1) < max_spec_len:
            pad_len = max_spec_len - spectra.size(1)
            spectra = F.pad(spectra, (0, 0, 0, pad_len), value=0.0)
            spectra_mask = F.pad(spectra_mask, (0, pad_len), value=True)

        if tokens.size(1) < max_tok_len:
            tok_pad = max_tok_len - tokens.size(1)
            tokens = F.pad(tokens, (0, tok_pad), value=0)

        padded_spectra.append(spectra)
        padded_masks.append(spectra_mask)
        padded_tokens.append(tokens)

    spectra = torch.cat(padded_spectra, dim=0)
    spectra_mask = torch.cat(padded_masks, dim=0)
    precursors = torch.cat(precursors_list, dim=0)
    tokens = torch.cat(padded_tokens, dim=0)
    aux_features = torch.cat(aux_feature_list, dim=0)
    label = torch.cat(label_list, dim=0).float()
    weight = torch.cat(weight_list, dim=0).float()
    peptides = [peptide for chunk in peptide_lists for peptide in chunk]

    return spectra, spectra_mask, precursors, tokens, aux_features, label, weight, peptides


@dataclass
class TrainConfig:
    pkl_dir: str
    parquet_dir: Optional[str]
    output_dir: str
    vocab_size: int
    batch_size: int = 1024
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.1
    num_workers: int = 0
    seed: int = 42
    token_embed_dim: int = 128
    precursor_dim: int = 64
    hidden_dim: int = 256
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.1
    max_token_len: int = 64
    grad_clip: float = 1.0
    amp: bool = True


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    return True, rank, world_size, device


def cleanup_distributed():
    if is_distributed():
        dist.barrier()
        dist.destroy_process_group()


def reduce_scalar(value: float, device: torch.device, op=dist.ReduceOp.SUM) -> float:
    if not is_distributed():
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def gather_variable_length_array(array: np.ndarray, device: torch.device) -> np.ndarray:
    if not is_distributed():
        return array

    local_size = torch.tensor([array.shape[0]], device=device, dtype=torch.int64)
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(get_world_size())]
    dist.all_gather(gathered_sizes, local_size)
    max_size = int(max(int(x.item()) for x in gathered_sizes))

    padded = np.zeros(max_size, dtype=array.dtype)
    padded[: array.shape[0]] = array
    tensor = torch.from_numpy(padded).to(device)
    gathered = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    dist.all_gather(gathered, tensor)

    arrays = []
    for size, part in zip(gathered_sizes, gathered):
        arrays.append(part[: int(size.item())].cpu().numpy())
    return np.concatenate(arrays, axis=0)


def gather_string_list(strings: List[str], device: torch.device) -> List[str]:
    if not is_distributed():
        return strings
    payload = [None for _ in range(get_world_size())]
    dist.all_gather_object(payload, list(strings))
    merged = []
    for part in payload:
        merged.extend(part)
    return merged


def save_checkpoint(path: str, model: nn.Module, config: TrainConfig, extra: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    payload = {
        "state_dict": state_dict,
        "config": config.__dict__,
        "extra": extra,
    }
    torch.save(payload, path)


def find_pkl_files(pkl_dir: str) -> List[str]:
    logging.info("Scanning PKL files under: %s", pkl_dir)
    files = sorted(glob.glob(os.path.join(pkl_dir, "*.pkl")))
    if not files:
        raise FileNotFoundError(f"No .pkl files found under: {pkl_dir}")
    logging.info("Finished scanning. Total PKL files=%d", len(files))
    return files


def train_val_split(files: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    files = files[:]
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_ratio)) if len(files) > 1 else 0
    val_files = files[:n_val]
    train_files = files[n_val:] if n_val > 0 else files
    if not train_files:
        train_files = val_files
    return train_files, val_files


def count_targets_at_fdr(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.01) -> int:
    if labels.size == 0:
        return 0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.int32, copy=False)
    decoy = 1 - sorted_labels
    target_cum = np.cumsum(sorted_labels)
    decoy_cum = np.cumsum(decoy)
    denom = np.maximum(target_cum, 1)
    q_values = np.minimum.accumulate((decoy_cum / denom)[::-1])[::-1]
    return int(np.sum((sorted_labels == 1) & (q_values <= threshold)))


def count_unique_peptides_at_fdr(labels: np.ndarray, scores: np.ndarray, peptides: List[str], threshold: float = 0.01) -> int:
    if labels.size == 0:
        return 0
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order].astype(np.int32, copy=False)
    sorted_peptides = [clean_sequence(peptides[i]) for i in order]
    decoy = 1 - sorted_labels
    target_cum = np.cumsum(sorted_labels)
    decoy_cum = np.cumsum(decoy)
    denom = np.maximum(target_cum, 1)
    q_values = np.minimum.accumulate((decoy_cum / denom)[::-1])[::-1]
    keep = (sorted_labels == 1) & (q_values <= threshold)
    return len({pep for pep, flag in zip(sorted_peptides, keep) if flag and pep})


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled: bool):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    logging.info(
        "Validation start | rank=%d | steps=%d | local_chunks=%d",
        get_rank(),
        len(loader),
        len(loader.dataset),
    )

    total_loss_sum = 0.0
    total_examples = 0.0
    total_correct = 0.0
    prob_parts = []
    label_parts = []
    peptide_parts: List[str] = []

    for step, (spectra, spectra_mask, precursors, tokens, aux_features, label, weight, peptides) in enumerate(loader, start=1):
        spectra = spectra.to(device, non_blocking=True)
        spectra_mask = spectra_mask.to(device, non_blocking=True)
        precursors = precursors.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        aux_features = aux_features.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(spectra, spectra_mask, precursors, tokens, aux_features)
            loss = criterion(logits, label)
            loss = (loss * weight).mean()

        prob = torch.sigmoid(logits)
        pred = (prob >= 0.5).float()

        batch_size = float(label.numel())
        total_loss_sum += float(loss.item()) * batch_size
        total_examples += batch_size
        total_correct += float((pred == label).sum().item())
        prob_parts.append(prob.detach().cpu().numpy().astype(np.float32))
        label_parts.append(label.detach().cpu().numpy().astype(np.float32))
        peptide_parts.extend(peptides)

        if step == 1 or step % 20 == 0 or step == len(loader):
            logging.info(
                "Validation rank=%d step=%d/%d (%.2f%%)",
                get_rank(),
                step,
                len(loader),
                100.0 * step / max(1, len(loader)),
            )

    total_loss_sum = reduce_scalar(total_loss_sum, device)
    total_examples = reduce_scalar(total_examples, device)
    total_correct = reduce_scalar(total_correct, device)

    if total_examples == 0:
        return {
            "loss": 0.0,
            "acc": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "targets_at_fdr01": 0,
            "unique_peptides_at_fdr01": 0,
        }

    probs = np.concatenate(prob_parts, axis=0) if prob_parts else np.empty(0, dtype=np.float32)
    labels = np.concatenate(label_parts, axis=0) if label_parts else np.empty(0, dtype=np.float32)

    probs = gather_variable_length_array(probs, device)
    labels = gather_variable_length_array(labels, device)
    peptides = gather_string_list(peptide_parts, device)

    roc_auc = 0.0
    pr_auc = 0.0
    targets_at_fdr01 = 0
    unique_peptides_at_fdr01 = 0
    if is_main_process() and labels.size > 0 and len(np.unique(labels)) > 1:
        roc_auc = float(roc_auc_score(labels, probs))
        pr_auc = float(average_precision_score(labels, probs))
        targets_at_fdr01 = count_targets_at_fdr(labels.astype(np.int32), probs)
        unique_peptides_at_fdr01 = count_unique_peptides_at_fdr(labels.astype(np.int32), probs, peptides)

    roc_auc = reduce_scalar(roc_auc, device)
    pr_auc = reduce_scalar(pr_auc, device)
    targets_at_fdr01 = int(reduce_scalar(float(targets_at_fdr01), device))
    unique_peptides_at_fdr01 = int(reduce_scalar(float(unique_peptides_at_fdr01), device))

    return {
        "loss": total_loss_sum / total_examples,
        "acc": total_correct / total_examples,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "targets_at_fdr01": targets_at_fdr01,
        "unique_peptides_at_fdr01": unique_peptides_at_fdr01,
    }


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory, sampler=None):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": (shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
        "collate_fn": collate_pkl_chunks,
        "persistent_workers": (num_workers > 0),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Train an AIPC reranking model from PKL chunks on 1 or multiple GPUs.")
    parser.add_argument("--pkl_dir", required=True, help="Directory containing .pkl files from 3_convert_parquet2pkl.py")
    parser.add_argument("--parquet_dir", default="", help="Matching parquet split directory for auxiliary tabular features")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size used during encoding")
    parser.add_argument("--batch_size", type=int, default=4096, help="Global target effective sample batch size across all GPUs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--token_embed_dim", type=int, default=128)
    parser.add_argument("--precursor_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_token_len", type=int, default=64)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()

    distributed, rank, world_size, device = setup_distributed()
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    os.makedirs(args.output_dir, exist_ok=True)
    handlers = []
    if is_main_process():
        handlers.append(logging.FileHandler(os.path.join(args.output_dir, "train.log"), encoding="utf-8"))
    handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | rank={rank} | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )

    cfg = TrainConfig(
        pkl_dir=args.pkl_dir,
        parquet_dir=args.parquet_dir or None,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        token_embed_dim=args.token_embed_dim,
        precursor_dim=args.precursor_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_token_len=args.max_token_len,
        grad_clip=args.grad_clip,
        amp=not args.disable_amp,
    )

    set_seed(cfg.seed + rank)
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    logging.info("Using device=%s | distributed=%s | world_size=%d", device, distributed, world_size)

    files = find_pkl_files(cfg.pkl_dir)
    train_files, val_files = train_val_split(files, cfg.val_ratio, cfg.seed)
    logging.info("Found %d PKL files | train=%d | val=%d", len(files), len(train_files), len(val_files))

    train_ds = PklChunkBatchDataset(train_files, parquet_dir=cfg.parquet_dir, name="train")
    val_source = val_files if val_files else train_files[:1]
    val_ds = PklChunkBatchDataset(val_source, parquet_dir=cfg.parquet_dir, name="val")
    logging.info("Indexed samples | train~=%d | val~=%d", train_ds.total_rows, val_ds.total_rows)

    rows_per_chunk = train_ds.rows_per_chunk if train_ds.rows_per_chunk is not None else 256
    global_chunk_batch_size = max(1, cfg.batch_size // max(1, rows_per_chunk))
    per_rank_chunk_batch_size = max(1, global_chunk_batch_size // world_size)

    logging.info(
        "rows_per_chunk=%d | requested_global_batch=%d | global_chunk_batch_size=%d | per_rank_chunk_batch_size=%d | per_rank_effective_samples~=%d | global_effective_samples~=%d",
        rows_per_chunk,
        cfg.batch_size,
        global_chunk_batch_size,
        per_rank_chunk_batch_size,
        per_rank_chunk_batch_size * rows_per_chunk,
        per_rank_chunk_batch_size * rows_per_chunk * world_size,
    )

    actual_workers = max(0, cfg.num_workers)
    pin_memory = device.type == "cuda"

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False) if distributed else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False) if distributed else None

    train_loader = build_dataloader(
        train_ds,
        batch_size=per_rank_chunk_batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=pin_memory,
        sampler=train_sampler,
    )
    val_loader = build_dataloader(
        val_ds,
        batch_size=per_rank_chunk_batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=pin_memory,
        sampler=val_sampler,
    )

    model = AIPCRerankNet(
        vocab_size=cfg.vocab_size,
        token_embed_dim=cfg.token_embed_dim,
        precursor_dim=cfg.precursor_dim,
        hidden_dim=cfg.hidden_dim,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        max_token_len=cfg.max_token_len,
    ).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_metric = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        model.train()
        running_loss_sum = 0.0
        seen = 0.0
        start = time.time()

        logging.info(
            "epoch=%d start | train_steps=%d | per_rank_effective_samples~=%d | local_train_samples~=%d",
            epoch,
            len(train_loader),
            per_rank_chunk_batch_size * rows_per_chunk,
            train_ds.total_rows,
        )

        for step, (spectra, spectra_mask, precursors, tokens, aux_features, label, weight, _) in enumerate(train_loader, start=1):
            spectra = spectra.to(device, non_blocking=True)
            spectra_mask = spectra_mask.to(device, non_blocking=True)
            precursors = precursors.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            aux_features = aux_features.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(spectra, spectra_mask, precursors, tokens, aux_features)
                loss = criterion(logits, label)
                loss = (loss * weight).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size_now = float(label.numel())
            running_loss_sum += float(loss.item()) * batch_size_now
            seen += batch_size_now

            if step == 1 or step % 20 == 0 or step == len(train_loader):
                logging.info(
                    "epoch=%d step=%d/%d (%.2f%%) train_loss=%.6f lr=%.6e",
                    epoch,
                    step,
                    len(train_loader),
                    100.0 * step / max(1, len(train_loader)),
                    running_loss_sum / max(1.0, seen),
                    optimizer.param_groups[0]["lr"],
                )

        scheduler.step()

        train_loss_sum_all = reduce_scalar(running_loss_sum, device)
        seen_all = reduce_scalar(seen, device)
        train_loss = train_loss_sum_all / max(1.0, seen_all)
        val_metrics = evaluate(model, val_loader, device, amp_enabled)
        elapsed = time.time() - start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_targets_at_fdr01": val_metrics["targets_at_fdr01"],
            "val_unique_peptides_at_fdr01": val_metrics["unique_peptides_at_fdr01"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
            "world_size": world_size,
        }

        if is_main_process():
            history.append(record)
            logging.info(
                "epoch=%d done | train_loss=%.6f | val_loss=%.6f | val_acc=%.4f | val_auc=%.4f | val_pr_auc=%.4f | val_targets@1%%=%d | val_unique_peptides@1%%=%d | time=%.1fs",
                epoch,
                train_loss,
                val_metrics["loss"],
                val_metrics["acc"],
                val_metrics["roc_auc"],
                val_metrics["pr_auc"],
                val_metrics["targets_at_fdr01"],
                val_metrics["unique_peptides_at_fdr01"],
                elapsed,
            )

            latest_path = os.path.join(cfg.output_dir, "latest.pt")
            save_checkpoint(latest_path, model, cfg, {"history": history, "best_metric": best_metric})

            metric = float(val_metrics["unique_peptides_at_fdr01"])
            if metric > best_metric:
                best_metric = metric
                best_path = os.path.join(cfg.output_dir, "best.pt")
                save_checkpoint(best_path, model, cfg, {"history": history, "best_metric": best_metric})
                logging.info("Saved new best checkpoint to %s", best_path)

            with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)

        if is_distributed():
            dist.barrier()

    if is_main_process():
        logging.info("Training finished. Best val_unique_peptides@1%%=%d", int(best_metric))

    cleanup_distributed()


if __name__ == "__main__":
    main()
