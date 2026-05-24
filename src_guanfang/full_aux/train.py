from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .common import VOCAB, compute_batch_aux_features, count_targets_and_unique_peptides
from .model import FullAuxRescorer

LOGGER = logging.getLogger("full_aux_train")
PKL_ROWS = 256


@dataclass
class TrainConfig:
    pkl_dir: str
    output_dir: str
    init_model_path: str = ""
    batch_size: int = 4096
    epochs: int = 3
    lr: float = 5e-4
    weight_decay: float = 1e-4
    val_ratio: float = 0.05
    num_workers: int = 4
    seed: int = 42
    token_embed_dim: int = 128
    precursor_dim: int = 64
    hidden_dim: int = 256
    n_heads: int = 8
    n_layers: int = 2
    dropout: float = 0.1
    max_token_len: int = 50
    grad_clip: float = 1.0
    amp: bool = True


class PklBatchDataset(Dataset):
    def __init__(self, file_paths: list[str]) -> None:
        self.file_paths = file_paths

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> dict:
        with open(self.file_paths[index], "rb") as handle:
            return pickle.load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(rank: int) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_main_process(rank: int) -> bool:
    return rank == 0


def collate_pkl_batches(batch: list[dict]) -> dict:
    max_peaks = max(item["spectra"].shape[1] for item in batch)
    spectra_list = []
    spectra_mask_list = []
    for item in batch:
        current_peaks = item["spectra"].shape[1]
        if current_peaks < max_peaks:
            pad_width = max_peaks - current_peaks
            spectra = np.pad(item["spectra"], ((0, 0), (0, pad_width), (0, 0)), mode="constant")
            spectra_mask = np.pad(item["spectra_mask"], ((0, 0), (0, pad_width)), mode="constant", constant_values=True)
        else:
            spectra = item["spectra"]
            spectra_mask = item["spectra_mask"]
        spectra_list.append(spectra)
        spectra_mask_list.append(spectra_mask)
    merged = {
        "spectra": np.concatenate(spectra_list, axis=0),
        "spectra_mask": np.concatenate(spectra_mask_list, axis=0),
        "precursors": np.concatenate([item["precursors"] for item in batch], axis=0),
        "tokens": np.concatenate([item["tokens"] for item in batch], axis=0),
        "label": np.concatenate([item["label"] for item in batch], axis=0),
        "weight": np.concatenate([item["weight"] for item in batch], axis=0),
        "aux_features": np.concatenate([item["aux_features"] for item in batch], axis=0),
        "peptides": sum((list(item["peptides"]) for item in batch), []),
    }
    return merged


def build_precursor_inputs(precursors: torch.Tensor) -> torch.Tensor:
    precursor_mass = precursors[:, 0]
    precursor_charge = precursors[:, 1]
    precursor_inputs = torch.stack(
        [precursor_mass / 4000.0, precursor_charge / 10.0],
        dim=1,
    )
    return precursor_inputs


def build_batch_tensors(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    spectra = torch.from_numpy(batch["spectra"]).to(device=device, dtype=torch.float32, non_blocking=True)
    spectra_mask = torch.from_numpy(batch["spectra_mask"]).to(device=device, dtype=torch.bool, non_blocking=True)
    precursors = torch.from_numpy(batch["precursors"]).to(device=device, dtype=torch.float32, non_blocking=True)
    tokens = torch.from_numpy(batch["tokens"]).to(device=device, dtype=torch.long, non_blocking=True)
    labels = torch.from_numpy(batch["label"]).to(device=device, dtype=torch.float32, non_blocking=True)
    weights = torch.from_numpy(batch["weight"]).to(device=device, dtype=torch.float32, non_blocking=True)
    aux_seed = torch.from_numpy(batch["aux_features"]).to(device=device, dtype=torch.float32, non_blocking=True)
    predicted_rt = aux_seed[:, 3]
    delta_rt = aux_seed[:, 4]
    aux_features = compute_batch_aux_features(
        spectra=spectra,
        spectra_mask=spectra_mask,
        precursor_masses=precursors[:, 0],
        precursor_charges=precursors[:, 1],
        predicted_rt=predicted_rt,
        delta_rt=delta_rt,
        sequences=batch["peptides"],
    )
    precursor_inputs = build_precursor_inputs(precursors)
    return spectra, spectra_mask, precursor_inputs, tokens, aux_features, labels, weights


def make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    losses: list[float] = []
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_sequences: list[str] = []

    for batch in val_loader:
        spectra, spectra_mask, precursor_inputs, tokens, aux_features, labels, weights = build_batch_tensors(batch, device)
        autocast_enabled = use_amp and device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model(spectra, spectra_mask, precursor_inputs, tokens, aux_features)
            loss = loss_fn(logits, labels) * weights
        losses.append(float(loss.mean().item()))
        all_scores.append(torch.sigmoid(logits).float().cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_sequences.extend(batch["peptides"])

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    targets_at_fdr01, unique_peptides_at_fdr01 = count_targets_and_unique_peptides(scores, labels, all_sequences)
    metrics = {
        "val_loss": float(np.mean(losses)),
        "val_acc": float(((scores >= 0.5) == labels).mean()),
        "val_roc_auc": float(roc_auc_score(labels, scores)),
        "val_pr_auc": float(average_precision_score(labels, scores)),
        "val_targets_at_fdr01": float(targets_at_fdr01),
        "val_unique_peptides_at_fdr01": float(unique_peptides_at_fdr01),
    }
    model.train()
    return metrics


def split_files(file_paths: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = file_paths[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_count:], shuffled[:val_count]


def save_checkpoint(
    model: nn.Module,
    config: TrainConfig,
    output_dir: str,
    history: list[dict],
    best_metric: float,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "extra": {
            "history": history,
            "best_metric": best_metric,
        },
    }
    torch.save(checkpoint, os.path.join(output_dir, "best.pt"))
    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as handle:
        json.dump(history, handle, ensure_ascii=False, indent=2)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl_dir", default="./data/mzml_pkl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--init_model_path", default="")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
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
    parser.add_argument("--max_token_len", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()
    amp = True
    if args.no_amp:
        amp = False
    elif args.amp:
        amp = True
    return TrainConfig(
        pkl_dir=args.pkl_dir,
        output_dir=args.output_dir,
        init_model_path=args.init_model_path,
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
        amp=amp,
    )


def main() -> None:
    rank, local_rank, world_size = init_distributed()
    setup_logging(rank)
    config = parse_args()
    set_seed(config.seed + rank)
    os.makedirs(config.output_dir, exist_ok=True)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    file_paths = sorted(
        os.path.join(config.pkl_dir, file_name)
        for file_name in os.listdir(config.pkl_dir)
        if file_name.endswith(".pkl")
    )
    train_files, val_files = split_files(file_paths, config.val_ratio, config.seed)
    if is_main_process(rank):
        LOGGER.info("train files=%s val files=%s", len(train_files), len(val_files))

    files_per_step = max(1, config.batch_size // PKL_ROWS)
    train_dataset = PklBatchDataset(train_files)
    val_dataset = PklBatchDataset(val_files)
    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=files_per_step,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_pkl_batches,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=files_per_step,
        shuffle=False,
        num_workers=max(1, min(2, config.num_workers)),
        pin_memory=True,
        collate_fn=collate_pkl_batches,
        persistent_workers=config.num_workers > 0,
    )

    model = FullAuxRescorer(
        vocab_size=len(VOCAB),
        token_embed_dim=config.token_embed_dim,
        precursor_dim=config.precursor_dim,
        hidden_dim=config.hidden_dim,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
    ).to(device)
    if config.init_model_path:
        checkpoint = torch.load(config.init_model_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        load_info = model.load_state_dict(state_dict, strict=False)
        if is_main_process(rank):
            LOGGER.info("loaded init model from %s", config.init_model_path)
            LOGGER.info("missing_keys=%s unexpected_keys=%s", load_info.missing_keys, load_info.unexpected_keys)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = make_optimizer(model, config)
    total_steps = max(1, len(train_loader) * config.epochs)
    scheduler = make_scheduler(optimizer, total_steps)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    history: list[dict] = []
    best_metric = -1.0
    global_step = 0

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if world_size > 1:
            dist.barrier()
        model.train()
        running_losses: list[float] = []

        for batch in train_loader:
            spectra, spectra_mask, precursor_inputs, tokens, aux_features, labels, weights = build_batch_tensors(batch, device)
            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = config.amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                logits = model(spectra, spectra_mask, precursor_inputs, tokens, aux_features)
                loss = loss_fn(logits, labels)
                loss = (loss * weights).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_losses.append(float(loss.item()))

        distributed_barrier()

        if is_main_process(rank):
            metrics = evaluate(model.module if isinstance(model, DDP) else model, val_loader, device, config.amp)
            epoch_record = {
                "epoch": epoch,
                "train_loss": float(np.mean(running_losses)),
                "lr": float(scheduler.get_last_lr()[0]),
                "seconds": float(time.time() - epoch_start),
                "world_size": world_size,
                **metrics,
            }
            history.append(epoch_record)
            LOGGER.info("epoch=%s metrics=%s", epoch, epoch_record)
            current_metric = metrics["val_unique_peptides_at_fdr01"]
            if current_metric > best_metric:
                best_metric = current_metric
                save_checkpoint(model.module if isinstance(model, DDP) else model, config, config.output_dir, history, best_metric)

        distributed_barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
