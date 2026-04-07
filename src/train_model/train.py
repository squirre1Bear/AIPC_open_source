import os
import glob
import json
import time
import copy
import random
import pickle
import logging
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from model_rerank import AIPCRerankNet

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PklChunkBatchDataset(Dataset):
    """
    Each PKL file already stores a pre-collated mini-batch:
      spectra:      [B, L, 2]
      spectra_mask: [B, L]
      precursors:   [B, 2]
      tokens:       [B, T]
      label:        [B]
      weight:       [B]

    So the right abstraction is:
      one PKL file == one dataset item
    instead of expanding into row-level items.
    """

    def __init__(self, pkl_files: List[str], log_every: int = 500, name: str = "dataset"):
        self.pkl_files = pkl_files
        self.name = name
        self.chunk_sizes: List[int] = []
        self.total_rows = 0
        self.rows_per_chunk: Optional[int] = None

        total_files = len(pkl_files)
        logging.info("[%s] Start indexing %d PKL files...", name, total_files)

        for file_idx, path in enumerate(pkl_files, start=1):
            with open(path, "rb") as f:
                obj = pickle.load(f)

            n = int(len(obj["label"]))
            self.chunk_sizes.append(n)
            self.total_rows += n
            if self.rows_per_chunk is None:
                self.rows_per_chunk = n

            if file_idx == 1 or file_idx % log_every == 0 or file_idx == total_files:
                logging.info(
                    "[%s] Indexed %d/%d PKL files (%.2f%%), accumulated samples=%d, latest=%s",
                    name,
                    file_idx,
                    total_files,
                    100.0 * file_idx / max(1, total_files),
                    self.total_rows,
                    os.path.basename(path),
                )

        logging.info("[%s] Finished indexing. Total samples=%d", name, self.total_rows)

    def __len__(self) -> int:
        return len(self.pkl_files)

    def __getitem__(self, idx: int):
        path = self.pkl_files[idx]
        with open(path, "rb") as f:
            obj = pickle.load(f)

        n = int(len(obj["label"]))
        weight_arr = obj.get("weight", np.ones(n, dtype=np.float32))

        # Copy to make tensors safe/contiguous
        spectra = torch.from_numpy(np.asarray(obj["spectra"], dtype=np.float32).copy())            # [B, L, 2]
        spectra_mask = torch.from_numpy(np.asarray(obj["spectra_mask"], dtype=np.bool_).copy())    # [B, L]
        precursors = torch.from_numpy(np.asarray(obj["precursors"], dtype=np.float32).copy())      # [B, 2]
        tokens = torch.from_numpy(np.asarray(obj["tokens"], dtype=np.int64).copy())                 # [B, T]
        label = torch.from_numpy(np.asarray(obj["label"], dtype=np.float32).copy())                 # [B]
        weight = torch.from_numpy(np.asarray(weight_arr, dtype=np.float32).copy())                  # [B]

        return spectra, spectra_mask, precursors, tokens, label, weight


def collate_pkl_chunks(batch):
    """
    Merge multiple pre-collated PKL chunks into one training batch.

    Each item in batch:
      spectra:      [B_i, L_i, 2]
      spectra_mask: [B_i, L_i]
      precursors:   [B_i, 2]
      tokens:       [B_i, T_i]
      label:        [B_i]
      weight:       [B_i]

    Need to pad across chunks because different PKL files may have different L_i.
    """
    spectra_list, spectra_mask_list, precursors_list, tokens_list, label_list, weight_list = zip(*batch)

    max_spec_len = max(x.size(1) for x in spectra_list)
    max_tok_len = max(x.size(1) for x in tokens_list)

    padded_spectra = []
    padded_masks = []
    padded_tokens = []

    for spectra, spectra_mask, tokens in zip(spectra_list, spectra_mask_list, tokens_list):
        # spectra: [B, L, 2]
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
    label = torch.cat(label_list, dim=0).float()
    weight = torch.cat(weight_list, dim=0).float()

    return spectra, spectra_mask, precursors, tokens, label, weight


@dataclass
class TrainConfig:
    pkl_dir: str
    output_dir: str
    vocab_size: int
    batch_size: int = 256   # target effective sample batch size, not DataLoader chunk count
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


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled: bool):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_score = []
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    logging.info("Validation start | steps=%d | chunks=%d", len(loader), len(loader.dataset))

    for step, (spectra, spectra_mask, precursors, tokens, label, weight) in enumerate(loader, start=1):
        spectra = spectra.to(device, non_blocking=True)
        spectra_mask = spectra_mask.to(device, non_blocking=True)
        precursors = precursors.to(device, non_blocking=True)
        tokens = tokens.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(spectra, spectra_mask, precursors, tokens)
            loss = criterion(logits, label)
            loss = (loss * weight).mean()

        total_loss += loss.item() * label.size(0)
        y_true.append(label.detach().cpu())
        y_score.append(torch.sigmoid(logits).detach().cpu())

        if step == 1 or step % 20 == 0 or step == len(loader):
            logging.info(
                "Validation step=%d/%d (%.2f%%)",
                step,
                len(loader),
                100.0 * step / max(1, len(loader)),
            )

    if len(y_true) == 0:
        return {"loss": 0.0, "acc": 0.0}

    y_true = torch.cat(y_true).numpy()
    y_score = torch.cat(y_score).numpy()
    y_pred = (y_score >= 0.5).astype(np.float32)
    acc = float((y_pred == y_true).mean())
    return {"loss": total_loss / max(1, len(y_true)), "acc": acc}


def save_checkpoint(path: str, model: nn.Module, config: TrainConfig, extra: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": config.__dict__,
        "extra": extra,
    }
    torch.save(payload, path)


def main():
    parser = argparse.ArgumentParser(description="Train an AIPC reranking model from pre-encoded PKL chunks.")
    parser.add_argument("--pkl_dir", required=True, help="Directory containing .pkl files from 3_convert_parquet2pkl.py")
    parser.add_argument("--output_dir", required=True, help="Directory to save checkpoints and logs")
    parser.add_argument("--vocab_size", type=int, required=True, help="Vocabulary size used during encoding")
    parser.add_argument("--batch_size", type=int, default=256, help="Target effective sample batch size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
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

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, "train.log"), encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    cfg = TrainConfig(
        pkl_dir=args.pkl_dir,
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

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    logging.info("Using device: %s", device)

    files = find_pkl_files(cfg.pkl_dir)
    train_files, val_files = train_val_split(files, cfg.val_ratio, cfg.seed)
    logging.info("Found %d PKL files | train=%d | val=%d", len(files), len(train_files), len(val_files))

    train_ds = PklChunkBatchDataset(train_files, name="train")
    val_ds = PklChunkBatchDataset(val_files, name="val") if val_files else PklChunkBatchDataset(train_files[:1], name="val_fallback")
    logging.info("Indexed samples | train=%d | val=%d", train_ds.total_rows, val_ds.total_rows)

    rows_per_chunk = train_ds.rows_per_chunk if train_ds.rows_per_chunk is not None else 256
    # interpret --batch_size as target effective sample batch size
    chunk_batch_size = max(1, cfg.batch_size // max(1, rows_per_chunk))
    logging.info(
        "PKL rows_per_chunk=%d | requested batch_size=%d | DataLoader chunk_batch_size=%d | effective_samples_per_step≈%d",
        rows_per_chunk,
        cfg.batch_size,
        chunk_batch_size,
        chunk_batch_size * rows_per_chunk,
    )

    actual_workers = max(0, cfg.num_workers)
    use_persistent = actual_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=chunk_batch_size,
        shuffle=True,
        num_workers=actual_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=use_persistent,
        collate_fn=collate_pkl_chunks,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=chunk_batch_size,
        shuffle=False,
        num_workers=actual_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=use_persistent,
        collate_fn=collate_pkl_chunks,
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_state = None
    best_metric = -1.0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        start = time.time()

        logging.info(
            "epoch=%d start | train_steps=%d | effective_samples_per_step≈%d | train_samples=%d",
            epoch,
            len(train_loader),
            chunk_batch_size * rows_per_chunk,
            train_ds.total_rows,
        )

        for step, (spectra, spectra_mask, precursors, tokens, label, weight) in enumerate(train_loader, start=1):
            spectra = spectra.to(device, non_blocking=True)
            spectra_mask = spectra_mask.to(device, non_blocking=True)
            precursors = precursors.to(device, non_blocking=True)
            tokens = tokens.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(spectra, spectra_mask, precursors, tokens)
                loss = criterion(logits, label)
                loss = (loss * weight).mean()

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * label.size(0)
            seen += label.size(0)

            if step == 1 or step % 20 == 0 or step == len(train_loader):
                logging.info(
                    "epoch=%d step=%d/%d (%.2f%%) train_loss=%.6f lr=%.6e",
                    epoch,
                    step,
                    len(train_loader),
                    100.0 * step / max(1, len(train_loader)),
                    running_loss / max(1, seen),
                    optimizer.param_groups[0]["lr"],
                )

        scheduler.step()
        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, val_loader, device, amp_enabled)
        elapsed = time.time() - start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        history.append(record)

        logging.info(
            "epoch=%d done | train_loss=%.6f | val_loss=%.6f | val_acc=%.4f | time=%.1fs",
            epoch,
            train_loss,
            val_metrics["loss"],
            val_metrics["acc"],
            elapsed,
        )

        metric = val_metrics["acc"]
        latest_path = os.path.join(cfg.output_dir, "latest.pt")
        save_checkpoint(latest_path, model, cfg, {"history": history, "best_metric": best_metric})

        if metric > best_metric:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            best_path = os.path.join(cfg.output_dir, "best.pt")
            torch.save(
                {
                    "state_dict": best_state,
                    "config": cfg.__dict__,
                    "extra": {"history": history, "best_metric": best_metric},
                },
                best_path,
            )
            logging.info("Saved new best checkpoint to %s", best_path)

    with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    logging.info("Training finished. Best val_acc=%.4f", best_metric)


if __name__ == "__main__":
    main()


#python src/train_model/train.py --pkl_dir E:/AIPC_dataset/ms_data_pkl --output_dir E:/AIPC_runs/run1 --vocab_size 29 --batch_size 256 --epochs 3 --lr 1e-3 --num_workers 8

