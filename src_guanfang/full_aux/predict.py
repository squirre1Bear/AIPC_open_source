from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.distributed as dist

from ..transformer.dataset import padding
from .common import PROTON_MASS_AMU, VOCAB, build_prediction_frame, tokenize_sequence
from .model import FullAuxRescorer

LOGGER = logging.getLogger("full_aux_predict")


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


def assign_files_by_size(file_paths: list[str], world_size: int) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(world_size)]
    totals = [0 for _ in range(world_size)]
    sized_files = sorted(
        ((path, os.path.getsize(path)) for path in file_paths),
        key=lambda item: item[1],
        reverse=True,
    )
    for path, size in sized_files:
        target_rank = min(range(world_size), key=lambda idx: totals[idx])
        buckets[target_rank].append(path)
        totals[target_rank] += size
    return buckets


def fast_process_spectrum(
    mz_array: Any,
    intensity_array: Any,
    precursor_mz: float,
    n_peaks: int = 300,
    min_mz: float = 50.0,
    max_mz: float = 2500.0,
    min_intensity: float = 0.01,
    remove_precursor_tol: float = 2.0,
) -> np.ndarray:
    mz = np.asarray(mz_array, dtype=np.float32)
    intensity = np.asarray(intensity_array, dtype=np.float32)
    if mz.size == 0 or intensity.size == 0:
        return np.array([[0.0, 1.0]], dtype=np.float32)

    keep = (mz >= min_mz) & (mz <= max_mz)
    mz = mz[keep]
    intensity = intensity[keep]
    if mz.size == 0:
        return np.array([[0.0, 1.0]], dtype=np.float32)

    keep = np.abs(mz - precursor_mz) > remove_precursor_tol
    mz = mz[keep]
    intensity = intensity[keep]
    if mz.size == 0:
        return np.array([[0.0, 1.0]], dtype=np.float32)

    max_int = float(intensity.max())
    if max_int > 0.0:
        keep = intensity >= max_int * min_intensity
        mz = mz[keep]
        intensity = intensity[keep]
    if mz.size == 0:
        return np.array([[0.0, 1.0]], dtype=np.float32)

    if mz.size > n_peaks:
        keep_idx = np.argpartition(intensity, -n_peaks)[-n_peaks:]
        keep_idx = np.sort(keep_idx)
        mz = mz[keep_idx]
        intensity = intensity[keep_idx]

    intensity = np.sqrt(intensity)
    norm = float(np.linalg.norm(intensity))
    if norm > 0.0:
        intensity = intensity / norm
    return np.stack([mz, intensity], axis=1).astype(np.float32)


def load_model(model_path: str, device: torch.device) -> tuple[FullAuxRescorer, dict]:
    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]
    model = FullAuxRescorer(
        vocab_size=len(VOCAB),
        token_embed_dim=config["token_embed_dim"],
        precursor_dim=config["precursor_dim"],
        hidden_dim=config["hidden_dim"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config


@torch.no_grad()
def encode_spectra_batch(
    model: FullAuxRescorer,
    spectra_arrays: list[np.ndarray],
    device: torch.device,
    chunk_size: int = 2048,
) -> np.ndarray:
    all_features: list[np.ndarray] = []
    autocast_enabled = device.type == "cuda"
    for start in range(0, len(spectra_arrays), chunk_size):
        chunk = [torch.tensor(item, dtype=torch.float32) for item in spectra_arrays[start : start + chunk_size]]
        spectra, spectra_mask = padding(chunk)
        spectra = spectra.to(device=device, dtype=torch.float32, non_blocking=True)
        spectra_mask = spectra_mask.to(device=device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            spec_feat = model.encode_spectrum_branch(spectra, spectra_mask)
        all_features.append(spec_feat.float().cpu().numpy())
    return np.concatenate(all_features, axis=0)


@torch.no_grad()
def encode_token_batch(
    model: FullAuxRescorer,
    token_tensors: list[torch.Tensor],
    device: torch.device,
    chunk_size: int = 4096,
) -> np.ndarray:
    all_features: list[np.ndarray] = []
    autocast_enabled = device.type == "cuda"
    for start in range(0, len(token_tensors), chunk_size):
        tokens = torch.stack(token_tensors[start : start + chunk_size], dim=0).to(device=device, dtype=torch.long, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            token_feat = model.encode_token_branch(tokens)
        all_features.append(token_feat.float().cpu().numpy())
    return np.concatenate(all_features, axis=0)


@torch.no_grad()
def score_classifier_batch(
    model: FullAuxRescorer,
    token_features: np.ndarray,
    spectrum_features: np.ndarray,
    precursor_inputs: np.ndarray,
    aux_features: np.ndarray,
    device: torch.device,
    chunk_size: int = 16384,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    autocast_enabled = device.type == "cuda"
    for start in range(0, len(token_features), chunk_size):
        token_feat = torch.tensor(token_features[start : start + chunk_size], dtype=torch.float32, device=device)
        spec_feat = torch.tensor(spectrum_features[start : start + chunk_size], dtype=torch.float32, device=device)
        precursors = torch.tensor(precursor_inputs[start : start + chunk_size], dtype=torch.float32, device=device)
        aux = torch.tensor(aux_features[start : start + chunk_size], dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = model.score_from_branch_features(token_feat, spec_feat, precursors, aux)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(scores, axis=0)


@torch.no_grad()
def predict_file(
    model: FullAuxRescorer,
    file_path: str,
    max_token_len: int,
    batch_size: int,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parquet_file = pq.ParquetFile(file_path)
    pred_parts: list[pd.DataFrame] = []
    meta_parts: list[pd.DataFrame] = []

    spectrum_cache: dict[int, tuple[np.ndarray, float, float, np.ndarray]] = {}
    token_cache: dict[str, np.ndarray] = {}

    for record_batch in parquet_file.iter_batches(batch_size=batch_size):
        batch = record_batch.to_pydict()
        row_count = len(batch["index"])
        meta_parts.append(
            pd.DataFrame(
                {
                    "scan_number": batch["scan_number"],
                    "precursor_mz": batch["precursor_mz"],
                    "charge": batch["charge"],
                    "precursor_sequence": batch["precursor_sequence"],
                    "label": batch.get("label", [1.0] * row_count),
                    "index": batch["index"],
                }
            )
        )

        missing_scans: list[int] = []
        missing_spectra: list[np.ndarray] = []
        missing_scan_stats: list[tuple[float, float]] = []
        seen_missing_scans: set[int] = set()
        for row_idx in range(row_count):
            scan_number = int(batch["scan_number"][row_idx])
            if scan_number in spectrum_cache or scan_number in seen_missing_scans:
                continue
            processed = fast_process_spectrum(
                batch["mz_array"][row_idx],
                batch["intensity_array"][row_idx],
                float(batch["precursor_mz"][row_idx]),
            )
            missing_scans.append(scan_number)
            missing_spectra.append(processed)
            missing_scan_stats.append(
                (
                    float(len(processed)) / 300.0,
                    float(processed[:, 0].mean()) / 2000.0,
                )
            )
            seen_missing_scans.add(scan_number)
        if missing_scans:
            spec_features = encode_spectra_batch(model, missing_spectra, device)
            for scan_number, processed, (valid_peak_ratio, mean_peak_mz), spec_feat in zip(
                missing_scans,
                missing_spectra,
                missing_scan_stats,
                spec_features,
            ):
                spectrum_cache[scan_number] = (processed, valid_peak_ratio, mean_peak_mz, spec_feat.astype(np.float32))

        missing_sequences: list[str] = []
        missing_tokens: list[torch.Tensor] = []
        seen_missing_sequences: set[str] = set()
        for sequence in batch["precursor_sequence"]:
            if sequence in token_cache or sequence in seen_missing_sequences:
                continue
            token_tensor = tokenize_sequence(sequence, max_token_len)
            missing_sequences.append(sequence)
            missing_tokens.append(token_tensor)
            seen_missing_sequences.add(sequence)
        if missing_sequences:
            token_features = encode_token_batch(model, missing_tokens, device)
            for sequence, token_feat in zip(missing_sequences, token_features):
                token_cache[sequence] = token_feat.astype(np.float32)

        token_features: list[np.ndarray] = []
        spectrum_features: list[np.ndarray] = []
        precursor_inputs: list[list[float]] = []
        aux_features: list[list[float]] = []
        indices: list[int] = []
        labels: list[float] = []

        for row_idx in range(row_count):
            scan_number = int(batch["scan_number"][row_idx])
            sequence = batch["precursor_sequence"][row_idx]
            _, valid_peak_ratio, mean_peak_mz, spec_feat = spectrum_cache[scan_number]
            token_feat = token_cache[sequence]
            precursor_mz = float(batch["precursor_mz"][row_idx])
            charge = float(batch["charge"][row_idx])
            precursor_mass = (precursor_mz - PROTON_MASS_AMU) * charge
            predicted_rt = float(batch.get("predicted_rt", [0.0] * row_count)[row_idx] or 0.0)
            delta_rt = float(batch.get("delta_rt_model", [0.0] * row_count)[row_idx] or 0.0)
            peptide_length = len(
                sequence.replace("n[42]", "")
                .replace("C[57.02]", "C")
                .replace("M[15.99]", "M")
                .replace("N[.98]", "N")
                .replace("Q[.98]", "Q")
            )
            mod_count = sequence.count("[")

            token_features.append(token_feat)
            spectrum_features.append(spec_feat)
            precursor_inputs.append([precursor_mass / 4000.0, charge / 10.0])
            aux_features.append(
                [
                    precursor_mz / 2000.0,
                    charge / 10.0,
                    float(np.clip(predicted_rt - delta_rt, 0.0, 1.0)),
                    predicted_rt,
                    delta_rt,
                    abs(delta_rt),
                    valid_peak_ratio,
                    mean_peak_mz,
                    peptide_length / 50.0,
                    mod_count / 5.0,
                ]
            )
            indices.append(int(batch["index"][row_idx]))
            labels.append(float(batch.get("label", [1.0] * row_count)[row_idx]))

        scores = score_classifier_batch(
            model=model,
            token_features=np.asarray(token_features, dtype=np.float32),
            spectrum_features=np.asarray(spectrum_features, dtype=np.float32),
            precursor_inputs=np.asarray(precursor_inputs, dtype=np.float32),
            aux_features=np.asarray(aux_features, dtype=np.float32),
            device=device,
        )
        pred_parts.append(
            pd.DataFrame(
                {
                    "index": np.asarray(indices, dtype=np.int64),
                    "score": scores,
                    "label": np.asarray(labels, dtype=np.float32),
                    "weight": np.ones(len(indices), dtype=np.float32),
                }
            )
        )

    pred_df = pd.concat(pred_parts, ignore_index=True)
    pred_df["index"] = pred_df["index"].astype(int)
    pred_df = pred_df.drop_duplicates(subset="index", keep="first").sort_values("index").reset_index(drop=True)
    meta_df = pd.concat(meta_parts, ignore_index=True)
    meta_df["index"] = meta_df["index"].astype(int)
    return pred_df, meta_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--q_value_threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    rank, local_rank, world_size = init_distributed()
    logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    os.makedirs(args.out_path, exist_ok=True)
    model, config = load_model(args.model_path, device)
    max_token_len = config.get("max_token_len", 50)

    file_paths = sorted(
        os.path.join(args.parquet_dir, file_name)
        for file_name in os.listdir(args.parquet_dir)
        if file_name.endswith(".parquet")
    )
    assigned_files = assign_files_by_size(file_paths, world_size)[rank]
    LOGGER.info("rank=%s assigned_files=%s", rank, len(assigned_files))

    for file_path in assigned_files:
        file_name = os.path.basename(file_path)[: -len(".parquet")]
        pred_path = os.path.join(args.out_path, f"{file_name}_pred.csv")
        result_path = os.path.join(args.out_path, f"{file_name}_result.tsv")
        if os.path.exists(pred_path) and os.path.exists(result_path):
            LOGGER.info("rank=%s skipping finished %s", rank, file_name)
            continue
        LOGGER.info("rank=%s parsing %s", rank, file_name)
        pred_df, meta_df = predict_file(
            model=model,
            file_path=file_path,
            max_token_len=max_token_len,
            batch_size=args.batch_size,
            device=device,
        )
        pred_df.to_csv(pred_path, index=False)
        result_df = build_prediction_frame(meta_df, pred_df, q_value_threshold=args.q_value_threshold)
        result_df.to_csv(result_path, sep="\t", index=False)

    distributed_barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
