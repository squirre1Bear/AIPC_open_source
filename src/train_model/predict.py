import os
import glob
import argparse
import logging
from typing import List, Tuple

import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import SpectrumDataset, padding, PROTON_MASS_AMU
from train import AIPCRerankNet


def build_vocab_from_yaml(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    vocab = ['<pad>', '<mask>'] + list(cfg['residues'].keys()) + ['<unk>']
    s2i = {v: i for i, v in enumerate(vocab)}
    return cfg, vocab, s2i


def collate_predict(batch: List[Tuple]):
    spectra, precursor_mzs, precursor_charges, tokens, peptides = zip(*batch)
    spectra, spectra_mask = padding(list(spectra))
    tokens = torch.stack(tokens, dim=0)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - PROTON_MASS_AMU) * precursor_charges
    precursors = torch.vstack([precursor_masses, precursor_charges]).T.float()
    return spectra, spectra_mask, precursors, tokens, list(peptides)


@torch.no_grad()
def predict_one_file(model, file_path: str, s2i: dict, cfg: dict, batch_size: int, num_workers: int, device):
    df = pd.read_parquet(file_path)
    ds = SpectrumDataset(
        df,
        s2i,
        n_peaks=cfg.get('n_peaks', 300),
        max_length=cfg.get('max_length', 50),
        min_mz=cfg.get('min_mz', 50.0),
        max_mz=cfg.get('max_mz', 2500.0),
        min_intensity=cfg.get('min_intensity', 0.01),
        remove_precursor_tol=cfg.get('remove_precursor_tol', 2.0),
        need_label=False,
    )
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
    for spectra, spectra_mask, precursors, tokens, _ in loader:
        spectra = spectra.to(device)
        spectra_mask = spectra_mask.to(device)
        precursors = precursors.to(device)
        tokens = tokens.to(device)
        logits = model(spectra, spectra_mask, precursors, tokens)
        prob = torch.sigmoid(logits)
        scores.append(logits.cpu())
        probs.append(prob.cpu())

    score = torch.cat(scores).numpy()
    prob = torch.cat(probs).numpy()
    out_df = df.copy()
    out_df['score'] = score
    out_df['pred_score'] = score
    out_df['prob'] = prob
    return out_df


def main():
    parser = argparse.ArgumentParser(description='Predict AIPC scores for parquet files.')
    parser.add_argument('--model_path', required=True, help='Checkpoint path produced by aipc_train.py')
    parser.add_argument('--parquet_dir', required=True, help='Directory containing test parquet files')
    parser.add_argument('--config', required=True, help='YAML used for peptide vocabulary and spectrum preprocessing')
    parser.add_argument('--out_path', required=True, help='Directory to save per-file scored parquet/tsv')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    cfg_yaml, vocab, s2i = build_vocab_from_yaml(args.config)

    ckpt = torch.load(args.model_path, map_location='cpu')
    model_cfg = ckpt['config']
    model = AIPCRerankNet(
        vocab_size=model_cfg['vocab_size'],
        token_embed_dim=model_cfg['token_embed_dim'],
        precursor_dim=model_cfg['precursor_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        n_heads=model_cfg['n_heads'],
        n_layers=model_cfg['n_layers'],
        dropout=model_cfg['dropout'],
        max_token_len=model_cfg['max_token_len'],
    )
    model.load_state_dict(ckpt['state_dict'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    parquet_files = sorted(glob.glob(os.path.join(args.parquet_dir, '*.parquet')))
    if not parquet_files:
        raise FileNotFoundError(f'No parquet files found under {args.parquet_dir}')

    for idx, file_path in enumerate(parquet_files, start=1):
        logging.info('[%d/%d] scoring %s', idx, len(parquet_files), os.path.basename(file_path))
        out_df = predict_one_file(model, file_path, s2i, cfg_yaml, args.batch_size, args.num_workers, device)

        stem = os.path.splitext(os.path.basename(file_path))[0]
        out_parquet = os.path.join(args.out_path, f'{stem}.parquet')
        out_tsv = os.path.join(args.out_path, f'{stem}.tsv')
        out_df.to_parquet(out_parquet, index=False)

        cols = []
        for c in ['psm_id', 'scan', 'precursor_sequence', 'modified_sequence', 'score', 'pred_score', 'prob']:
            if c in out_df.columns:
                cols.append(c)
        if 'score' not in cols:
            cols.append('score')
        out_df[cols].to_csv(out_tsv, sep='\t', index=False)

    logging.info('Finished scoring. Outputs written to %s', args.out_path)


if __name__ == '__main__':
    main()

# python aipc_predict.py --model_path E:/AIPC_runs/run1/best.pt --parquet_dir E:/AIPC_test/parquet_dir --config E:/path/to/aipc_test_mzml.yaml --out_path E:/AIPC_test/score_dir --batch_size 1024
