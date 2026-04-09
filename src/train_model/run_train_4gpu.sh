#!/usr/bin/env bash
set -euo pipefail

# Example:
# bash run_train_4gpu.sh /data/ms_data_pkl /data/run1 29

PKL_DIR=${1:?need pkl_dir}
OUTPUT_DIR=${2:?need output_dir}
VOCAB_SIZE=${3:?need vocab_size}

torchrun --standalone --nproc_per_node=4 /mnt/data/train_4gpu.py \
  --pkl_dir "$PKL_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --vocab_size "$VOCAB_SIZE" \
  --batch_size 1024 \
  --epochs 10 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --val_ratio 0.1 \
  --num_workers 4 \
  --seed 42 \
  --token_embed_dim 128 \
  --precursor_dim 64 \
  --hidden_dim 256 \
  --n_heads 8 \
  --n_layers 2 \
  --dropout 0.1 \
  --max_token_len 64 \
  --grad_clip 1.0
