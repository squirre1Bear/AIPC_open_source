from __future__ import annotations

import os
import polars as pl
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import logging
import argparse

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import lightning.pytorch as ptl
from lightning.pytorch.strategies import DDPStrategy

from transformer.dataset import SpectrumDataset, mkdir_p, collate_batch_weight_deltaRT_index
from transformer.model import MSGPT

logger = logging.getLogger()
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")


def mkdir_p(dirs, delete=True):
    """
    make a directory (dir) if it doesn't exist
    """    
    # 如果文件夹不存在，则递归新建
    if not os.path.exists(dirs):
        try:
            # 递归创建文件夹
            os.makedirs(dirs)
        except:
            pass

    return True, 'OK'

class Evalute(ptl.LightningModule):
    """evaluate for model."""

    def __init__(
            self,
            out_path: str,
            file_name: str,
            model: MSGPT,
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
        """Single test step."""
        spectra, spectra_mask, precursors, tokens, peptide, label, weight, index = batch
        
        spectra = spectra.to(self.device).to(torch.bfloat16)
        spectra_mask = spectra_mask.to(self.device).to(torch.bfloat16)
        precursors = precursors.to(self.device).to(torch.bfloat16)
        tokens = tokens.to(self.device).to(torch.long)
 
        # Loss
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred, _ = self.model.pred(spectra, spectra_mask, precursors, tokens)

        label = label.to(torch.int32).cpu().detach().numpy()
        pred = pred.to(torch.float32).cpu().detach().numpy()
        index = index.to(torch.int32).cpu().detach().numpy()
        weight = weight.to(torch.float32).cpu().detach().numpy()
      
        if label.shape[0] > 1 and (not isinstance(pred, float)) and pred is not None:
            self.pred_list.extend(pred.tolist())
            self.label_list.extend(label.tolist())
            self.index_list.extend(index.tolist())
            self.weight_list.extend(weight.tolist())

    def on_test_end(self) -> None:
        df = pd.DataFrame({"index": self.index_list,
                           "score": self.pred_list,
                           # "label": self.label_list,
                           # "weight": self.weight_list
                          })
        df.to_csv(os.path.join(self.out_path, f"%s_pred.csv" % self.file_name), mode='a+', header=False, index=None)
        
        auc = roc_auc_score(self.label_list, self.pred_list)
        logging.info(f"auc: {auc}")

    def _reset_metrics(self) -> None:
        self.pred_list = []
        self.label_list = []
        self.index_list = []
        self.weight_list = []

    

def gen_dl(df, config):
    s2i = {v: i for i, v in enumerate(config["vocab"])}
    logging.info(f"gen Vocab: {s2i}")
    
    # add index when dataloader 
    ds = SpectrumDataset(df, s2i, config["n_peaks"], need_label=True, need_weight=True, need_deltaRT=True, need_index=True)
    dl = DataLoader(ds,
                    batch_size=config["predict_batch_size"],
                    num_workers=0,
                    shuffle=False,
                    collate_fn=collate_batch_weight_deltaRT_index,
                    pin_memory=True)
    logging.info(f"Data: {len(ds):,} samples, DataLoader: {len(dl):,}")
    return dl


def main() -> None:
    """Predict with the model."""
    logging.info("Initializing inference.")

    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--parquet_dir") # 输入路径
    parser.add_argument("--config", default="/zhangxiaofan/bohr_pred_260430/yaml/dataset62_sample600_deltaRT_20250320.yaml")
    parser.add_argument("--out_path", default='') # 若输出路径为空，则与输入路径一致；否则，设置为指定的输出路径
    args = parser.parse_args()

    # load model
    logging.info(f"inference use model path: {args.model_path}")
    model_type = args.model_path.split('.')[-1]
    if model_type in ('pth', 'bin','pt'):
        # load config
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        vocab = ['<pad>', '<mask>', 'A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'C[57.02]', 'M[15.99]', 'N[.98]', 'Q[.98]', 'X', '<unk>']
        config["vocab"] = vocab
        s2i = {v: i for i, v in enumerate(vocab)}
        logging.info(f"Vocab: {s2i}, n_peaks: {config['n_peaks']}")
        
        if model_type == 'pth':
            model = torch.load(args.model_path)
        elif model_type == 'pt':
            model = MSGPT.load_pt(args.model_path, config)
            model.to(torch.bfloat16)
        else:
            model = torch.load_bin(args.model_path, config)
    elif model_type == 'ckpt':
        model, config = MSGPT.load_ckpt(args.model_path)
        
    logging.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    # 评测模式
    model.eval()
    model = model.to(torch.bfloat16) # 映射模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 检测gpu数量
    gpu_num = torch.cuda.device_count()
    logging.info(
        f"eval model with {gpu_num} gpus"
    )
    
    # update out_path
    if args.out_path == '':
        out_path = args.parquet_dir
    else:
        out_path = args.out_path
        
    mkdir_p(out_path)
    logging.info(f'**************out_path: {out_path}**************************')
    
    # load data
    data_path_list = [os.path.join(args.parquet_dir, f) for f in os.listdir(args.parquet_dir) if f.endswith('.parquet')]
    for file_path in data_path_list:
        file_name = os.path.basename(file_path).split('.')[0]
        logging.info(f'parse: {file_path}, file_name: {file_name}')
        
        # 如果文件已存在，则跳过/删除
        out_file = os.path.join(out_path, f"%s_pred.csv" % file_name)
        if os.path.exists(out_file):
            logging.info(f'out_file: {out_file} exist!!!!!!!!!!!!')
            continue
        
        df_pandas = pd.read_parquet(file_path)
        df_pandas['label'] = 1
        df = pl.from_pandas(df_pandas)
        
        try:
            # 判断是否存在 modified_sequence 列
            if 'modified_sequence' not in df.columns:
                # 若没有 modified_sequence，则将 precursor_sequence 赋值给 modified_sequence
                df = df.with_columns(df['precursor_sequence'].alias('modified_sequence'))

            if 'index' not in df.columns:
                # 20250311 若没有index，增加一列index
                df = df.with_columns((pl.arange(0, pl.count()).alias("index")))

            # 检查是否存在"weight"列，若不存在则添加
            if "weight" not in df.columns:
                df = df.with_columns(pl.lit(1.0).alias("weight"))

            dl = gen_dl(df, config)

            # evaluate
            logging.info(f'total {gpu_num} gpu ......')
            strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
            trainer = ptl.Trainer(
                accelerator="auto",
                devices="auto",
                strategy=strategy,
            )
            evaluate = Evalute(out_path, file_name, model)
            trainer.test(evaluate, dataloaders=dl)
        except Exception as e:
            logger.info(f'load {file_path} ipc error: {e}!!!')
            continue
            
    pred_files = [
        os.path.join(out_path, f)
        for f in os.listdir(out_path)
        if f.endswith("_pred.csv")
    ]

    if len(pred_files) == 0:
        logging.info("No pred files found. Skip merging.")
        return

    df_list = []
    for f in pred_files:
        try:
            tmp_df = pd.read_csv(f,header=None,names=['index','score'])
            df_list.append(tmp_df)
        except Exception as e:
            logging.info(f"Error reading {f}: {e}")

    if len(df_list) == 0:
        logging.info("No valid pred files to merge.")
        return

    merged_df = pd.concat(df_list, ignore_index=True)

    merged_path = os.path.join(out_path, "all_pred.tsv")

    merged_df.to_csv(
        merged_path,
        sep="\t",
        index=False
    )

    logging.info(f"Merged TSV saved to: {merged_path}")

if __name__ == "__main__":
    main()
