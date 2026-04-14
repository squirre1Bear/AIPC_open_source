from __future__ import annotations

import os
import polars as pl # Importing polars, though not directly used in the current functions
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm # Importing tqdm, though not directly used in the current functions
from sklearn.metrics import roc_auc_score
import logging
import argparse
import sys # Need to import sys for sys.exit()

import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import lightning.pytorch as ptl
from lightning.pytorch.strategies import DDPStrategy

# Assuming these are custom modules
from transformer.dataset import SpectrumDataset, mkdir_p, collate_batch_weight_deltaRT_index
from transformer.model import MSGPT

# Configure logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import warnings
warnings.filterwarnings("ignore")

def get_fdr_result(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the q-value (False Discovery Rate) for PSMs."""
    df = df.copy()
    # Assuming label 1 is Target (non-decoy) and 0 is Decoy
    df['decoy'] = np.where(df['label'] == 1, 0, 1) 
    
    # Cumulative sums
    target_num = (df['decoy'] == 0).cumsum()
    decoy_num = (df['decoy'] == 1).cumsum()
    
    # Replace 0 with 1 to avoid division by zero (for the first entry)
    target_num = target_num.replace(0, 1)
    decoy_num = decoy_num.replace(0, 1)
    
    # Calculate initial q-value (FDR formula: Decoy / Target)
    df['q_value'] = decoy_num / target_num
    
    # Apply the reverse cumulative minimum to ensure monotonicity
    df['q_value'] = df['q_value'][::-1].cummin()[::-1]
    return df

class Evalute(ptl.LightningModule):
    """Lightning Module for model evaluation and result aggregation."""

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
        """Single test step to perform prediction and accumulate results."""
        # Unpack the batch including all necessary inputs and metadata
        spectra, spectra_mask, precursors, tokens, peptide, label, weight, index = batch
        
        # Move inputs to device and convert to bfloat16 for mixed precision
        spectra = spectra.to(self.device).to(torch.bfloat16)
        spectra_mask = spectra_mask.to(self.device).to(torch.bfloat16)
        precursors = precursors.to(self.device).to(torch.bfloat16)
        tokens = tokens.to(self.device).to(torch.long)
        
        # Perform prediction (Loss calculation is often skipped in test_step for pure prediction)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            pred, _ = self.model.pred(spectra, spectra_mask, precursors, tokens)

        # Move outputs and metadata to CPU and convert to numpy for list extension
        label = label.to(torch.int32).cpu().detach().numpy()
        pred = pred.to(torch.float32).cpu().detach().numpy()
        index = index.to(torch.int32).cpu().detach().numpy()
        weight = weight.to(torch.float32).cpu().detach().numpy()
        
        # Accumulate predictions and metadata
        if label.shape[0] > 1 and (not isinstance(pred, float)) and pred is not None:
            self.pred_list.extend(pred.tolist())
            self.label_list.extend(label.tolist())
            self.index_list.extend(index.tolist())
            self.weight_list.extend(weight.tolist())

    def on_test_end(self) -> None:
        """Called when the testing is complete. Saves results and calculates AUC."""
        # Create DataFrame from accumulated results
        df = pd.DataFrame({"index": self.index_list,
                           "score": self.pred_list,
                           "label": self.label_list,
                           "weight": self.weight_list})
        
        # Save the prediction scores to a CSV file (appending mode is used, which is risky
        # if the process is not carefully managed, especially with DDP)
        df.to_csv(os.path.join(self.out_path, f"%s_pred.csv" % self.file_name), mode='a+', index=None)
        
        # Calculate and log ROC AUC score
        auc = roc_auc_score(self.label_list, self.pred_list)
        logging.info(f"auc: {auc}")

    def _reset_metrics(self) -> None:
        """Initializes/resets lists for collecting metrics."""
        self.pred_list = []
        self.label_list = []
        self.index_list = []
        self.weight_list = []
        
# Added sys import at the top of the file

def gen_dl(df, config):
    """Generates the DataLoader from a Pandas DataFrame and config."""
    # Create symbol-to-index mapping (vocabulary)
    s2i = {v: i for i, v in enumerate(config["vocab"])}
    logging.info(f"gen Vocab: {s2i}")
    
    # Initialize the Dataset with necessary flags (label, weight, deltaRT, index)
    ds = SpectrumDataset(df, s2i, config["n_peaks"], need_label=True, need_weight=True, need_deltaRT=True, need_index=True)
    
    # Initialize the DataLoader
    dl = DataLoader(ds,
                    batch_size=config["predict_batch_size"],
                    num_workers=0, # Set to 0 to potentially avoid multiprocessing issues during evaluation
                    shuffle=False,
                    collate_fn=collate_batch_weight_deltaRT_index, # Custom collate function
                    pin_memory=True)
    logging.info(f"Data: {len(ds):,} samples, DataLoader: {len(dl):,}")
    return dl

def postprocess_file(file_path, score_dir):
    """Integrates parquet data with predicted scores, calculates FDR, and generates final DataFrame."""
    file_name = os.path.basename(file_path)[:-len('.parquet')]
    score_path = os.path.join(score_dir,file_name+"_pred.csv")

    # Load original parquet data and prediction scores
    sage_parquet = pd.read_parquet(file_path)
    # Load scores and remove duplicates, which can occur when using multiple GPUs for scoring
    sage_score = pd.read_csv(score_path).drop_duplicates(subset='index') 

    # Sanity check for data consistency
    assert len(sage_parquet) == len(sage_score), f"{file_name}: parquet length ({len(sage_parquet)}) and score length ({len(sage_score)}) are inconsistent"

    # Merge original data with scores based on 'index'
    sage_parquet = sage_parquet.merge(sage_score[['index', 'score']], on="index", how="left")

    def _post(df):
        """Internal function to calculate PSM IDs, sort, and calculate FDR."""
        df = df.copy()
        # Create precursor_id and PSM_id
        df['precursor_id'] = df['charge'].astype(str) + "_" + df['precursor_sequence']
        df['psm_id'] = df['scan_number'].astype(str) + "_" + df['charge'].astype(str) + "_" + df['precursor_sequence']
        # Sort by score for FDR calculation
        df = df.sort_values("score", ascending=False)
        # Calculate q-values
        df = get_fdr_result(df)
        # Filter PSMs where q_value is less than 1.0 (retains all PSMs that passed the FDR check)
        df = df[df['q_value'] < 1.0]
        return df

    sage_parquet = _post(sage_parquet)

    # Clean the sequence strings by removing mass modifications (e.g., n[42] -> "")
    sage_parquet["cleaned_sequence"] = (
        sage_parquet["precursor_sequence"]
        .str.replace(r"n\[42\]", "", regex=True)
        .str.replace(r"N\[\.98\]", "N", regex=True)
        .str.replace(r"Q\[\.98\]", "Q", regex=True)
        .str.replace(r"M\[15\.99\]", "M", regex=True)
        .str.replace(r"C\[57\.02\]", "C", regex=True)
    )

    # Drop spectrum arrays to save memory/space
    sage_parquet = sage_parquet.drop(columns=["mz_array", "intensity_array"], errors="ignore")
    # Filter for Target PSMs only (label == 1)
    sage_parquet_target = sage_parquet[sage_parquet["label"] == 1].copy()

    # Rename columns to standard proteomics output names
    sage_parquet_target = sage_parquet_target.rename(columns={
        'precursor_sequence': 'modified_sequence',
        'charge': 'precursor_charge'
    })

    required_columns = [
        "cleaned_sequence", "precursor_mz", "precursor_charge",
        "modified_sequence", "label", "score", "q_value", "scan_number"
    ]
    
    # Check for missing required columns before proceeding
    missing_columns = [col for col in required_columns if col not in sage_parquet_target.columns]
    if missing_columns:
        print(f"Warning: File {file_name} is missing columns: {', '.join(missing_columns)}")
        sys.exit() # Exit if required columns are missing

    # Standardize the modification format to UniMod style (e.g., n[42] -> n(UniMod:1))
    sage_parquet_target["modified_sequence"] = (
        sage_parquet_target["modified_sequence"]
        .str.replace(r"n\[42\]", "n(UniMod:1)", regex=True) # N-term Acetylation
        .str.replace(r"N\[\.98\]", "N(UniMod:7)", regex=True) # Deamidation of N
        .str.replace(r"Q\[\.98\]", "Q(UniMod:7)", regex=True) # Deamidation of Q
        .str.replace(r"M\[15\.99\]", "M(UniMod:35)", regex=True) # Oxidation of M
        .str.replace(r"C\[57\.02\]", "C(UniMod:4)", regex=True) # Carbamidomethylation of C
    )

    # Sort again and select the best PSM per spectrum (scan_number)
    sage_parquet_target = sage_parquet_target.sort_values(by="score", ascending=False)
    sage_parquet_target_unique = sage_parquet_target.drop_duplicates(subset="scan_number", keep="first")
    
    # Return only the required columns
    return sage_parquet_target_unique[required_columns]


def main() -> None:
    """Main function to perform prediction and post-processing."""
    logging.info("Initializing inference.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--parquet_dir") # Input directory containing parquet files
    parser.add_argument("--config", default="/zhangxiaofan/DDA_BERT_deltaRT/yaml/ajun_dataset62_deltart.yaml")
    parser.add_argument("--out_path", default='') # If empty, output path is same as input; otherwise, use the specified path
    args = parser.parse_args()

    # Load model and config
    logging.info(f"Inference use model path: {args.model_path}")
    model_type = args.model_path.split('.')[-1]
    
    if model_type in ('pth', 'bin','pt'):
        # Load config from YAML
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # Define the fixed vocabulary
        vocab = ['<pad>', '<mask>', 'A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'C[57.02]', 'M[15.99]', 'N[.98]', 'Q[.98]', 'X', '<unk>']
        config["vocab"] = vocab
        s2i = {v: i for i, v in enumerate(vocab)}
        logging.info(f"Vocab: {s2i}, n_peaks: {config['n_peaks']}")
        
        # Load the model based on file extension
        if model_type == 'pth':
            model = torch.load(args.model_path)
        elif model_type == 'pt':
            model = MSGPT.load_pt(args.model_path, config)
            model.to(torch.bfloat16)
        else:
            # Assuming a custom function to load from 'bin' format
            model = torch.load_bin(args.model_path, config) 
    elif model_type == 'ckpt':
        # Load model and config from a Lightning checkpoint
        model, config = MSGPT.load_ckpt(args.model_path)
        
    logging.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )

    # Set model to evaluation mode and map to device
    model.eval()
    model = model.to(torch.bfloat16) # Map model weights to bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Check available GPU count
    gpu_num = torch.cuda.device_count()
    logging.info(
        f"Evaluate model with {gpu_num} gpus"
    )
    
    # Determine the output path
    if args.out_path == '':
        out_path = args.parquet_dir
    else:
        out_path = args.out_path
        
    mkdir_p(out_path) # Create output directory if it doesn't exist
    logging.info(f'**************out_path: {out_path}**************************')
    
    # Load data file paths
    data_path_list = [os.path.join(args.parquet_dir, f) for f in os.listdir(args.parquet_dir) if f.endswith('.parquet')]
    
    for file_path in data_path_list:
        file_name = os.path.basename(file_path)[:-len('.parquet')]
        logging.info(f'Parse: {file_path}, file_name: {file_name}')
        
        # Check if the prediction score file already exists, and skip if it does
        out_file = os.path.join(out_path, f"%s_pred.csv" % file_name)
        if os.path.exists(out_file):
            logging.info(f'Output score file: {out_file} exists. Skipping prediction.')
            continue
        
        df = pd.read_parquet(file_path)
        
        try:
            # Check for "weight" column and add a default if missing
            if "weight" not in df.columns:
                df['weight'] = 1.0

            dl = gen_dl(df, config)

            # Initialize and run the Lightning Trainer for evaluation
            logging.info(f'Total {gpu_num} GPU(s) available ......')
            # Use DDP strategy for multi-GPU prediction
            strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
            trainer = ptl.Trainer(
                accelerator="auto",
                devices="auto",
                strategy=strategy,
            )
            evaluate = Evalute(out_path, file_name, model)
            trainer.test(evaluate, dataloaders=dl)
            
        except Exception as e:
            # Log error and skip to the next file if loading/prediction fails
            logger.info(f'Loading {file_path} parquet error: {e}!!! Skipping file.')
            continue
        
        # Post-process the results (merge scores, calculate FDR, clean up)
        result_data = postprocess_file(file_path, out_path)
        
        # Save the final result to a TSV file
        result_path = os.path.join(out_path, file_name+"_result.tsv")
        result_data.to_csv(result_path, sep='\t', index=False)
        

if __name__ == "__main__":
    main()

# cd /zhangxiaofan/DDA_BERT_deltaRT/; python aipc_test_baseline.py --model_path /zhangxiaofan/DDA_BERT_deltaRT/pred/mzml_tims_wiff.pt --parquet_dir /zhangxiaofan/DDA_BERT_deltaRT/bohr_test/parquet_dir --out_path /zhangxiaofan/DDA_BERT_deltaRT/bohr_test/score_dir