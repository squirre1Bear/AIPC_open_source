from __future__ import annotations

import os
import sys
import yaml
import logging
import argparse
import warnings
from typing import Any

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.data import DataLoader
import lightning.pytorch as ptl
from lightning.pytorch.strategies import DDPStrategy

from ..transformer.dataset import (
    SpectrumDataset,
    mkdir_p,
    collate_batch_weight_deltaRT_index,
)
from ..transformer.model import MSGPT

warnings.filterwarnings("ignore")

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_barrier() -> None:
    if is_dist_initialized():
        dist.barrier()


def get_rank() -> int:
    if is_dist_initialized():
        return dist.get_rank()
    return 0


def is_global_zero() -> bool:
    return get_rank() == 0


def get_fdr_result(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates q-value (FDR) for PSMs."""
    df = df.copy()

    # label=1 为 target，label=0 为 decoy
    df["decoy"] = np.where(df["label"] == 1, 0, 1)

    target_num = (df["decoy"] == 0).cumsum()
    decoy_num = (df["decoy"] == 1).cumsum()

    # 防止分母为 0
    target_num = target_num.replace(0, 1)
    decoy_num = decoy_num.replace(0, 1)

    # 运行中的 FDR
    df["q_value"] = decoy_num / target_num

    # 反向 cumulative min，保证 q_value 单调
    df["q_value"] = df["q_value"][::-1].cummin()[::-1]
    return df


class Evaluate(ptl.LightningModule):
    """Lightning Module for distributed inference and result aggregation."""

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

    def forward(
        self,
        spectra: Tensor,
        spectra_mask: Tensor,
        precursors: Tensor,
        tokens: Tensor,
    ) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, _ = self.model.pred(spectra, spectra_mask, precursors, tokens)
        return pred

    def test_step(
        self,
        batch: tuple[Tensor, Tensor, Tensor, Tensor, list, list],
        batch_idx: int,
    ) -> None:
        spectra, spectra_mask, precursors, tokens, peptide, label, weight, index = batch

        spectra = spectra.to(self.device, non_blocking=True).to(torch.bfloat16)
        spectra_mask = spectra_mask.to(self.device, non_blocking=True).to(torch.bfloat16)
        precursors = precursors.to(self.device, non_blocking=True).to(torch.bfloat16)
        tokens = tokens.to(self.device, non_blocking=True).to(torch.long)

        pred = self.forward(spectra, spectra_mask, precursors, tokens)

        label_np = label.detach().to(torch.int32).cpu().numpy()
        pred_np = pred.detach().to(torch.float32).cpu().numpy()
        index_np = index.detach().to(torch.int32).cpu().numpy()
        weight_np = weight.detach().to(torch.float32).cpu().numpy()

        # 不再丢掉 batch size = 1 的情况
        self.pred_list.extend(np.atleast_1d(pred_np).tolist())
        self.label_list.extend(np.atleast_1d(label_np).tolist())
        self.index_list.extend(np.atleast_1d(index_np).tolist())
        self.weight_list.extend(np.atleast_1d(weight_np).tolist())

    def on_test_epoch_end(self) -> None:
        """
        Gather all ranks' predictions, and only rank 0 writes the final CSV.
        """
        local_data = {
            "index": self.index_list,
            "score": self.pred_list,
            "label": self.label_list,
            "weight": self.weight_list,
        }

        if is_dist_initialized():
            gathered: list[Any] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_data)
        else:
            gathered = [local_data]

        if is_global_zero():
            all_index: list[int] = []
            all_score: list[float] = []
            all_label: list[int] = []
            all_weight: list[float] = []

            for item in gathered:
                if item is None:
                    continue
                all_index.extend(item["index"])
                all_score.extend(item["score"])
                all_label.extend(item["label"])
                all_weight.extend(item["weight"])

            df = pd.DataFrame(
                {
                    "index": all_index,
                    "score": all_score,
                    "label": all_label,
                    "weight": all_weight,
                }
            )

            # 保险起见，做一次清洗和去重
            df["index"] = pd.to_numeric(df["index"], errors="coerce")
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            df["label"] = pd.to_numeric(df["label"], errors="coerce")
            df["weight"] = pd.to_numeric(df["weight"], errors="coerce")

            df = df.dropna(subset=["index", "score", "label", "weight"])
            df["index"] = df["index"].astype(int)
            df["label"] = df["label"].astype(int)

            # 同一个 index 只保留一条，理论上正确 gather 后不应有重复
            if df["index"].duplicated().any():
                dup_count = int(df["index"].duplicated().sum())
                logging.warning(
                    f"{self.file_name}: found {dup_count} duplicated indices after gather, keeping first."
                )
                df = df.drop_duplicates(subset="index", keep="first")

            df = df.sort_values("index").reset_index(drop=True)

            out_csv = os.path.join(self.out_path, f"{self.file_name}_pred.csv")
            df.to_csv(out_csv, index=False)

            if len(df) > 0 and df["label"].nunique() > 1:
                auc = roc_auc_score(df["label"].tolist(), df["score"].tolist())
                logging.info(f"global auc: {auc}")
            else:
                logging.warning(
                    f"{self.file_name}: unable to calculate global auc because labels are empty or single-class."
                )

        dist_barrier()
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        self.pred_list: list[float] = []
        self.label_list: list[int] = []
        self.index_list: list[int] = []
        self.weight_list: list[float] = []


def gen_dl(df: pd.DataFrame, config: dict) -> DataLoader:
    """Generate DataLoader."""
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


def postprocess_file(
    file_path: str,
    score_dir: str,
    q_value_threshold: float = 1.0,
) -> pd.DataFrame:
    """Merge parquet with predicted scores and generate final result."""
    file_name = os.path.basename(file_path)[:-len(".parquet")]
    score_path = os.path.join(score_dir, f"{file_name}_pred.csv")

    sage_parquet = pd.read_parquet(file_path)
    sage_score = pd.read_csv(score_path)

    # 安全清洗
    sage_score["index"] = pd.to_numeric(sage_score["index"], errors="coerce")
    sage_score["score"] = pd.to_numeric(sage_score["score"], errors="coerce")
    sage_score = sage_score.dropna(subset=["index", "score"])
    sage_score["index"] = sage_score["index"].astype(int)
    sage_score = sage_score.drop_duplicates(subset="index", keep="first")

    assert len(sage_parquet) == len(
        sage_score
    ), (
        f"{file_name}: parquet length ({len(sage_parquet)}) "
        f"and score length ({len(sage_score)}) are inconsistent"
    )

    sage_parquet = sage_parquet.merge(
        sage_score[["index", "score"]],
        on="index",
        how="left",
    )

    def _post(df: pd.DataFrame) -> pd.DataFrame:
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
        df = df[df["q_value"] < q_value_threshold]
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
        raise ValueError(
            f"File {file_name} is missing columns: {', '.join(missing_columns)}"
        )

    sage_parquet_target["modified_sequence"] = (
        sage_parquet_target["modified_sequence"]
        .str.replace(r"n\[42\]", "n(UniMod:1)", regex=True)
        .str.replace(r"N\[\.98\]", "N(UniMod:7)", regex=True)
        .str.replace(r"Q\[\.98\]", "Q(UniMod:7)", regex=True)
        .str.replace(r"M\[15\.99\]", "M(UniMod:35)", regex=True)
        .str.replace(r"C\[57\.02\]", "C(UniMod:4)", regex=True)
    )

    sage_parquet_target = sage_parquet_target.sort_values(by="score", ascending=False)
    sage_parquet_target_unique = sage_parquet_target.drop_duplicates(
        subset="scan_number",
        keep="first",
    )

    return sage_parquet_target_unique[required_columns]


def load_model_and_config(model_path: str, config_path: str) -> tuple[MSGPT, dict]:
    """Load model and config."""
    logging.info(f"Inference use model path: {model_path}")
    model_type = model_path.split(".")[-1]

    if model_type in ("pth", "bin", "pt"):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)

        vocab = [
            "<pad>", "<mask>", "A", "D", "E", "F", "G", "H", "I", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "Y",
            "C[57.02]", "M[15.99]", "N[.98]", "Q[.98]", "X", "<unk>",
        ]
        config["vocab"] = vocab
        s2i = {v: i for i, v in enumerate(vocab)}
        logging.info(f"Vocab: {s2i}, n_peaks: {config['n_peaks']}")

        if model_type == "pth":
            model = torch.load(model_path, map_location="cpu")
        elif model_type == "pt":
            model = MSGPT.load_pt(model_path, config)
        else:
            model = torch.load_bin(model_path, config)

    elif model_type == "ckpt":
        model, config = MSGPT.load_ckpt(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.eval()
    model = model.to(torch.bfloat16)

    logging.info(
        f"Model loaded with {np.sum([p.numel() for p in model.parameters()]):,d} parameters"
    )
    return model, config


def main() -> None:
    logging.info("Initializing inference.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--parquet_dir", required=True)
    parser.add_argument("--config", default="src_guanfang/aipc_test_mzml.yaml")
    parser.add_argument("--out_path", default="")
    parser.add_argument("--q_value_threshold", type=float, default=1.0)
    args = parser.parse_args()

    model, config = load_model_and_config(args.model_path, args.config)

    gpu_num = torch.cuda.device_count()
    logging.info(f"Evaluate model with {gpu_num} gpus")

    out_path = args.parquet_dir if args.out_path == "" else args.out_path
    mkdir_p(out_path)
    logging.info(f"**************out_path: {out_path}**************************")

    data_path_list = sorted(
        [
            os.path.join(args.parquet_dir, f)
            for f in os.listdir(args.parquet_dir)
            if f.endswith(".parquet")
        ]
    )

    if torch.cuda.is_available() and gpu_num > 1:
        strategy = DDPStrategy(
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        )
        trainer = ptl.Trainer(
            accelerator="gpu",
            devices=gpu_num,
            strategy=strategy,
            logger=False,
            enable_checkpointing=False,
        )
    else:
        trainer = ptl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )

    for file_path in data_path_list:
        file_name = os.path.basename(file_path)[:-len(".parquet")]
        pred_csv = os.path.join(out_path, f"{file_name}_pred.csv")
        result_tsv = os.path.join(out_path, f"{file_name}_result.tsv")

        logging.info(f"Parse: {file_path}, file_name: {file_name}")

        # 为了避免 DDP 下不同 rank 因为 skip 条件不一致产生分叉，
        # 这里采用“统一覆盖写”的安全策略。
        # 若你后续确实需要 resume/skip，必须做成所有 rank 一致的决定。
        df = pd.read_parquet(file_path)

        if "weight" not in df.columns:
            df["weight"] = 1.0

        dl = gen_dl(df, config)

        try:
            evaluate = Evaluate(out_path, file_name, model)
            trainer.test(evaluate, dataloaders=dl)

            # 确保 pred.csv 已由 rank0 写完
            dist_barrier()

            if trainer.is_global_zero:
                result_data = postprocess_file(
                    file_path=file_path,
                    score_dir=out_path,
                    q_value_threshold=args.q_value_threshold,
                )
                result_data.to_csv(result_tsv, sep="\t", index=False)

            # 确保 rank0 后处理完成，再进入下一个文件
            dist_barrier()

        except Exception as e:
            logging.exception(f"Processing {file_path} failed: {e}")
            dist_barrier()
            continue


if __name__ == "__main__":
    main()

# 多卡运行示例：
# python3 -m src_guanfang.test.gpt_test_baseline \
#   --model_path /home/yhc/projects/AIPC/model/MSGPT/5/mp_rank_00_model_states.pt \
#   --parquet_dir data/bas_test_dataset \
#   --out_path result \
#   --q_value_threshold 1.0