from __future__ import annotations

import torch
import torch.utils.data as pt_data
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, ConcatDataset
from torch import Tensor
from torch.utils.data import random_split

import os
import numpy as np
import polars as pl
import pickle
import random
from random import sample
import math
import re
from collections import defaultdict
import glob
import time  ##20250713 add


def augment_spectrum(spectrum: torch.Tensor):
    """
    对 spectrum 最后一维的每个 channel 应用不同范围的随机因子。
    
    Args:
        spectrum (Tensor): shape [..., C]，C 是特征数
    
    Returns:
        Tensor: 扰动后的张量，shape 与输入相同
    """
    # 创建一个独立的随机数生成器，并基于时间设置 seed
    gen = torch.Generator(device=spectrum.device) if spectrum.is_cuda else torch.Generator()
    seed = int(round(time.time() % 1, 5) * 1e5)
    gen.manual_seed(seed)

    # 获取最后维度的大小（假设为特征维度）
    C = spectrum.size(-1)

    # 定义( m/z and intensity)每个 channel 的随机缩放范围
    scale_ranges = [(0.999, 1.001), (0.95, 1.05)]
    assert len(scale_ranges) == C, "scale_ranges 的长度必须等于特征维度"

    # 构建每个通道的随机因子
    scale_factors = []
    for i in range(C):
        low, high = scale_ranges[i]
        scale = torch.rand(spectrum.shape[:-1], generator=gen, device=spectrum.device).uniform_(low, high)
        scale_factors.append(scale)

    # 堆叠并扩展维度以匹配输入张量
    scale_factors = torch.stack(scale_factors, dim=-1)

    # 应用扰动
    return spectrum * scale_factors

class Dataset(pt_data.Dataset):
    def __init__(self):
        self.spectra = None
        self.spectra_mask = None
        self.precursors = None
        self.tokens = None
        self.label = None

    def __getitem__(self, idx):
        return_dict = {"spectra": self.spectra[idx],
                       "spectra_mask": self.spectra_mask[idx],
                       "precursors": self.precursors[idx],
                       "tokens": self.tokens[idx],
                       "label": self.label[idx]}
        return return_dict

    def __len__(self):
        return len(self.label)

    def fit_scale(self):
        pass

def dict2pt(
    batch: dict,
    s2i: dict,
    max_length: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """dict to pt_data.Dataset."""
    # gen dataset
    data_set = Dataset()
    data_set.spectra = np.nan_to_num(batch['spectra'])
    data_set.spectra_mask = np.nan_to_num(batch['spectra_mask'])
    data_set.precursors = np.nan_to_num(batch['precursors'])
    # data_set.tokens = torch.stack([aa_tokenize(p, s2i, max_length) for p in batch['peptides']]).numpy()
    data_set.tokens = np.nan_to_num(batch['tokens'])
    data_set.tokens = np.nan_to_num(batch['precursors'])
    data_set.label = np.nan_to_num(batch['label'])
    return data_set


def shuffle_file_list(file_list, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    idx = torch.randperm(len(file_list), generator=generator).numpy()
    file_list = (np.array(file_list)[idx]).tolist()
    return file_list


class Dataset_weight(pt_data.Dataset):
    def __init__(self):
        self.spectra = None
        self.spectra_mask = None
        self.precursors = None
        self.tokens = None
        self.label = None
        self.weight = None

    def __getitem__(self, idx):
        return_dict = {"spectra": self.spectra[idx],
                       "spectra_mask": self.spectra_mask[idx],
                       "precursors": self.precursors[idx],
                       "tokens": self.tokens[idx],
                       "label": self.label[idx],
                       "weight": self.weight[idx]}
        return return_dict

    def __len__(self):
        return len(self.label)

    def fit_scale(self):
        pass


def dict2pt_weight(
    batch: dict,
    s2i: dict,
    max_length: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """dict to pt_data.Dataset."""
    # gen dataset
    data_set = Dataset_weight()
    data_set.spectra = np.nan_to_num(batch['spectra'])
    data_set.spectra_mask = np.nan_to_num(batch['spectra_mask'])
    data_set.precursors = np.nan_to_num(batch['precursors'])
    data_set.tokens = np.nan_to_num(batch['tokens'])
    data_set.weight = np.nan_to_num(batch['weight'])
    data_set.label = np.nan_to_num(batch['label'])
    return data_set


class Dataset_weight_unmask(pt_data.Dataset):
    def __init__(self):
        self.spectra = None
        self.spectra_mask = None
        self.precursors = None
        self.tokens = None
        self.label = None
        self.weight = None
        self.unmask = None

    def __getitem__(self, idx):
        return_dict = {"spectra": self.spectra[idx],
                       "spectra_mask": self.spectra_mask[idx],
                       "precursors": self.precursors[idx],
                       "tokens": self.tokens[idx],
                       "label": self.label[idx],
                       "weight": self.weight[idx],
                       "unmask": self.unmask[idx]}
        return return_dict

    def __len__(self):
        return len(self.label)

    def fit_scale(self):
        pass


def dict2pt_weight_unmask(
    batch: dict,
    s2i: dict,
    max_length: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """dict to pt_data.Dataset."""
    # gen dataset
    data_set = Dataset_weight_unmask()
    data_set.spectra = np.nan_to_num(batch['spectra'])
    data_set.spectra_mask = np.nan_to_num(batch['spectra_mask'])
    data_set.precursors = np.nan_to_num(batch['precursors'])
    data_set.tokens = np.nan_to_num(batch['tokens'])
    data_set.weight = np.nan_to_num(batch['weight'])
    data_set.label = np.nan_to_num(batch['label'])
    data_set.unmask = np.nan_to_num(batch['unmask'])
    return data_set



class Dataset_weight_no_fdr01_target(pt_data.Dataset):
    def __init__(self):
        self.spectra = None
        self.spectra_mask = None
        self.precursors = None
        self.tokens = None
        self.label = None
        self.weight = None
        self.no_fdr01_target = None

    def __getitem__(self, idx):
        return_dict = {"spectra": self.spectra[idx],
                       "spectra_mask": self.spectra_mask[idx],
                       "precursors": self.precursors[idx],
                       "tokens": self.tokens[idx],
                       "label": self.label[idx],
                       "weight": self.weight[idx],
                       "no_fdr01_target": self.no_fdr01_target[idx]}
        return return_dict

    def __len__(self):
        return len(self.label)

    def fit_scale(self):
        pass


def dict2pt_weight_no_fdr01_target(
    batch: dict,
    s2i: dict,
    max_length: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """dict to pt_data.Dataset."""
    # gen dataset
    data_set = Dataset_weight_no_fdr01_target()
    data_set.spectra = np.nan_to_num(batch['spectra'])
    data_set.spectra_mask = np.nan_to_num(batch['spectra_mask'])
    data_set.precursors = np.nan_to_num(batch['precursors'])
    data_set.tokens = np.nan_to_num(batch['tokens'])
    data_set.weight = np.nan_to_num(batch['weight'])
    data_set.label = np.nan_to_num(batch['label'])
    data_set.no_fdr01_target = np.nan_to_num(batch['no_fdr01_target'])
    return data_set

    
def collate_numpy_batch(batch_data):
    """Collate batch of samples."""
    one_batch_spectra = torch.tensor(np.array([batch["spectra"] for batch in batch_data]), dtype=torch.float)
    one_batch_spectra_mask = torch.tensor(np.array([batch["spectra_mask"] for batch in batch_data]), dtype=torch.float)
    one_batch_precursors = torch.tensor(np.array([batch["precursors"] for batch in batch_data]), dtype=torch.float)
    one_batch_tokens = torch.tensor(np.array([batch["tokens"] for batch in batch_data]), dtype=torch.float)
    one_batch_label = torch.tensor(np.array([batch["label"] for batch in batch_data]), dtype=torch.float)

    return one_batch_spectra, one_batch_spectra_mask, one_batch_precursors, one_batch_tokens, one_batch_label

def collate_numpy_batch_weight(batch_data):
    """Collate batch of samples."""
    one_batch_spectra = torch.tensor(np.array([batch["spectra"] for batch in batch_data]), dtype=torch.float)
    one_batch_spectra_mask = torch.tensor(np.array([batch["spectra_mask"] for batch in batch_data]), dtype=torch.float)
    one_batch_precursors = torch.tensor(np.array([batch["precursors"] for batch in batch_data]), dtype=torch.float)
    one_batch_tokens = torch.tensor(np.array([batch["tokens"] for batch in batch_data]), dtype=torch.float)
    one_batch_label = torch.tensor(np.array([batch["label"] for batch in batch_data]), dtype=torch.float)
    one_batch_weight = torch.tensor(np.array([batch["weight"] for batch in batch_data]), dtype=torch.float)

    return one_batch_spectra, one_batch_spectra_mask, one_batch_precursors, one_batch_tokens, one_batch_label, one_batch_weight


def collate_numpy_batch_weight_augment(batch_data):
    """Collate batch of samples."""
    one_batch_spectra = torch.tensor(np.array([batch["spectra"] for batch in batch_data]), dtype=torch.float)
    one_batch_spectra_mask = torch.tensor(np.array([batch["spectra_mask"] for batch in batch_data]), dtype=torch.float)
    one_batch_precursors = torch.tensor(np.array([batch["precursors"] for batch in batch_data]), dtype=torch.float)
    one_batch_tokens = torch.tensor(np.array([batch["tokens"] for batch in batch_data]), dtype=torch.float)
    one_batch_label = torch.tensor(np.array([batch["label"] for batch in batch_data]), dtype=torch.float)
    one_batch_weight = torch.tensor(np.array([batch["weight"] for batch in batch_data]), dtype=torch.float)

    ## augment_spectrum
    one_batch_spectra = augment_spectrum(one_batch_spectra)
    
    return one_batch_spectra, one_batch_spectra_mask, one_batch_precursors, one_batch_tokens, one_batch_label, one_batch_weight


def collate_numpy_batch_weight_unmask(batch_data):
    """Collate batch of samples."""
    one_batch_spectra = torch.tensor(np.array([batch["spectra"] for batch in batch_data]), dtype=torch.float)
    one_batch_spectra_mask = torch.tensor(np.array([batch["spectra_mask"] for batch in batch_data]), dtype=torch.float)
    one_batch_precursors = torch.tensor(np.array([batch["precursors"] for batch in batch_data]), dtype=torch.float)
    one_batch_tokens = torch.tensor(np.array([batch["tokens"] for batch in batch_data]), dtype=torch.float)
    one_batch_label = torch.tensor(np.array([batch["label"] for batch in batch_data]), dtype=torch.float)
    one_batch_weight = torch.tensor(np.array([batch["weight"] for batch in batch_data]), dtype=torch.float)
    one_batch_unmask = torch.tensor(np.array([batch["unmask"] for batch in batch_data]), dtype=torch.float)

    return one_batch_spectra, one_batch_spectra_mask, one_batch_precursors, one_batch_tokens, one_batch_label, one_batch_weight, one_batch_unmask

def collate_numpy_batch_weight_no_fdr01_target(batch_data):
    """Collate batch of samples."""
    one_batch_spectra = torch.tensor(np.array([batch["spectra"] for batch in batch_data]), dtype=torch.float)
    one_batch_spectra_mask = torch.tensor(np.array([batch["spectra_mask"] for batch in batch_data]), dtype=torch.float)
    one_batch_precursors = torch.tensor(np.array([batch["precursors"] for batch in batch_data]), dtype=torch.float)
    one_batch_tokens = torch.tensor(np.array([batch["tokens"] for batch in batch_data]), dtype=torch.float)
    one_batch_label = torch.tensor(np.array([batch["label"] for batch in batch_data]), dtype=torch.float)
    one_batch_weight = torch.tensor(np.array([batch["weight"] for batch in batch_data]), dtype=torch.float)
    one_batch_no_fdr01_target = torch.tensor(np.array([batch["no_fdr01_target"] for batch in batch_data]), dtype=torch.float)

    return one_batch_spectra, one_batch_spectra_mask, one_batch_precursors, one_batch_tokens, one_batch_label, one_batch_weight, one_batch_no_fdr01_target


# https://blog.csdn.net/zhang19990111/article/details/131636456
def create_iterable_dataset(logging,
                            config,
                            s2i,
                            parse='train',
                            multi_node=False,
                            need_augment=False,
                            need_weight=False,
                            need_unmask=False,
                            need_no_fdr01_target=False,
                            seed=123):
    """
    Note: If you want to load all data in the memory, please set "read_part" to False.
    Args:
        :param logging: out logging.
        :param config: data from the yaml file.
        :param s2i: vocab.
        :param buffer_size: An integer. the size of file_name buffer.
    :return:
    """
    # update gpu_num
    if multi_node:
        gpu_num = int(config['gpu_num'])
    else:
        gpu_num = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
    # logging.info(f"******************multi_node: {multi_node}, need_weight: {need_weight}, need_unmask: {need_unmask},  need_no_fdr01_target: {need_no_fdr01_target}, gpu_num: {gpu_num};**********")
    logging.info(f"******************multi_node: {multi_node}, need_augment: {need_augment}, gpu_num: {gpu_num};**********")

    if parse == 'train':
        # 训练阶段
        if ';' in config['train_path']:
            total_train_path = config['train_path'].split(';')
            data_file_list = []
            for train_path in total_train_path:
                train_part_file_list = glob.glob(f'{train_path}/*.pkl')
                if len(train_part_file_list) > 0:
                    data_file_list.extend(train_part_file_list)
            logging.info(f"******************{parse} {config['train_path']}, total loaded: {len(data_file_list)};**********")
        else:
            data_file_list = glob.glob(f"{config['train_path']}/*pkl")
            logging.info(f"******************{parse} {config['train_path']}, loaded: {len(data_file_list)};**********")
        
        random.shuffle(data_file_list)
        data_file_list = shuffle_file_list(data_file_list, config['seed'])
        
        # 按照gpu_num数量，对数据集截断
        # if multi_node:
        file_bin_num = len(data_file_list) // gpu_num
        file_truncation_num = file_bin_num * gpu_num
        data_file_list = data_file_list[:file_truncation_num]
        # data_file_list = data_file_list[:1000] # debug

        train_dl = IterableDiartDataset(data_file_list,
                                        config["train_batch_size"],
                                        s2i, # vocab
                                        max_length=config["max_length"], # 序列最大长度
                                        buffer_size=config["buffer_size"], # 蓄水池深度
                                        gpu_num=gpu_num,
                                        shuffle=True,
                                        multi_node=multi_node,
                                        need_augment=need_augment,
                                        need_weight=need_weight,
                                        need_unmask=need_unmask,
                                        need_no_fdr01_target=need_no_fdr01_target,
                                        seed=config['seed'])
        logging.info(f"Data loaded: {len(train_dl) * config['train_batch_size']:,} training samples")
        return train_dl
    else:
        # 验证阶段
        data_file_list = glob.glob(f"{config['valid_path']}/*pkl")
        logging.info(f"******************{parse} loaded: {len(data_file_list)};**********")
        
        val_dl = IterableDiartDataset(data_file_list,
                                      config["predict_batch_size"],
                                      s2i, # vocab
                                      max_length=config["max_length"], # 序列最大长度
                                      gpu_num=gpu_num,
                                      shuffle=False,
                                      multi_node=multi_node,
                                      need_augment=need_augment,
                                      need_weight=need_weight,
                                      need_unmask=need_unmask,
                                      need_no_fdr01_target=need_no_fdr01_target)
        logging.info(f"{len(val_dl) * config['predict_batch_size']:,} validation samples")
        return val_dl


class IterableDiartDataset(IterableDataset):
    """
    Custom dataset class for dataset in order to use efficient
    dataloader tool provided by PyTorch.
    """

    def __init__(self,
                 file_list: list,
                 batch_size,
                 s2i, # vocab
                 max_length=50,
                 buffer_size=1,
                 gpu_num=1,
                 shuffle=False,
                 multi_node=False,
                 need_weight=False,
                 need_unmask=False,
                 need_no_fdr01_target=False,
                 need_augment=False,
                 seed=0,
                 bath_file_size=1,
                 **kwargs):
        super(IterableDiartDataset).__init__()
        # 文件列表
        self.file_list = file_list
        self.batch_size = batch_size
        self.s2i = s2i
        self.max_length = max_length

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # 单次抽样的文件大小
        self.bath_file_size = bath_file_size
        self.buffer_size = buffer_size

        self.gpu_num = gpu_num
        self.multi_node = multi_node
        self.need_augment = need_augment
        self.need_weight = need_weight
        self.need_unmask = need_unmask
        self.need_no_fdr01_target = need_no_fdr01_target
        
    def parse_file(self, file_name):
        # 加载pkl
        f = open(file_name, "rb")
        ds = pickle.loads(f.read())
        # print('file_name: ', file_name, 'ds: ', ds.keys(), 'fdr: ', ds['fdr'].shape, 'need_fdr:', self.need_fdr, flush=True)
        f.close()

        if self.need_augment:
            dpt = dict2pt_weight(ds, self.s2i, self.max_length)
            return DataLoader(dpt,
                              batch_size=self.batch_size,
                              shuffle=self.shuffle,
                              collate_fn=collate_numpy_batch_weight_augment,
                              num_workers=2,
                              drop_last=True,
                              pin_memory=True)
        
        elif self.need_no_fdr01_target:
            dpt = dict2pt_weight_no_fdr01_target(ds, self.s2i, self.max_length)
            return DataLoader(dpt,
                              batch_size=self.batch_size,
                              shuffle=self.shuffle,
                              collate_fn=collate_numpy_batch_weight_no_fdr01_target,
                              num_workers=2,
                              drop_last=True,
                              pin_memory=True)
        elif self.need_unmask:
            dpt = dict2pt_weight_unmask(ds, self.s2i, self.max_length)
            return DataLoader(dpt,
                              batch_size=self.batch_size,
                              shuffle=self.shuffle,
                              collate_fn=collate_numpy_batch_weight_unmask,
                              num_workers=2,
                              drop_last=True,
                              pin_memory=True)
        elif self.need_weight:
            dpt = dict2pt_weight(ds, self.s2i, self.max_length)
            return DataLoader(dpt,
                              batch_size=self.batch_size,
                              shuffle=self.shuffle,
                              collate_fn=collate_numpy_batch_weight,
                              num_workers=2,
                              drop_last=True,
                              pin_memory=True)
        else:
            dpt = dict2pt(ds, self.s2i, self.max_length)
            return DataLoader(dpt,
                              batch_size=self.batch_size,
                              shuffle=self.shuffle,
                              collate_fn=collate_numpy_batch,
                              num_workers=2,
                              drop_last=True,
                              pin_memory=True)

    def file_mapper(self, file_list):
        idx = 0
        file_num = len(file_list)
        while idx < file_num:
            yield self.parse_file(file_list[idx])
            idx += 1

    def __iter__(self):
        if self.gpu_num > 1:
            if self.multi_node:# 多机多卡
                if 'RANK' in os.environ:
                    rank = int(os.environ['RANK'])
                else:
                    rank = 0
            else:# 单机多卡
                if 'LOCAL_RANK' in os.environ:
                    rank = int(os.environ['LOCAL_RANK'])
                else:
                    rank = 0
            
            file_itr = self.file_list[rank::self.gpu_num]
            # print('rank: ', rank, 'file_itr', file_itr[0], flush=True)
            
        else:
            # 单卡
            file_itr = self.file_list

        file_mapped_itr = self.file_mapper(file_itr)

        if self.shuffle:
            return self._shuffle(file_mapped_itr)
        else:
            return file_mapped_itr

    def __len__(self):
        if self.gpu_num > 1:
            return math.ceil(len(self.file_list) / self.gpu_num)
        else:
            return len(self.file_list)
        
    def set_epoch(self, epoch):
        self.epoch = epoch

    def generate_random_num(self):
        while True:
            random.seed(self.epoch + self.seed)
            random_nums = random.sample(range(self.buffer_size), self.bath_file_size)
            yield from random_nums

    # 蓄水池抽样（每个数据都是以m/N的概率获得的）
    def _shuffle(self, mapped_itr):
        buffer = []
        for dt in mapped_itr:
            # 如果接收的数据量小于m，则依次放入蓄水池。
            if len(buffer) < self.buffer_size:
                buffer.append(dt)
            # 当接收到第i个数据时，i >= m，在[0, i]范围内取以随机数d。
            # 若d的落在[0, m-1]范围内，则用接收到的第i个数据替换蓄水池中的第d个数据。
            else:
                i = next(self.generate_random_num())
                yield buffer[i]
                buffer[i] = dt
        random.shuffle(buffer)
        yield from buffer

def aa_tokenize(sequence, s2i, max_length):
    """Transform a peptide sequence into tokens

    Parameters
    ----------
    sequence : str
        A peptide sequence.
    s2i : str
        amino acid s2i.
    max_length : int
        default is 50.

    Returns
    -------
    torch.Tensor
        The token for each amino acid in the peptide sequence.
    """
    sequence = sequence.replace("I", "L").replace('n[42]', 'X')
    sequence = re.split(r"(?<=.)(?=[A-Z])", sequence)

    tokens = torch.tensor([s2i[aa] for aa in sequence])
    tokens = F.pad(tokens, (0, max_length - tokens.shape[0]), 'constant', 0)# padding
    return tokens
        
def mask_unk_decoy_tokens(tokens, label, mask_index, unk_index, token_mask_ratio=0.15, device='cpu'):
    """Args:
            tokens: float Tensor (batch, 50)
            label: float Tensor (batch)
            mask_index: int=1
            unk_index: int=28
            token_mask_ratio: float=0.15
            device: cpu or cuda
        Returns:
            tokens_mask: float Tensor (batch, 50)
            tokens_label: float Tensor (batch)
    """
    # 生成mask
    mask = torch.rand(tokens.shape[0], tokens.shape[1]).to(device)
    mask = torch.where(mask < token_mask_ratio, 1, 0)
    
    tokens = tokens.to(device)

    # 生成输入模型的tokens_mask
    tokens_mask = torch.where((mask==1) & (tokens > mask_index), mask_index, tokens).to(device)

    # 调整label为[-1, 1]，方便映射
    label_adjust = torch.where(label==0, -1, label).to(device)
    label_mask = torch.mul(mask, label_adjust.reshape(label_adjust.shape[0], -1)).to(torch.int8).to(device)
    
    # target：预测被mask的氨基酸；
    # decoy：预测固定的替换符
    tokens_label = torch.where((label_mask==1) & (tokens > mask_index), tokens, \
                   torch.where((label_mask==-1) & (tokens > mask_index), unk_index, 0)).to(device)
    return tokens_mask, tokens_label



def mask_batch_data(batch, pad_index=0, mask_index=1, unk_index=28, mask_ratio=0.9, token_mask_ratio=0.15, device='cpu'):
    """Args:
            batch: spectra, spectra_mask, precursors, tokens, label, weight. the dim is 
            float Tensor (batch, 300, 2), float Tensor (batch, 300), float Tensor (batch, 3), float Tensor (batch, 50), float Tensor (batch)
            pad_index: int=0
            mask_index: int=1
            unk_index: int=28
            mask_ratio: float=0.9（样本mask的比例 ）
            token_mask_ratio: float=0.15 (每个样本，sequence mask比例)
            device: cpu or cuda
        Returns:
            batch: spectra, spectra_mask, precursors, tokens, label. 
    """
    spectra, spectra_mask, precursors, tokens, label, weight = batch
    
    # 生成掩码对应的index
    full_indices = list(range(label.shape[0]))
    mask_size = int(mask_ratio * len(full_indices))
    remain_size = len(full_indices) - mask_size
    mask_index_list, remain_index_list = random_split(full_indices, [mask_size, remain_size])

    # 按照是否掩码分组
    mask_spectra = spectra[mask_index_list].to(device)
    mask_spectra_mask = spectra_mask[mask_index_list].to(device)
    mask_precursors = precursors[mask_index_list].to(device)
    mask_tokens = tokens[mask_index_list].to(device)
    mask_label = label[mask_index_list].to(device)
    mask_weight = weight[mask_index_list].to(device)

    remain_spectra = spectra[remain_index_list].to(device)
    remain_spectra_mask = spectra_mask[remain_index_list].to(device)
    remain_precursors = precursors[remain_index_list].to(device)
    remain_tokens = tokens[remain_index_list].to(device)
    remain_label = label[remain_index_list].to(device)
    remain_weight = weight[remain_index_list].to(device)
    
    tokens_masked, tokens_label_masked = mask_unk_decoy_tokens(mask_tokens,
                                                               mask_label,
                                                               mask_index,
                                                               unk_index,
                                                               token_mask_ratio=token_mask_ratio,
                                                               device=device)
    remain_tokens_label = torch.zeros(remain_tokens.shape[0], remain_tokens.shape[1]).to(device)
    
    # 合并batch
    spectra = torch.cat((mask_spectra, remain_spectra))
    spectra_mask = torch.cat((mask_spectra_mask, remain_spectra_mask))
    precursors = torch.cat((mask_precursors, remain_precursors))
    tokens = torch.cat((tokens_masked, remain_tokens))
    tokens_label = torch.cat((tokens_label_masked, remain_tokens_label))
    label = torch.cat((mask_label, remain_label))
    weight = torch.cat((mask_weight, remain_weight))
    return spectra, spectra_mask, precursors, tokens, tokens_label, label, weight


def mask_decoy_1unk(tokens, unk_index=28, device='cpu'):
    # 获取矩阵的形状
    num_rows, num_cols = tokens.shape

    # 随机生成每行保留的索引
    random_indices = torch.randint(0, num_cols, (num_rows,)).to(device)  # 每行随机生成一个索引

    # 创建掩码矩阵
    mask = torch.zeros_like(tokens, dtype=torch.bool).to(device)  # 初始化全 False 的掩码
    mask[torch.arange(num_rows), random_indices] = True  # 标记需要保留的位置

    # 更新矩阵：非保留位置设置为目标值
    result = torch.where(mask, tokens, torch.tensor(unk_index)).to(device)
    return result


def mask_target_decoy_1unk_tokens(tokens, label, mask_index, unk_index, token_mask_ratio=0.15, device='cpu'):
    """Args:
            tokens: float Tensor (batch, 50)
            label: float Tensor (batch)
            mask_index: int=1
            unk_index: int=28
            token_mask_ratio: float=0.15
            device: cpu or cuda
        Returns:
            tokens_mask: float Tensor (batch, 50)
            tokens_label: float Tensor (batch)
    """
    # 生成mask
    mask = torch.rand(tokens.shape[0], tokens.shape[1]).to(device)
    mask = torch.where(mask < token_mask_ratio, 1, 0)
    
    tokens = tokens.to(device)

    # 生成输入模型的tokens_mask
    tokens_mask = torch.where((mask==1) & (tokens > mask_index), mask_index, tokens).to(device)

    # 调整label为[-1, 1]，方便映射
    label_adjust = torch.where(label==0, -1, label).to(device)
    label_mask = torch.mul(mask, label_adjust.reshape(label_adjust.shape[0], -1)).to(torch.int8).to(device)
    
    # decoy每行留一个，其它均设置为unk
    decoy_tokens = mask_decoy_1unk(tokens, unk_index, device)
    
    # target：预测被mask的氨基酸；
    # decoy：预测固定的替换符
    tokens_label = torch.where((label_mask==1) & (tokens > mask_index), tokens, \
                   torch.where((label_mask==-1) & (tokens > mask_index), decoy_tokens, 0)).to(device)
    return tokens_mask, tokens_label


def mask_batch_decoy_1unk_data(batch, pad_index=0, mask_index=1, unk_index=28, mask_ratio=0.9, token_mask_ratio=0.15, device='cpu'):
    """Args:
            batch: spectra, spectra_mask, precursors, tokens, label, weight. the dim is 
            float Tensor (batch, 300, 2), float Tensor (batch, 300), float Tensor (batch, 3), float Tensor (batch, 50), float Tensor (batch)
            pad_index: int=0
            mask_index: int=1
            unk_index: int=28
            mask_ratio: float=0.9（样本mask的比例 ）
            token_mask_ratio: float=0.15 (每个样本，sequence mask比例)
            device: cpu or cuda
        Returns:
            batch: spectra, spectra_mask, precursors, tokens, label. 
    """
    spectra, spectra_mask, precursors, tokens, label, weight = batch
    
    # 生成掩码对应的index
    full_indices = list(range(label.shape[0]))
    mask_size = int(mask_ratio * len(full_indices))
    remain_size = len(full_indices) - mask_size
    mask_index_list, remain_index_list = random_split(full_indices, [mask_size, remain_size])

    # 按照是否掩码分组
    mask_spectra = spectra[mask_index_list].to(device)
    mask_spectra_mask = spectra_mask[mask_index_list].to(device)
    mask_precursors = precursors[mask_index_list].to(device)
    mask_tokens = tokens[mask_index_list].to(device)
    mask_label = label[mask_index_list].to(device)
    mask_weight = weight[mask_index_list].to(device)

    remain_spectra = spectra[remain_index_list].to(device)
    remain_spectra_mask = spectra_mask[remain_index_list].to(device)
    remain_precursors = precursors[remain_index_list].to(device)
    remain_tokens = tokens[remain_index_list].to(device)
    remain_label = label[remain_index_list].to(device)
    remain_weight = weight[remain_index_list].to(device)
    
    tokens_masked, tokens_label_masked = mask_target_decoy_1unk_tokens(mask_tokens,
                                                                       mask_label,
                                                                       mask_index,
                                                                       unk_index,
                                                                       token_mask_ratio=token_mask_ratio,
                                                                       device=device)
    remain_tokens_label = torch.zeros(remain_tokens.shape[0], remain_tokens.shape[1]).to(device)
    
    # 合并batch
    spectra = torch.cat((mask_spectra, remain_spectra))
    spectra_mask = torch.cat((mask_spectra_mask, remain_spectra_mask))
    precursors = torch.cat((mask_precursors, remain_precursors))
    tokens = torch.cat((tokens_masked, remain_tokens))
    tokens_label = torch.cat((tokens_label_masked, remain_tokens_label))
    label = torch.cat((mask_label, remain_label))
    weight = torch.cat((mask_weight, remain_weight))
    return spectra, spectra_mask, precursors, tokens, tokens_label, label, weight


def mask_batch_decoy_1unk_data_no_fdr01_target(batch, pad_index=0, mask_index=1, unk_index=28, mask_ratio=0.9, token_mask_ratio=0.15, device='cpu'):
    """Args:
            batch: spectra, spectra_mask, precursors, tokens, label, weight. the dim is 
            float Tensor (batch, 300, 2), float Tensor (batch, 300), float Tensor (batch, 3), float Tensor (batch, 50), float Tensor (batch)
            pad_index: int=0
            mask_index: int=1
            unk_index: int=28
            mask_ratio: float=0.9（样本mask的比例 ）
            token_mask_ratio: float=0.15 (每个样本，sequence mask比例)
            device: cpu or cuda
        Returns:
            batch: spectra, spectra_mask, precursors, tokens, label. 
    """
    spectra, spectra_mask, precursors, tokens, label, weight, no_fdr01_target = batch
    
    # 生成掩码对应的index
    full_indices = list(range(label.shape[0]))
    mask_size = int(mask_ratio * len(full_indices))
    remain_size = len(full_indices) - mask_size
    mask_index_list, remain_index_list = random_split(full_indices, [mask_size, remain_size])

    # 按照是否掩码分组
    mask_spectra = spectra[mask_index_list].to(device)
    mask_spectra_mask = spectra_mask[mask_index_list].to(device)
    mask_precursors = precursors[mask_index_list].to(device)
    mask_tokens = tokens[mask_index_list].to(device)
    mask_label = label[mask_index_list].to(device)
    mask_weight = weight[mask_index_list].to(device)
    mask_no_fdr01_target = no_fdr01_target[mask_index_list].to(device)

    remain_spectra = spectra[remain_index_list].to(device)
    remain_spectra_mask = spectra_mask[remain_index_list].to(device)
    remain_precursors = precursors[remain_index_list].to(device)
    remain_tokens = tokens[remain_index_list].to(device)
    remain_label = label[remain_index_list].to(device)
    remain_weight = weight[remain_index_list].to(device)
    remain_no_fdr01_target = no_fdr01_target[remain_index_list].to(device)
    
    tokens_masked, tokens_label_masked = mask_target_decoy_1unk_tokens(mask_tokens,
                                                                       mask_label,
                                                                       mask_index,
                                                                       unk_index,
                                                                       token_mask_ratio=token_mask_ratio,
                                                                       device=device)
    remain_tokens_label = torch.zeros(remain_tokens.shape[0], remain_tokens.shape[1]).to(device)
    
    # 合并batch
    spectra = torch.cat((mask_spectra, remain_spectra))
    spectra_mask = torch.cat((mask_spectra_mask, remain_spectra_mask))
    precursors = torch.cat((mask_precursors, remain_precursors))
    tokens = torch.cat((tokens_masked, remain_tokens))
    tokens_label = torch.cat((tokens_label_masked, remain_tokens_label))
    label = torch.cat((mask_label, remain_label))
    weight = torch.cat((mask_weight, remain_weight))
    no_fdr01_target = torch.cat((mask_no_fdr01_target, remain_no_fdr01_target))
    return spectra, spectra_mask, precursors, tokens, tokens_label, label, weight, no_fdr01_target


def mask_target_decoy_1unk_tft_maskratio_tokens(tokens, label, weight, mask_index, unk_index, decoy_token_mask_ratio=0.4, tft_token_mask_ratio=0.5, device='cpu'):
    """Args:
            tokens: float Tensor (batch, 50)
            label: float Tensor (batch)
            mask_index: int=1
            unk_index: int=28
            decoy_token_mask_ratio: float=0.4
            tft_token_mask_ratio: float=0.5
            device: cpu or cuda
        Returns:
            tokens_mask: float Tensor (batch, 50)
            tokens_label: float Tensor (batch)
    """
    # 标记decoy
    decoy_tag = (label < 0.5) & (weight > 0.9).to(torch.long).to(device)
    
    
    # 生成mask
    mask = torch.rand(tokens.shape[0], tokens.shape[1]).to(device)
    
    # decoy按照token_mask_ratio进行mask；target和false target按照tft_token_mask_ratio进行mask；
    mask[decoy_tag, :] = torch.where(mask[decoy_tag, :] < decoy_token_mask_ratio, 1.0, 0.0)
    mask[~decoy_tag, :] = torch.where(mask[~decoy_tag, :] < tft_token_mask_ratio, 1.0, 0.0)

    tokens = tokens.to(device)

    # 生成输入模型的tokens_mask
    tokens_mask = torch.where((mask > 0.9) & (tokens > mask_index), mask_index, tokens).to(device)

    # 调整label为[-1, 1]，方便映射
    label_adjust = torch.where(label==0, -1, label).to(device)
    label_mask = torch.mul(mask, label_adjust.reshape(label_adjust.shape[0], -1)).to(torch.int8).to(device)
    
    # decoy每行留一个，其它均设置为unk
    decoy_tokens = mask_decoy_1unk(tokens, unk_index, device)
    
    # target：预测被mask的氨基酸；
    # decoy：预测固定的替换符
    tokens_label = torch.where((label_mask==1) & (tokens > mask_index), tokens, \
                   torch.where((label_mask==-1) & (tokens > mask_index), decoy_tokens, 0)).to(device)
    return tokens_mask, tokens_label


def mask_batch_decoy_1unk_tft_maskratio_data(batch, pad_index=0, mask_index=1, unk_index=28, mask_ratio=0.9, decoy_token_mask_ratio=0.4, tft_token_mask_ratio=0.5, device='cpu'):
    """Args:
            batch: spectra, spectra_mask, precursors, tokens, label, weight. the dim is 
            float Tensor (batch, 300, 2), float Tensor (batch, 300), float Tensor (batch, 3), float Tensor (batch, 50), float Tensor (batch)
            pad_index: int=0
            mask_index: int=1
            unk_index: int=28
            mask_ratio: float=0.9（样本mask的比例 ）
            decoy_token_mask_ratio: float=0.4 (decoy，sequence mask比例)
            tft_token_mask_ratio: float=0.5 (target和false target，sequence mask比例)
            device: cpu or cuda
        Returns:
            batch: spectra, spectra_mask, precursors, tokens, label. 
    """
    spectra, spectra_mask, precursors, tokens, label, weight = batch
    
    # 生成掩码对应的index
    full_indices = list(range(label.shape[0]))
    mask_size = int(mask_ratio * len(full_indices))
    remain_size = len(full_indices) - mask_size
    mask_index_list, remain_index_list = random_split(full_indices, [mask_size, remain_size])

    # 按照是否掩码分组
    mask_spectra = spectra[mask_index_list].to(device)
    mask_spectra_mask = spectra_mask[mask_index_list].to(device)
    mask_precursors = precursors[mask_index_list].to(device)
    mask_tokens = tokens[mask_index_list].to(device)
    mask_label = label[mask_index_list].to(device)
    mask_weight = weight[mask_index_list].to(device)

    remain_spectra = spectra[remain_index_list].to(device)
    remain_spectra_mask = spectra_mask[remain_index_list].to(device)
    remain_precursors = precursors[remain_index_list].to(device)
    remain_tokens = tokens[remain_index_list].to(device)
    remain_label = label[remain_index_list].to(device)
    remain_weight = weight[remain_index_list].to(device)
    
    tokens_masked, tokens_label_masked = mask_target_decoy_1unk_tft_maskratio_tokens(mask_tokens,
                                                                                     mask_label,
                                                                                     mask_weight,
                                                                                     mask_index,
                                                                                     unk_index,
                                                                                     decoy_token_mask_ratio=decoy_token_mask_ratio,
                                                                                     tft_token_mask_ratio=tft_token_mask_ratio,
                                                                                     device=device)
    remain_tokens_label = torch.zeros(remain_tokens.shape[0], remain_tokens.shape[1]).to(device)
    
    # 合并batch
    spectra = torch.cat((mask_spectra, remain_spectra))
    spectra_mask = torch.cat((mask_spectra_mask, remain_spectra_mask))
    precursors = torch.cat((mask_precursors, remain_precursors))
    tokens = torch.cat((tokens_masked, remain_tokens))
    tokens_label = torch.cat((tokens_label_masked, remain_tokens_label))
    label = torch.cat((mask_label, remain_label))
    weight = torch.cat((mask_weight, remain_weight))
    return spectra, spectra_mask, precursors, tokens, tokens_label, label, weight



def mask_adjust_maskratio_tokens(tokens, label, weight, mask_index, unk_index, decoy_token_mask_ratio=0.4, target_token_mask_ratio=0.5, ft_token_mask_ratio=0.5, device='cpu'):
    """Args:
            tokens: float Tensor (batch, 50)
            label: float Tensor (batch)
            mask_index: int=1
            unk_index: int=28
            decoy_token_mask_ratio: float=0.4
            target_token_mask_ratio: float=0.5
            ft_token_mask_ratio: float=0.5
            device: cpu or cuda
        Returns:
            tokens_mask: float Tensor (batch, 50)
            tokens_label: float Tensor (batch)
    """
    # 标记decoy
    decoy_tag = (label < 0.5) & (weight > 0.9).to(torch.long).to(device)
    target_tag = (label > 0.5) & (weight > 0.9).to(torch.long).to(device)
    false_target_tag = (label < 0.5) & (weight < 0.9).to(torch.long).to(device)
    
    # 生成mask
    mask = torch.rand(tokens.shape[0], tokens.shape[1]).to(device)
    
    # decoy/target和false target按照各自token_mask_ratio进行mask
    mask[decoy_tag, :] = torch.where(mask[decoy_tag, :] < decoy_token_mask_ratio, 1.0, 0.0)
    mask[target_tag, :] = torch.where(mask[target_tag, :] < target_token_mask_ratio, 1.0, 0.0)
    mask[false_target_tag, :] = torch.where(mask[false_target_tag, :] < ft_token_mask_ratio, 1.0, 0.0)

    tokens = tokens.to(device)

    # 生成输入模型的tokens_mask
    tokens_mask = torch.where((mask > 0.9) & (tokens > mask_index), mask_index, tokens).to(device)

    # 调整label为[-1, 1]，方便映射
    label_adjust = torch.where(label==0, -1, label).to(device)
    label_mask = torch.mul(mask, label_adjust.reshape(label_adjust.shape[0], -1)).to(torch.int8).to(device)
    
    # decoy每行留一个，其它均设置为unk
    decoy_tokens = mask_decoy_1unk(tokens, unk_index, device)
    
    # target：预测被mask的氨基酸；
    # decoy：预测固定的替换符
    tokens_label = torch.where((label_mask==1) & (tokens > mask_index), tokens, \
                   torch.where((label_mask==-1) & (tokens > mask_index), decoy_tokens, 0)).to(device)
    return tokens_mask, tokens_label


def mask_batch_adjust_maskratio_data(batch, pad_index=0, mask_index=1, unk_index=28, mask_ratio=0.9, decoy_token_mask_ratio=0.4, target_token_mask_ratio=0.5, ft_token_mask_ratio=0.5, device='cpu'):
    """Args:
            batch: spectra, spectra_mask, precursors, tokens, label, weight. the dim is 
            float Tensor (batch, 300, 2), float Tensor (batch, 300), float Tensor (batch, 3), float Tensor (batch, 50), float Tensor (batch)
            pad_index: int=0
            mask_index: int=1
            unk_index: int=28
            mask_ratio: float=0.9（样本mask的比例 ）
            decoy_token_mask_ratio: float=0.4 (decoy，sequence mask比例)
            target_token_mask_ratio: float=0.5 (target，sequence mask比例)
            ft_token_mask_ratio: float=0.4 (false target，sequence mask比例)
            device: cpu or cuda
        Returns:
            batch: spectra, spectra_mask, precursors, tokens, label. 
    """
    spectra, spectra_mask, precursors, tokens, label, weight = batch
    
    # 生成掩码对应的index
    full_indices = list(range(label.shape[0]))
    mask_size = int(mask_ratio * len(full_indices))
    remain_size = len(full_indices) - mask_size
    mask_index_list, remain_index_list = random_split(full_indices, [mask_size, remain_size])

    # 按照是否掩码分组
    mask_spectra = spectra[mask_index_list].to(device)
    mask_spectra_mask = spectra_mask[mask_index_list].to(device)
    mask_precursors = precursors[mask_index_list].to(device)
    mask_tokens = tokens[mask_index_list].to(device)
    mask_label = label[mask_index_list].to(device)
    mask_weight = weight[mask_index_list].to(device)

    remain_spectra = spectra[remain_index_list].to(device)
    remain_spectra_mask = spectra_mask[remain_index_list].to(device)
    remain_precursors = precursors[remain_index_list].to(device)
    remain_tokens = tokens[remain_index_list].to(device)
    remain_label = label[remain_index_list].to(device)
    remain_weight = weight[remain_index_list].to(device)
    
    tokens_masked, tokens_label_masked = mask_adjust_maskratio_tokens(mask_tokens,
                                                                      mask_label,
                                                                      mask_weight,
                                                                      mask_index,
                                                                      unk_index,
                                                                      decoy_token_mask_ratio=decoy_token_mask_ratio,
                                                                      target_token_mask_ratio=target_token_mask_ratio,
                                                                      ft_token_mask_ratio=ft_token_mask_ratio,
                                                                      device=device)
    remain_tokens_label = torch.zeros(remain_tokens.shape[0], remain_tokens.shape[1]).to(device)
    
    # 合并batch
    spectra = torch.cat((mask_spectra, remain_spectra))
    spectra_mask = torch.cat((mask_spectra_mask, remain_spectra_mask))
    precursors = torch.cat((mask_precursors, remain_precursors))
    tokens = torch.cat((tokens_masked, remain_tokens))
    tokens_label = torch.cat((tokens_label_masked, remain_tokens_label))
    label = torch.cat((mask_label, remain_label))
    weight = torch.cat((mask_weight, remain_weight))
    return spectra, spectra_mask, precursors, tokens, tokens_label, label, weight



def mask_batch_data_by_unmask(batch, pad_index=0, mask_index=1, unk_index=28, token_mask_ratio=0.4, device='cpu'):
    """Args:
            batch: spectra, spectra_mask, precursors, tokens, label, unmask. the dim is 
             (float Tensor (batch, 300, 2), float Tensor (batch, 300), float Tensor (batch, 3), float Tensor (batch, 50), float Tensor (batch), float Tensor (batch)
            pad_index: int=0
            mask_index: int=1
            unk_index: int=28
            token_mask_ratio: float=0.4 (每个样本，sequence mask比例从15%，提升至40%)
            device: cpu or cuda
        Returns:
            batch: spectra, spectra_mask, precursors, tokens, tokens_label, label, weight. 
    """
    spectra, spectra_mask, precursors, tokens, label, weight, unmask = batch
    
    # 生成掩码对应的index
    full_indices = list(range(label.shape[0]))
    
    # 被标记为unmask的样本，不进行序列掩码；其它，进行序列掩码
    remain_index_list = [i for i, value in enumerate(unmask.cpu().numpy().tolist()) if value >= 1]
    mask_index_list = list(set(full_indices) - set(remain_index_list))

    # 按照是否掩码分组
    mask_spectra = spectra[mask_index_list].to(device)
    mask_spectra_mask = spectra_mask[mask_index_list].to(device)
    mask_precursors = precursors[mask_index_list].to(device)
    mask_tokens = tokens[mask_index_list].to(device)
    mask_label = label[mask_index_list].to(device)
    mask_weight = weight[mask_index_list].to(device)

    remain_spectra = spectra[remain_index_list].to(device)
    remain_spectra_mask = spectra_mask[remain_index_list].to(device)
    remain_precursors = precursors[remain_index_list].to(device)
    remain_tokens = tokens[remain_index_list].to(device)
    remain_label = label[remain_index_list].to(device)
    remain_weight = weight[remain_index_list].to(device)
    
    tokens_masked, tokens_label_masked = mask_unk_decoy_tokens(mask_tokens,
                                                               mask_label,
                                                               mask_index,
                                                               unk_index,
                                                               token_mask_ratio=token_mask_ratio,
                                                               device=device)
    remain_tokens_label = torch.zeros(remain_tokens.shape[0], remain_tokens.shape[1]).to(device)
    
    # 合并batch
    spectra = torch.cat((mask_spectra, remain_spectra))
    spectra_mask = torch.cat((mask_spectra_mask, remain_spectra_mask))
    precursors = torch.cat((mask_precursors, remain_precursors))
    tokens = torch.cat((tokens_masked, remain_tokens))
    tokens_label = torch.cat((tokens_label_masked, remain_tokens_label))
    label = torch.cat((mask_label, remain_label))
    weight = torch.cat((mask_weight, remain_weight))
    return spectra, spectra_mask, precursors, tokens, tokens_label, label, weight


def mask_batch_data_by_unmask_adjust(batch, pad_index=0, mask_index=1, unk_index=28, token_mask_ratio=0.4, device='cpu'):
    """Args:
            batch: spectra, spectra_mask, precursors, tokens, label, unmask. the dim is 
             (float Tensor (batch, 300, 2), float Tensor (batch, 300), float Tensor (batch, 3), float Tensor (batch, 50), float Tensor (batch), float Tensor (batch)
            pad_index: int=0
            mask_index: int=1
            unk_index: int=28
            token_mask_ratio: float=0.4 (每个样本，sequence mask比例从15%，提升至40%)
            device: cpu or cuda
        Returns:
            batch: spectra, spectra_mask, precursors, tokens, tokens_label, label, weight. 
    """
    spectra, spectra_mask, precursors, tokens, label, weight, unmask = batch
    
    # 生成掩码对应的index
    full_indices = list(range(label.shape[0]))
    
    # 被标记为unmask的样本，不进行序列掩码；其它，进行序列掩码
    remain_index_list = [i for i, value in enumerate(unmask.cpu().numpy().tolist()) if value >= 1]
    mask_index_list = list(set(full_indices) - set(remain_index_list))
    
    # 被标记为unmask的样本，随机选一半，token_mask_ratio设置为0.3
    if len(remain_index_list) > 2:
        adjust_index_list = random.sample(remain_index_list, len(remain_index_list) // 2)
        remain_index_list = list(set(remain_index_list) - set(adjust_index_list)) # update
        
        adjust_spectra = spectra[adjust_index_list].to(device)
        adjust_spectra_mask = spectra_mask[adjust_index_list].to(device)
        adjust_precursors = precursors[adjust_index_list].to(device)
        adjust_tokens = tokens[adjust_index_list].to(device)
        adjust_label = label[adjust_index_list].to(device)
        adjust_weight = weight[adjust_index_list].to(device)
        
        
        tokens_adjusted, tokens_label_adjusted = mask_unk_decoy_tokens(adjust_tokens,
                                                                       adjust_label,
                                                                       mask_index,
                                                                       unk_index,
                                                                       token_mask_ratio=0.3,
                                                                       device=device)
        
        

    # 按照是否掩码分组
    mask_spectra = spectra[mask_index_list].to(device)
    mask_spectra_mask = spectra_mask[mask_index_list].to(device)
    mask_precursors = precursors[mask_index_list].to(device)
    mask_tokens = tokens[mask_index_list].to(device)
    mask_label = label[mask_index_list].to(device)
    mask_weight = weight[mask_index_list].to(device)
    

    remain_spectra = spectra[remain_index_list].to(device)
    remain_spectra_mask = spectra_mask[remain_index_list].to(device)
    remain_precursors = precursors[remain_index_list].to(device)
    remain_tokens = tokens[remain_index_list].to(device)
    remain_label = label[remain_index_list].to(device)
    remain_weight = weight[remain_index_list].to(device)
    
    tokens_masked, tokens_label_masked = mask_unk_decoy_tokens(mask_tokens,
                                                               mask_label,
                                                               mask_index,
                                                               unk_index,
                                                               token_mask_ratio=token_mask_ratio,
                                                               device=device)
    

    
    remain_tokens_label = torch.zeros(remain_tokens.shape[0], remain_tokens.shape[1]).to(device)
    
    if len(remain_index_list) > 2:
        # 合并batch
        spectra = torch.cat((mask_spectra, adjust_spectra, remain_spectra))
        spectra_mask = torch.cat((mask_spectra_mask, adjust_spectra_mask, remain_spectra_mask))
        precursors = torch.cat((mask_precursors, adjust_precursors, remain_precursors))
        tokens = torch.cat((tokens_masked, tokens_adjusted, remain_tokens))
        tokens_label = torch.cat((tokens_label_masked, tokens_label_adjusted, remain_tokens_label))
        label = torch.cat((mask_label, adjust_label, remain_label))
        weight = torch.cat((mask_weight, adjust_weight, remain_weight))
    else:
        spectra = torch.cat((mask_spectra, remain_spectra))
        spectra_mask = torch.cat((mask_spectra_mask, remain_spectra_mask))
        precursors = torch.cat((mask_precursors, remain_precursors))
        tokens = torch.cat((tokens_masked, remain_tokens))
        tokens_label = torch.cat((tokens_label_masked, remain_tokens_label))
        label = torch.cat((mask_label, remain_label))
        weight = torch.cat((mask_weight, remain_weight))
    return spectra, spectra_mask, precursors, tokens, tokens_label, label, weight


def mask_batch_data_by_unmask_psmID(batch, pad_index=0, mask_index=1, unk_index=28, token_mask_ratio=0.4, device='cpu'):
    """Args:
            batch: spectra, spectra_mask, precursors, tokens, label, unmask. the dim is 
             (float Tensor (batch, 300, 2), float Tensor (batch, 300), float Tensor (batch, 3), float Tensor (batch, 50), float Tensor (batch), float Tensor (batch)
            pad_index: int=0
            mask_index: int=1
            unk_index: int=28
            token_mask_ratio: float=0.4 (每个样本，sequence mask比例从15%，提升至40%)
            device: cpu or cuda
        Returns:
            batch: spectra, spectra_mask, precursors, tokens, tokens_label, label, weight. 
    """
    spectra, spectra_mask, precursors, tokens, label, weight, unmask, psm_id = batch
    
    # 生成掩码对应的index
    full_indices = list(range(label.shape[0]))
    
    # 被标记为unmask的样本，不进行序列掩码；其它，进行序列掩码
    remain_index_list = [i for i, value in enumerate(unmask.cpu().numpy().tolist()) if value >= 1]
    mask_index_list = list(set(full_indices) - set(remain_index_list))

    # 按照是否掩码分组
    mask_spectra = spectra[mask_index_list].to(device)
    mask_spectra_mask = spectra_mask[mask_index_list].to(device)
    mask_precursors = precursors[mask_index_list].to(device)
    mask_tokens = tokens[mask_index_list].to(device)
    mask_label = label[mask_index_list].to(device)
    mask_weight = weight[mask_index_list].to(device)
    mask_psm_id = [psm_id[i] for i in mask_index_list]

    remain_spectra = spectra[remain_index_list].to(device)
    remain_spectra_mask = spectra_mask[remain_index_list].to(device)
    remain_precursors = precursors[remain_index_list].to(device)
    remain_tokens = tokens[remain_index_list].to(device)
    remain_label = label[remain_index_list].to(device)
    remain_weight = weight[remain_index_list].to(device)
    remain_psm_id = [psm_id[i] for i in remain_index_list]
    
    tokens_masked, tokens_label_masked = mask_unk_decoy_tokens(mask_tokens,
                                                               mask_label,
                                                               mask_index,
                                                               unk_index,
                                                               token_mask_ratio=token_mask_ratio,
                                                               device=device)
    remain_tokens_label = torch.zeros(remain_tokens.shape[0], remain_tokens.shape[1]).to(device)
    
    # 合并batch
    spectra = torch.cat((mask_spectra, remain_spectra))
    spectra_mask = torch.cat((mask_spectra_mask, remain_spectra_mask))
    precursors = torch.cat((mask_precursors, remain_precursors))
    tokens = torch.cat((tokens_masked, remain_tokens))
    tokens_label = torch.cat((tokens_label_masked, remain_tokens_label))
    label = torch.cat((mask_label, remain_label))
    weight = torch.cat((mask_weight, remain_weight))
    psm_id = mask_psm_id + remain_psm_id
    return spectra, spectra_mask, precursors, tokens, tokens_label, label, weight, psm_id


def mask_spectra_data(spectra, spectra_mask, remain_ratio=0.1, spectra_zero_ratio=0.1, device='cpu'):
    """Args:
            spectra: float Tensor (batch, 300, 2)
            spectra_mask: float Tensor (batch, 300)
            remain_ratio: float=0.1（样本不进行mask的比例）
            spectra_zero_ratio: float=0.1（每个样本，spectrum mask比例）
            device: cpu or cuda
        Returns:
            spectra: float Tensor (batch, 300, 2)
            spectra_mask: float Tensor (batch, 300)
    """
    # 随机删除其中10%的<mz、intensity>对
    spectra_mask_backup = torch.rand(spectra.shape[0], spectra.shape[1]).to(device)
    spectra_mask_backup = torch.where(spectra_mask_backup < spectra_zero_ratio, 0, 1)

    # 10%的spectrum保持不变
    full_indices = list(range(spectra.shape[0]))
    remain_size = int(remain_ratio * len(full_indices))
    mask_size = len(spectra) - remain_size
    remain_spectra_index_list, mask_spectra_index_list = random_split(full_indices, [remain_size, mask_size])

    # remain对应的mask值为1（不变）
    for i in remain_spectra_index_list:
        spectra_mask_backup[i] = 1

    # 根据掩码矩阵，映射spectrum和spectra_mask
    spectra = torch.mul(spectra, torch.unsqueeze(spectra_mask_backup, dim=-1)).to(device)
    spectra_mask = torch.mul(spectra_mask, spectra_mask_backup).to(device)
    return spectra, spectra_mask
