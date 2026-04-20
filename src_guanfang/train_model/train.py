import torch
import argparse
import datetime
import logging
import os
import sys

import deepspeed
from deepspeed.accelerator import get_accelerator
import matplotlib.pyplot as plt
import numpy as np
import json
import yaml
import random
import pandas as pd
import polars as pl
import math
from tqdm import tqdm

from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_ as clip_grad_norm
from sklearn.metrics import accuracy_score
from torch.optim import Optimizer

from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
from ..transformer.dataset import mkdir_p
from ..transformer.iterable_dataset_online_parquet import create_iterable_dataset, mask_batch_decoy_1unk_data, \
    mask_spectra_data
from ..transformer.model import MSGPT

import warnings

warnings.filterwarnings("ignore")

# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Command-line argument parsing
def add_argument():
    """Use a fixed parameter configuration to avoid issues with command-line argument parsing"""

    class Args:
        def __init__(self):
            # CUDA
            self.with_cuda = True
            self.node_num = 1  # Modify based on the execution command
            self.gpu_num = 4  # Modify based on the execution command
            self.local_rank = -1

            # training
            self.config = "src_guanfang/aipc_test_mzml.yaml"  # Modify based on the execution command
            self.file_size = 128

            # DeepSpeed
            self.dtype = 'bf16'
            self.stage = 0
            # DeepSpeed config
            self.deepspeed = True
            self.deepspeed_config = "ds_config.json"  # Modify based on the execution command

    args = Args()
    return args


# get parameters
args = add_argument()


# set seeds
def set_seeds(seed):
    logging.info('Unified seeds !!')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # multi GPUs
    np.random.seed(seed)  # Numpy random seed
    random.seed(seed)  # Python random seed

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# get PyTorch optimizer
def get_torch_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer

    if hasattr(optimizer, 'optimizer') and isinstance(optimizer.optimizer, Optimizer):
        return optimizer.optimizer

    raise TypeError('{} is not a subclass of torch.optim.Optimizer'.format(type(optimizer).__name__))


# save checkpoint
# model_engine为DeepSpeed封装后的模型引擎。
# client_state自定义的其他要保存的信息
# 可选是否排除被冻结的参数
def save_checkpoint(model_engine, save_dir, tag, client_state={}, exclude_frozen_parameters=False):
    save_path = model_engine._get_ckpt_name(save_dir, tag)
    mkdir_p(os.path.join(save_dir, str(tag)))

    # 判断是否启用了零冗余优化器
    zero_optimizer_state = model_engine.zero_optimization() or model_engine.bfloat16_enabled()
    save_frozen_param = model_engine.zero_optimization_partition_gradients() and not exclude_frozen_parameters

    # get model states
    # stata_dict：状态字典，存储模型所有的参数
    module = model_engine.module_state_dict(exclude_frozen_parameters=exclude_frozen_parameters)

    # 用字典记录了一堆模型当前信息
    state = dict(
        module=module,
        buffer_names=model_engine._get_buffer_names(),
        optimizer=model_engine.optimizer.state_dict() if model_engine.optimizer and not zero_optimizer_state else None,
        param_shapes=model_engine._get_zero_param_shapes() if model_engine.optimizer and zero_optimizer_state else None,
        frozen_param_shapes=model_engine._get_zero_frozen_param_attributes(model_engine._get_param_shape_func)
        if save_frozen_param else None,
        shared_params=model_engine._get_shared_params() if model_engine.optimizer and zero_optimizer_state else None,
        frozen_param_fragments=model_engine._get_zero_frozen_param_attributes(model_engine._get_param_fragment_func)
        if save_frozen_param else None,
        lr_scheduler=model_engine.lr_scheduler.state_dict() if model_engine.lr_scheduler is not None else None,
        data_sampler=model_engine.training_dataloader.data_sampler.state_dict() if
        (model_engine.training_dataloader is not None and model_engine.curriculum_learning_enabled()) else None,
        random_ltd=model_engine.random_ltd_scheduler.state_dict() if model_engine.random_ltd_enabled() else None,
        sparse_tensor_module_names=model_engine.sparse_tensor_module_names,
        skipped_steps=model_engine.skipped_steps,
        global_steps=model_engine.global_steps,
        global_samples=model_engine.global_samples,
        dp_world_size=model_engine.seq_dp_world_size,
        mp_world_size=model_engine.mp_world_size,
        ds_config=model_engine.config
    )
    # 把client_state合并到state字典里
    state.update(client_state)

    # 存储刚才保存的state，记录存档点
    checkpoint_engine = TorchCheckpointEngine()
    checkpoint_engine.save(state, save_path)


class WarmupScheduler(object):
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_iter: int,
                 max_iter: int,
                 max_lr: int,
                 min_lr: int,
                 second_min_lr: int,  # 第二小学习率。如果学习率小于min_lr，则直接设为second_min_lr
                 warmup_type: str,
                 last_batch_iteration: int = -1):  # 上一个batch的编号

        self.optimizer = get_torch_optimizer(optimizer)

        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.warmup_type = warmup_type
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.second_min_lr = second_min_lr

        self.last_batch_iteration = last_batch_iteration
        # param_groups为[[],[],[]...]，存储[每一层]的参数。
        # 每一层可能有不同学习率，这里记录各层的初始学习率。
        self.org_lrs = [group['lr'] for group in self.optimizer.param_groups]

    # 指数衰减因子，用于乘到学习率前面
    def get_exponential_lr_factor(self) -> float:
        """Get the learning rate factor for the current step (exponential decay)"""
        lr_factor = 1.0

        # 当前轮数较小，还在warmup段。使用线性学习率，逐步增长到1.0
        if self.last_batch_iteration <= self.warmup_iter:
            lr_factor *= self.last_batch_iteration / self.warmup_iter
        # warmup结束后，学习率开始衰减
        else:
            lr_factor = (1 - (self.last_batch_iteration - self.warmup_iter) / (self.max_iter - self.warmup_iter)) ** 0.9
        return lr_factor

    # 另一种学习率变化策略，余弦退火因子
    def get_cosine_lr_factor(self) -> float:
        """Get the learning rate factor for the current step (cosine annealing)"""
        lr_factor = 1.0
        # warmup期间线性增长
        if self.last_batch_iteration <= self.warmup_iter:
            lr_factor *= self.last_batch_iteration / self.warmup_iter
        # 余弦下降
        else:
            lr = self.min_lr + \
                 0.5 * (self.max_lr - self.min_lr) * (
                             1 + np.cos(self.last_batch_iteration / (self.max_iter - self.warmup_iter) * np.pi))
            lr_factor = lr / self.max_lr
        return lr_factor

    def get_lr_ratio(self):
        # 还没开始训练（last_batch_初始值为-1）
        if self.last_batch_iteration < 0:
            logger.warning("Attempting to get learning rate from scheduler before it has started")
            return [0.0]

        if self.warmup_type == 'exp':
            ratio = self.get_exponential_lr_factor()
        elif self.warmup_type == 'cos':
            ratio = self.get_cosine_lr_factor()
        else:
            ratio = 1.0

        if isinstance(ratio, float):
            # 防止异常值
            ratio = min(1.0, ratio)
        else:
            # Set to 0 when the learning rate factor is complex
            ratio = 0.0
        ratio = max(0.0, ratio)
        return ratio

    def step(self, last_batch_iteration=None):
        # 更新本次步数
        if last_batch_iteration is None:
            last_batch_iteration = self.last_batch_iteration + 1
        self.last_batch_iteration = last_batch_iteration

        lrs = self.get_lr()
        # zip(a, b)会进行迭代，每次返回(a[i],b[i])，只需a、b长度相同即可
        # 这里optimizer.param_groups长度是层数，lrs长度也是层数（表示每一层的学习率）
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        if self.last_batch_iteration < 0:
            logger.warning("Attempting to get learning rate from scheduler before it has started")
            return [0.0]
        lr_ratio = self.get_lr_ratio()
        # 下面为列表推导式 [当前位置的值 for x in list]，会遍历list中的每一个元素x，并通过最开始的表达式填入新list的当前位置
        return [org_lr * lr_ratio if float(org_lr * lr_ratio) > self.min_lr else self.second_min_lr for org_lr in
                self.org_lrs]

    def get_last_lr(self):
        """Return the last learning rate computed by the current scheduler"""
        assert getattr(self, '_last_lr', None) is not None, "need to call step() first"
        return self._last_lr

    # 保存检查点的时候使用
    def state_dict(self):
        return {'last_batch_iteration': self.last_batch_iteration}

    def load_state_dict(self, sd):
        self.last_batch_iteration = sd['last_batch_iteration']

    def _format_param(self, optimizer, param_value, param_name):
        if isinstance(param_value, list) or isinstance(param_value, tuple):
            if len(param_value) != len(optimizer.param_groups):
                raise ValueError("expected {} value for {}, got {}".format(len(optimizer.param_groups), param_name,
                                                                           FileNotFoundError(param_value)))
            return list(param_value)
        return [param_value] * len(optimizer.param_groups)


# Loss computation helper function
def calc_loss(mask, weight, label, pred):
    mask_weight = weight[mask]
    mask_label = label[mask]
    mask_pred = pred[mask]

    if len(mask_weight) > 0:
        dda_criterion = nn.BCEWithLogitsLoss(weight=mask_weight)
        dda_loss = dda_criterion(mask_pred, mask_label.flatten())
        return dda_loss.item()
    else:
        return 0


# train function
def train(model_engine, trainloader, sw, optim, scheduler, local_device, target_dtype, local_rank, config, epoch,
          sw_step):
    running_loss, dda_running_loss, mask_running_loss = None, None, None
    target_running_loss, false_target_running_loss, decoy_running_loss = None, None, None

    trainloader.set_epoch(epoch)

    train_bar = tqdm(trainloader, total=len(trainloader))
    for batch_data in train_bar:
        for i, (spectra, spectra_mask, precursors, tokens, label, weight) in enumerate(batch_data):
            # mask data
            spectra, spectra_mask, precursors, tokens, tokens_label, label, weight = mask_batch_decoy_1unk_data(
                (spectra, spectra_mask, precursors, tokens, label, weight),
                token_mask_ratio=0.4,
                device=local_device
            )

            # mask on spectra
            spectra, spectra_mask = mask_spectra_data(spectra, spectra_mask, device=local_device)

            # move data to the device
            spectra = spectra.to(local_device)
            spectra_mask = spectra_mask.to(local_device)
            precursors = precursors.to(local_device)
            tokens = tokens.to(local_device)
            tokens_label = tokens_label.to(local_device).to(torch.long)
            label = label.to(local_device)
            weight = weight.to(local_device)

            # Data type conversion
            if target_dtype != None:
                spectra = spectra.to(target_dtype)
                spectra_mask = spectra_mask.to(target_dtype)
                precursors = precursors.to(target_dtype)
                tokens = tokens.to(target_dtype)
                label = label.to(target_dtype)
                weight = weight.to(target_dtype)

            # forward
            dda_pred, mask_pred = model_engine(spectra, spectra_mask, precursors, tokens)

            # cal DDA loss
            dda_criterion = nn.BCEWithLogitsLoss(weight=weight)
            dda_loss = dda_criterion(dda_pred, label.flatten())

            # Compute loss by type
            target_mask = (label > 0.5)
            target_dda_loss = calc_loss(target_mask, weight, label, dda_pred)

            false_target_mask = (label < 0.5) & (weight < 0.9)
            false_target_dda_loss = calc_loss(false_target_mask, weight, label, dda_pred)

            decoy_mask = (label < 0.5) & (weight > 0.9)
            decoy_dda_loss = calc_loss(decoy_mask, weight, label, dda_pred)

            # cal mask loss
            mask_criterion = nn.CrossEntropyLoss(ignore_index=0)
            mask_loss = mask_criterion(mask_pred, tokens_label)

            # Total loss (DDA loss weight = 0.18)
            loss = mask_loss + 0.18 * dda_loss

            try:
                # backward and update parameters
                model_engine.backward(loss)
                model_engine.step()
                scheduler.step()
            except Exception as e:
                logging.info(f"epoch: {epoch}, rank: {int(os.environ['RANK'])}, error: {e}!!!")
                continue

            # 更新运行损失（平滑处理）
            if running_loss is None:
                running_loss = loss.item()
                dda_running_loss = dda_loss.item()
                mask_running_loss = mask_loss.item()

                target_running_loss = target_dda_loss
                false_target_running_loss = false_target_dda_loss
                decoy_running_loss = decoy_dda_loss
            else:
                running_loss = 0.99 * running_loss + (1 - 0.99) * loss.item()
                dda_running_loss = 0.99 * dda_running_loss + (1 - 0.99) * dda_loss.item()
                mask_running_loss = 0.99 * mask_running_loss + (1 - 0.99) * mask_loss.item()

                target_running_loss = 0.99 * target_running_loss + (1 - 0.99) * target_dda_loss
                false_target_running_loss = 0.99 * false_target_running_loss + (1 - 0.99) * false_target_dda_loss
                decoy_running_loss = 0.99 * decoy_running_loss + (1 - 0.99) * decoy_dda_loss

            # Main process logging and TensorBoard
            if int(os.environ.get('RANK', 0)) == 0:
                sw_step += 1

                if sw_step % 1000 == 0:
                    scheduler_lr = scheduler.get_last_lr()[0]
                    train_bar.set_description(
                        f"[Epoch={epoch}, sw_step={sw_step}, rank={int(os.environ.get('RANK', 0))}]")
                    train_bar.set_postfix(loss=running_loss, lr=scheduler_lr)

                    # 写入TensorBoard
                    sw.add_scalar("train/loss", loss.item(), sw_step)
                    sw.add_scalar("train/loss_smooth", running_loss, sw_step)

                    sw.add_scalar("train/dda_loss", dda_loss.item(), sw_step)
                    sw.add_scalar("train/dda_loss_smooth", dda_running_loss, sw_step)

                    sw.add_scalar("train/target_loss", target_running_loss, sw_step)
                    sw.add_scalar("train/false_target_loss", false_target_running_loss, sw_step)
                    sw.add_scalar("train/decoy_loss", decoy_running_loss, sw_step)

                    sw.add_scalar("train/mask_loss", mask_loss.item(), sw_step)
                    sw.add_scalar("train/mask_loss_smooth", mask_running_loss, sw_step)

                    sw.add_scalar("optim/scheduler_lr", scheduler_lr, sw_step)
                    sw.add_scalar("optim/epoch", epoch, sw_step)
                    sw.flush()

            # save checkpoint
            if (sw_step > 0) and (sw_step % config["ckpt_interval"] == 0):
                logging.info('[%d, %d, %d] save model: %s ' % (
                epoch, int(os.environ.get('RANK', 0)), sw_step, config["model_save_path"]))
                save_checkpoint(model_engine, save_dir=config["model_save_path"], tag=epoch,
                                client_state={'epoch': epoch})

    sw.add_scalar("train/train_loss", running_loss, epoch)
    return sw_step


def main():
    # deepspeed初始化分布式训练环境
    deepspeed.init_distributed(timeout=datetime.timedelta(seconds=5400))

    # 从环境中读取当前是几号GPU
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # 加载配置
    config_path = args.config
    with open(config_path) as f_in:
        config = yaml.safe_load(f_in)

    # Synchronize across processes
    # 如果当前进程不是主进程（编号0），就停下来等待0号进程。barrier为同步点
    if torch.distributed.get_rank() != 0:
        torch.distributed.barrier()

    # 词表，用于将str类型的token转为整数ID。residue 残基
    vocab = ['<pad>', '<mask>'] + list(config["residues"].keys()) + ['<unk>']
    config["vocab"] = vocab
    config["node_num"] = args.node_num
    config["gpu_num"] = args.gpu_num
    # enumerate(list)返回 下标i，原值v
    s2i = {v: i for i, v in enumerate(vocab)}
    logging.info(f"Vocab: {s2i}")

    # set seeds
    set_seeds(config['seed'])

    # mkdir save dirs
    mkdir_p(config["tb_summarywriter"])
    mkdir_p(config["model_save_path"])

    # set TensorBoard logging dir
    config["tb_summarywriter"] = config["tb_summarywriter"] + datetime.datetime.now().strftime("MSGPT_%y_%m_%d_%H_%M")
    sw = SummaryWriter(config["tb_summarywriter"])

    # Synchronize across processes
    if torch.distributed.get_rank() == 0:
        torch.distributed.barrier()

    # create datasets
    multi_node = args.node_num > 1
    train_dl = create_iterable_dataset(logging, config, s2i, parse='train', multi_node=multi_node, need_weight=True)

    ########################################################################
    # define model
    model = MSGPT(
        dim_model=config["dim_model"],  # 每个token被映射到了dim_model维度
        n_head=config["n_head"],
        dim_feedforward=config["dim_feedforward"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        vocab_size=len(vocab),
        max_charge=config["max_charge"],
    )

    # 如果模型已经进行过训练，则加载ckpt接着训练
    if (config['init_model_path'] is not None) and config['init_model_path'] != '':
        logging.info(f"Loading model checkpoint from '{config['init_model_path']}'")
        model = MSGPT.load_pt(config['init_model_path'], config)
        model = model.to(torch.bfloat16)

    # 找可训练参数（.requires_grad），被冻结的参数不算入
    # filter(函数func，可迭代对象iter)，从iter依次取值放入func，如果返回true则iter[i]会保留到结果list
    # lambda p: 返回值。是个函数
    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # Initialize the optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    # Update configuration parameters
    one_epoch_iters = math.ceil(max(len(train_dl), 1) * int(args.file_size) / int(config['train_batch_size']))
    if args.node_num > 1:  # 多机
        max_iters = config["epochs"] * one_epoch_iters * args.node_num
    else:  # 单机
        max_iters = config["epochs"] * one_epoch_iters * 2

    # Use an exponential learning rate schedule for the first 20 epochs, then switch to a minimum learning rate schedule for the next 20 epochs
    max_iters = max_iters // 2

    warmup_iters = max(int(config["warmup_ratio"] * max_iters), 1)
    config["train_step_scale"] = max(int(one_epoch_iters * float(config["train_step_ratio"])), 1)
    config["ckpt_interval"] = int(one_epoch_iters * 0.8)
    logging.info(f"Updates max_iters of per epoch is : {max_iters:,},"
                 f" train_step_scale={config['train_step_scale']}, "
                 f" warmup_iters={warmup_iters}, "
                 f" ckpt interval={config['ckpt_interval']}")

    # Initialize the learning rate scheduler
    scheduler = WarmupScheduler(
        optim,
        warmup_iters,
        max_iters,
        float(config['learning_rate']),
        float(config['min_lr']),
        float(config['second_min_lr']),
        config['warmup_strategy']
    )

    # initialize DeepSpeed
    print('args: ', args)
    logging.info(f"rank : {int(os.environ.get('RANK', 0))}, local_rank: {local_rank}")
    model_engine, optim, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters,
        optimizer=optim,
        lr_scheduler=scheduler
    )

    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank

    # Set the target data type
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half
    logging.info(f"target_dtype: {target_dtype}")

    ########################################################################
    # training network
    sw_step = 0
    for epoch in range(config['epochs']):  # multi rounds
        try:
            sw_step = train(
                model_engine,
                train_dl,
                sw,
                optim,
                scheduler,
                local_device,
                target_dtype,
                local_rank,
                config,
                epoch,
                sw_step
            )
        except Exception as e:
            logging.info(f"epoch: {epoch}, rank: {int(os.environ.get('RANK', 0))}, error: {e}!!!")
            continue

    logging.info('Finished Training')
    if int(os.environ.get('RANK', 0)) == 0:
        logging.info(f"Saving final model to {config['model_save_path']}")
        save_checkpoint(
            model_engine,
            save_dir=config["model_save_path"],
            tag="final",
            client_state={'epoch': config['epochs']}
        )


# run
if __name__ == '__main__':
    main()

# deepspeed --num_gpus 4 --module src_guanfang.train_model.train
