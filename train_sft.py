import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer

from model.model import MiniR1
from model.MiniR1Config import MiniR1Config
from model.dataset import SFTDataset

warnings.filterwarnings('ignore')


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def train_epoch(epoch, writer):
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()


        if (step + 1) % args.accumulation_steps == 0: # 累计梯度下降，用时间换内存
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0 and (not ddp or dist.get_rank() == 0):
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

        if (writer is not None) and (not ddp or dist.get_rank() == 0):
            writer.add_scalar('Training Loss',
                    loss.item() * args.accumulation_steps,
                    global_step=epoch * len(train_loader) + step+1)

        
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/sft_tcm_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train()


def init_model(is_continue_pretrain=False):

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def estimate_model_size(model):
        # 计算模型参数所占的内存大小（以字节为单位）
        num_params = count_parameters(model)
        # 假设每个参数是32位浮点数，即4字节
        size_per_param = 4
        # 模型大小（未压缩）
        model_size = num_params * size_per_param
        # 转换为兆字节（MB）
        model_size_mb = model_size / (1024 ** 2)
        return model_size_mb
    
    model_from = 1  # 1从权重，2用transformers
    
    if model_from == 1 and not is_continue_pretrain:
        model = MiniR1(lm_config)
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(args.device)
    
    elif model_from == 1 and is_continue_pretrain:
        model = MiniR1(lm_config)
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/sft_long_{lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(args.device)


    tokenizer = AutoTokenizer.from_pretrained('./model/minir1_tokenizer')

    Logger(f'LLM总参数量：{count_parameters(model) / 1e9:.3f} B')
   
   # 估计模型保存大小（MB）
    model_size_mb = estimate_model_size(model)
    Logger(f'估计模型保存大小：{model_size_mb:.3f} MB')

    return model, tokenizer


def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)


# torchrun --nproc_per_node 2 1-pretrain.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniLLM Pretraining")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use_tensorboard")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default="/home/mth/project_llm/mini_llm/data/origin_data/ChatMed_lora.jsonl", help="Path to training data") # 根据不同的数据确定是单论对话还是多轮对话
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
    parser.add_argument('--ntk', type=int, default=8, help='')

    args = parser.parse_args()

    lm_config = MiniR1Config()
    max_seq_len = 1500
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 根据设备类型（CPU 或 GPU）来选择适当的上下文管理器（ctx），用于控制混合精度训练中的计算精度。
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type)

    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))

    # ddp_local_rank, DEVICE = 1, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_tensorboard and (not ddp or dist.get_rank() == 0):
        tensorboard_file = os.path.join(args.save_dir, 'log')
        os.makedirs(tensorboard_file, exist_ok=True)
        writer = SummaryWriter(tensorboard_file)
    else:
        writer = None

    model, tokenizer = init_model(is_continue_pretrain=True)
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 训练过程中管理梯度缩放，以支持混合精度训练。
    scaler = torch.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, writer)
    
    if (writer is not None) and (not ddp or dist.get_rank() == 0):
        writer.close()