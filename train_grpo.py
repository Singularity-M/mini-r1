import os
import platform
import argparse
import time
import math
import warnings

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer

from model.model import MiniR1
from model.MiniR1Config import MiniR1Config
from model.dataset import PretrainDataset, BatchLoadingPretrainDataset, PretrainDataset_1, GRPODataset
from grpo.reward import REWARD_FUNCS 
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed

warnings.filterwarnings('ignore')


def safe_generate(model, *args, **kwargs):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module.generate(*args, **kwargs)
    else:
        return model.generate(*args, **kwargs)


def repeat_collate_fn(batch, repeat_times=4):
    """
    将每条数据重复 repeat_times 次，用于 GRPO 算法中“一条 Prompt 采样多条 Completion”的场景
    """
    expanded_batch = []
    for item in batch:
        expanded_batch.extend([item] * repeat_times)
    return expanded_batch

def Logger(content):
    """简单的日志记录函数"""
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, base_lr):
    """
    根据当前 step，使用 Cosine 退火策略计算学习率
    """
    return base_lr / 10 + 0.5 * base_lr * (1 + math.cos(math.pi * current_step / total_steps))


def get_per_token_logps(model, input_ids, attention_mask, logits_to_keep):
    """
    计算给定模型在 prompt+completion 上的对数概率（仅保留 completion 部分的 logits）。
    """
    # 这里 logits_to_keep+1 是因为最后一个 logit 对应下一个 token 的预测，将其裁剪掉
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
    ).logits
    # (B, L-1, V)，排除最后一个时间步的 logit
    logits = logits[:, :-1, :]

    # 只保留 completion 对应的 logits
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    log_probs = F.log_softmax(logits, dim=2)  # (B, C, V)
    # 选出实际 token 的对数概率
    token_log_probs = torch.gather(
        log_probs, dim=2, index=input_ids.unsqueeze(2)
    ).squeeze(-1)
    return token_log_probs

def prepare_inputs(
    policy_model, ref_policy_model, prompt_ids, prompt_mask, reward_func_list, raw_batch_prompts, args, tokenizer
):
    """
    1. 使用 policy_model 生成 completion
    2. 计算 ref_policy_model 上的对数概率
    3. 计算各个 Reward 并在各进程间进行 gather
    4. 计算 advantages
    5. 返回训练所需的各项
    """

    # 生成完整的 prompt+completion
    with torch.no_grad():
        # 生成的序列 (B, P+C)
        prompt_completion_ids = safe_generate(policy_model,
            idx=prompt_ids,
            attention_mask=prompt_mask,
            eos=tokenizer.eos_token_id,
            stream=False,
            max_new_tokens=args.max_completion_length,
        )

    # 先分割出 prompt 和 completion
    prompt_length = prompt_ids.size(1)
    real_prompt_ids = prompt_completion_ids[:, :prompt_length]
    completion_ids = prompt_completion_ids[:, prompt_length:]

    # 判断 completion 中是否有 eos，得到每条完成序列的有效长度
    is_eos = completion_ids == tokenizer.eos_token_id  # (B, C)
    eos_idx = torch.full(
        (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device
    )
    # 若某行存在 eos，则取第一个 eos 的位置；否则视为最后一个位置
    has_eos_line = is_eos.any(dim=1)
    eos_idx[has_eos_line] = is_eos.int().argmax(dim=1)[has_eos_line]
    sequence_indices = torch.arange(is_eos.size(1), device=args.device).expand(
        is_eos.size(0), -1
    )
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()  # (B, C)

    # 拼接新的 attention_mask
    new_attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    with torch.no_grad():
    # 计算参考模型在 completion 上的对数概率
        ref_logps  = get_per_token_logps(
            ref_policy_model,
            prompt_completion_ids,
            new_attention_mask,
            logits_to_keep,
        )
    

    # 将生成的 completion 解码为文本，用于计算各种 Reward
    completion_text_list = tokenizer.batch_decode(completion_ids)

    # 计算各条数据在多个 Reward 函数下的分值 (B, num_reward_funcs)
    rewards_per_func = torch.zeros(
        prompt_ids.size(0), len(reward_func_list), device=args.device
    )

    # 注意：下面对每个 reward_func 都先提取 prompt 的其他信息，并调用对应的函数
    for i, reward_func_name in enumerate(reward_func_list):
        # 收集除 "prompt" 和 "completion" 以外的其他信息
        keys = [key for key in raw_batch_prompts[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [ex[key] for ex in raw_batch_prompts] for key in keys}
        # 计算该条数据对应的 Reward
        reward_values = REWARD_FUNCS[reward_func_name](
            prompt=raw_batch_prompts,
            completions=completion_text_list,
            **reward_kwargs
        )
        # 将 Python 列表转换为 tensor
        rewards_per_func[:, i] = torch.tensor(
            reward_values, dtype=torch.float32, device=args.device
        )

    # # 将所有进程上的 Reward gather 到每个进程
    # rewards_per_func = gather(rewards_per_func)
    # 按列累加（所有 reward 加起来）
    rewards_all = rewards_per_func.sum(dim=1)

    # 将 batch_size * num_generations 的数据 reshape 为 (batch_size, num_generations)
    # 注：此处 batch_size 实际上是 "原始 Prompt 数" * "num_generations"
    # 但逻辑上为了每组 num_generations 进行分组求平均与方差
    grouped_rewards = rewards_all.view(-1, args.num_generations)
    mean_grouped_rewards = grouped_rewards.mean(dim=1)
    std_grouped_rewards = grouped_rewards.std(dim=1)

    # 将每组的均值与方差重复回去，以对应到每个样本
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_generations)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_generations)

    # 计算 Advantage
    advantages = (rewards_all - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    # ------ 计算平均 Reward（全局） ------
    # 此时 rewards_all 是已经 gather 后的全体进程数据，直接求平均即可
    avg_reward_global = rewards_all.mean().item()

    # ------ 计算平均生成长度（全局） ------
    # 首先得到每条数据的生成长度，完成再 gather 后求平均
    with torch.no_grad():
        local_lengths = completion_mask.sum(dim=1).float()  # 每条数据的长度
    # gather 到所有进程
    all_lengths = gather(local_lengths)
    avg_length_global = all_lengths.mean().item()

    # 返回需要的所有信息
    return {
        "prompt_ids": real_prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "ref_logps": ref_logps,
        "advantages": advantages,
        "avg_reward_global": avg_reward_global,
        "avg_length_global": avg_length_global,
    }

    
    
def compute_grpo_loss(
    policy_model, ref_policy_model, prompt_ids, prompt_mask, reward_func_list,
    raw_prompts
):
    """
    计算 GRPO 的损失，并返回损失值、全局平均 reward、全局平均生成长度等信息
    """
    # 准备输入数据并拿到各项
    inputs_data = prepare_inputs(
        policy_model, ref_policy_model, 
        prompt_ids, prompt_mask, 
        reward_func_list, raw_prompts, args, tokenizer
    )

    with ctx:
        # 拼接 prompt + completion 用于计算模型在 completion 的 logP
        complete_input_ids = torch.cat(
            [inputs_data["prompt_ids"], inputs_data["completion_ids"]], dim=1
        )
        complete_attention_mask = torch.cat(
            [inputs_data["prompt_mask"], inputs_data["completion_mask"]], dim=1
        )
        logits_to_keep = inputs_data["completion_ids"].size(1)

        # 模型在 prompt+completion 上的对数概率
        per_token_logps = get_per_token_logps(
            policy_model,
            complete_input_ids,
            complete_attention_mask,
            logits_to_keep
        )
        # 参考模型在 completion 上的对数概率
        ref_per_token_logps = inputs_data["ref_logps"]

        # KL散度 (ref - policy)
        # 这里使用 GRPO 论文中的公式：exp(r-p) - (r-p) - 1
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (
            ref_per_token_logps - per_token_logps
        ) - 1.0

        # Advantage
        advantages = inputs_data["advantages"]

        # exp(policy_logp - 停梯度的policy_logp) * advantages 
        # 减去 beta * KL
        policy_ratio = torch.exp(per_token_logps - per_token_logps.detach())
        per_token_loss = policy_ratio * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - args.beta * per_token_kl)

        # 只在 completion_mask (有效 token) 区域做平均
        cmask = inputs_data["completion_mask"]
        loss = ((per_token_loss * cmask).sum(dim=1) / cmask.sum(dim=1)).mean()

    return {
        "loss": loss,
        "avg_reward": inputs_data["avg_reward_global"],
        "avg_length": inputs_data["avg_length_global"],
    }

def build_prompt_inputs(batch_prompts, tokenizer, max_prompt_len=None):
    """
    将输入的 batch_prompts（其中包含 prompt 字段）进行 tokenize 并截断到指定长度
    """
    # 使用内部的 apply_chat_template 添加用户对话等特殊格式
    prompt_texts = [
        tokenizer.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True)
        for ex in batch_prompts
    ]
    tokenized_result = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False
    )
    # 截断
    prompt_ids, prompt_mask = tokenized_result["input_ids"], tokenized_result["attention_mask"]
    if max_prompt_len is not None:
        prompt_ids = prompt_ids[:, -max_prompt_len:]
        prompt_mask = prompt_mask[:, -max_prompt_len:]
    return prompt_ids, prompt_mask

def train_epoch(epoch, writer):
    start_time = time.time()
    for step, batch_prompts in enumerate(train_loader):

        prompt_ids, prompt_mask = build_prompt_inputs(batch_prompts, tokenizer, args.max_prompt_length)
        X = prompt_ids.to(args.device)
        attention_masks = prompt_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        loss_dict = compute_grpo_loss(model, ref_model, X, attention_masks, REWARD_FUNCS, batch_prompts)

        loss = loss_dict["loss"] / args.accumulation_steps


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
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
        
        
        if (step+1) % args.ref_model_sync_steps == 0 and (not ddp or dist.get_rank() == 0):
            ref_model.load_state_dict(model.state_dict())
            ref_model.to(model.device)
            ref_model.eval()
        # 使用 TensorBoard 记录指标（仅在主进程进行）
        if writer is not None and (not args.ddp or dist.get_rank() == 0):
            step_global = epoch * len(train_loader) + step+1
            writer.add_scalar("Train/Loss", loss_dict["loss"].item(), step_global)
            writer.add_scalar("Train/AvgReward", loss_dict["avg_reward"], step_global)
            writer.add_scalar("Train/AvgGenLen", loss_dict["avg_length"], step_global)

        
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/tcm_r1_{lm_config.dim}{moe_path}.pth'

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


    tokenizer = AutoTokenizer.from_pretrained('/home/mth/project_llm/mini_llm/model/minir1_tokenizer')
    
    # 构建训练模型
    model_from = 1  # 1从权重，2用transformers
    
    if model_from == 1 and not is_continue_pretrain:
        model = MiniR1(lm_config)
        moe_path = '_moe' if lm_config.use_moe else ''
        ckp = f'./out/sft_rl_{lm_config.dim}{moe_path}.pth'
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
        ckp = f'./out/r1_{lm_config.dim}{moe_path}.pth'
        state_dict = torch.load(ckp, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(args.device)
    
    ref_model = MiniR1(lm_config)
    ref_model.load_state_dict(model.state_dict())
    ref_model.to(args.device)
    ref_model.eval()

    Logger(f'LLM总参数量：{count_parameters(model) / 1e9:.3f} B')
   
   # 估计模型保存大小（MB）
    model_size_mb = estimate_model_size(model)
    Logger(f'估计模型保存大小：{model_size_mb:.3f} MB')

    return model, ref_model, tokenizer


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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Use_tensorboard")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default="/home/mth/project_llm/mini_llm/data/origin_data/grpo_tcm_train.jsonl", help="Path to training data")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold")
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=100, help="Model saving interval")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')
    parser.add_argument("--num_generations", type=int, default=6,help="Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size) must be divisible by this value.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,help="The beta1 hyperparameter for the [`AdamW`] optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="The beta2 hyperparameter for the [`AdamW`] optimizer.")
    parser.add_argument( "--weight_decay", type=float, default=0.1, help="The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the optimizer.")
    parser.add_argument("--ref_model_sync_steps", type=int, default=100, help="Update the comments of the reference model")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.")
    parser.add_argument("--max_completion_length", type=int, default=512, required=False, help="Maximum length of the generated completion.")
    parser.add_argument("--beta", type=float, default=1e-3, required=False, help="")


    args = parser.parse_args()

    lm_config = MiniR1Config()
    max_seq_len = 512
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
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_tensorboard and (not ddp or ddp_local_rank == 0):
        tensorboard_file = os.path.join(args.save_dir, 'log')
        os.makedirs(tensorboard_file, exist_ok=True)
        writer = SummaryWriter(tensorboard_file)
    else:
        writer = None

    model, ref_model, tokenizer = init_model(is_continue_pretrain=False)
    
    train_ds = GRPODataset(args.data_path, args.num_generations)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        collate_fn=lambda x: repeat_collate_fn(x, repeat_times=args.num_generations),
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 训练过程中管理梯度缩放，以支持混合精度训练。
    scaler = torch.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    optimizer = optim.AdamW(model.parameters(),betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.weight_decay, lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])
        ref_model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
        ref_model = DistributedDataParallel(ref_model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, writer)
    
    if (writer is not None) and (not ddp or dist.get_rank() == 0):
        writer.close()