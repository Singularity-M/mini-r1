import json
import random
import re

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio

from datasets import IterableDataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.samples = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 3

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        
        sample = self.samples[index]

        input_id = sample['tokens']
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)


class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1024):
        super().__init__()
        self.file_path = file_path
        self.max_length = max_length
    
        #
        self.tokenizer = tokenizer
        self.bos_id = tokenizer('<bos>assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<eos>\n', add_special_tokens=False).input_ids

        self.samples = self.load_data()

    
    def load_data(self):
        samples = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def _generate_loss_mask(self, input_ids):
        # 找到句子中需要计算Loss的部分
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到句子开始的地方（从self.bos_id第一个开始）
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        #
        sample = self.samples[index]
        messages = sample['conversations']
        new_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_special_tokens=False
        )
        input_ids = self.tokenizer(new_prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        loss_mask = self._generate_loss_mask(input_ids)

        X_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X_tensor, Y_tensor, loss_mask


class GRPODataset(Dataset):
    def __init__(self, file_path, repeat_times):
        super().__init__()
        self.file_path = file_path
        self.repeat_times = repeat_times
        self.samples = self.load_data()

    
    def load_data(self):
        samples = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        #
        sample = self.samples[index]

        return sample


def get_grpo_dataset(file_path) -> IterableDataset:
    data = []
    with open(file_path, mode="r", encoding="utf-8") as file:
        for f in file:
            d = json.loads(f.strip())
            data.append(d)
    from datasets import Dataset
    data = Dataset.from_list(data)
    return data

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./model/minir1_tokenizer")
    messages = [
        {"role": "system", "content": "好好好"},
        {"role": "user", "content": "你是一个优秀的聊天机器人，总是给我正确的回应！"}]
    new_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )[-(512 - 1):]
    prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )[-(512 - 1):]
    data_path = ""

    train_ds = SFTDataset(data_path, tokenizer, max_length=4096)
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=1
    )
    for X_tensor, Y_tensor, loss_mask_tensor in train_loader:
        pass