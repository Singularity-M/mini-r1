import torch
import math
from torch import nn
from typing import Optional, Dict, List
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from .model import MiniR1
from .MiniR1Config import MiniR1Config


def find_all_linear_names(model):
    cls = torch.nn.Linear
    linear_layers = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            linear_layers.add(name.split('.')[-1])  # 取最后一层名称
    return list(linear_layers)



# 自定义LoRA层
class LoRALayer(nn.Module):
    """
    标准LoRA层实现
    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        rank: LoRA的秩（内在维度）
        alpha: 缩放系数（通常设置为rank的倍数）
        dropout: Dropout概率
    """
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        # 初始化参数
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha  # 缩放系数
        
        # LoRA参数A（输入侧的低秩矩阵）
        self.lora_A = nn.Parameter(torch.empty((rank, in_features)))
        # LoRA参数B（输出侧的低秩矩阵）
        self.lora_B = nn.Parameter(torch.zeros((out_features, rank)))  
        self.dropout = nn.Dropout(dropout)
        
        # 使用Kaiming初始化参数A
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播计算
        Args:
            x: 输入张量，形状为(batch_size, seq_len, in_features)
        Returns:
            LoRA调整后的输出张量
        """
        x = self.dropout(x)
        # 计算LoRA调整量：(x @ A.T) @ B.T * scaling
        lora_adaptation = (x @ self.lora_A.T) @ self.lora_B.T
        return self.scaling * lora_adaptation

# LoRA包装的线性层
class LoRALinear(nn.Module):
    """
    包装原始线性层并添加LoRA适配器
    Args:
        original_layer: 原始线性层
        rank: LoRA秩
        alpha: 缩放系数
        dropout: Dropout概率
    """
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.linear = original_layer
        self.lora = LoRALayer(
            in_features=original_layer.in_features,
            out_features=original_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

        # 冻结原始权重
        self.linear.weight.requires_grad = False
        # 启用LoRA参数梯度
        self.lora.lora_A.requires_grad = True
        self.lora.lora_B.requires_grad = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：原始输出 + LoRA调整量
        """
        return self.linear(x) + self.lora(x)

# 主要的模型包装类
class ModelWithLoRA(nn.Module):
    """
    LoRA包装的Transformer模型
    Args:
        base_model: 基础Transformer模型
        target_modules: 需要应用LoRA的模块列表（例如['q_proj', 'v_proj']）
        lora_rank: LoRA秩
        lora_alpha: 缩放系数
        lora_dropout: Dropout概率
    """
    def __init__(
        self,
        base_model: MiniR1,
        target_modules: List[str] = ['down_proj_kv', 'up_proj_k', 'up_proj_v', 'down_proj_q', 'up_proj_q','proj_qr', 'proj_kr','wo', 'w1', 'w2', 'w3'],
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0
    ):
        super().__init__()
        self.base_model = base_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
       
            
        # 替换目标模块为LoRA线性层
        self._replace_layers(target_modules)
        self.lora_params = []
        for name, _ in self.base_model.named_parameters():
            if 'lora_A' in name or 'lora_B' in name:
                self.lora_params.append(name)
        # 冻结非 LoRA 参数
        for name, param in self.base_model.named_parameters():
            param.requires_grad = name in self.lora_params
        
        # 打印可训练参数信息
        self.print_trainable_parameters()

        self.OUT = CausalLMOutputWithPast()

    def _replace_layers(self, target_modules):
        """
        使用 for 循环遍历模型结构并替换目标模块
        """
        # 使用栈来模拟递归
        stack = [(name, module) for name, module in self.base_model.named_children()]
        
        while stack:
            name, module = stack.pop()
            
            # 如果当前模块是需要替换的目标模块
            if name in target_modules and isinstance(module, nn.Linear):
                # 用 LoRALinear 层替换原始线性层
                new_layer = LoRALinear(
                    module,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout
                )
                setattr(self.base_model, name, new_layer)
            
            # 将子模块加入栈中，继续遍历
            for child_name, child_module in module.named_children():
                stack.append((child_name, child_module))

    def print_trainable_parameters(self):
        """
        打印可训练参数信息
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"可训练参数: {trainable_params} || 总参数: {all_param} || 训练比例: {100 * trainable_params / all_param:.2f}%"
        )

    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, use_kv_cache=False, **keyargs):
        """
        前向传播函数，兼容Hugging Face格式
        """
        # 调用基础模型的前向传播
        outputs = self.base_model(tokens, targets, **keyargs)

        logits = outputs.logits
        loss  = None
        if targets is not None:
            loss  = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=3)

        # return self.OUT  # 返回输出对象
        return CausalLMOutputWithPast(
        loss=loss,          # 确保参数名是 loss
        logits=logits)
        
    @torch.inference_mode()  # 推理模式
    def generate(self, idx, eos, max_new_tokens=1024, temperature=0.7, top_k=None, repetition_penalty=1., attention_mask: Optional[torch.Tensor] = None, use_kv_cache=False, stream=False, **keyargs):
        return self.base_model.generate(idx, eos, max_new_tokens, temperature, top_k, repetition_penalty, attention_mask, use_kv_cache, stream, **keyargs)

    def merge_lora_weights(self):
        """
        合并LoRA权重到原始线性层（用于推理加速）
        """
        for name, module in self.base_model.named_modules():
            if isinstance(module, LoRALinear):
                # 计算合并后的权重
                merged_weight = module.linear.weight + module.lora.scaling * (module.lora.lora_B @ module.lora.lora_A)
                # 替换原始权重
                module.linear.weight.data = merged_weight
                # 删除LoRA参数
                del module.lora

    def save_lora_weights(self, save_path: str):
        """
        保存LoRA权重（仅保存可训练参数）
        """
        lora_state_dict = {k: v for k, v in self.state_dict().items() if 'lora' in k}
        torch.save(lora_state_dict, save_path)
        print(f"LoRA参数已保存至: {save_path}")

    def load_lora_weights(self, load_path: str):
        """
        加载LoRA权重
        """
        lora_state_dict = torch.load(load_path, map_location='cpu')
        self.base_model.load_state_dict(lora_state_dict, strict=False)
        print(f"已从 {load_path} 加载LoRA参数")


if __name__ == "__main__":
    lm_config = MiniR1Config()
    model = MiniR1(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/tcm_r1_{lm_config.dim}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location="cuda:0")
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model = model.to("cuda:0")
    target_modules = find_all_linear_names(model)
    model = ModelWithLoRA(model)
    model.print_trainable_parameters()
    model.train()
    model.print_trainable_parameters()
    model.eval()
    model.print_trainable_parameters()
