import math
import struct
import inspect
import torch

import torch.nn.functional as F

from .MiniR1Config import MiniR1Config
from torch import nn
from transformers import PreTrainedModel
from typing import Any, Optional, Tuple
from transformers.modeling_outputs import CausalLMOutputWithPast


# 使用RMSNorm类，实现归一化
class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 8e-5):
    super(RMSNorm, self).__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
  
  def _norm(self, x):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + self.eps)
  
  def forward(self, x):
    return self.weight * self._norm(x.float()).type_as(x)


# 定义 precompute_pos_cis 函数，用于预计算位置编码的复数形式
def precompute_pos_cis(dim: int, end: int = int(8 * 1024), theta: float = 1e5):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # 计算频率
    t = torch.arange(end, device=freqs.device)  # 生成时间序列
    freqs = torch.outer(t, freqs).float()  # 计算外积
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # 计算复数形式的位置编码（使用极化编码的方式）
    return pos_cis

# 定义 apply_rotary_emb 函数，用于应用旋转位置编码
def apply_rotary_emb(xq, xk, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # 将 xq 转换为复数形式
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # 将 xk 转换为复数形式
    pos_cis = unite_shape(pos_cis, xq_)  # 调整 pos_cis 的形状
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)  # 应用旋转位置编码
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)  # 应用旋转位置编码
    return xq_out.type_as(xq), xk_out.type_as(xk)  # 返回结果


def apply_rotaryemb(x, pos_cis):
    def unite_shape(pos_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))  # 将 xq 转换为复数形式
   
    pos_cis = unite_shape(pos_cis, x_)  # 调整 pos_cis 的形状
    x_out = torch.view_as_real(x_ * pos_cis).flatten(3)  # 应用旋转位置编码
    
    return x_out.type_as(x)


# 多头潜在注意力机制
class MLA(nn.Module):
    def __init__(self, args: MiniR1Config):
        super(MLA, self).__init__()

        self.dim = args.dim
        self.down_dim = args.down_dim
        self.up_dim = args.up_dim
        self.num_heads = args.n_heads
        self.rope_head_dim = args.rope_head_dim
        self.v_head_dim = args.up_dim // args.n_heads
        self.dropout = args.dropout
        # 初始化kv联合以及q对应的dow,up projection
        self.down_proj_kv = nn.Linear(self.dim, self.down_dim) # W^{DKV}
        self.up_proj_k = nn.Linear(self.down_dim, self.up_dim)# W^{UK}
        self.up_proj_v = nn.Linear(self.down_dim, self.up_dim) # W^{UV}
        self.down_proj_q = nn.Linear(self.dim, self.down_dim) #W^{DQ}
        self.up_proj_q = nn.Linear(self.down_dim, self.up_dim) # W^{UQ}  
        
        # 初始化解耦的q,k进行MQA计算的映射矩阵
        self.proj_qr = nn.Linear(self.down_dim, self.rope_head_dim * self.num_heads)
        # 对于k来说，位置编码向量是共享的
        self.proj_kr = nn.Linear(self.dim, self.rope_head_dim*1)
          
        #最终输出层
        self.attn_dropout = nn.Dropout(self.dropout)
        self.wo = nn.Linear(self.num_heads * self.v_head_dim, self.dim)
        self.res_dropout = nn.Dropout(self.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn  # 判断是否使用 Flash Attention同时判断是否支持
        if not self.flash:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))  # 初始化掩码
            mask = torch.triu(mask, diagonal=1)  # 生成上三角掩码
            self.register_buffer("mask", mask)  # 注册掩码

        self.kv_cache, self.k_pe_cache = None, None  # 初始化浅向量缓存
    
    def forward(self, x: torch.tensor, pos_cis: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, use_kv_cache=False):
        bs, seqlen, _ = x.shape
        

        if use_kv_cache and self.eval():
            if self.kv_cache is None or self.kv_cache.shape[1] != x.shape[1] - 1:
                c_t_kv = self.down_proj_kv(x)
                # c_t_q = self.down_proj_q(x)
                k_pe = self.proj_kr(x)
                # q_t_c = self.up_proj_q(c_t_q)
            else:
                token = x[:, -1:, :]  # 获取最后一个 token
                c_t_kv = torch.cat((self.kv_cache, self.down_proj_kv(token)), dim=1)  # 更新 K
                k_pe = torch.cat((self.k_pe_cache, self.proj_kr(token)), dim=1)  # 更新 K
            
            k_t_c = self.up_proj_k(c_t_kv)
            v_t_c = self.up_proj_v(c_t_kv)
            self.kv_cache = c_t_kv
            self.k_pe_cache = k_pe# 更新 KV 缓存
        else:
            # setp1 :低秩转换
            c_t_kv = self.down_proj_kv(x)
            k_t_c = self.up_proj_k(c_t_kv)
            v_t_c = self.up_proj_v(c_t_kv)
            k_pe = self.proj_kr(x)
           

        
        c_t_q = self.down_proj_q(x)
        q_t_c = self.up_proj_q(c_t_q)
        q_pe = self.proj_qr(c_t_q)

        
        #step2:解耦的q,k进行MQA计算，同时引入ROPE

        #q_t_r,k_t_r施加rope时均扩展了n_h_r维度->[bs,n_h_r,seq_len,rope_head_dim]
        q_pe = q_pe.view(bs, seqlen, self.num_heads, self.rope_head_dim)
        k_pe = k_pe.view(bs, seqlen, -1, self.rope_head_dim)
        
        q_t_r = apply_rotaryemb(q_pe, pos_cis).transpose(1, 2)
        k_t_r = apply_rotaryemb(k_pe, pos_cis).transpose(1, 2)
        
        #step3:拼接step1，step2得到的q,k,进行sdpa计算
        #q_t_c扩展出num_heads为4维，以便于和q_t_r拼接
        q_t_c = q_t_c.reshape(bs, seqlen, self.num_heads, -1).transpose(1, 2)
        #head_dim,rope_head_dim拼接
        xq = torch.cat([q_t_c, q_t_r], dim=-1)
        #k_t_c扩展出num_heads为4维，以便于和k_t_r拼接
        k_t_c = k_t_c.reshape(bs, seqlen, self.num_heads, -1).transpose(1, 2)
        #k_t_r为MQA,n_h_k_r=1,为了和q_t_r计算，需要在n_h_k_r维度复制
        #k_t_r:[bs,n_h_r_k,seq_len,rope_head_dim]->[bs,num_heads,seq_len,rope_head_dim]
        k_t_r=k_t_r.repeat(1,self.num_heads,1,1)
        #head_dim,rope_head_dim拼接
        xk = torch.cat([k_t_c, k_t_r], dim=-1) 

        xv = v_t_c.reshape(bs, seqlen, self.num_heads, self.v_head_dim).transpose(1, 2)

        if attention_mask is not None:
        # 将形状调整为 (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask[:, None, None, :]
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=attention_mask.bool() if attention_mask is not None else None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)  # 使用 Flash Attention
        else:
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)  # 计算注意力分数
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]  # 应用掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # 计算 softmax
            scores = self.attn_dropout(scores)  # 应用注意力 dropout
            output = torch.matmul(scores, xv)  # 计算输出

        #压缩num_head,送入最终统一映射层
        output = output.transpose(1, 2).reshape(bs, seqlen, -1)
        
        output = self.wo(output)
        output = self.res_dropout(output)
        return output

# 前馈神经网络
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super(FeedForward, self).__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim  # 设置隐藏层维度
            hidden_dim = int(2 * hidden_dim / 3)  # 调整隐藏层维度
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # 调整隐藏层维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # 初始化第一层线性变换
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # 初始化第二层线性变换
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # 初始化第三层线性变换
        self.dropout = nn.Dropout(dropout)  # 初始化 dropout

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))  # 前向传播

# 定义 MoEGate 类，实现专家混合（MoE）的门控机制
class MoEGate(nn.Module):
    def __init__(self, config: MiniR1Config) -> None:
        super(MoEGate, self).__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok #设置每个token的专家数量
        self.n_routed_experts = config.n_routed_experts #设置路由专家的数量
        
        self.scoring_func = config.scoring_func #设置评分函数，默认为softmax
         # 无辅助损失更新的参数
        self.b = torch.nn.Parameter(torch.zeros(self.n_routed_experts), requires_grad=False)
        self.u = config.u  # 更新自学习率

        self.norm_topk_prob = config.norm_topk_prob  # 设置是否归一化 top-k 概率
        self.gating_dim = config.dim  # 设置门控维度
        self.u = config.u
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 初始化权重参数
        self.reset_parameters()  # 重置参数

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 使用 Kaiming 初始化权重
    
    def forward(self, hidden_states):
        b, seq_len, h = hidden_states.shape

        hidden_states = hidden_states.view(-1, h) # 调整隐藏状态的形状

        logits = F.linear(hidden_states, self.weight, None)  # 计算 logits

        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 计算 softmax 评分
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')
        
        # topk_weight：这是每个token选择的top-k专家的得分，topk_idx专家的id
        
        if self.training:
            # 训练模式下，添加 self.b 的调整项
            b_factor = self.b.view(1, -1).repeat(b*seq_len, 1)  # 调整形状为 (b, n_routed_experts)
            adjusted_scores = scores + b_factor
            topk_weight, topk_idx = torch.topk(adjusted_scores, k=self.top_k, dim=-1, sorted=False)  # 选择 top-k 专家

            # 计算辅助损失相关的统计信息
            topk_idx_for_aux = topk_idx.view(-1)  # 展平所有 token 的专家索引
            counts_per_expert = torch.bincount(topk_idx_for_aux, minlength=self.n_routed_experts)  # 统计每个专家被选中的次数
            mean_frequency = counts_per_expert.float().mean()  # 计算专家选择次数的均值
            
            # 更新 self.b，实现负载均衡
            difference = (counts_per_expert.float() - mean_frequency).unsqueeze(0)  # (n_routed_experts,)
            self.b.data += self.u * torch.sign(difference).squeeze(0)  # 使用 torch.sign 并更新 self.b
        else:
            topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # 选择 top-k 专家
        
        
        return topk_idx, topk_weight  # 返回 top-k 专家索引、权重



# 定义 MOEFeedForward 类，实现专家混合（MoE）的前馈神经网络
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniR1Config):
        super(MOEFeedForward, self).__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )
            for _ in range(config.n_routed_experts)
        ])  # 初始化专家列表

        self.gate = MoEGate(config)  # 初始化门控机制
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(
                dim=config.dim,
                hidden_dim=config.hidden_dim,
                multiple_of=config.multiple_of,
                dropout=config.dropout,
            )  # 初始化共享专家
    
    def forward(self, x):
        identity = x
        orig_shape = x.shape
        b, seq_len, _ = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight = self.gate(x)

        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)

        if self.training:
            # 训练模式下，重复输入数据
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                # y[flat_topk_idx == i] = expert(x[flat_topk_idx == i])
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理模式下，只选择最优专家
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)

        return y
    
    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 例如当tokens_per_expert=[6, 15, 20, 26, 33, 38, 46, 52]
        # 当token_idxs=[3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...]
        # 意味着当token_idxs[:6] -> [3,  7, 19, 21, 24, 25,  4]位置的token都由专家0处理，token_idxs[6:15]位置的token都由专家1处理......
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 使用 scatter_add_ 进行 sum 操作
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


# 定义 TransformerBlock 类，实现 Transformer 的一个块，包括自注意力和前馈神经网络
class MiniR1Block(nn.Module):
    def __init__(self, layer_id: int, args: MiniR1Config):
        super(MiniR1Block, self).__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = MLA(args)  # 初始化自注意力机制

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 初始化注意力归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)  # 初始化前馈神经网络归一化

        if args.use_moe:
            self.feed_forward = MOEFeedForward(args)  # 初始化专家混合前馈神经网络
        else:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )  # 初始化前馈神经网络

    def forward(self, x, pos_cis, attention_mask: Optional[torch.Tensor] = None, use_kv_cache=False):
        h = x + self.attention(self.attention_norm(x), pos_cis, attention_mask, use_kv_cache)  # 计算自注意力
        out = h + self.feed_forward(self.ffn_norm(h))  # 计算前馈神经网络
        return out  # 返回输出


class MiniR1(PreTrainedModel):
    config_class = MiniR1Config
    # last_loss: Optional[torch.Tensor]

    def __init__(self, params: MiniR1Config=None):
        # if not config:
        #     config = MiniR1Config()
        self.params = params or MiniR1Config()
        super().__init__(self.params)
        
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)  # 初始化词嵌入层
        self.dropout = nn.Dropout(params.dropout)  # 初始化 dropout 层
        self.layers = torch.nn.ModuleList()  # 初始化 Transformer 块列表
        for layer_id in range(self.n_layers):
            self.layers.append(MiniR1Block(layer_id, params))  # 添加 Transformer 块
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)  # 初始化归一化层
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)  # 初始化输出层
        self.tok_embeddings.weight = self.output.weight  # 共享词嵌入和输出层的权重
        pos_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, theta=1e5 * params.ntk)  # 预计算位置编码
        self.register_buffer("pos_cis", pos_cis, persistent=False)  # 注册位置编码缓冲区


        self.apply(self._init_weights)  # 初始化模型权重

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))  # 对特定权重进行初始化

        self.last_loss = None  # 初始化最后一个损失
        self.OUT = CausalLMOutputWithPast()  # 初始化输出对象

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化线性层的权重
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)  # 初始化线性层的偏置
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  # 初始化嵌入层的权重

    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, use_kv_cache=False, **keyargs):
        if 'input_ids' in keyargs:
            tokens = keyargs['input_ids']  # 如果传入了 input_ids，则使用 input_ids
        if 'logits_to_keep' in keyargs:
            logits_to_keep = keyargs['logits_to_keep']
        else:
            logits_to_keep = None

        b, seqlen = tokens.shape  # 获取批量大小和序列长度
        h = self.tok_embeddings(tokens)  # 获取词嵌入
        h = self.dropout(h)  # 应用 dropout
        pos_cis = self.pos_cis[:seqlen]
        for idx, layer in enumerate(self.layers):
            h = layer(h, pos_cis, attention_mask, use_kv_cache)  # 逐层应用 Transformer 块

        h = self.norm(h)  # 应用归一化

        if targets is not None:
            logits = self.output(h)  # 计算 logits
            if logits_to_keep is not None:
                logits = logits[:, -logits_to_keep:, :]
        else:
            if logits_to_keep is not None:
                logits = self.output(h)[:, -logits_to_keep:, :]
            else:
                logits = self.output(h[:, [-1], :])  # 计算最后一个 token 的 logits

        self.OUT.__setitem__('logits', logits)  # 设置输出对象的 logits

        return self.OUT  # 返回输出对象

    @torch.inference_mode()  # 推理模式
    def generate(self, idx, eos, max_new_tokens=1024, temperature=0.7, top_k=None, repetition_penalty=1., attention_mask: Optional[torch.Tensor] = None, use_kv_cache=False, stream=False, **keyargs):
        if stream:
            return self.generate_stream(idx, eos, max_new_tokens, temperature, top_k, repetition_penalty, attention_mask, use_kv_cache, **keyargs)
        else:
            batch_size, _ = idx.shape
            generated_tokens = [[] for _ in range(batch_size)]  # 记录每个batch生成的token
            ended = torch.zeros(batch_size, dtype=torch.bool, device=idx.device)  # 记录每个序列是否终止
            original_tonkens_len = idx.shape[1]
            original_idx = idx
            while idx.shape[1] < max_new_tokens + original_tonkens_len:
                inference_res = self(idx, attention_mask=attention_mask, use_kv_cache=use_kv_cache)
                
                if attention_mask is not None:
                    #更新mask
                    additional_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat((attention_mask, additional_mask), dim=1)
                logits = inference_res.logits[:, -1, :]

                # 逐batch重复惩罚
                for i in range(batch_size):
                    for token in set(idx[i].tolist()):
                        logits[i, token] /= repetition_penalty

                # 采样逻辑
                if temperature == 0.0:
                    _, idx_next = torch.topk(logits, k=1, dim=-1)
                else:
                    logits = logits / temperature
                    if top_k is not None:
                        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits[logits < v[:, [-1]]] = -float('Inf')

                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)

                # 记录生成结果
                for i in range(batch_size):
                    if not ended[i]:  # 还没终止的才继续记录
                        generated_tokens[i].append(idx_next[i])

                # 判断终止
                eos_mask = idx_next.squeeze(-1) == eos
                ended |= eos_mask

                # 所有都终止了就提前结束
                if ended.all():
                    break

                # 拼接到原序列
                idx = torch.cat((idx, idx_next), dim=1)

            # 整理生成结果
            max_len = max([len(g) for g in generated_tokens])
            padded_gen_tokens = torch.full((batch_size, max_len), eos, device=idx.device)
            for i, tokens in enumerate(generated_tokens):
                padded_gen_tokens[i, :len(tokens)] = torch.cat(tokens)

            prompt_completion_ids = torch.cat((original_idx, padded_gen_tokens), dim=1)
            return prompt_completion_ids


    def generate_stream(self, idx, eos, max_new_tokens=1024, temperature=0.7, top_k=None, repetition_penalty=1., attention_mask: Optional[torch.Tensor] = None,  use_kv_cache=False, **keyargs):
        
        batch_size, _ = idx.shape
        _, start_id = idx.shape
        ended = torch.zeros(batch_size, dtype=torch.bool, device=idx.device)  # 记录每个序列是否终止
        original_tonkens_len = idx.shape[1]
        while idx.shape[1] < max_new_tokens + original_tonkens_len:
            inference_res = self(idx, attention_mask=attention_mask, use_kv_cache=use_kv_cache)
            if attention_mask is not None:
                    #更新mask
                    additional_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                    attention_mask = torch.cat((attention_mask, additional_mask), dim=1)
            logits = inference_res.logits[:, -1, :]

            # 逐batch重复惩罚
            for i in range(batch_size):
                for token in set(idx[i].tolist()):
                    logits[i, token] /= repetition_penalty

            # 采样逻辑
            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            # 判断终止
            eos_mask = idx_next.squeeze(-1) == eos
            ended |= eos_mask

            # 拼接到原序列
            idx = torch.cat((idx, idx_next), dim=1)

            # 所有都终止了就提前结束
            if ended.all():
                break

            yield idx[:, start_id:]  # 每次返回新生成的部分



    @torch.inference_mode()  # 推理模式
    def eval_answer(self, idx):
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]  # 截取序列
        inference_res = self(idx_cond)  # 进行前向传播
        logits = inference_res.logits  # 获取 logits
        logits = logits[:, -1, :]  # 获取最后一个 token 的 logits
        return logits  # 返回 logits

if __name__ == "__main__":

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

    # bs, seq_len, d_model = 4, 10, 512
    # arg = MiniR1Config()
    # model = MoEGate(arg)

    # # 输入数据
    # batch_size = 2
    # seq_len = 3
    # hidden_dim = arg.dim

    # 随机生成隐藏状态
    # hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

    # # 前向传播
    # topk_idx, topk_weight = model(hidden_states)

    # # 打印结果
    # print("Top-k 专家索引:", topk_idx)
    # print("Top-k 专家权重:", topk_weight)

    # # 检查负载均衡更新
    # print("\n初始 self.b:", model.b)
    # model(hidden_states)  # 第二次调用，触发负载均衡更新
    # print("更新后的 self.b:", model.b)

    # #测试MLA
    # mla = MLA(arg)
    # pos_cis = precompute_pos_cis(arg.rope_head_dim, 512)
    # h = torch.randn(bs, seq_len, d_model)
    # output = mla(h, pos_cis[:h.shape[1]])


    lm_config = MiniR1Config()
    test_model = MiniR1(lm_config).to("cuda:0")

    # test_tensor = torch.randint(0, 1000, (32, 10)).to("cuda:0")

    # test_model(test_tensor)
    # # 将模型设置为评估模式
    # test_model.eval()

    print(f'LLM总参数量：{count_parameters(test_model) / 1e9:.3f} B')

   
   # 估计模型保存大小（MB）
    model_size_mb = estimate_model_size(test_model)
    print(f'估计模型保存大小：{model_size_mb:.3f} MB')

    # 进行前向传播
    # with torch.no_grad():
    #     outputs = test_model(test_tensor)

    # # 打印输出的形状，以验证模型是否正常工作
    # print(outputs.logits.shape)  # 应该输出：torch.Size([32, 10, 1000])