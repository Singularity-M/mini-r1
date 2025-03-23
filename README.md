![LOGO](logo.png)

### 📌 项目简介  

本项目提供了一个完整的 **数据清洗、训练、评估** 流程，并使用 **100% PyTorch** 实现整个训练过程，无任何第三方高阶封装。  

- 数据清洗包括去重、格式规范化等步骤，确保高质量输入数据。  
- 训练过程涵盖 **预训练（Pretrain）、监督微调（SFT）、强化学习（GPRO）、知识蒸馏（Distillation）、LoRA 微调** 等完整流程。
- 本项目所开源的GRPO训练代码适合快速上手和本地适配。
- 代码完全基于 **PyTorch** 从零实现，避免黑盒封装，适合研究和深入理解 LLM 训练。 

---

### 📌 关键特性  

- **💡 数据清洗**
  - 提供完整的数据清理与预处理代码，确保数据高质量输入。
  - 采用 `jsonl` 格式，支持多种数据来源的整合。
  - 处理缺失值、去重、格式规范化，确保数据一致性。

- **🛠 训练流程**
  - **预训练（Pretrain）**：基于大规模语料进行无监督训练，让模型学习基础知识。
  - **监督微调（SFT）**：使用高质量问答数据训练模型，提升对话能力。
  - **强化学习（GRPO）**：基于偏好数据优化模型，使其输出更加符合人类需求。
  - **LoRA 微调（LoRA）**：低秩适配，使模型适应特定领域知识。

- **🚀 100% PyTorch**
  - 训练过程完全使用 PyTorch 实现，不依赖 HuggingFace 等封装库。
  - 适合深度理解 LLM 训练机制，适合学习和研究。
  - 支持 **单卡、多卡（DDP/DeepSpeed）训练**，可扩展至大规模训练。

---

### 📌 快速上手  

#### **1️⃣ 环境准备**

```bash
git clone <你的项目地址>
cd <你的项目目录>
pip install -r requirements.txt
```

#### **2️⃣ 数据准备**  
- 下载清洗后的数据集，并放入 `./dataset/` 目录。  
- 预训练数据 (`pretrain.jsonl`)、微调数据 (`sft.jsonl`)、强化学习数据 (`grpo.jsonl`)。

#### **3️⃣ 训练步骤**  

✅ **预训练** (Pretrain)  
```bash
python train_pretrain.py
```

✅ **监督微调** (SFT)  
```bash
python train_sft.py
```

✅ **强化学习（DPO）**  
```bash
python train_dpo.py
```

✅ **知识蒸馏** (Distillation)  
```bash
python train_distill.py
```

✅ **LoRA 微调**  
```bash
python train_lora.py
```

✅ **评估测试**  
```bash
python eval_model.py
```

---

### 📌 模型结构  

本项目基于 **Transformer Decoder-Only** 结构，并做了以下优化：  

- **使用 `RMSNorm`** 进行归一化，替代传统的 LayerNorm。  
- **采用 `SwiGLU` 激活函数**，提升训练稳定性。  
- **支持 `RoPE 位置编码`**，增强长文本建模能力。  
- **支持 `Dense` 和 `MoE`（混合专家）两种架构**，适应不同计算需求。  

待补充图片：  
📌 **[Transformer 架构图]**  
📌 **[MoE 结构示意图]**  

---

### 📌 参考与致谢  

本项目受到 **[MiniMind](https://github.com/jingyaogong/minimind)** 启发，感谢其开源的训练思路和实现方式。  
希望本项目也能为更多人提供 **简洁、完整、100% PyTorch** 的 LLM 训练示例。  

🚀 **欢迎 Star & PR！**