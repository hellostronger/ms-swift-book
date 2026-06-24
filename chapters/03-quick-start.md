# 第 3 章 快速入门

## 3.1 安装

```bash
# 基础安装
pip install ms-swift

# 完整安装（包含所有可选依赖）
pip install ms-swift[all]

# 从源码安装
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

## 3.2 SFT 微调示例

```bash
# 使用 LoRA 微调 Qwen2.5-7B
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --tuner_type lora \
  --output_dir output
```

等价 Python 代码：

```python
from swift import SwiftPipeline

pipeline = SwiftPipeline.from_args({
    'task_type': 'causal_lm',
    'model': 'Qwen/Qwen2.5-7B-Instruct',
    'dataset': 'AI-ModelScope/alpaca-gpt4-data-zh',
    'tuner_type': 'lora',
    'output_dir': 'output',
})
pipeline.run()
```

## 3.3 RLHF 训练示例

```bash
# DPO 训练
swift rlhf \
  --rlhf_type dpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset Anthropic/hh-rlhf \
  --tuner_type lora \
  --output_dir output

# GRPO 训练
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset openai/gsm8k \
  --reward_funcs accuracy format \
  --tuner_type lora \
  --output_dir output
```

## 3.4 推理与部署

```bash
# 模型推理
swift infer \
  --model output/qwen2_5-7b-instruct/vx_xxx/checkpoint-xxx \
  --infer_backend transformers

# 部署为 API 服务
swift deploy \
  --model output/qwen2_5-7b-instruct/vx_xxx/checkpoint-xxx \
  --infer_backend vllm \
  --port 8000
```

## 3.5 训练流程概览

```
数据准备 → 模型加载 → 训练循环 → 评估验证 → 模型保存
    ↓           ↓           ↓           ↓           ↓
  Dataset     Model       Trainer     Metrics     Checkpoint
  Template    Tuner       Loss        Eval        Export
```

## 小结

- **要点 1**：ms-swift 提供 CLI 和 Python API 两种使用方式
- **要点 2**：SFT 和 RLHF 训练只需几行命令即可启动
- **要点 3**：推理和部署支持多种后端（Transformers、vLLM、LMDeploy）
