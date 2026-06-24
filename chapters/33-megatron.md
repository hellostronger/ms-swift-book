# 第 33 章 Megatron 集成

## 33.1 Megatron 概述

Megatron-LM 是 NVIDIA 开发的大规模语言模型训练框架，提供：
- 张量并行（Tensor Parallelism）
- 流水线并行（Pipeline Parallelism）
- 序列并行（Sequence Parallelism）
- 专家并行（Expert Parallelism）

## 33.2 ms-swift Megatron 支持

ms-swift 提供 Megatron 集成：

```
swift/megatron/
├── arguments/          # Megatron 参数
├── callbacks/          # 回调函数
├── model/              # 模型定义
├── pipelines/          # 训练流水线
│   ├── export/         # 导出
│   └── train/          # 训练
├── trainers/           # 训练器
└── utils/              # 工具函数
```

## 33.3 并行策略

### 张量并行（TP）

将模型参数切分到多个 GPU：

```
GPU 0: [W1_a, W2_a, W3_a]
GPU 1: [W1_b, W2_b, W3_b]
GPU 2: [W1_c, W2_c, W3_c]
GPU 3: [W1_d, W2_d, W3_d]
```

```bash
--tensor_model_parallel_size 4
```

### 流水线并行（PP）

将模型层切分到多个 GPU：

```
GPU 0: [Layer 0-7]
GPU 1: [Layer 8-15]
GPU 2: [Layer 16-23]
GPU 3: [Layer 24-31]
```

```bash
--pipeline_model_parallel_size 4
```

### 数据并行（DP）

每个 GPU 处理不同的数据：

```
GPU 0: [Batch 0]
GPU 1: [Batch 1]
GPU 2: [Batch 2]
GPU 3: [Batch 3]
```

```bash
# 自动计算
world_size = TP * PP * DP
```

## 33.4 3D 并行示例

```bash
# 8 GPU: TP=2, PP=2, DP=2
swift sft \
  --model Qwen/Qwen2.5-72B-Instruct \
  --megatron true \
  --tensor_model_parallel_size 2 \
  --pipeline_model_parallel_size 2 \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh \
  ...
```

## 33.5 Megatron 参数

### 并行参数

```python
tensor_model_parallel_size: int = 1
pipeline_model_parallel_size: int = 1
context_parallel_size: int = 1
sequence_parallel: bool = False
```

### 模型参数

```python
num_layers: int = None
hidden_size: int = None
num_attention_heads: int = None
intermediate_size: int = None
```

### 训练参数

```python
micro_batch_size: int = 1
global_batch_size: int = None
recompute_method: str = None
recompute_granularity: str = None
```

## 33.6 内存优化

### 激活重计算

```bash
--recompute_method uniform \
--recompute_granularity full
```

### 序列并行

```bash
--sequence_parallel true \
--tensor_model_parallel_size 4
```

## 33.7 Megatron vs DeepSpeed

| 特性 | Megatron | DeepSpeed |
|------|----------|-----------|
| 张量并行 | 原生支持 | 不支持 |
| 流水线并行 | 原生支持 | 不支持 |
| 易用性 | 复杂 | 简单 |
| 性能 | 极高 | 高 |
| 适用规模 | 超大模型 | 中大模型 |

### 选择建议

- **Megatron**：超大规模模型（100B+），多节点训练
- **DeepSpeed**：中小规模模型，单节点或少量节点

## 33.8 完整训练示例

```bash
# 16 GPU: TP=4, PP=2, DP=2
torchrun --nproc_per_node=8 --nnodes=2 \
  swift/cli/_megatron/train.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --tensor_model_parallel_size 4 \
  --pipeline_model_parallel_size 2 \
  --sequence_parallel true \
  --micro_batch_size 1 \
  --global_batch_size 64 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --bf16 true \
  --num_train_epochs 3 \
  --output_dir output/qwen2_5-72b-megatron
```

## 小结

- **要点 1**：Megatron 支持 3D 并行（TP、PP、DP）
- **要点 2**：适合超大规模模型的多节点训练
- **要点 3**：配置复杂但性能极高
- **要点 4**：与 DeepSpeed 互补，根据规模选择
