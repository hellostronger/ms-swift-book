# 第 32 章 DeepSpeed 与 FSDP

## 32.1 DeepSpeed 概述

DeepSpeed 是微软开源的分布式训练库，提供 ZeRO（Zero Redundancy Optimizer）优化器：

| ZeRO 阶段 | 优化器状态 | 梯度 | 参数 | 显存节省 |
|----------|-----------|------|------|---------|
| ZeRO-0 | 不切分 | 不切分 | 不切分 | 0% |
| ZeRO-1 | 切分 | 不切分 | 不切分 | 4x |
| ZeRO-2 | 切分 | 切分 | 不切分 | 8x |
| ZeRO-3 | 切分 | 切分 | 切分 | 24x+ |

## 32.2 DeepSpeed 配置

ms-swift 提供预设配置：

```python
deepspeed: Optional[str] = None

# 预设值
'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload'
```

### 使用示例

```bash
# ZeRO-2
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --deepspeed zero2 \
  ...

# ZeRO-3 with CPU offload
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --deepspeed zero3_offload \
  ...
```

## 32.3 ZeRO 配置详解

### ZeRO-2 配置

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "none",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": false,
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 32,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

### ZeRO-3 配置

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "none",
      "pin_memory": true
    },
    "offload_param": {
      "device": "none",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

## 32.4 CPU Offload

CPU offload 将优化器状态和参数卸载到 CPU 内存：

```bash
# ZeRO-2 with optimizer offload
swift sft \
  --deepspeed zero2_offload \
  ...

# ZeRO-3 with full offload
swift sft \
  --deepspeed zero3_offload \
  ...
```

### 性能影响

| 配置 | 显存 | 速度 |
|------|------|------|
| ZeRO-2 | 中 | 快 |
| ZeRO-2 + offload | 低 | 慢 |
| ZeRO-3 | 低 | 中 |
| ZeRO-3 + offload | 最低 | 最慢 |

## 32.5 ZeRO++ 特性

ZeRO++ 提供额外的优化：

```python
zero_hpz_partition_size: Optional[int] = None
```

```bash
swift sft \
  --deepspeed zero3 \
  --zero_hpz_partition_size 8 \
  ...
```

特点：
- 节点内模型分片
- 节点间数据分片
- 减少通信开销

## 32.6 FSDP2 配置

FSDP2 是 PyTorch 原生的分布式训练方案：

```python
fsdp: Optional[str] = None

# 预设值
'fsdp2'
```

### 使用示例

```bash
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --fsdp fsdp2 \
  ...
```

## 32.7 FSDP 配置文件

```json
{
  "fsdp": "full_shard auto_wrap",
  "fsdp_config": {
    "fsdp_version": 2,
    "auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    "min_num_params": 1e7,
    "backward_prefetch": "BACKWARD_PRE",
    "forward_prefetch": false,
    "use_orig_params": true,
    "state_dict_type": "SHARDED_STATE_DICT",
    "activation_checkpointing": true
  }
}
```

## 32.8 DeepSpeed vs FSDP

| 特性 | DeepSpeed | FSDP2 |
|------|-----------|-------|
| 成熟度 | 高 | 中 |
| 功能丰富度 | 高 | 中 |
| 原生支持 | 否 | 是 |
| 易用性 | 中 | 高 |
| 性能 | 高 | 高 |

### 选择建议

- **DeepSpeed**：需要丰富功能（offload、ZeRO++ 等）
- **FSDP2**：追求简单、原生支持

## 32.9 完整训练示例

### DeepSpeed ZeRO-3

```bash
swift sft \
  --model Qwen/Qwen2.5-72B-Instruct \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --tuner_type full \
  --deepspeed zero3 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --bf16 true \
  --num_train_epochs 3 \
  --output_dir output/qwen2_5-72b-sft
```

### FSDP2

```bash
swift sft \
  --model Qwen/Qwen2.5-72B-Instruct \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --tuner_type full \
  --fsdp fsdp2 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --gradient_accumulation_steps 4 \
  --per_device_train_batch_size 1 \
  --bf16 true \
  --num_train_epochs 3 \
  --output_dir output/qwen2_5-72b-fsdp
```

## 小结

- **要点 1**：DeepSpeed ZeRO 通过切分优化器状态、梯度和参数减少显存
- **要点 2**：ZeRO-3 with CPU offload 可以训练超大模型
- **要点 3**：FSDP2 是 PyTorch 原生方案，更易用
- **要点 4**：根据模型大小和硬件选择合适的分布式策略
