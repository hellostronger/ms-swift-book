# 第 34 章 模型导出与部署

## 34.1 导出类型

ms-swift 支持多种导出类型：

```python
export_type: Literal['checkpoint', 'full', 'merged', 'quant'] = 'merged'
```

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| checkpoint | 导出 checkpoint | 继续训练 |
| full | 导出完整模型 | 全参数训练 |
| merged | 合并 LoRA 权重 | LoRA 训练 |
| quant | 量化导出 | 推理部署 |

## 34.2 LoRA 合并

将 LoRA 权重合并到基础模型：

```bash
swift export \
  --model Qwen/Qwen2.5-7B \
  --adapters output/qwen2_5-7b-lora/checkpoint-1000 \
  --export_type merged \
  --output_dir output/qwen2_5-7b-merged
```

### 合并原理

```python
# LoRA: W' = W + BA
# 合并后: W_merged = W + BA

merged_weight = base_weight + lora_B @ lora_A * scaling
```

## 34.3 量化导出

### GPTQ 量化

```bash
swift export \
  --model output/qwen2_5-7b-merged \
  --export_type quant \
  --quant_method gptq \
  --quant_bits 4 \
  --output_dir output/qwen2_5-7b-gptq
```

### AWQ 量化

```bash
swift export \
  --model output/qwen2_5-7b-merged \
  --export_type quant \
  --quant_method awq \
  --quant_bits 4 \
  --output_dir output/qwen2_5-7b-awq
```

### BNB 量化

```bash
swift export \
  --model output/qwen2_5-7b-merged \
  --export_type quant \
  --quant_method bnb \
  --quant_bits 4 \
  --output_dir output/qwen2_5-7b-bnb
```

## 34.4 量化方法对比

| 方法 | 速度 | 精度 | 显存 | 适用场景 |
|------|------|------|------|---------|
| GPTQ | 快 | 高 | 4x 压缩 | GPU 推理 |
| AWQ | 快 | 高 | 4x 压缩 | GPU 推理 |
| BNB | 中 | 中 | 4x 压缩 | CPU/GPU |
| GGUF | 慢 | 中 | 可变 | CPU 推理 |

## 34.5 GGUF 导出

导出为 llama.cpp 格式：

```bash
swift export \
  --model output/qwen2_5-7b-merged \
  --export_type quant \
  --quant_method gguf \
  --quant_bits q4_k_m \
  --output_dir output/qwen2_5-7b-gguf
```

### GGUF 量化类型

| 类型 | 大小 | 质量 |
|------|------|------|
| q2_k | 最小 | 最低 |
| q4_0 | 小 | 中 |
| q4_k_m | 小 | 高 |
| q5_k_m | 中 | 很高 |
| q8_0 | 大 | 接近原始 |

## 34.6 模型推送

推送模型到 ModelScope Hub：

```bash
swift export \
  --model output/qwen2_5-7b-merged \
  --export_type merged \
  --push_to_hub true \
  --hub_model_id my-organization/my-model \
  --hub_token xxx
```

## 34.7 完整导出流程

### 1. LoRA 训练

```bash
swift sft \
  --model Qwen/Qwen2.5-7B \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --tuner_type lora \
  --output_dir output/qwen2_5-7b-lora
```

### 2. 合并权重

```bash
swift export \
  --model Qwen/Qwen2.5-7B \
  --adapters output/qwen2_5-7b-lora/checkpoint-1000 \
  --export_type merged \
  --output_dir output/qwen2_5-7b-merged
```

### 3. 量化导出

```bash
swift export \
  --model output/qwen2_5-7b-merged \
  --export_type quant \
  --quant_method awq \
  --quant_bits 4 \
  --output_dir output/qwen2_5-7b-awq
```

### 4. 部署推理

```bash
swift deploy \
  --model output/qwen2_5-7b-awq \
  --infer_backend vllm \
  --quant_method awq \
  --port 8000
```

## 34.8 ONNX 导出

导出为 ONNX 格式：

```bash
swift export \
  --model output/qwen2_5-7b-merged \
  --export_type full \
  --to_onnx true \
  --output_dir output/qwen2_5-7b-onnx
```

## 34.9 部署最佳实践

### 1. 性能优化

```bash
swift deploy \
  --model output/qwen2_5-7b-awq \
  --infer_backend vllm \
  --tensor_parallel_size 4 \
  --gpu_memory_utilization 0.9 \
  --max_model_len 32768 \
  --port 8000
```

### 2. 多实例部署

```bash
# 实例 1
swift deploy --port 8000 ...

# 实例 2
swift deploy --port 8001 ...

# 负载均衡
nginx -> [8000, 8001, 8002]
```

### 3. 监控与日志

```bash
swift deploy \
  --model output/qwen2_5-7b-awq \
  --log_level info \
  --metrics_port 9090 \
  --port 8000
```

## 小结

- **要点 1**：支持 merged、quant 等多种导出类型
- **要点 2**：GPTQ/AWQ 是推荐的量化方法
- **要点 3**：完整流程：训练 → 合并 → 量化 → 部署
- **要点 4**：根据硬件选择合适的量化方法
