# 第 31 章 部署与推理超参数

## 31.1 推理后端选择

ms-swift 支持多种推理后端：

```python
infer_backend: Literal['transformers', 'vllm', 'lmdeploy', 'sglang'] = 'transformers'
```

### 后端对比

| 后端 | 吞吐 | 显存 | 适用场景 |
|------|------|------|---------|
| transformers | 低 | 中 | 开发调试 |
| vllm | 高 | 中 | 生产部署 |
| lmdeploy | 高 | 中 | NVIDIA GPU |
| sglang | 高 | 中 | 高并发 |

## 31.2 vLLM 参数

### 基础参数

```python
# 张量并行
tensor_parallel_size: int = 1

# GPU 显存利用率
gpu_memory_utilization: float = 0.9

# 最大模型长度
max_model_len: Optional[int] = None

# 批处理大小
max_num_seqs: int = 256
```

### 使用示例

```bash
swift deploy \
  --model Qwen/Qwen2.5-7B-Instruct \
  --infer_backend vllm \
  --tensor_parallel_size 4 \
  --gpu_memory_utilization 0.9 \
  --max_model_len 32768 \
  --port 8000
```

## 31.3 生成参数

### 采样参数

```python
temperature: float = 0.7      # 温度
top_p: float = 0.9            # Top-p 采样
top_k: int = 50               # Top-k 采样
repetition_penalty: float = 1.0  # 重复惩罚
max_tokens: int = 2048        # 最大生成 token 数
```

### 推荐配置

| 场景 | temperature | top_p | top_k |
|------|-------------|-------|-------|
| 确定性任务 | 0.0 | 1.0 | 1 |
| 创意写作 | 0.9 | 0.95 | 50 |
| 代码生成 | 0.2 | 0.9 | 20 |
| 对话 | 0.7 | 0.9 | 50 |

## 31.4 LoRA 部署

部署 LoRA 模型时，需要指定 adapter：

```bash
swift deploy \
  --model Qwen/Qwen2.5-7B \
  --adapters output/qwen2_5-7b-lora/checkpoint-1000 \
  --infer_backend vllm \
  --port 8000
```

## 31.5 多模态部署

部署多模态模型时，需要指定视觉参数：

```bash
swift deploy \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --infer_backend vllm \
  --limit_mm_per_prompt '{"image": 5, "video": 2}' \
  --port 8000
```

## 31.6 量化推理

### GPTQ 量化

```bash
# 使用 GPTQ 量化模型
swift deploy \
  --model TheBloke/Llama-2-7B-GPTQ \
  --infer_backend vllm \
  --quant_method gptq \
  --port 8000
```

### AWQ 量化

```bash
swift deploy \
  --model TheBloke/Llama-2-7B-AWQ \
  --infer_backend vllm \
  --quant_method awq \
  --port 8000
```

## 31.7 API 服务参数

### 服务器参数

```python
host: str = '0.0.0.0'        # 监听地址
port: int = 8000             # 端口
api_key: Optional[str] = None # API 密钥
```

### 使用示例

```bash
swift deploy \
  --model Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api_key sk-xxx \
  --infer_backend vllm
```

## 31.8 完整部署示例

```bash
swift deploy \
  --model Qwen/Qwen2.5-7B-Instruct \
  --infer_backend vllm \
  --tensor_parallel_size 4 \
  --gpu_memory_utilization 0.9 \
  --max_model_len 32768 \
  --max_num_seqs 256 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 50 \
  --repetition_penalty 1.0 \
  --max_tokens 2048 \
  --host 0.0.0.0 \
  --port 8000 \
  --api_key sk-xxx
```

## 小结

- **要点 1**：vLLM 是生产部署的推荐选择
- **要点 2**：tensor_parallel_size 控制张量并行度
- **要点 3**：temperature/top_p/top_k 控制生成多样性
- **要点 4**：支持 GPTQ/AWQ 量化推理
