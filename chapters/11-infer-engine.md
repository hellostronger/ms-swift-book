# 第 11 章 推理引擎

## 11.1 推理引擎架构

ms-swift 支持多种推理后端：

```
swift/infer_engine/
├── base.py                  # 基类
├── infer_engine.py          # 推理引擎接口
├── transformers_engine.py   # Transformers 后端
├── vllm_engine.py           # vLLM 后端
├── lmdeploy_engine.py       # LMDeploy 后端
├── sglang_engine.py         # SGLang 后端
├── grpo_vllm_engine.py      # GRPO vLLM 引擎
├── infer_client.py          # 推理客户端
└── protocol.py              # 协议定义
```

## 11.2 推理后端对比

| 后端 | 特点 | 适用场景 | 吞吐 |
|------|------|---------|------|
| Transformers | 原生支持，无需额外安装 | 开发调试 | 低 |
| vLLM | PagedAttention，高吞吐 | 生产部署 | 高 |
| LMDeploy | TurboMind 加速 | NVIDIA GPU | 高 |
| SGLang | RadixAttention | 高并发 | 高 |

## 11.3 TransformersEngine

```python
class TransformersEngine(InferEngine):
    """基于 Transformers 的推理引擎"""
    
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = kwargs.get('generation_config')
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        # 1. 模板编码
        input_ids = self.template.encode(messages)
        
        # 2. 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                generation_config=self.generation_config,
                **kwargs,
            )
        
        # 3. 解码
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
```

## 11.4 vLLMEngine

```python
class VLLMEngine(InferEngine):
    """基于 vLLM 的高性能推理引擎"""
    
    def __init__(self, model, tokenizer, **kwargs):
        from vllm import LLM, SamplingParams
        
        self.llm = LLM(
            model=model,
            tokenizer=tokenizer,
            tensor_parallel_size=kwargs.get('tensor_parallel_size', 1),
            gpu_memory_utilization=kwargs.get('gpu_memory_utilization', 0.9),
            max_model_len=kwargs.get('max_model_len'),
        )
        
        self.sampling_params = SamplingParams(
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            max_tokens=kwargs.get('max_tokens', 2048),
        )
    
    def generate(self, messages: List[Dict], **kwargs) -> str:
        prompt = self.template.encode(messages, return_text=True)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0].outputs[0].text
```

## 11.5 推理协议

```python
@dataclass
class InferRequest:
    """推理请求"""
    messages: List[Dict]
    images: Optional[List[str]] = None
    audios: Optional[List[str]] = None
    
@dataclass
class RequestConfig:
    """请求配置"""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 2048
    repetition_penalty: float = 1.0

@dataclass  
class InferResponse:
    """推理响应"""
    text: str
    usage: Dict
    finish_reason: str
```

## 11.6 API 服务部署

```bash
# 部署为 OpenAI 兼容 API
swift deploy \
  --model Qwen/Qwen2.5-7B-Instruct \
  --infer_backend vllm \
  --port 8000 \
  --host 0.0.0.0

# 调用
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}]}'
```

## 小结

- **要点 1**：ms-swift 支持 4 种推理后端，覆盖从开发到生产的全部场景
- **要点 2**：vLLM 通过 PagedAttention 实现高吞吐推理
- **要点 3**：部署为 OpenAI 兼容 API，可无缝替换现有应用
