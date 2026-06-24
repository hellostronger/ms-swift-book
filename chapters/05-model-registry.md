# 第 5 章 模型注册系统

## 5.1 MODEL_MAPPING 架构

ms-swift 通过 `MODEL_MAPPING` 统一管理所有支持的模型。每个模型由 `ModelMeta` 对象描述其元信息。

```python
# swift/model/register.py

MODEL_MAPPING: Dict[str, ModelMeta] = {}

def register_model(model_meta: ModelMeta, *, exist_ok: bool = False):
    """注册模型"""
    if not exist_ok and model_meta.model_type in MODEL_MAPPING:
        raise ValueError(f'Model type {model_meta.model_type} already registered.')
    MODEL_MAPPING[model_meta.model_type] = model_meta
```

## 5.2 ModelMeta 结构

```python
# swift/model/model_meta.py

@dataclass
class ModelMeta:
    """模型元信息"""
    
    # 基本信息
    model_type: str                    # 模型类型标识
    model_arch: Optional[str] = None   # 模型架构
    models: List[str] = field(default_factory=list)  # 模型 ID 列表
    
    # Template
    template: Optional[str] = None     # 默认模板
    
    # 能力
    is_multimodal: bool = False        # 是否多模态
    is_agent: bool = False             # 是否支持 Agent
    
    # 加载配置
    torch_dtype: Optional[torch.dtype] = None
    requires_pkgs: List[str] = field(default_factory=list)
```

## 5.3 模型加载流程

```
用户输入 model 参数
        ↓
  匹配 ModelMeta
        ↓
  加载模型配置
        ↓
  实例化模型类
        ↓
  应用量化配置
        ↓
  加载权重
```

核心代码：

```python
# swift/model/utils.py

def get_model_tokenizer(model: str, model_type: Optional[str] = None, ...):
    # 1. 获取 ModelMeta
    if model_type is None:
        model_meta = match_model_meta(model)
    else:
        model_meta = MODEL_MAPPING[model_type]
    
    # 2. 加载配置
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    
    # 3. 实例化模型
    model_cls = get_model_class(model_meta)
    model = model_cls.from_pretrained(model, config=config, ...)
    
    # 4. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, ...)
    
    return model, tokenizer
```

## 5.4 模型类型匹配

ms-swift 支持通过多种方式匹配模型：

```python
def match_model_meta(model: str) -> ModelMeta:
    """匹配模型"""
    # 1. 精确匹配
    for model_meta in MODEL_MAPPING.values():
        if model in model_meta.models:
            return model_meta
    
    # 2. 模糊匹配（支持路径）
    model_lower = model.lower()
    for model_meta in MODEL_MAPPING.values():
        for model_id in model_meta.models:
            if model_id.lower() in model_lower:
                return model_meta
    
    # 3. 从 config 推断
    config = AutoConfig.from_pretrained(model)
    model_type = infer_model_type(config)
    return MODEL_MAPPING.get(model_type)
```

## 5.5 自定义模型注册

开发者可以注册自定义模型：

```python
from swift.model import register_model, ModelMeta

# 定义模型元信息
my_model_meta = ModelMeta(
    model_type='my_model',
    models=['my-org/my-model'],
    template='my_template',
    is_multimodal=False,
)

# 注册
register_model(my_model_meta)
```

## 5.6 支持的模型类别

| 类别 | 代表模型 | 数量 |
|------|----------|------|
| Qwen 系列 | Qwen2.5, Qwen2-VL, QwQ | 20+ |
| LLaMA 系列 | LLaMA3, LLaMA3.1 | 10+ |
| ChatGLM 系列 | ChatGLM3, GLM-4 | 10+ |
| Baichuan 系列 | Baichuan2 | 5+ |
| Yi 系列 | Yi, Yi-1.5 | 10+ |
| 多模态 | Qwen-VL, InternVL, LLaVA | 50+ |
| 其他 | Mistral, Phi, Gemma | 100+ |

## 小结

- **要点 1**：MODEL_MAPPING 统一管理所有支持的模型
- **要点 2**：ModelMeta 描述模型的元信息，包括类型、架构、能力等
- **要点 3**：支持精确匹配、模糊匹配和从 config 推断三种模型匹配方式
