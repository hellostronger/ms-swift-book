# 第 8 章 Tuner 微调方法

## 8.1 Tuner 体系概览

ms-swift 支持多种微调方法，通过 `tuner_type` 参数选择：

| Tuner 类型 | 原理 | 显存占用 | 效果 |
|-----------|------|---------|------|
| full | 全参数训练 | 100% | 最佳 |
| lora | 低秩适配 | ~10% | 接近全参数 |
| qlora | 量化+LoRA | ~5% | 良好 |
| longlora | 长上下文LoRA | ~10% | 长文本 |
| llamapro | LLaMA Pro | ~15% | 渐进式 |
| adapter | Adapter层 | ~12% | 传统PEFT |
| prompt | Prompt Tuning | ~3% | 轻量 |
| reft | REFT | ~8% | 表示微调 |
| side | Side Network | ~10% | 侧网络 |

## 8.2 LoRA 实现

```python
# swift/tuners/lora.py

@dataclass
class LoRAConfig(LoraConfig, SwiftConfig):
    """LoRA 配置"""
    
    # 基本参数
    r: int = 8                    # 秩
    lora_alpha: int = 32          # 缩放因子
    lora_dropout: float = 0.05    # dropout
    
    # 目标模块
    target_modules: List[str] = None  # 应用 LoRA 的模块
    
    # 高级参数
    use_qa_lora: bool = False     # QA-LoRA
    lora_dtype: Optional[str] = None  # LoRA 权重精度
    lorap_lr_ratio: float = 16.0  # LoRA+ 学习率比例

class LoRA(SwiftAdapter):
    """LoRA 实现"""
    
    @staticmethod
    def prepare_model(model, config, adapter_name):
        # 1. 找到目标模块
        target_modules = config.target_modules or ['q_proj', 'v_proj']
        
        # 2. 注入 LoRA 层
        for name, module in model.named_modules():
            if any(t in name for t in target_modules):
                lora_layer = LoRALayer(module, config)
                replace_module(model, name, lora_layer)
        
        return model
```

### LoRA 数学原理

LoRA 通过低秩分解来近似权重更新：

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d, k)$

缩放因子：$\frac{\alpha}{r}$ 控制 LoRA 更新的强度

## 8.3 QLoRA 实现

QLoRA 在 4-bit 量化基础上使用 LoRA：

```python
class QLoRA(LoRA):
    """QLoRA = 4-bit quantization + LoRA"""
    
    @staticmethod
    def prepare_model(model, config, adapter_name):
        # 1. 量化基础模型
        from bitsandbytes import Linear4bit
        quantize_model(model, Linear4bit)
        
        # 2. 注入 LoRA
        super().prepare_model(model, config, adapter_name)
        
        # 3. 启用 gradient checkpointing
        model.gradient_checkpointing_enable()
        
        return model
```

## 8.4 LoRA+ 学习率策略

```python
# LoRA+ 为 A 和 B 矩阵使用不同的学习率
# lora_A: lr
# lora_B: lr * lorap_lr_ratio (默认 16x)
```

## 8.5 TunerArguments 关键参数

```python
@dataclass
class TunerArguments:
    tuner_type: Literal['full', 'lora', 'qlora', ...] = 'lora'
    
    # LoRA 参数
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = ['ALL']  # 默认所有线性层
    
    # 多 LoRA
    adapters: List[str] = []  # 多个 adapter 路径
    
    # 冻结参数
    freeze_vit: bool = True        # 冻结视觉编码器
    freeze_llm: bool = False       # 冻结语言模型
    freeze_aligner: bool = True    # 冻结对齐层
```

## 小结

- **要点 1**：ms-swift 支持 9 种微调方法，覆盖从全参数到参数高效的所有场景
- **要点 2**：LoRA 通过低秩分解大幅减少可训练参数，效果接近全参数训练
- **要点 3**：LoRA+ 通过为 A/B 矩阵设置不同学习率进一步提升效果
