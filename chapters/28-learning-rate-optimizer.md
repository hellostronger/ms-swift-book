# 第 28 章 学习率与优化器

## 28.1 学习率配置

ms-swift 提供灵活的学习率配置，支持不同训练方式的默认值：

```python
# swift/arguments/sft_args.py

@dataclass
class SftArguments:
    learning_rate: Optional[float] = None
    
    def _init_override(self):
        if self.learning_rate is None:
            if self.tuner_type == 'full':
                self.learning_rate = 1e-5  # 全参数训练
            else:
                self.learning_rate = 1e-4  # LoRA 等参数高效方法
```

### 推荐学习率

| 训练方式 | 推荐学习率 | 说明 |
|---------|-----------|------|
| full | 1e-5 ~ 5e-5 | 全参数微调需要较小的学习率 |
| lora | 1e-4 ~ 3e-4 | LoRA 可以使用较大的学习率 |
| qlora | 1e-4 ~ 2e-4 | 量化模型需要稍小的学习率 |
| dpo | 1e-6 ~ 5e-6 | RLHF 需要更小的学习率 |
| grpo | 1e-6 ~ 1e-5 | GRPO 对学习率敏感 |

## 28.2 学习率调度器

ms-swift 支持多种学习率调度策略：

```bash
# 余弦调度（推荐）
--lr_scheduler_type cosine

# 线性调度
--lr_scheduler_type linear

# 常数调度
--lr_scheduler_type constant

# 余弦带重启
--lr_scheduler_type cosine_with_restarts

# 多项式调度
--lr_scheduler_type polynomial
```

### 余弦调度详解

余弦调度是最常用的策略，学习率按余弦曲线衰减：

```
lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * t / T))
```

参数：
- `lr_max`：初始学习率
- `lr_min`：最小学习率（默认为 0）
- `T`：总训练步数

## 28.3 优化器选择

ms-swift 支持多种优化器：

```python
# swift/arguments/sft_args.py

optimizer: Optional[str] = None

def __post_init__(self):
    if self.optimizer is None:
        if self.lorap_lr_ratio:
            self.optimizer = 'lorap'  # LoRA+ 优化器
        elif self.use_galore:
            self.optimizer = 'galore'  # GaLore 优化器
```

### 内置优化器

| 优化器 | 适用场景 | 显存占用 |
|-------|---------|---------|
| adamw_hf | 通用 | 高 |
| adamw_torch | PyTorch 原生 | 高 |
| sgd | 简单任务 | 低 |
| adafactor | 大模型 | 中 |
| lorap | LoRA+ | 中 |
| galore | 内存高效 | 低 |

### LoRA+ 优化器

LoRA+ 为 A 和 B 矩阵使用不同的学习率：

```python
@dataclass
class LoRAConfig:
    lorap_lr_ratio: float = 16.0  # B 矩阵学习率倍率
    lorap_emb_lr: float = 1e-6    # embedding 学习率
```

使用方式：

```bash
swift sft \
  --tuner_type lora \
  --lorap_lr_ratio 16.0 \
  --learning_rate 1e-4
```

效果：
- A 矩阵：lr = 1e-4
- B 矩阵：lr = 1e-4 * 16.0 = 1.6e-3

### GaLore 优化器

GaLore（Gradient Low-Rank Projection）通过低秩投影减少优化器状态：

```bash
swift sft \
  --use_galore true \
  --galore_rank 256 \
  --galore_update_proj_gap 200 \
  --galore_scale 0.25
```

参数：
- `galore_rank`：低秩近似的秩
- `galore_update_proj_gap`：投影矩阵更新间隔
- `galore_scale`：缩放因子

## 28.4 权重衰减

权重衰减可以防止过拟合：

```bash
--weight_decay 0.01  # 推荐值
```

对于不同层可以使用不同的权重衰减：

```python
# 通常对 bias 和 layer_norm 不使用权重衰减
no_decay = ["bias", "layer_norm.weight", "LayerNorm.weight"]
```

## 28.5 梯度累积

梯度累积可以在不增加显存的情况下增大 batch size：

```bash
--gradient_accumulation_steps 4
```

有效 batch size = per_device_batch_size × num_gpus × gradient_accumulation_steps

示例：
- per_device_batch_size = 2
- num_gpus = 4
- gradient_accumulation_steps = 4
- 有效 batch size = 2 × 4 × 4 = 32

## 28.6 梯度裁剪

梯度裁剪可以防止梯度爆炸：

```bash
--max_grad_norm 1.0  # 默认值
```

当梯度范数超过阈值时，会按比例缩放：

```python
if grad_norm > max_grad_norm:
    grad = grad * (max_grad_norm / grad_norm)
```

## 28.7 Warmup

Warmup 阶段逐渐增加学习率，防止训练初期不稳定：

```bash
--warmup_ratio 0.1      # warmup 步数占总步数的比例
--warmup_steps 100      # 或者直接指定步数
```

推荐使用 `warmup_ratio`，因为它会自动适配总训练步数。

## 28.8 混合精度训练

混合精度可以加速训练并减少显存：

```bash
# bf16（推荐，需要 Ampere 或更新架构）
--bf16 true

# fp16（需要 loss scaling）
--fp16 true
```

bf16 vs fp16：
- bf16：动态范围大，不需要 loss scaling，推荐
- fp16：动态范围小，需要 loss scaling，可能溢出

## 28.9 完整训练示例

```bash
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --tuner_type lora \
  --learning_rate 2e-4 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --weight_decay 0.01 \
  --gradient_accumulation_steps 4 \
  --max_grad_norm 1.0 \
  --bf16 true \
  --num_train_epochs 3 \
  --output_dir output/qwen2_5-7b-sft
```

## 小结

- **要点 1**：学习率根据训练方式不同，推荐值不同（full: 1e-5, lora: 1e-4）
- **要点 2**：余弦调度是最常用的学习率策略
- **要点 3**：LoRA+ 通过为 A/B 矩阵设置不同学习率提升效果
- **要点 4**：梯度累积和混合精度是加速训练的关键技术
