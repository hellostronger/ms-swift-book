# 第 25 章 GKD 知识蒸馏

## 25.1 GKD 核心原理

GKD（Generalized Knowledge Distillation）是通用知识蒸馏算法，通过让学生模型学习教师模型的分布来实现知识迁移。

**核心公式：**

$$\mathcal{L}_{GKD} = \text{JSD}(\pi_\theta \| \pi_{teacher}) + \alpha \cdot \mathcal{L}_{SFT}$$

其中 JSD（Jensen-Shannon Divergence）：

$$\text{JSD}(P \| Q) = \beta \cdot D_{KL}(M \| Q) + (1-\beta) \cdot D_{KL}(M \| P)$$

$$M = \beta P + (1-\beta) Q$$

**参数说明：**
- $\beta$：JSD 插值参数（0=forward KL, 1=reverse KL, 0<β<1=JSD）
- $\alpha$：SFT 损失权重

## 25.2 GKD vs 其他方法

| 维度 | SFT | DPO | GKD |
|------|-----|-----|-----|
| 教师模型 | 不需要 | 不需要 | 需要 |
| 训练数据 | 人类标注 | 偏好数据 | 教师生成 |
| 知识来源 | 数据 | 偏好 | 教师分布 |
| 适用场景 | 基础训练 | 对齐 | 蒸馏 |

## 25.3 教师模型配置

ms-swift 支持三种教师模型配置：

### 25.3.1 自蒸馏（动态教师）

```bash
# 不指定 teacher_model，使用当前学生权重作为教师
swift rlhf \
  --rlhf_type gkd \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset student_data \
  ...
```

**特点：** 教师 = 学生（当前权重），动态更新。

### 25.3.2 自蒸馏（固定教师）

```bash
# teacher_model 与 model 相同，使用 LoRA
swift rlhf \
  --rlhf_type gkd \
  --model Qwen/Qwen2.5-7B-Instruct \
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --tuner_type lora \
  ...
```

**特点：** 教师 = 基础模型（固定），学生 = LoRA 微调。

### 25.3.3 标准蒸馏

```bash
# 使用不同的教师模型
swift rlhf \
  --rlhf_type gkd \
  --model Qwen/Qwen2.5-7B-Instruct \
  --teacher_model Qwen/Qwen2.5-72B-Instruct \
  ...
```

**特点：** 教师和大模型，学生是小模型。

### 25.3.4 教师 API 服务

```bash
# 使用远程教师 API
swift rlhf \
  --rlhf_type gkd \
  --model Qwen/Qwen2.5-7B-Instruct \
  --teacher_model_server http://localhost:8000 \
  --gkd_logits_topk 20 \
  ...
```

**特点：** 教师通过 API 提供 logits，无需本地加载。

## 25.4 关键超参数

### beta

```python
# GKD 默认值
beta: Optional[float] = 0.5
```

**作用：** JSD 插值参数。
- 0：forward KL（$D_{KL}(student \| teacher)$）
- 1：reverse KL（$D_{KL}(teacher \| student)$）
- 0.5：标准 JSD

### temperature

```python
temperature: float = 0.9
```

**作用：** 温度缩放参数，控制分布平滑度。

### lmbda

```python
lmbda: float = 0.5
```

**作用：** 平衡策略损失和价值损失。

### sft_alpha

```python
sft_alpha: float = 0
```

**作用：** SFT 损失权重。

$$\mathcal{L} = \mathcal{L}_{GKD} + \alpha \cdot \mathcal{L}_{SFT}$$

### gkd_logits_topk

```python
gkd_logits_topk: Optional[int] = None
```

**作用：** 仅使用 top-k logits 计算 KL 散度。
- None：使用完整词表（更准确，内存占用大）
- 正整数：仅使用 top-k（更高效）

**注意：** 使用 `teacher_model_server` 时必须设置。

### offload_teacher_model

```python
offload_teacher_model: bool = False
```

**作用：** 是否将教师模型卸载到 CPU 内存，节省 GPU 显存。

## 25.5 JSD 损失实现

```python
# swift/rlhf_trainers/gkd_loss.py

def jsd_loss(s_logits, t_logits, beta, log_softmax_fn, kl_div_fn, chunk_size=512):
    """分块计算 JSD"""
    N = s_logits.size(0)
    total = s_logits.new_zeros(())
    
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        s_log = log_softmax_fn(s_logits[start:end])
        t_log = log_softmax_fn(t_logits[start:end])
        
        if beta == 0:
            jsd = kl_div_fn(s_log, t_log)  # forward KL
        elif beta == 1:
            jsd = kl_div_fn(t_log, s_log)  # reverse KL
        else:
            m_log = torch.logsumexp(torch.stack([s_log + log_1_minus_beta, t_log + log_beta]), dim=0)
            jsd = beta_t * kl_div_fn(m_log, t_log) + (1 - beta_t) * kl_div_fn(m_log, s_log)
        
        total = total + jsd.sum()
    
    return total
```

## 25.6 训练流程

```
1. 加载学生模型和教师模型
2. 对于每个 batch:
   a. 学生模型前向传播，获得 logits
   b. 教师模型前向传播，获得 logits（或从 API 获取）
   c. 计算 JSD 损失
   d. 可选：计算 SFT 损失
   e. 反向传播，更新学生模型
```

## 25.7 训练示例

```bash
# 标准蒸馏（大模型 -> 小模型）
swift rlhf \
  --rlhf_type gkd \
  --model Qwen/Qwen2.5-7B-Instruct \
  --teacher_model Qwen/Qwen2.5-72B-Instruct \
  --dataset student_data \
  --beta 0.5 \
  --temperature 0.9 \
  --output_dir output/gkd

# 自蒸馏（LoRA）
swift rlhf \
  --rlhf_type gkd \
  --model Qwen/Qwen2.5-7B-Instruct \
  --teacher_model Qwen/Qwen2.5-7B-Instruct \
  --tuner_type lora \
  --beta 0.5 \
  --output_dir output/gkd_self

# 使用教师 API
swift rlhf \
  --rlhf_type gkd \
  --model Qwen/Qwen2.5-7B-Instruct \
  --teacher_model_server http://localhost:8000 \
  --gkd_logits_topk 20 \
  --beta 0.5 \
  --output_dir output/gkd_api
```

## 25.8 Liger Kernel 加速

```bash
# 使用 Liger Kernel 加速 GKD
swift rlhf \
  --rlhf_type gkd \
  --use_liger_kernel true \
  ...
```

**注意：** Liger Kernel 不支持 `gkd_logits_topk`。

## 小结

- **要点 1**：GKD 通过 JSD 损失实现知识蒸馏
- **要点 2**：支持三种教师配置（动态、固定、API）
- **要点 3**：beta 控制 JSD 类型（0=forward KL, 1=reverse KL, 0.5=JSD）
- **要点 4**：gkd_logits_topk 可提高效率，但使用 API 时必须设置
- **要点 5**：sft_alpha 可混合 SFT 损失，防止遗忘
