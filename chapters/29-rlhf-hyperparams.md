# 第 29 章 RLHF 超参数

## 29.1 beta 参数

beta 是 RLHF 中最重要的超参数，控制策略偏离参考模型的程度：

```python
beta: Optional[float] = None

def _set_default(self):
    if self.beta is None:
        if self.rlhf_type == 'gkd':
            self.beta = 0.5
        elif self.rlhf_type == 'simpo':
            self.beta = 2.0
        elif self.rlhf_type == 'grpo':
            self.beta = 0.04
        else:
            self.beta = 0.1  # dpo, kto, cpo, orpo
```

### 不同算法的推荐值

| 算法 | 推荐值 | 说明 |
|------|--------|------|
| DPO | 0.05 ~ 0.2 | 默认 0.1 |
| CPO | 0.05 ~ 0.2 | 默认 0.1 |
| SimPO | 1.0 ~ 3.0 | 默认 2.0 |
| ORPO | 0.05 ~ 0.2 | 默认 0.1 |
| KTO | 0.05 ~ 0.2 | 默认 0.1 |
| PPO | 0.01 ~ 0.1 | 默认 0.05 |
| GRPO | 0.01 ~ 0.1 | 默认 0.04 |
| GKD | 0.3 ~ 0.7 | 默认 0.5 |

### beta 的影响

```python
# DPO 损失公式
losses = -F.logsigmoid(self.beta * logits)

# logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
```

- beta 大：更严格约束策略，更新更保守
- beta 小：允许更大偏离，更新更激进

## 29.2 temperature 参数

temperature 控制采样的随机性：

```python
temperature: float = 0.9  # PPO/GRPO/GKD 默认值
```

### 推荐值

| 算法 | 推荐值 | 说明 |
|------|--------|------|
| PPO | 0.8 ~ 1.0 | 生成响应时使用 |
| GRPO | 0.8 ~ 1.0 | 生成响应时使用 |
| GKD | 0.5 ~ 1.0 | JSD 计算时使用 |

### 数学意义

```python
# 采样时
probs = softmax(logits / temperature)

# temperature = 0: 贪婪解码
# temperature = 1: 标准采样
# temperature > 1: 更随机
# temperature < 1: 更确定
```

## 29.3 label_smoothing 参数

label_smoothing 用于 DPO，防止过拟合：

```python
label_smoothing: float = 0  # 默认不启用

# DPO sigmoid 损失
if loss_type == 'sigmoid':
    losses = (-F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) 
              - F.logsigmoid(-self.beta * logits) * self.label_smoothing)
```

### 推荐值

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 标准训练 | 0 | 不启用 |
| 噪声数据 | 0.01 ~ 0.05 | 轻度平滑 |
| 严重过拟合 | 0.05 ~ 0.1 | 强平滑 |

注意：某些 loss_type（如 hinge, ipo）不支持 label_smoothing。

## 29.4 rpo_alpha 参数

rpo_alpha 在 DPO 损失基础上添加 SFT 损失：

```python
rpo_alpha: Optional[float] = None

# 损失计算
if self.args.rpo_alpha is not None:
    losses = losses + self.args.rpo_alpha * model_output['nll_loss']
```

### 推荐值

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 不启用 | None | 纯 DPO |
| 轻度 SFT | 0.1 ~ 0.5 | 保留部分语言能力 |
| 强 SFT | 0.5 ~ 1.0 | 更多保留语言能力 |

### 使用示例

```bash
swift rlhf \
  --rlhf_type dpo \
  --rpo_alpha 0.1 \
  --beta 0.1 \
  ...
```

## 29.5 ld_alpha 参数

ld_alpha 来自 LD-DPO 论文，缓解长度偏好：

```python
ld_alpha: Optional[float] = None

# 计算逻辑
if self.args.ld_alpha is not None:
    # 将序列分为前缀和后缀
    front_logps = per_token_logps[:, start:start + public_length].sum()
    rear_logps = per_token_logps[:, start + public_length:end].sum()
    
    # 对后缀应用 ld_alpha
    all_logps[i] = front_logps + self.args.ld_alpha * rear_logps
```

### 推荐值

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 不启用 | None | 标准 DPO |
| 缓解长度偏好 | 0.5 ~ 1.0 | 降低后缀权重 |

## 29.6 cpo_alpha 参数

cpo_alpha 控制 CPO/SimPO 中 NLL 损失的权重：

```python
cpo_alpha: float = 1.0

# SimPO 警告
if args.loss_type == 'simpo' and self.cpo_alpha > 0:
    warnings.warn('You are using CPO-SimPO method because you set a non-zero cpo_alpha.')
```

### 推荐值

| 算法 | 推荐值 | 说明 |
|------|--------|------|
| CPO | 0.5 ~ 1.0 | 默认 1.0 |
| SimPO | 0 | 纯 SimPO |
| CPO-SimPO | 0.1 ~ 0.5 | 混合方法 |

## 29.7 simpo_gamma 参数

simpo_gamma 是 SimPO 的 margin 参数：

```python
simpo_gamma: float = 1.0

# SimPO 损失
# loss = -log(sigmoid(beta * (avg_logp_chosen - avg_logp_rejected) - gamma))
```

### 推荐值

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 标准 | 0.5 ~ 1.5 | 论文推荐范围 |
| 默认 | 1.0 | 常用值 |

## 29.8 desirable_weight / undesirable_weight

这两个参数用于 KTO，平衡好坏样本的权重：

```python
desirable_weight: float = 1.0
undesirable_weight: float = 1.0

# 损失计算
loss = (desirable_weight * desirable_loss + 
        undesirable_weight * undesirable_loss) / 
       (desirable_weight + undesirable_weight)
```

### 推荐值

| 数据分布 | desirable_weight | undesirable_weight |
|---------|-----------------|-------------------|
| 均衡 | 1.0 | 1.0 |
| 好样本多 | 0.5 | 1.5 |
| 坏样本多 | 1.5 | 0.5 |

## 29.9 loss_type 和 loss_weights

DPO 支持多种 loss_type，可以混合使用：

```python
loss_type: Optional[List[str]] = None
loss_weights: Optional[List[float]] = None

# 处理逻辑
def _process_loss_type(self):
    if self.loss_type is None:
        # 默认值
        if self.rlhf_type in ['dpo', 'cpo']:
            self.loss_type = 'sigmoid'
        elif self.rlhf_type == 'kto':
            self.loss_type = 'kto'
```

### 混合损失示例

```bash
swift rlhf \
  --rlhf_type dpo \
  --loss_type sigmoid sft \
  --loss_weights 0.8 0.2 \
  ...
```

## 29.10 完整 DPO 训练示例

```bash
swift rlhf \
  --rlhf_type dpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --ref_model Qwen/Qwen2.5-7B \
  --dataset Anthropic/hh-rlhf \
  --tuner_type lora \
  --beta 0.1 \
  --loss_type sigmoid \
  --label_smoothing 0.0 \
  --rpo_alpha 0.1 \
  --learning_rate 5e-6 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --gradient_accumulation_steps 4 \
  --bf16 true \
  --num_train_epochs 1 \
  --output_dir output/qwen2_5-7b-dpo
```

## 小结

- **要点 1**：beta 是 RLHF 最重要的超参数，不同算法推荐值不同
- **要点 2**：temperature 控制采样随机性，PPO/GRPO 推荐 0.8-1.0
- **要点 3**：label_smoothing、rpo_alpha、ld_alpha 用于正则化
- **要点 4**：cpo_alpha 和 simpo_gamma 是 CPO/SimPO 特有的超参数
