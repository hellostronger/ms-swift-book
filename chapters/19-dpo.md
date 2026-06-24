# 第 19 章 DPO 及其 14 种变体

## 19.1 DPO 核心原理

DPO（Direct Preference Optimization）是最流行的对齐算法，直接从偏好数据优化策略模型，无需训练奖励模型。

**核心公式：**

$$\mathcal{L}_{DPO} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

其中：
- $y_w$：chosen response（优选回答）
- $y_l$：rejected response（拒绝回答）
- $\beta$：温度参数，控制偏离参考模型的程度

**ms-swift 实现：**

```python
# swift/rlhf_trainers/dpo_trainer.py

def dpo_loss(self, chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, loss_type='sigmoid'):
    # 计算 log ratios
    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps
    
    # 计算 logits
    logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
    
    # 根据 loss_type 计算损失
    if loss_type == 'sigmoid':
        losses = -F.logsigmoid(self.beta * logits)
    
    return losses
```

## 19.2 14 种 DPO Loss 变体

ms-swift 支持 14 种 DPO 损失函数变体：

### 19.2.1 sigmoid（默认）

标准 DPO 损失：

$$\mathcal{L} = -\log\sigma(\beta \cdot \text{logits})$$

```python
losses = -F.logsigmoid(self.beta * logits)
```

**特点：** 最稳定，效果最好，适合大多数场景。

### 19.2.2 hinge

铰链损失：

$$\mathcal{L} = \max(0, 1 - \beta \cdot \text{logits})$$

```python
losses = torch.relu(1 - self.beta * logits)
```

**特点：** 更关注难样本，但训练可能不稳定。

### 19.2.3 ipo

Identity Preference Optimization：

$$\mathcal{L} = \left(\text{logits} - \frac{1}{2\beta}\right)^2$$

```python
losses = (logits - 1 / (2 * self.beta))**2
```

**特点：** 避免 DPO 的过拟合问题。

### 19.2.4 exo_pair

Exponential Preference Optimization：

```python
losses = (self.beta * logits).sigmoid() * (F.logsigmoid(self.beta * logits) - math.log(1 - self.label_smoothing)) + \
         (-self.beta * logits).sigmoid() * (F.logsigmoid(-self.beta * logits) - math.log(self.label_smoothing))
```

**特点：** 指数加权，对极端值更敏感。

### 19.2.5 bco_pair

BCO (Binary Cross-entropy with Offset)：

```python
chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps)
rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps)
rewards = torch.cat((chosen_rewards, rejected_rewards), 0).mean().detach()
delta = self.running.mean
losses = -F.logsigmoid(chosen_rewards - delta) - F.logsigmoid(-(rejected_rewards - delta))
```

**特点：** 使用运行均值作为偏移，更稳定。

### 19.2.6 sppo_hard

SPPO (Self-Play Preference Optimization)：

```python
a = chosen_logps - ref_chosen_logps
b = rejected_logps - ref_rejected_logps
losses = (a - 0.5 / self.beta)**2 + (b + 0.5 / self.beta)**2
```

**特点：** 自博弈优化，适合对抗训练。

### 19.2.7 nca_pair

NCA (Noise Contrastive Alignment)：

```python
chosen_rewards = (chosen_logps - ref_chosen_logps) * self.beta
rejected_rewards = (rejected_logps - ref_rejected_logps) * self.beta
losses = -F.logsigmoid(chosen_rewards) - 0.5 * F.logsigmoid(-chosen_rewards) - 0.5 * F.logsigmoid(-rejected_rewards)
```

**特点：** 噪声对比学习，增强鲁棒性。

### 19.2.8 aot / aot_unpaired

AOT (Aligned Optimal Transport)：

```python
# 排序后计算
chosen_logratios_sorted, _ = torch.sort(chosen_logratios, dim=0)
rejected_logratios_sorted, _ = torch.sort(rejected_logratios, dim=0)
delta = chosen_logratios_sorted - rejected_logratios_sorted
losses = -F.logsigmoid(self.beta * delta)
```

**特点：** 最优传输对齐，处理分布偏移。

### 19.2.9 apo_zero / apo_down

APO (Aligning Preference Optimization)：

```python
# apo_zero
losses_chosen = 1 - F.sigmoid(self.beta * chosen_logratios)
losses_rejected = F.sigmoid(self.beta * rejected_logratios)
losses = losses_chosen + losses_rejected

# apo_down
losses_chosen = F.sigmoid(self.beta * chosen_logratios)
losses_rejected = 1 - F.sigmoid(self.beta * (chosen_logratios - rejected_logratios))
losses = losses_chosen + losses_rejected
```

**特点：** 自适应优化，动态调整权重。

### 19.2.10 discopop

DiscoPOP：

```python
logits = logits * self.beta
log_ratio_modulation = torch.sigmoid(logits / self.args.discopop_tau)
logistic_component = -F.logsigmoid(logits)
exp_component = torch.exp(-logits)
losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
```

**特点：** 双组件加权，`discopop_tau` 控制调制强度。

### 19.2.11 robust

鲁棒 DPO：

```python
losses = (-F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing) + \
          F.logsigmoid(-self.beta * logits) * self.label_smoothing) / (1 - 2 * self.label_smoothing)
```

**特点：** 标签平滑，增强鲁棒性。

### 19.2.12 sft

SFT 损失（作为辅助任务）：

```python
sft_loss = model_output['nll_loss']
losses = sft_loss.expand(batch_size)
```

**特点：** 混合 SFT 和 DPO，防止遗忘。

## 19.3 Loss Type 对比表

| Loss Type | 公式特点 | 稳定性 | 效果 | 适用场景 |
|-----------|---------|--------|------|---------|
| sigmoid | 标准对数 sigmoid | 高 | 好 | 默认推荐 |
| hinge | 铰链损失 | 中 | 中 | 难样本 |
| ipo | 平方损失 | 高 | 好 | 避免过拟合 |
| exo_pair | 指数加权 | 中 | 中 | 极端值 |
| bco_pair | 偏移交叉熵 | 高 | 好 | 稳定训练 |
| sppo_hard | 自博弈 | 低 | 中 | 对抗训练 |
| nca_pair | 噪声对比 | 中 | 好 | 鲁棒性 |
| aot | 最优传输 | 中 | 好 | 分布偏移 |
| apo_zero/down | 自适应 | 中 | 中 | 动态调整 |
| discopop | 双组件 | 中 | 好 | 精细控制 |
| robust | 标签平滑 | 高 | 好 | 噪声数据 |
| sft | SFT 辅助 | 高 | 中 | 混合训练 |

## 19.4 混合 Loss Type

ms-swift 支持混合多种 loss type：

```bash
# 混合 sigmoid 和 sft
swift rlhf \
  --rlhf_type dpo \
  --loss_type sigmoid sft \
  --loss_weights 0.8 0.2 \
  ...
```

**实现：**

```python
for idx, loss_type in enumerate(loss_types):
    _losses = self.dpo_loss(..., loss_type)
    weight = loss_weights[idx] if loss_weights else 1.0
    losses = losses + _losses * weight
```

## 19.5 关键超参数

### beta

```python
beta: Optional[float] = None
# 默认值：
# - DPO: 0.1
# - SimPO: 2.0
# - GRPO: 0.04
# - GKD: 0.5
```

**作用：** 控制策略偏离参考模型的程度。
- beta 大：偏离小，更保守
- beta 小：偏离大，更激进

### label_smoothing

```python
label_smoothing: float = 0
```

**作用：** 标签平滑，防止过拟合。
- 0：不平滑
- 0.1：轻度平滑
- 0.2：重度平滑

### rpo_alpha

```python
rpo_alpha: Optional[float] = None
```

**作用：** RPO（Reward Penalized Optimization）的 alpha 参数，在 DPO 损失基础上添加 SFT 损失：

$$\mathcal{L} = \mathcal{L}_{DPO} + \alpha \cdot \mathcal{L}_{SFT}$$

### ld_alpha

```python
ld_alpha: Optional[float] = None
```

**作用：** LD-DPO（Length-biased DPO）的 alpha 参数，缓解长度偏好：

$$\mathcal{L} = \text{front\_logps} + \alpha \cdot \text{rear\_logps}$$

## 19.6 实际应用示例

```bash
# 标准 DPO
swift rlhf --rlhf_type dpo --loss_type sigmoid --beta 0.1 ...

# IPO
swift rlhf --rlhf_type dpo --loss_type ipo --beta 0.1 ...

# 混合 DPO + SFT
swift rlhf --rlhf_type dpo --loss_type sigmoid sft --loss_weights 0.8 0.2 ...

# 鲁棒 DPO
swift rlhf --rlhf_type dpo --loss_type robust --label_smoothing 0.1 ...
```

## 小结

- **要点 1**：DPO 是最流行的对齐算法，ms-swift 支持 14 种变体
- **要点 2**：sigmoid 是默认推荐，适合大多数场景
- **要点 3**：混合 loss type 可以结合多种优势
- **要点 4**：beta、label_smoothing、rpo_alpha 是关键超参数
