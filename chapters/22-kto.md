# 第 22 章 KTO

## 22.1 KTO 核心原理

KTO（Kahneman-Tversky Optimization）是一种**基于二元反馈**的对齐方法，不需要成对的偏好数据，只需要"好/坏"标签。

**核心公式：**

$$\mathcal{L}_{KTO} = \begin{cases} 
1 - \sigma\left(\beta\left(\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} - D_{KL}\right)\right) & \text{if } y \text{ is desirable} \\
1 - \sigma\left(\beta\left(D_{KL} - \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right)\right) & \text{if } y \text{ is undesirable}
\end{cases}$$

其中：
- $D_{KL}$：KL 散度估计
- $\beta$：温度参数

**核心思想：**
- 基于前景理论（Prospect Theory）
- 人类对损失和收益的敏感度不同
- 只需要二元标签（好/坏），不需要成对数据

## 22.2 KTO vs DPO

| 维度 | DPO | KTO |
|------|-----|-----|
| 数据格式 | 成对（chosen/rejected） | 二元（好/坏） |
| 参考模型 | 需要 | 需要 |
| 数据收集 | 困难（需要对比） | 简单（只需标签） |
| 训练稳定性 | 高 | 中 |
| 适用场景 | 偏好对齐 | 二元反馈 |

## 22.3 实现细节

```python
# swift/rlhf_trainers/kto_trainer.py

class KTOTrainer(RLHFTrainerMixin, SwiftMixin, HFKTOTrainer):
    def __init__(self, model, ref_model, *_args, **kwargs):
        args = kwargs['args']
        args.disable_dropout = True
        self.desirable_weight = args.desirable_weight
        self.undesirable_weight = args.undesirable_weight
        
        # 计算 KL 散度
        self.calculate_KL = True
        if self.loss_type in ['apo_zero_unpaired']:
            self.calculate_KL = False
```

**数据格式：**

```json
{
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "label": true  // true: 好, false: 坏
}
```

## 22.4 关键超参数

### desirable_weight / undesirable_weight

```python
desirable_weight: float = 1.0
undesirable_weight: float = 1.0
```

**作用：** 平衡好/坏样本的权重，应对数据不平衡。

```python
loss = desirable_weight * desirable_loss + undesirable_weight * undesirable_loss
```

**调优建议：**
- 好样本多：降低 desirable_weight
- 坏样本多：降低 undesirable_weight

### beta

```python
# KTO 默认值
beta: Optional[float] = None  # 默认 0.1
```

### loss_type

```python
# 支持的 loss type
loss_type: Literal['kto', 'apo_zero_unpaired'] = 'kto'
```

- `kto`：标准 KTO 损失
- `apo_zero_unpaired`：APO 变体，不计算 KL

## 22.5 训练示例

```bash
# 标准 KTO
swift rlhf \
  --rlhf_type kto \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset binary_feedback_dataset \
  --desirable_weight 1.0 \
  --undesirable_weight 1.0 \
  --output_dir output/kto
```

## 22.6 KL 散度计算

KTO 需要估计 KL 散度：

```python
def _compute_kl_logps(self, model, batch):
    """计算 KL log probabilities"""
    KL_logps = None
    if self.calculate_KL:
        KL_model_kwargs, labels = self._get_model_kwargs(batch, 'KL_completion_')
        with torch.no_grad():
            KL_logits = model(**KL_model_kwargs).logits
        KL_logps, _ = self.get_batch_logps(KL_model_kwargs, KL_logits, labels)
    return KL_logps
```

**作用：** 估计当前策略与参考模型的差距，用于正则化。

## 小结

- **要点 1**：KTO 只需要二元反馈（好/坏），不需要成对数据
- **要点 2**：基于前景理论，人类对损失和收益的敏感度不同
- **要点 3**：desirable/undesirable_weight 用于平衡数据不平衡
- **要点 4**：需要计算 KL 散度，训练开销略大
