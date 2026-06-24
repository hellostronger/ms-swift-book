# 第 13 章 CrossEntropyLoss 详解

## 13.1 CustomCrossEntropyLoss 实现

ms-swift 的 CrossEntropyLoss 是对标准交叉熵的定制实现，支持 token 级别的损失计算。

```python
# swift/loss/causal_lm.py

class CustomCrossEntropyLoss(BaseLoss):

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        from swift.trainers import per_token_loss_func
        
        # 计算每个 token 的损失
        token_loss = per_token_loss_func(outputs, labels)
        
        # 计算有效 token 数（排除 padding 和 -100 位置）
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        
        # 归一化
        return token_loss.sum() / num_items_in_batch
```

## 13.2 数学原理

交叉熵损失的标准形式：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\log p(y_i|x_i)$$

在语言模型中，对每个 token 位置计算：

$$\mathcal{L}_{token} = -\log \frac{\exp(z_{y_t})}{\sum_{j=1}^{V}\exp(z_j)}$$

其中 $z_{y_t}$ 是正确 token 的 logits，$V$ 是词表大小。

## 13.3 与标准 CrossEntropyLoss 的区别

| 特性 | PyTorch CE | Custom CE |
|------|-----------|-----------|
| 归一化方式 | 按 batch 平均 | 按有效 token 数归一化 |
| 支持 loss_scale | 否 | 是 |
| 梯度累积 | 需手动处理 | 自动处理 |
| Padding 处理 | 需设置 ignore_index | 自动忽略 -100 |

## 13.4 应用场景

**场景一：标准 SFT 训练**
```bash
swift sft --loss_type cross_entropy ...
```

**场景二：Agent 训练（配合 loss_scale）**
```bash
swift sft --loss_type cross_entropy --loss_scale react ...
```
此时，Agent 的 Thought/Final Answer 部分 loss_scale=1.0，Action/Action Input 部分 loss_scale=2.0，Observation 部分 loss_scale=0.0。

**场景三：嵌入学习**
```bash
swift sft --task_type embedding --loss_type infonce ...
```

## 13.5 性能优化

ms-swift 的 CustomCrossEntropyLoss 支持：

1. **Liger Kernel 加速**：通过 `--use_liger_kernel true` 启用
2. **Flash Attention 集成**：与 FlashAttention 协同工作
3. **Sequence Parallel**：支持长序列并行训练

## 小结

- **要点 1**：CustomCrossEntropyLoss 支持 token 级别损失计算和自动归一化
- **要点 2**：与 loss_scale 系统深度集成，支持细粒度损失控制
- **要点 3**：适用于标准 SFT、Agent 训练、嵌入学习等多种场景
