# 第 15 章 重排损失函数

## 15.1 概述

重排（Reranker）任务的目标是对候选文档进行排序。ms-swift 支持两种重排损失：

| 损失函数 | 原理 | 适用场景 |
|---------|------|---------|
| PointwiseReranker | 独立打分 | 简单排序 |
| ListwiseReranker | 组内对比 | 精确排序 |

## 15.2 PointwiseRerankerLoss

```python
class PointwiseRerankerLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        logits = outputs.logits.squeeze(1)
        labels = labels.to(logits.dtype)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return loss
```

**数学原理：**

使用二元交叉熵损失，将每个文档独立分类为相关（1）或不相关（0）：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\sigma(x_i)) + (1-y_i)\log(1-\sigma(x_i))\right]$$

**特点：**
- 简单直接，计算效率高
- 不考虑文档间的相对关系
- 适合二分类场景

## 15.3 ListwiseRerankerLoss ⭐

```python
class ListwiseRerankerLoss(BaseLoss):
    """列表级重排损失"""

    def __call__(self, outputs, labels, **kwargs):
        logits = outputs.logits.squeeze(-1)  # [batch_size]
        labels = labels.float()

        # 配置参数（通过环境变量）
        temperature = float(os.environ.get('LISTWISE_RERANKER_TEMPERATURE', '1.0'))
        min_group_size = int(os.environ.get('LISTWISE_RERANKER_MIN_GROUP_SIZE', '2'))

        # 找到正样本位置，确定分组边界
        positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)

        if len(positive_indices) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        total_loss = 0.0
        num_groups = 0

        for i, pos_idx in enumerate(positive_indices):
            # 确定当前组边界
            group_start = pos_idx.item()
            group_end = positive_indices[i + 1].item() if i + 1 < len(positive_indices) else len(labels)

            # 提取组内 logits
            group_logits = logits[group_start:group_end]
            group_labels = labels[group_start:group_end]

            # 跳过过小的组
            if len(group_logits) < min_group_size:
                continue

            # 温度缩放
            scaled_logits = group_logits / temperature

            # 正样本应在索引 0
            target = torch.tensor(0, dtype=torch.long, device=logits.device)

            # 交叉熵损失：正样本应获得最高分
            loss_fct = CrossEntropyLoss()
            group_loss = loss_fct(scaled_logits.unsqueeze(0), target.unsqueeze(0))

            total_loss += group_loss
            num_groups += 1

        return total_loss / num_groups if num_groups > 0 else torch.tensor(0.0, requires_grad=True)
```

**数学原理：**

将排序问题转化为分类问题，在每个查询的候选文档中，正样本应获得最高分：

$$\mathcal{L} = -\log \frac{\exp(s_{pos}/\tau)}{\sum_{i=1}^{N}\exp(s_i/\tau)}$$

**关键超参数：**

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| LISTWISE_RERANKER_TEMPERATURE | 1.0 | 温度参数 |
| LISTWISE_RERANKER_MIN_GROUP_SIZE | 2 | 最小组大小 |

**数据格式：**
```
labels: [1, 0, 0, 0, 1, 0, 0, ...]
         ↑ 第一组   ↑ 第二组
```

**特点：**
- 考虑文档间的相对关系
- 更符合排序任务的本质
- 计算复杂度略高

## 15.4 两种损失对比

| 维度 | Pointwise | Listwise |
|------|-----------|----------|
| 计算方式 | 独立打分 | 组内对比 |
| 考虑文档关系 | 否 | 是 |
| 训练效率 | 高 | 中 |
| 排序精度 | 一般 | 更高 |
| 适用场景 | 简单二分类 | 精确排序 |

## 15.5 实际应用

```bash
# Pointwise 重排
swift sft \
  --task_type reranker \
  --loss_type pointwise_reranker \
  --model BAAI/bge-reranker-v2-m3 \
  --dataset mteb/reranking \
  --output_dir output

# Listwise 重排
swift sft \
  --task_type generative_reranker \
  --loss_type listwise_reranker \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset mteb/reranking \
  --output_dir output
```

## 小结

- **要点 1**：Pointwise 独立打分，Listwise 组内对比
- **要点 2**：Listwise 更符合排序任务本质，精度更高
- **要点 3**：通过环境变量可配置 Listwise 的温度和最小组大小
