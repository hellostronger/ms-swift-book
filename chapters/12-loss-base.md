# 第 12 章 BaseLoss 基础架构

## 12.1 损失函数体系概览

ms-swift 的损失函数体系分为三个层次：

```
swift/loss/
├── base.py         # BaseLoss 抽象基类
├── causal_lm.py    # 语言模型损失 (CustomCrossEntropyLoss)
├── embedding.py    # 嵌入学习损失 (5种)
├── reranker.py     # 重排损失 (2种)
└── mapping.py      # 损失函数注册表
```

## 12.2 BaseLoss 抽象基类

```python
# swift/loss/base.py

class BaseLoss(ABC):
    """损失函数抽象基类
    
    所有自定义损失函数都应继承此基类并实现 __call__ 方法。
    """
    
    def __init__(self, args: 'TrainingArguments', trainer: 'Trainer'):
        self.args = args
        self.trainer = trainer
        
        # 判断是否为 Megatron 训练
        mro_class_names = [cls.__name__ for cls in trainer.__class__.__mro__]
        self.is_megatron = 'BaseMegatronTrainer' in mro_class_names
    
    @abstractmethod
    def __call__(self, outputs, labels, *, num_items_in_batch=None, 
                 loss_scale=None, **kwargs) -> torch.Tensor:
        """计算损失值
        
        Args:
            outputs: 模型输出
            labels: 标签
            num_items_in_batch: 批次中的 token 数（用于归一化）
            loss_scale: 损失缩放因子
            
        Returns:
            标量损失张量
        """
        pass
```

## 12.3 损失函数注册表

```python
# swift/loss/mapping.py

loss_map = {
    # 语言模型
    'cross_entropy': CustomCrossEntropyLoss,
    
    # 嵌入学习
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    
    # 重排
    'pointwise_reranker': PointwiseRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
}
```

## 12.4 损失函数选择流程

```
task_type
  ├── causal_lm → CustomCrossEntropyLoss
  ├── embedding → 根据 loss_type 选择
  │     ├── cosine_similarity
  │     ├── contrastive
  │     ├── online_contrastive
  │     └── infonce
  ├── reranker → PointwiseRerankerLoss
  └── generative_reranker → ListwiseRerankerLoss
```

## 12.5 自定义损失函数

开发者可以注册自定义损失函数：

```python
from swift.loss import BaseLoss, loss_map

class MyCustomLoss(BaseLoss):
    def __call__(self, outputs, labels, **kwargs):
        logits = outputs.logits
        # 自定义损失计算
        loss = my_loss_fn(logits, labels)
        return loss

# 注册
loss_map['my_loss'] = MyCustomLoss
```

然后在训练时指定：

```bash
swift sft --loss_type my_loss ...
```

## 12.6 损失函数与 Task Type 的关系

`--task_type` 与 `--loss_type` 是**两个正交开关**：`task_type` 决定走哪条数据管线和评估指标，`loss_type` 决定用哪个损失函数。二者没有强制绑定，框架也不校验组合合法性——**配错了要到运行时才会报错**，因此对照下表使用：

| Task Type | 数据形态 | 常用 loss_type | 模型前向输出 | 默认评估指标 |
|-----------|---------|---------------|-------------|------------|
| `causal_lm` | 单条 messages | `cross_entropy` | logits `[B, S, V]` | `nlg` / `acc` |
| `seq_cls` | messages + label | `loss_type=None`（走模型自带 loss） | logits `[B, num_labels]` | `acc` |
| `embedding` | anchor + positive（±negative） | `infonce` | `last_hidden_state` `[B, D]` | `infonce` 或 `paired` |
| `reranker` | query + positive + negative | `pointwise_reranker` | logits `[B, 1]` | `reranker` |
| `generative_reranker` | query + positive + negative | `pointwise_reranker` | logits `[B, S]` → 取末位 | `reranker` |

两点补充：

- **`embedding` 家族有 4 个可选损失**（`infonce` / `cosine_similarity` / `contrastive` / `online_contrastive`），其中只有 `infonce` 的评估指标会自动切到 `infonce`，其余三个共用 `paired`。详见[第 14 章](/chapters/14-loss-embedding)。
- **`generative_reranker` 的 logits 会被二次加工**：`RerankerTrainer` 先取最后一个有效 token 的位置，再由 `get_generative_reranker_logits` 把词表投影压缩成 `P(yes) - P(no)` 的标量差。详见[第 15 章](/chapters/15-loss-reranker)。

## 12.7 num_items_in_batch 的作用

在梯度累积场景下，`num_items_in_batch` 用于正确归一化损失：

```python
# 标准做法
token_loss = per_token_loss_func(outputs, labels)
if num_items_in_batch is None:
    num_items_in_batch = (labels[:, 1:] != -100).sum()
return token_loss.sum() / num_items_in_batch
```

这确保了不同 batch 大小下的损失值一致性。

## 小结

- **要点 1**：BaseLoss 定义了统一的损失函数接口，支持 Megatron 适配
- **要点 2**：loss_map 注册表支持 7 种内置损失函数
- **要点 3**：num_items_in_batch 确保梯度累积时的正确归一化
