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

| Task Type | 默认损失 | 可选损失 | 输出格式 |
|-----------|---------|---------|---------|
| causal_lm | CrossEntropy | - | logits [B, S, V] |
| embedding | InfoNCE | Cosine, Contrastive | last_hidden_state [B, D] |
| reranker | Pointwise | Listwise | logits [B, 1] |
| generative_reranker | Listwise | - | logits [B, S] |

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
