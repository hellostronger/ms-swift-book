# 第 9 章 Trainer 架构

## 9.1 Trainer 类层次

```
HfTrainer (transformers)
   ↓
Trainer (swift/trainers/trainer.py)
   ├── EmbeddingTrainer      # 嵌入模型训练
   ├── RerankerTrainer       # 重排模型训练
   └── Seq2SeqTrainer        # 序列到序列训练
   ↓
RLHFTrainerMixin
   ├── DPOTrainer            # DPO 训练
   ├── CPOTrainer            # CPO 训练
   ├── ORPOTrainer           # ORPO 训练
   ├── KTOTrainer            # KTO 训练
   ├── PPOTrainer            # PPO 训练
   ├── GRPOTrainer           # GRPO 训练
   ├── GKDTrainer            # GKD 知识蒸馏
   └── RewardTrainer         # 奖励模型训练
```

## 9.2 核心 Trainer 实现

```python
# swift/trainers/trainer.py

class Trainer(SwiftMixin, DataLoaderMixin, HfTrainer):
    """ms-swift 基础训练器"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        
        # 计算准确率
        if inputs.get('labels') is not None:
            self._compute_acc(outputs, inputs['labels'])
        
        # 梯度累积缩放
        if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
            loss = loss / self.args.gradient_accumulation_steps
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, *args, **kwargs):
        # 应用模板上下文
        with self.template.forward_context(self.model, inputs):
            return super().training_step(model, inputs, *args, **kwargs)
```

## 9.3 SwiftMixin

SwiftMixin 提供 ms-swift 特有的训练功能：

```python
class SwiftMixin:
    """ms-swift 训练器混入类"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 自定义损失函数
        self.compute_loss_func = kwargs.get('compute_loss_func')
        
        # 自定义指标
        self.custom_metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
    
    def _compute_acc(self, outputs, labels):
        """计算 token 级别准确率"""
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
        preds = logits.argmax(dim=-1)
        
        # 忽略 padding 和 -100 位置
        mask = labels != -100
        acc = (preds[mask] == labels[mask]).float().mean()
        
        self.custom_metrics['train']['token_acc'].update(acc)
```

## 9.4 自定义损失函数

ms-swift 支持通过 `compute_loss_func` 注入自定义损失：

```python
class EmbeddingTrainer(Trainer):
    """嵌入模型训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Matryoshka 表示学习
        mrl_dims = self.args.mrl_dims
        if mrl_dims and self.compute_loss_func is not None:
            origin_loss_func = self.compute_loss_func
            
            def mrl_loss_func(outputs, labels, **kwargs):
                last_hidden_state = outputs['last_hidden_state']
                loss = None
                for dim, weight in mrl_dims.items():
                    if dim > last_hidden_state.shape[-1]:
                        continue
                    sliced = F.normalize(last_hidden_state[..., :dim], p=2, dim=-1)
                    cur_loss = weight * origin_loss_func(
                        {'last_hidden_state': sliced}, labels, **kwargs)
                    loss = cur_loss if loss is None else loss + cur_loss
                return loss
            
            self.compute_loss_func = mrl_loss_func
```

## 9.5 TrainerFactory

```python
# swift/trainers/trainer_factory.py

class TrainerFactory:
    """训练器工厂"""
    
    TRAINER_MAPPING = {
        'causal_lm': Trainer,
        'seq_cls': Seq2SeqTrainer,
        'embedding': EmbeddingTrainer,
        'reranker': RerankerTrainer,
        'generative_reranker': RerankerTrainer,
        'reward_model': RewardTrainer,
        # RLHF
        'dpo': DPOTrainer,
        'cpo': CPOTrainer,
        'orpo': ORPOTrainer,
        'kto': KTOTrainer,
        'ppo': PPOTrainer,
        'grpo': GRPOTrainer,
        'gkd': GKDTrainer,
    }
    
    @classmethod
    def get_trainer(cls, task_type: str) -> type:
        return cls.TRAINER_MAPPING.get(task_type, Trainer)
```

## 小结

- **要点 1**：Trainer 采用 Mixin 模式，SwiftMixin 提供 ms-swift 特有功能
- **要点 2**：TrainerFactory 根据任务类型返回对应的训练器
- **要点 3**：支持自定义损失函数注入，EmbeddingTrainer 实现了 Matryoshka 学习
