# 第 10 章 训练流水线

## 10.1 Pipeline 架构

ms-swift 的训练流水线通过 Pipeline 模式组织：

```
swift/pipelines/
├── base.py          # 基础 Pipeline
├── train/           # 训练 Pipeline
│   ├── sft.py       # SFT 训练
│   └── rlhf.py      # RLHF 训练
├── infer/           # 推理 Pipeline
├── eval/            # 评估 Pipeline
├── export/          # 导出 Pipeline
├── app/             # 应用 Pipeline
└── sampling/        # 采样 Pipeline
```

## 10.2 SFT Pipeline

```python
class SwiftSft(SwiftPipeline):
    """SFT 训练流水线"""
    
    def run(self):
        # 1. 解析参数
        args = self.args
        
        # 2. 加载模型和 tokenizer
        model, tokenizer = load_model_and_tokenizer(args)
        
        # 3. 准备数据集
        train_dataset, eval_dataset = self._prepare_dataset(args, tokenizer)
        
        # 4. 创建 Trainer
        trainer = self._create_trainer(args, model, tokenizer, train_dataset, eval_dataset)
        
        # 5. 开始训练
        trainer.train()
        
        # 6. 保存模型
        trainer.save_model()
    
    def _prepare_dataset(self, args, tokenizer):
        # 加载数据集
        dataset = load_dataset(args.dataset)
        
        # 应用模板
        template = get_template(args.template, tokenizer)
        
        # 编码
        encoded = [template.encode(sample) for sample in dataset]
        
        # 划分训练集/验证集
        return split_dataset(encoded, args.split_dataset_ratio)
```

## 10.3 RLHF Pipeline

```python
class SwiftRLHF(SwiftSft):
    """RLHF 训练流水线"""
    
    def run(self):
        args = self.args
        
        # 1. 根据 rlhf_type 选择 Trainer
        trainer_cls = TrainerFactory.get_trainer(args.rlhf_type)
        
        # 2. 加载参考模型（DPO/KTO/PPO/GRPO 需要）
        ref_model = None
        if args.rlhf_type in ['dpo', 'kto', 'ppo', 'grpo']:
            ref_model = load_ref_model(args)
        
        # 3. 加载奖励模型（PPO 需要）
        reward_model = None
        if args.rlhf_type == 'ppo':
            reward_model = load_reward_model(args)
        
        # 4. 准备偏好数据
        train_dataset = self._prepare_preference_data(args)
        
        # 5. 创建 Trainer
        trainer = trainer_cls(
            model=model,
            ref_model=ref_model,
            reward_model=reward_model,
            args=args,
            train_dataset=train_dataset,
        )
        
        # 6. 训练
        trainer.train()
```

## 10.4 训练循环详解

```
┌────────────────────────────────────────────┐
│              训练循环                        │
│                                            │
│  for epoch in epochs:                      │
│    for batch in dataloader:                │
│      ┌──────────────────────┐              │
│      │ 1. Forward           │              │
│      │    outputs = model(  │              │
│      │      input_ids,      │              │
│      │      attention_mask, │              │
│      │      labels)         │              │
│      └──────────┬───────────┘              │
│                 ↓                          │
│      ┌──────────────────────┐              │
│      │ 2. Compute Loss      │              │
│      │    loss = loss_fn(   │              │
│      │      outputs, labels)│              │
│      └──────────┬───────────┘              │
│                 ↓                          │
│      ┌──────────────────────┐              │
│      │ 3. Backward          │              │
│      │    loss.backward()   │              │
│      └──────────┬───────────┘              │
│                 ↓                          │
│      ┌──────────────────────┐              │
│      │ 4. Update            │              │
│      │    optimizer.step()  │              │
│      │    scheduler.step()  │              │
│      └──────────┬───────────┘              │
│                 ↓                          │
│      ┌──────────────────────┐              │
│      │ 5. Log & Evaluate    │              │
│      └──────────────────────┘              │
└────────────────────────────────────────────┘
```

## 小结

- **要点 1**：Pipeline 模式组织训练流程，SFT 和 RLHF 有不同的 Pipeline
- **要点 2**：RLHF Pipeline 额外处理参考模型和奖励模型的加载
- **要点 3**：训练循环遵循标准的 forward → loss → backward → update 流程
