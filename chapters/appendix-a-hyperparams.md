# 附录 A：超参数速查表

## A.1 基础训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | 必填 | 模型名称或路径 |
| `dataset` | str/list | 必填 | 数据集名称或路径 |
| `output_dir` | str | 'output' | 输出目录 |
| `seed` | int | 42 | 随机种子 |
| `num_train_epochs` | int | 3 | 训练轮数 |
| `per_device_train_batch_size` | int | 1 | 训练批次大小 |
| `per_device_eval_batch_size` | int | 1 | 评估批次大小 |

## A.2 学习率参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `learning_rate` | float | full:1e-5, lora:1e-4 | 学习率 |
| `lr_scheduler_type` | str | 'cosine' | 学习率调度器 |
| `warmup_ratio` | float | 0.0 | 预热比例 |
| `warmup_steps` | int | 0 | 预热步数 |
| `weight_decay` | float | 0.0 | 权重衰减 |
| `max_grad_norm` | float | 1.0 | 梯度裁剪阈值 |

## A.3 优化器参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `optimizer` | str | 'adamw' | 优化器类型 |
| `lorap_lr_ratio` | float | 2.0**4 | LoRA+ 学习率比例 |
| `use_galore` | bool | False | 使用 GaLore 优化器 |
| `galore_rank` | int | 128 | GaLore 秩 |
| `galore_update_proj_gap` | int | 200 | GaLore 投影更新间隔 |

## A.4 微调参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tuner_type` | str | 'lora' | 微调类型 |
| `lora_rank` | int | 8 | LoRA 秩 |
| `lora_alpha` | int | 32 | LoRA alpha |
| `lora_dropout` | float | 0.05 | LoRA dropout |
| `target_modules` | list | None | 目标模块 |
| `modules_to_save` | list | None | 保留训练的模块 |

## A.5 RLHF 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rlhf_type` | str | 'dpo' | RLHF 算法类型 |
| `ref_model` | str | None | 参考模型 |
| `beta` | float | DPO:0.1, GRPO:0.04, GKD:0.5 | 温度参数 |
| `label_smoothing` | float | 0.0 | 标签平滑 |
| `max_completion_length` | int | 512 | 最大生成长度 |
| `temperature` | float | 0.9 | 采样温度 |

## A.6 DPO 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `loss_type` | list | ['sigmoid'] | DPO 损失类型 |
| `loss_weights` | list | None | 损失权重 |
| `rpo_alpha` | float | None | RPO alpha 参数 |
| `ld_alpha` | float | None | LD-DPO alpha 参数 |
| `discopop_tau` | float | 0.05 | DiscoPOP tau 参数 |

## A.7 CPO/SimPO 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cpo_alpha` | float | 1.0 | CPO NLL 损失权重 |
| `simpo_gamma` | float | 1.0 | SimPO margin |

## A.8 KTO 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `desirable_weight` | float | 1.0 | 正样本权重 |
| `undesirable_weight` | float | 1.0 | 负样本权重 |

## A.9 GRPO 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_generations` | int | 8 | 每个 prompt 生成数 |
| `reward_funcs` | list | [] | 奖励函数列表 |
| `reward_weights` | list | None | 奖励权重 |
| `advantage_estimator` | str | 'grpo' | 优势估计器 |
| `scale_rewards` | str | 'group' | 奖励缩放方式 |
| `kl_in_reward` | bool | False | 是否在奖励中包含 KL |

## A.10 GKD 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `teacher_model` | str | None | 教师模型 |
| `sft_alpha` | float | 0.0 | SFT 损失权重 |
| `lmbda` | float | 0.5 | JSD lambda 参数 |
| `gkd_logits_topk` | int | None | 仅保留 top-k logits |

## A.11 PPO 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_ppo_epochs` | int | 4 | PPO 更新轮数 |
| `kl_coef` | float | 0.05 | KL 系数 |
| `vf_coef` | float | 0.1 | 价值函数系数 |
| `cliprange` | float | 0.2 | 裁剪范围 |
| `gamma` | float | 1.0 | 折扣因子 |
| `lam` | float | 0.95 | GAE lambda |

## A.12 分布式训练参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `deepspeed` | str | None | DeepSpeed 配置 |
| `zero_hpz_partition_size` | int | None | ZeRO++ 分区大小 |
| `fsdp` | str | None | FSDP 配置 |
| `gradient_accumulation_steps` | int | 1 | 梯度累积步数 |

## A.13 混合精度参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bf16` | bool | False | 使用 bf16 |
| `fp16` | bool | False | 使用 fp16 |
| `fp16_opt_level` | str | 'O1' | FP16 优化级别 |

## A.14 评估参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `eval_strategy` | str | 'no' | 评估策略 |
| `eval_steps` | int | None | 评估间隔 |
| `metric_for_best_model` | str | 'loss' | 最佳模型指标 |
| `greater_is_better` | bool | False | 指标是否越大越好 |

## A.15 部署参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `infer_backend` | str | 'transformers' | 推理后端 |
| `tensor_parallel_size` | int | 1 | 张量并行大小 |
| `gpu_memory_utilization` | float | 0.9 | GPU 显存利用率 |
| `max_model_len` | int | None | 最大模型长度 |
| `temperature` | float | 0.7 | 采样温度 |
| `top_p` | float | 0.9 | Top-p 采样 |
| `top_k` | int | 50 | Top-k 采样 |

## A.16 参数推荐值速查

### 学习率推荐

| 训练方式 | 推荐学习率 |
|---------|-----------|
| full | 1e-5 ~ 5e-5 |
| lora | 1e-4 ~ 3e-4 |
| qlora | 1e-4 ~ 2e-4 |
| dpo | 1e-6 ~ 5e-6 |
| grpo | 1e-6 ~ 1e-5 |

### beta 推荐值

| 算法 | 推荐值 |
|------|--------|
| DPO | 0.05 ~ 0.2 |
| CPO | 0.05 ~ 0.2 |
| SimPO | 1.0 ~ 3.0 |
| ORPO | 0.05 ~ 0.2 |
| KTO | 0.05 ~ 0.2 |
| PPO | 0.01 ~ 0.1 |
| GRPO | 0.01 ~ 0.1 |
| GKD | 0.3 ~ 0.7 |

### 温度推荐值

| 场景 | 推荐值 |
|------|--------|
| 确定性任务 | 0.0 |
| 创意写作 | 0.9 |
| 代码生成 | 0.2 |
| 对话 | 0.7 |
