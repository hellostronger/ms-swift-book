# 第 18 章 RLHF 概论与算法对比

## 18.1 对齐技术演进

大语言模型的对齐技术经历了三个阶段：

```
阶段一：SFT（监督微调）
   └─ 用人类标注的对话数据训练模型
   └─ 问题：可能学到人类的偏见

阶段二：RLHF（人类反馈强化学习）
   └─ 训练奖励模型 + PPO 强化学习
   └─ 问题：训练不稳定，超参敏感

阶段三：DPO 及其变体
   └─ 直接在偏好数据上优化
   └─ 优势：训练稳定，实现简单
```

## 18.2 ms-swift 支持的 RLHF 算法

| 算法 | 类型 | 是否需要参考模型 | 是否需要奖励模型 | 适用场景 |
|------|------|-----------------|-----------------|---------|
| DPO | 离线偏好优化 | 是 | 否 | 通用对齐 |
| CPO | 离线偏好优化 | 否 | 否 | 无参考模型 |
| SimPO | 离线偏好优化 | 否 | 否 | 简化版 CPO |
| ORPO | 离线偏好优化 | 否 | 否 | 无需参考模型 |
| KTO | 离线偏好优化 | 是 | 否 | 二元反馈 |
| PPO | 在线强化学习 | 是 | 是 | 经典 RLHF |
| GRPO | 在线强化学习 | 是 | 否 | 分组相对优化 |
| GKD | 知识蒸馏 | 否 | 否 | 模型蒸馏 |
| RM | 奖励模型训练 | 否 | - | 训练奖励模型 |

## 18.3 算法选择指南

```
你的场景是什么？
│
├─ 有偏好数据（chosen/rejected）
│  │
│  ├─ 想要简单稳定
│  │  └─ 选择 DPO（默认推荐）
│  │
│  ├─ 不想加载参考模型
│  │  ├─ 选择 CPO
│  │  └─ 选择 SimPO（更简单）
│  │
│  └─ 数据是二元的（好/坏）
│     └─ 选择 KTO
│
├─ 只有评分数据（无对比）
│  └─ 选择 KTO
│
├─ 想要在线学习
│  ├─ 有奖励模型
│  │  └─ 选择 PPO
│  └─ 无奖励模型
│     └─ 选择 GRPO
│
└─ 想要知识蒸馏
   └─ 选择 GKD
```

## 18.4 核心概念

### 参考模型（Reference Model）

参考模型是 RLHF 中的关键组件，用于约束策略更新幅度：

$$\mathcal{L}_{KL} = \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

- DPO/KTO/PPO/GRPO 需要参考模型
- CPO/SimPO/ORPO 不需要参考模型

### 奖励模型（Reward Model）

奖励模型将人类偏好建模为标量分数：

$$r(x, y) = \text{RewardModel}(x, y)$$

- PPO 需要奖励模型
- DPO 系列算法不需要奖励模型

### 温度参数（Temperature）

控制策略更新的激进程度：

- 温度高：更新更平滑，探索更多
- 温度低：更新更保守，利用更多

默认值：`temperature = 0.9`（GRPO/PPO/GKD）

## 18.5 训练流程对比

### DPO 训练流程

```
1. 加载偏好数据 (chosen, rejected)
2. 加载策略模型和参考模型
3. 对于每个 batch:
   - 计算策略模型的 log probabilities
   - 计算参考模型的 log probabilities
   - 计算 DPO loss
   - 反向传播，更新策略模型
```

### PPO 训练流程

```
1. 加载 prompt 数据
2. 加载策略模型、参考模型、奖励模型、价值模型
3. 对于每个 batch:
   - 策略模型生成 responses
   - 奖励模型计算 rewards
   - 价值模型计算 advantages
   - 使用 PPO 算法更新策略模型
```

### GRPO 训练流程

```
1. 加载 prompt 数据
2. 加载策略模型和参考模型
3. 对于每个 batch:
   - 对每个 prompt 生成 G 个 responses
   - 计算每个 response 的 rewards
   - 在组内计算相对优势
   - 使用 GRPO 算法更新策略模型
```

## 18.6 性能对比

| 算法 | 训练速度 | 显存占用 | 稳定性 | 效果 |
|------|---------|---------|--------|------|
| DPO | 快 | 中（2倍模型） | 高 | 好 |
| CPO | 快 | 低（1倍模型） | 高 | 好 |
| SimPO | 快 | 低（1倍模型） | 高 | 好 |
| ORPO | 快 | 低（1倍模型） | 高 | 好 |
| KTO | 中 | 中（2倍模型） | 中 | 中 |
| PPO | 慢 | 高（4倍模型） | 低 | 最好 |
| GRPO | 中 | 中（2倍模型） | 中 | 好 |
| GKD | 中 | 中（2倍模型） | 高 | 好 |

## 18.7 实际使用示例

```bash
# DPO 训练
swift rlhf \
  --rlhf_type dpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset Anthropic/hh-rlhf \
  --beta 0.1 \
  --output_dir output/dpo

# GRPO 训练
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset openai/gsm8k \
  --reward_funcs accuracy format \
  --num_generations 8 \
  --beta 0.04 \
  --output_dir output/grpo

# CPO 训练（无需参考模型）
swift rlhf \
  --rlhf_type cpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset Anthropic/hh-rlhf \
  --cpo_alpha 1.0 \
  --output_dir output/cpo
```

## 小结

- **要点 1**：ms-swift 支持 9 种 RLHF 算法，覆盖离线/在线/蒸馏全部场景
- **要点 2**：DPO 是最简单稳定的选择，适合大多数场景
- **要点 3**：GRPO/PPO 适合在线学习，GRPO 不需要奖励模型
- **要点 4**：选择算法时考虑数据格式、计算资源、训练稳定性
