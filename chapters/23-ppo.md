# 第 23 章 PPO

## 23.1 PPO 核心原理

PPO（Proximal Policy Optimization）是经典的强化学习算法，用于 RLHF 的在线训练。

**核心公式：**

$$\mathcal{L}_{PPO} = -\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$：重要性采样比率
- $\hat{A}_t$：优势函数估计
- $\epsilon$：裁剪参数（通常 0.2）

**PPO 需要 4 个模型：**
1. **策略模型（Actor）**：生成 responses
2. **参考模型（Reference）**：约束策略更新
3. **奖励模型（Reward）**：评估 responses 质量
4. **价值模型（Critic）**：估计状态价值

## 23.2 PPO vs DPO vs GRPO

| 维度 | DPO | PPO | GRPO |
|------|-----|-----|------|
| 训练方式 | 离线 | 在线 | 在线 |
| 参考模型 | 需要 | 需要 | 需要 |
| 奖励模型 | 不需要 | 需要 | 不需要 |
| 价值模型 | 不需要 | 需要 | 不需要 |
| 显存占用 | 2倍模型 | 4倍模型 | 2倍模型 |
| 训练稳定性 | 高 | 低 | 中 |
| 效果 | 好 | 最好 | 好 |

## 23.3 关键超参数

### num_ppo_epochs

```python
num_ppo_epochs: int = 4
```

**作用：** 每个 batch 的 PPO 更新次数。

### kl_coef

```python
kl_coef: float = 0.05
```

**作用：** KL 散度系数，约束策略偏离参考模型。

$$\mathcal{L}_{KL} = \text{kl\_coef} \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

### vf_coef

```python
vf_coef: float = 0.1
```

**作用：** 价值函数损失系数。

### cliprange / cliprange_value

```python
cliprange: float = 0.2        # 策略裁剪范围
cliprange_value: float = 0.2  # 价值函数裁剪范围
```

### gamma / lam

```python
gamma: float = 1.0   # 折扣因子
lam: float = 0.95    # GAE lambda
```

**作用：** 计算优势函数时的超参数。

### whiten_rewards

```python
whiten_rewards: bool = False
```

**作用：** 是否对奖励进行白化处理。

## 23.4 训练流程

```
1. 加载 prompt 数据
2. 加载 4 个模型（actor, reference, reward, critic）
3. 对于每个 batch:
   a. Actor 生成 responses
   b. Reward 模型计算 rewards
   c. Critic 估计 values
   d. 计算 advantages (GAE)
   e. PPO 更新 actor 和 critic
   f. 同步 reference 模型
```

## 23.5 训练示例

```bash
swift rlhf \
  --rlhf_type ppo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --reward_model OpenAssistant/reward-model-deberta-v3-large-v2 \
  --dataset prompt_dataset \
  --num_ppo_epochs 4 \
  --kl_coef 0.05 \
  --output_dir output/ppo
```

## 23.6 PPO 的挑战

**显存占用大：**
- 需要同时加载 4 个模型
- 建议使用 DeepSpeed ZeRO-3 或 FSDP

**训练不稳定：**
- 超参数敏感
- 需要仔细调优

**实现复杂：**
- 需要处理 rollout、advantage 计算
- 需要多个模型的同步

## 23.7 何时使用 PPO

**适合 PPO 的场景：**
- 有高质量的奖励模型
- 有充足的计算资源
- 追求最优效果

**不适合 PPO 的场景：**
- 计算资源有限
- 没有奖励模型
- 希望简单稳定的训练

**替代方案：**
- 无奖励模型：使用 GRPO
- 资源有限：使用 DPO/CPO/SimPO

## 小结

- **要点 1**：PPO 是经典的在线强化学习算法，需要 4 个模型
- **要点 2**：显存占用大（4倍模型），训练不稳定
- **要点 3**：kl_coef、vf_coef、cliprange 是关键超参数
- **要点 4**：追求最优效果且有充足资源时使用 PPO，否则考虑 GRPO 或 DPO
