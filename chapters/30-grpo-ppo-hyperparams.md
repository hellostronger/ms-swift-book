# 第 30 章 GRPO 与 PPO 超参数

## 30.1 num_generations

num_generations 是 GRPO 的核心参数，控制每个 prompt 生成的响应数量：

```python
num_generations: int = 8  # GRPO 论文中的 G 值
```

### 推荐值

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 资源充足 | 8 ~ 16 | 更稳定的梯度估计 |
| 资源有限 | 4 ~ 8 | 平衡效果和效率 |
| 快速实验 | 2 ~ 4 | 快速验证想法 |

### 计算约束

```python
# 生成 batch size 必须能被 num_generations 整除
generation_batch_size = per_device_batch_size * num_processes
assert generation_batch_size % num_generations == 0
```

示例：
- per_device_batch_size = 2
- num_processes = 4
- num_generations = 8
- generation_batch_size = 8 ✓

## 30.2 reward_funcs 和 reward_weights

GRPO 使用规则奖励函数，而非学习的奖励模型：

```python
reward_funcs: List[str] = field(default_factory=list)
reward_weights: List[float] = None

# 默认奖励函数
# 'accuracy': 数学准确性
# 'format': 格式检查
# 'cosine': 余弦奖励（长度惩罚）
# 'repetition': 重复惩罚
# 'soft_overlong': 软长度限制
```

### 使用示例

```bash
swift rlhf \
  --rlhf_type grpo \
  --reward_funcs accuracy format repetition \
  --reward_weights 1.0 0.5 -0.1 \
  ...
```

### 奖励权重设计

| 奖励函数 | 正权重 | 负权重 | 说明 |
|---------|--------|--------|------|
| accuracy | 1.0 | - | 鼓励正确答案 |
| format | 0.5 | - | 鼓励正确格式 |
| repetition | - | -0.1 | 惩罚重复内容 |
| soft_overlong | - | -0.05 | 惩罚过长响应 |

## 30.3 advantage_estimator

advantage_estimator 控制优势函数的计算方式：

```python
advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus'] = 'grpo'
```

### 三种估计器

#### GRPO（默认）

```python
# 组内标准化
advantages = (rewards - group_mean) / (group_std + epsilon)
```

特点：
- 在每个 prompt 的 G 个响应内标准化
- 计算简单，效果稳定
- 推荐首选

#### RLOO（REINFORCE Leave-One-Out）

```python
# 留一法估计
if advantage_estimator == 'rloo' and K > 1:
    advantages = rewards * K / (K - 1) - group_mean * K / (K - 1)
```

特点：
- 方差更低
- 计算稍复杂
- 适合小样本

#### REINFORCE++

```python
# 批次或组内标准化
if scale_rewards == 'batch':
    std = advantages.std().expand_as(advantages)
elif scale_rewards == 'group':
    std = advantages.view(-1, K).std(dim=1).repeat_interleave(K)
```

特点：
- 更灵活的标准化
- 适合特定场景

## 30.4 scale_rewards

scale_rewards 控制奖励的标准化方式：

```python
scale_rewards: Optional[str] = None  # 'batch', 'group', 'none', 'gdpo'

# 默认值
if self.scale_rewards is None:
    if self.advantage_estimator == 'grpo':
        self.scale_rewards = 'group'
    elif self.advantage_estimator == 'rloo':
        self.scale_rewards = 'none'
    elif self.advantage_estimator == 'reinforce_plus_plus':
        self.scale_rewards = 'batch'
```

### 四种模式

| 模式 | 说明 | 适用场景 |
|------|------|---------|
| group | 组内标准化 | GRPO（默认） |
| batch | 批次标准化 | REINFORCE++ |
| none | 不标准化 | RLOO |
| gdpo | GDPO 标准化 | 特殊场景 |

## 30.5 kl_in_reward

kl_in_reward 控制是否在奖励中减去 KL 散度：

```python
kl_in_reward: Optional[bool] = None

# 默认值
if self.kl_in_reward is None:
    if self.advantage_estimator == 'grpo':
        self.kl_in_reward = False
    elif self.advantage_estimator in ['rloo', 'reinforce_plus_plus']:
        self.kl_in_reward = True
```

### 数学意义

```python
# kl_in_reward = True
rewards = rewards - beta * kl_values

# kl_in_reward = False
# rewards 保持不变
```

推荐：
- GRPO：False（默认）
- RLOO/REINFORCE++：True（默认）

## 30.6 loss_type

GRPO 支持不同的损失类型：

```python
loss_type: Literal['grpo', 'real'] = 'grpo'

# REAL 损失特殊处理
if self.loss_type == 'real':
    self.scale_rewards = 'none'
    logger.warning("scale_rewards='none' is forced for 'real' loss type.")
```

### GRPO vs REAL

| 损失类型 | 奖励标准化 | 适用场景 |
|---------|-----------|---------|
| grpo | 支持标准化 | 标准 GRPO |
| real | 强制不标准化 | REAL 论文方法 |

## 30.7 num_iterations

num_iterations 控制每个样本的更新次数：

```python
num_iterations: int = 1  # GRPO 论文中的 K 值
```

### 推荐值

| 场景 | 推荐值 | 说明 |
|------|--------|------|
| 标准 | 1 | 每个样本更新一次 |
| 多轮更新 | 2 ~ 4 | 更充分利用数据 |

注意：过大的 num_iterations 可能导致过拟合。

## 30.8 truncation_strategy

truncation_strategy 控制超长序列的处理：

```python
truncation_strategy: Literal['delete', 'left', 'right', 'split', None] = None

# GRPO 默认值
if self.truncation_strategy is None:
    self.truncation_strategy = 'left'
```

### 四种策略

| 策略 | 说明 | 适用场景 |
|------|------|---------|
| delete | 删除超长样本 | 多模态模型 |
| left | 左侧截断 | 标准场景（默认） |
| right | 右侧截断 | 特殊需求 |
| split | 分割为多个样本 | 长文本 |

注意：多模态模型建议使用 `delete`，避免截断图像 token。

## 30.9 vLLM 相关参数

GRPO 支持使用 vLLM 加速生成：

```python
use_vllm: bool = False
vllm_mode: Optional[str] = None  # 'colocal' or 'server'
vllm_server_host: Optional[str] = None
vllm_server_port: int = 8000
```

### 使用示例

```bash
# 内部 vLLM
swift rlhf \
  --rlhf_type grpo \
  --use_vllm true \
  --vllm_mode colocal \
  ...

# 外部 vLLM 服务器
swift rlhf \
  --rlhf_type grpo \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host localhost \
  --vllm_server_port 8000 \
  ...
```

## 30.10 async_generate

async_generate 启用异步生成：

```python
async_generate: bool = False
```

### 使用约束

```python
if self.async_generate:
    assert self.vllm_mode == 'server', 'async generate requires vllm_mode=server'
    logger.info('Using async mode. This will use old weights to generate responses.')
```

特点：
- 必须使用外部 vLLM 服务器
- 使用旧权重生成，可能影响稳定性
- 加速训练流程

## 30.11 完整 GRPO 训练示例

```bash
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --ref_model Qwen/Qwen2.5-7B \
  --dataset openai/gsm8k \
  --tuner_type lora \
  --reward_funcs accuracy format repetition \
  --reward_weights 1.0 0.5 -0.1 \
  --num_generations 8 \
  --beta 0.04 \
  --temperature 0.9 \
  --advantage_estimator grpo \
  --scale_rewards group \
  --kl_in_reward false \
  --loss_type grpo \
  --num_iterations 1 \
  --truncation_strategy left \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host localhost \
  --vllm_server_port 8000 \
  --learning_rate 1e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --gradient_accumulation_steps 1 \
  --bf16 true \
  --num_train_epochs 1 \
  --output_dir output/qwen2_5-7b-grpo
```

## 小结

- **要点 1**：num_generations 控制每个 prompt 的生成数量，推荐 4-16
- **要点 2**：reward_funcs 和 reward_weights 定义奖励函数及其权重
- **要点 3**：advantage_estimator 控制优势计算方式，GRPO 是默认选择
- **要点 4**：vLLM 集成可以显著加速生成过程
