# 第 24 章 GRPO

## 24.1 GRPO 核心原理

GRPO（Group Relative Policy Optimization）是 DeepSeek 提出的在线对齐算法，**不需要奖励模型**，使用组内相对优势进行优化。

**核心公式：**

$$\mathcal{L}_{GRPO} = -\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\min\left(r_i(\theta)\hat{A}_i, \text{clip}(r_i(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_i\right)\right]$$

其中：
- $G$：每个 prompt 生成的 response 数量（num_generations）
- $r_i(\theta)$：重要性采样比率
- $\hat{A}_i$：组内相对优势

**优势计算：**

$$\hat{A}_i = \frac{r_i - \text{mean}(r)}{\text{std}(r) + \epsilon}$$

其中 $r_i$ 是第 $i$ 个 response 的奖励，在组内进行标准化。

## 24.2 GRPO vs PPO

| 维度 | PPO | GRPO |
|------|-----|------|
| 奖励模型 | 需要 | 不需要 |
| 价值模型 | 需要 | 不需要 |
| 显存占用 | 4倍模型 | 2倍模型 |
| 训练稳定性 | 低 | 中 |
| 实现复杂度 | 高 | 中 |
| 效果 | 最好 | 好 |

## 24.3 奖励函数

GRPO 使用规则奖励函数，而非学习的奖励模型：

```python
# swift/rewards/orm.py

orms = {
    'accuracy': MathAccuracy,       # 数学准确性
    'format': Format,               # 格式检查
    'cosine': CosineReward,         # 余弦奖励
    'repetition': RepetitionPenalty,# 重复惩罚
    'soft_overlong': SoftOverlong,  # 软长度限制
}
```

**使用示例：**

```bash
swift rlhf \
  --rlhf_type grpo \
  --reward_funcs accuracy format \
  --reward_weights 1.0 0.5 \
  ...
```

## 24.4 关键超参数

### num_generations

```python
num_generations: int = 8  # G in the GRPO paper
```

**作用：** 每个 prompt 生成的 response 数量。
- 更大：优势估计更准确，但计算成本高
- 默认 8，推荐 4-16

### beta

```python
# GRPO 默认值
beta: Optional[float] = 0.04
```

**作用：** KL 散度系数。

### advantage_estimator

```python
advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus'] = 'grpo'
```

**三种估计器：**
- `grpo`：标准 GRPO
- `rloo`：REINFORCE Leave-One-Out
- `reinforce_plus_plus`：REINFORCE++

### scale_rewards

```python
scale_rewards: Optional[str] = None  # 'batch', 'group', 'none', 'gdpo'
```

**作用：** 奖励标准化方式。
- `group`：组内标准化（默认）
- `batch`：批内标准化
- `none`：不标准化
- `gdpo`：GDPO 标准化

### loss_type

```python
loss_type: Literal['grpo', 'real'] = 'grpo'
```

- `grpo`：标准 GRPO 损失
- `real`：REAL 损失（https://arxiv.org/abs/2602.05630）

## 24.5 奖励函数详解

### MathAccuracy

```python
class MathAccuracy(ORM):
    """数学准确性奖励"""
    
    def __call__(self, completions, solution, **kwargs):
        # 解析答案
        content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
        # 验证数学表达式
        gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
        answer_parsed = parse(content_to_parse, ...)
        reward = float(verify(gold_parsed, answer_parsed))
        return reward
```

### Format

```python
class Format(ORM):
    """格式检查奖励"""
    
    def __call__(self, completions, **kwargs):
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, ...) for content in completions]
        return [1.0 if match else 0.0 for match in matches]
```

### CosineReward

```python
class CosineReward(ORM):
    """余弦奖励（长度惩罚）"""
    
    def __call__(self, completions, solution, **kwargs):
        # 根据准确性和长度计算奖励
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
```

### RepetitionPenalty

```python
class RepetitionPenalty(ORM):
    """重复惩罚"""
    
    def __call__(self, completions, **kwargs):
        # 计算 n-gram 重复率
        ngrams = set()
        for ng in self.zipngram(completion, self.ngram_size):
            ngrams.add(ng)
        scaling = 1 - len(ngrams) / total
        reward = scaling * self.max_penalty
```

### SoftOverlong

```python
class SoftOverlong(ORM):
    """软长度限制"""
    
    def __call__(self, completions, **kwargs):
        completion_length = len(ids)
        expected_len = self.soft_max_length - self.soft_cache_length
        exceed_len = completion_length - expected_len
        rewards.append(min(-exceed_len / self.soft_cache_length, 0))
```

## 24.6 训练流程

```
1. 加载 prompt 数据
2. 加载策略模型和参考模型
3. 对于每个 batch:
   a. 对每个 prompt 生成 G 个 responses
   b. 计算每个 response 的奖励（使用规则函数）
   c. 在组内计算相对优势（标准化）
   d. 计算 GRPO 损失
   e. 反向传播，更新策略模型
```

## 24.7 训练示例

```bash
# 基础 GRPO
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset openai/gsm8k \
  --reward_funcs accuracy format \
  --reward_weights 1.0 0.5 \
  --num_generations 8 \
  --beta 0.04 \
  --output_dir output/grpo

# 使用 vLLM 加速生成
swift rlhf \
  --rlhf_type grpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset openai/gsm8k \
  --reward_funcs accuracy \
  --use_vllm true \
  --vllm_mode server \
  --output_dir output/grpo_vllm
```

## 24.8 vLLM 集成

GRPO 支持使用 vLLM 加速 response 生成：

```bash
# 内部 vLLM 模式
--use_vllm true --vllm_mode colocal

# 外部 vLLM 服务器模式
--use_vllm true --vllm_mode server --vllm_server_host localhost --vllm_server_port 8000
```

## 24.9 异步生成

```bash
# 启用异步生成
swift rlhf \
  --rlhf_type grpo \
  --async_generate true \
  --vllm_mode server \
  ...
```

**注意：** 异步生成使用旧权重生成 responses，可能影响训练稳定性。

## 小结

- **要点 1**：GRPO 不需要奖励模型，使用规则奖励函数
- **要点 2**：num_generations 控制每个 prompt 的生成数量，默认 8
- **要点 3**：支持 5 种内置奖励函数（accuracy, format, cosine, repetition, soft_overlong）
- **要点 4**：可与 vLLM 集成，加速 response 生成
- **要点 5**：beta、advantage_estimator、scale_rewards 是关键超参数
