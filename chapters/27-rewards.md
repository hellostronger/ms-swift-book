# 第 27 章 奖励函数体系

## 27.1 奖励函数概述

ms-swift 提供了丰富的奖励函数体系，用于 GRPO 等在线强化学习算法。

**奖励函数分类：**
- **Outcome Reward Model (ORM)**：基于最终结果的奖励
- **Process Reward Model (PRM)**：基于过程的奖励

**内置奖励函数：**
- `accuracy`：数学准确性
- `format`：格式检查
- `cosine`：余弦奖励（长度惩罚）
- `repetition`：重复惩罚
- `soft_overlong`：软长度限制
- `react_format`：ReAct 格式检查
- `math`：数学验证（旧版）
- `toolbench`：工具调用奖励

## 27.2 ORM 基类

```python
# swift/rewards/orm.py

class ORM:
    """同步奖励模型基类"""
    
    def __init__(self, args=None, **kwargs):
        self.args = args
    
    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError

class AsyncORM:
    """异步奖励模型基类（用于 I/O 密集型任务）"""
    
    async def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError
```

## 27.3 MathAccuracy

```python
class MathAccuracy(ORM):
    """数学准确性奖励"""
    
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify'.")
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        
        rewards = []
        for content, sol in zip(completions, solution):
            # 提取 <answer> 标签内容
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol
            
            # 解析和验证
            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                answer_parsed = parse(content_to_parse, ...)
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                reward = 0.0
            
            rewards.append(reward)
        return rewards
```

**使用场景：** 数学推理任务（GSM8K、MATH 等）

**依赖：** `pip install math_verify`

## 27.4 Format

```python
class Format(ORM):
    """格式检查奖励"""
    
    def __call__(self, completions, **kwargs) -> List[float]:
        """检查是否符合 <think>...</think><answer>...</answer> 格式"""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]
```

**使用场景：** 强制模型输出特定格式

## 27.5 CosineReward

```python
class CosineReward(ORM):
    """余弦奖励（长度惩罚）"""
    
    def __init__(self, args=None, accuracy_orm=None):
        super().__init__(args)
        self.min_len_value_wrong = args.cosine_min_len_value_wrong
        self.max_len_value_wrong = args.cosine_max_len_value_wrong
        self.min_len_value_correct = args.cosine_min_len_value_correct
        self.max_len_value_correct = args.cosine_max_len_value_correct
        self.max_len = args.cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()
    
    @staticmethod
    def cosfn(t, T, min_value, max_value):
        """余弦函数"""
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2
    
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            
            # 根据准确性选择参数
            if is_correct:
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        
        return rewards
```

**超参数：**
- `cosine_min_len_value_wrong`：错误答案的最小长度值
- `cosine_max_len_value_wrong`：错误答案的最大长度值
- `cosine_min_len_value_correct`：正确答案的最小长度值
- `cosine_max_len_value_correct`：正确答案的最大长度值
- `cosine_max_len`：最大长度

**核心思想：** 鼓励短而正确的回答，惩罚冗长的错误回答。

## 27.6 RepetitionPenalty

```python
class RepetitionPenalty(ORM):
    """重复惩罚"""
    
    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        self.ngram_size = args.repetition_n_grams
        self.max_penalty = args.repetition_max_penalty
    
    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])
    
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue
            
            # 计算 n-gram 重复率
            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1
            
            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        
        return rewards
```

**超参数：**
- `repetition_n_grams`：n-gram 大小（默认 3）
- `repetition_max_penalty`：最大惩罚值（默认 -0.1）

**核心思想：** 惩罚重复的 n-gram，鼓励多样性。

## 27.7 SoftOverlong

```python
class SoftOverlong(ORM):
    """软长度限制"""
    
    def __init__(self, args=None, **kwargs):
        super().__init__(args)
        assert args.soft_cache_length < args.soft_max_length
        self.soft_max_length = args.soft_max_length
        self.soft_cache_length = args.soft_cache_length
    
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        
        return rewards
```

**超参数：**
- `soft_max_length`：最大长度
- `soft_cache_length`：缓存长度（必须 < soft_max_length）

**核心思想：** 超过期望长度后，线性惩罚。

## 27.8 ReactORM

```python
class ReactORM(ORM):
    """ReAct Agent 奖励"""
    
    @staticmethod
    def evaluate_action_reward(action_pred, action_ref, cand_list, ref_list):
        """评估动作奖励"""
        f1 = []
        for i in range(len(action_pred)):
            # 比较动作和输入
            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                # JSON 比较
                half_match = 0
                full_match = 0
                for k, v in ref_input_json.items():
                    if k in cand_input_json.keys():
                        if cand_input_json[k] == v:
                            full_match += 1
                        else:
                            half_match += 1
                
                recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                f1.append((2 * recall * precision) / (recall + precision))
        
        return f1[0] == 1.0
    
    def __call__(self, infer_requests, solution, **kwargs) -> List[float]:
        """评估 ReAct 格式的动作"""
        rewards = []
        for prediction, ground_truth in zip(predictions, ground_truth):
            # 解析 Action 和 Action Input
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            
            # 评估奖励
            reward = ReactORM.evaluate_action_reward(...)
            rewards.append(float(reward))
        
        return rewards
```

**使用场景：** ToolBench 等工具调用任务

## 27.9 自定义奖励函数

```python
from swift.rewards import ORM

class MyCustomReward(ORM):
    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
    
    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            # 自定义奖励逻辑
            reward = compute_reward(completion)
            rewards.append(reward)
        return rewards

# 注册
from swift.rewards import orms
orms['my_reward'] = MyCustomReward
```

**使用：**

```bash
swift rlhf --reward_funcs my_reward ...
```

## 27.10 奖励函数组合

```bash
# 组合多个奖励函数
swift rlhf \
  --rlhf_type grpo \
  --reward_funcs accuracy format repetition \
  --reward_weights 1.0 0.5 -0.1 \
  ...
```

**说明：**
- `reward_weights`：每个奖励函数的权重
- 正权重：鼓励
- 负权重：惩罚

## 27.11 异步奖励函数

```python
from swift.rewards import AsyncORM

class MyAsyncReward(AsyncORM):
    """异步奖励（适用于 API 调用等 I/O 密集型任务）"""
    
    async def __call__(self, completions, **kwargs) -> List[float]:
        import aiohttp
        
        async def score_single(session, text):
            async with session.post(api_url, json={'text': text}) as resp:
                result = await resp.json()
                return result['score']
        
        async with aiohttp.ClientSession() as session:
            tasks = [score_single(session, c) for c in completions]
            rewards = await asyncio.gather(*tasks)
            return list(rewards)
```

**优势：** 并行执行多个 API 调用，显著加速奖励计算。

## 小结

- **要点 1**：ms-swift 提供 8 种内置奖励函数，覆盖数学、格式、长度、重复等场景
- **要点 2**：MathAccuracy 使用 math_verify 库验证数学表达式
- **要点 3**：CosineReward 鼓励短而正确的回答
- **要点 4**：支持自定义奖励函数和异步奖励函数
- **要点 5**：通过 reward_weights 可以组合多个奖励函数
