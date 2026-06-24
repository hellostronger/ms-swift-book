# 第 16 章 Loss Scale 体系

## 16.1 Loss Scale 概念

Loss Scale 是 ms-swift 的创新设计，用于控制训练时哪些 token 参与损失计算，以及每个 token 的损失权重。

```
原始对话：
System: You are a helpful assistant.
User: 你好
Assistant: 你好！有什么可以帮你的吗？

Loss Scale 策略：
- System: 0.0（不参与训练）
- User: 0.0（不参与训练）
- Assistant: 1.0（参与训练）
```

## 16.2 LossScale 基类

```python
# swift/loss_scale/base.py

class LossScale:
    """损失缩放基类"""
    
    is_binary = True  # 是否只有 0 和 1
    
    def __init__(self, base_strategy: Literal['default', 'last_round', 'all'] = 'default'):
        """
        Args:
            base_strategy: 基础策略
                - 'default': 只在 assistant 回复上计算损失
                - 'last_round': 只在最后一轮 assistant 回复上计算损失
                - 'all': 在所有部分计算损失
        """
        self.base_strategy = base_strategy
    
    def get_loss_scale(self, context: str, **kwargs) -> Tuple[List[str], List[float]]:
        """计算损失权重"""
        return [context], [1.]
    
    def __call__(self, context_list, context_types, messages, **kwargs):
        """处理完整对话上下文"""
        res_context_list = []
        res_loss_scale = []
        
        for context, context_type in zip(context_list, context_types):
            is_assistant = context_type in {ContextType.RESPONSE, ContextType.SUFFIX}
            
            # 根据策略决定是否计算损失
            if self.base_strategy == 'all' or \
               (self.base_strategy == 'default' and is_assistant) or \
               (self.base_strategy == 'last_round' and is_assistant and is_last_round):
                new_context, loss_scale = self.get_loss_scale(context, query=query)
            else:
                new_context, loss_scale = [context], [0.]
            
            res_context_list += new_context
            res_loss_scale += loss_scale
        
        return res_context_list, res_loss_scale
```

## 16.3 三种基础策略

| 策略 | 计算范围 | 适用场景 |
|------|---------|---------|
| default | 所有 assistant 回复 | 标准 SFT |
| last_round | 最后一轮 assistant | 多轮对话微调 |
| all | 所有部分 | 语言模型预训练 |

## 16.4 ConfigLossScale

从 JSON 配置文件加载损失权重映射：

```python
class ConfigLossScale(LossScale):
    """配置驱动的 Loss Scale"""
    
    is_binary = None
    loss_scale_config = None  # 配置文件路径
    
    def __init__(self, base_strategy='default'):
        super().__init__(base_strategy)
        self.loss_scale_map = None
        if self.loss_scale_config is not None:
            config_path = os.path.join('config', self.loss_scale_config)
            with open(config_path, 'r') as f:
                self.loss_scale_map = json.load(f)
    
    def get_loss_scale(self, context: str, *, query=None, **kwargs):
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map)
        return super().get_loss_scale(context)
```

## 16.5 ConcatLossScale

支持多个 Loss Scale 串联，权重相乘：

```python
class ConcatLossScale(LossScale):
    """串联多个 Loss Scale"""

    def __init__(self, loss_scales: List[LossScale], base_strategy='default'):
        super().__init__(base_strategy)
        self.loss_scales = loss_scales

    def get_loss_scale(self, context, **kwargs):
        contexts = [context]
        weights = [1.0]
        
        for ls in self.loss_scales:
            new_contexts = []
            new_weights = []
            for c, w in zip(contexts, weights):
                sub_contexts, sub_weights = ls.get_loss_scale(c, **kwargs)
                for sc, sw in zip(sub_contexts, sub_weights):
                    new_contexts.append(sc)
                    new_weights.append(w * sw)  # 权重相乘
            contexts = new_contexts
            weights = new_weights
        
        return contexts, weights
```

**使用示例：**
```bash
# 串联 hermes 和 ignore_empty_think
--loss_scale hermes+ignore_empty_think
```

## 16.6 注册表

```python
# swift/loss_scale/mapping.py

loss_scale_map = {
    'base': LossScale,
    'ignore_empty_think': IgnoreEmptyThinkLossScale,
    # Agent
    'react': REACTLossScale,
    'hermes': HermesLossScale,
    'qwen': QwenLossScale,
    'agentflan': AgentFlanLossScale,
    'alpha_umi': AlphaUmiLossScale,
}
```

## 16.7 工厂函数

```python
def get_loss_scale(loss_scale: str) -> LossScale:
    """
    支持的格式：
    1. 'default' - 基础策略
    2. 'react' - 使用 REACTLossScale
    3. 'default+react' - 策略 + Loss Scale
    4. 'hermes+ignore_empty_think' - 串联多个
    """
    parts = loss_scale.split('+')
    
    if parts[0] in ALL_BASE_STRATEGY:
        base_strategy = parts[0]
        ls_names = parts[1:] or ['base']
    else:
        base_strategy = 'default'
        ls_names = parts
    
    if len(ls_names) == 1:
        return loss_scale_map[ls_names[0]](base_strategy)
    
    # 串联多个
    sub_loss_scales = [loss_scale_map[name]('default') for name in ls_names]
    return ConcatLossScale(sub_loss_scales, base_strategy)
```

## 小结

- **要点 1**：Loss Scale 控制哪些 token 参与训练，是 ms-swift 的核心创新
- **要点 2**：三种基础策略覆盖标准 SFT、多轮对话、预训练场景
- **要点 3**：支持配置驱动和串联组合，灵活性强
