# 第 17 章 Agent Loss Scale

第 16 章讲了 `LossScale` 的通用框架：三种 `base_strategy`、`ConfigLossScale` 的 JSON 驱动、`ConcatLossScale` 的串联。这一章专门讲 Agent 场景——ms-swift 里最复杂、也最有特色的 Loss Scale 用法。

## 17.1 Agent 训练的特殊性

Agent 模型的输出不是一段连贯的自然语言，而是由多个**语义完全不同的片段**拼接而成：

```text
Thought: 我需要查询天气
Action: get_weather
Action Input: {city: Beijing}
Observation: 北京今天晴，25C          ← 环境返回，模型没有"说过"
Thought: 我知道答案了
Final Answer: 北京今天是晴天，25C
```

如果对整段回复用同一个权重（比如全部 1.0），会出两个问题：

1. **`Observation` 被污染**：`Observation` 的内容是外部工具（搜索引擎、数据库）返回的，不是模型生成的。训练模型去"预测"这些内容，等于在教它幻觉。
2. **格式信号被稀释**：Agent 场景中，`Action` / `Action Input` 的**格式精确性**比内容正确性更重要——工具名拼错、JSON 少一个逗号，整条链路就断了。但这些 token 在整段回复里占比很小，用统一权重时格式学习信号会被大量自然语言 token 淹没。

ms-swift 的 Agent Loss Scale 就是为了解决这个问题：**按语义片段分配不同的损失权重**。

| 片段 | 语义 | 默认权重 | 理由 |
|------|------|---------|------|
| `Action:` | 模型生成的工具调用名 | 2.0 | 格式必须精确 |
| `Action Input:` | 模型生成的参数 | 2.0 | 格式必须精确 |
| `Thought:` | 模型的推理过程 | 1.0 | 常规语义学习 |
| `Final Answer:` | 最终回答 | 1.0 | 常规语义学习 |
| `Observation:` | 外部环境返回 | 0.0 | 不训练，避免幻觉 |

## 17.2 REACTLossScale：最基础的 Agent 实现

源码只有 4 行（`swift/loss_scale/agent.py:19-20`）：

```python
# swift/loss_scale/agent.py

class REACTLossScale(ConfigLossScale):
    loss_scale_config = 'react.json'
```

所有逻辑都在配置文件里。`swift/loss_scale/config/react.json`：

```json
{
    "Action:": [2.0, 2.0],
    "Action Input:": [2.0, 2.0],
    "Thought:": [1.0, 1.0],
    "Final Answer:": [1.0, 1.0],
    "Observation:": [2.0, 0.0]
}
```

### 权重元组的语义：`[delimiter_weight, content_weight]`

这是最容易误解的地方。以 `"Observation:": [2.0, 0.0]` 为例，它被拆成了**两个独立的部分**：

```text
[2.0]                      [0.0]
 ↓                           ↓
"Observation:"          "北京今天晴，25C"
分隔符本身                  分隔符之后的内容
```

为什么 `Observation:` 这个词本身给 2.0 而不是 0.0？因为**模型确实需要学会"什么时候该产出 Observation 这个字段"**——它是 Agent 循环结构的语法信号。如果连分隔符都 mask 掉，模型就不知道工具返回后自己该接什么内容了。所以策略是：**保留结构的语法信号（高权重），屏蔽外部注入的内容（零权重）**。

注意 JSON 的 key 里带冒号（`"Action:"` 而不是 `"Action"`），因为分隔符就是模板里真实渲染出来的字符串。

## 17.3 核心算法：`calculate_loss_scale`

真正的实现在 `swift/loss_scale/utils.py`。这是整个 Agent Loss Scale 的心脏：

```python
# swift/loss_scale/utils.py:7

def calculate_loss_scale(query: str,
                         response: str,
                         response_loss_scale_map: Dict[str, list],
                         query_loss_scale_map: Optional[Dict[str, list]] = None
                         ) -> Tuple[List[str], List[float]]:
    """Calculate the loss scale by splitting the agent response.

    This algorithm comes from paper: https://arxiv.org/pdf/2309.00986.pdf
    """
    # ① query loss scale map：若命中，整段 response 使用同一个权重
    if query_loss_scale_map is not None:
        for key in query_loss_scale_map.keys():
            if key in query:
                if isinstance(query_loss_scale_map[key], (float, int)):
                    query_loss_scale_map[key] = [query_loss_scale_map[key]]
                loss_scale_value = query_loss_scale_map[key][0]
                return [response], [float(loss_scale_value)]

    # ② 长度为 2 的 value 是「字面量分隔符」，直接 split
    delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 2]
    if delimiters:
        agent_parts = split_str_parts_by(response, delimiters)
    else:
        # ③ 否则视为正则分隔符（regex_mode=True）
        regex_delimiters = [k for k, v in response_loss_scale_map.items() if len(v) == 1]
        agent_parts = split_str_parts_by(response, regex_delimiters, regex_mode=True)

    # ④ 按配置给每一段分配权重
    weights = []
    agent_content = []
    for c in agent_parts:
        if c['key'] in response_loss_scale_map:
            loss_scale = response_loss_scale_map[c['key']]
            assert len(loss_scale) in {1, 2}, f'loss_scale: {loss_scale}'
            if len(loss_scale) == 1:
                weights += loss_scale
                agent_content.append(c['content'])
            else:
                # 长度为 2：分隔符单独成一段，权重用第一个
                weights += loss_scale
                agent_content += [c['key'], c['content']]
        else:
            # 未匹配的片段默认权重 1.0
            weights.append(1.)
            agent_content.append(c['content'])
    return agent_content, weights
```

### 关键分支：长度决定「字面量模式」还是「正则模式」

这是一个隐式约定，看代码时容易忽略：

| JSON value 长度 | 含义 | 拆分方式 | 例子 |
|----------------|------|---------|------|
| `2` | 该配置项贡献**两段**（分隔符 + 内容） | 字面量 split | `[2.0, 2.0]` → react.json |
| `1` | 该配置项只贡献**一段内容**（分隔符本身不单独计权） | **正则** split | `[2.0]` → hermes.json |

`split_str_parts_by`（`swift/template/utils.py:137`）的实现印证了这一点：

```python
def split_str_parts_by(text: str, delimiters: List[str], regex_mode: bool = False) -> List[Dict[str, str]]:
    assert isinstance(text, str), f'text: {text}'
    delimiters_origin = delimiters
    if not regex_mode:
        # 字面量模式：对分隔符做 re.escape，避免 "Action:" 里的特殊字符被当正则
        delimiters = [re.escape(delimiter) for delimiter in delimiters]
    parts = _split_str_by_regex(text, delimiters) if delimiters else ['', text]
    res = []
    if regex_mode:
        # 正则模式：分隔符是 part 的一部分，所以要 re.match 探测它属于哪个 key
        parts = [part for part in parts if part]
        for part in parts:
            for delimiter, delimiter_origin in zip(delimiters, delimiters_origin):
                if re.match(delimiter, part, re.DOTALL):
                    break
            else:
                delimiter_origin = ''
            res.append({'key': delimiter_origin, 'content': part})
    else:
        # 字面量模式：偶数位置是分隔符，奇数位置是内容
        for key, content in zip(parts[::2], parts[1::2]):
            res.append({'key': key, 'content': content})
    return res
```

字面量模式下 `_split_str_by_regex` 返回的 `parts` 是 `[sep1, content1, sep2, content2, ..., sepN, contentN]`，所以 `parts[::2]` 是分隔符、`parts[1::2]` 是内容。

### 完整走一遍

对第 17.1 节那段回复，调用 `calculate_loss_scale(query, response, react_map)`：

```text
输入 response:
Thought: 我需要查询天气
Action: get_weather
Action Input: {city: Beijing}
Observation: 北京今天晴，25C
Thought: 我知道答案了
Final Answer: 北京今天是晴天，25C
```

`delimiters = ['Action:', 'Action Input:', 'Thought:', 'Final Answer:', 'Observation:']`，切分结果与权重分配：

| # | 分段内容 | 权重 | 说明 |
|---|---------|------|------|
| 0 | `Thought:` | 1.0 | 分隔符 |
| 1 | ` 我需要查询天气` | 1.0 | 内容 |
| 2 | `Action:` | 2.0 | 分隔符 |
| 3 | ` get_weather` | 2.0 | 内容 |
| 4 | `Action Input:` | 2.0 | 分隔符 |
| 5 | ` {city: Beijing}` | 2.0 | 内容 |
| 6 | `Observation:` | 2.0 | 分隔符 |
| 7 | ` 北京今天晴，25C` | **0.0** | 内容 ← 关键 |
| 8 | `Thought:` | 1.0 | 分隔符 |
| 9 | ` 我知道答案了` | 1.0 | 内容 |
| 10 | `Final Answer:` | 1.0 | 分隔符 |
| 11 | ` 北京今天是晴天，25C` | 1.0 | 内容 |

返回 `(agent_content, weights)` 是**一一对应**的两条列表——这正是 `LossScale.__call__` 契约要求的（第 16 章提到过「values in loss_scale_list correspond one-to-one with the values in context_list」）。分段结果会把新的 `context_list` 交还给上层，重新走一遍 encode 流程，因此每个子段会被独立 tokenize，权重精确到 token 边界。

## 17.4 HermesLossScale：正则模式

Hermes 格式用 XML 风格标签包裹工具调用，配置（`config/hermes.json`）只有一行：

```json
{
    "﻿tool_call>.+?<\/﻿tool_call>": [2.0]
}
```

> 注：上面的 key 首尾含不可见零宽字符（`\u200b`），这是 ms-swift 的一个**刻意的防御性设计**——防止模型生成的 `</tool_call>` 被误切，同时保证不会切到用户输入里的同名文本。这也是为什么必须走正则模式（`len(v) == 1`）而不是字面量模式。

解析过程：
- `len(v) == 1` → 走 `regex_mode=True`，用正则 `<tool_call>(.+?)</tool_call>` 匹配
- 匹配到的整段（含首尾标签）作为**一个 part**，`key` 是原正则字符串
- 因为 `len(loss_scale) == 1`，只追加内容不追加分隔符，权重取 `[0]` = 2.0

效果：整个工具调用块（含标签）权重 2.0，其余部分默认 1.0。

## 17.5 QwenLossScale：四段式

`config/qwen.json`（Qwen-Agent 的 XML 风格）：

```json
{
    "✿FUNCTION✿": [2.0, 2.0],
    "✿ARGS✿":    [2.0, 2.0],
    "✿RETURN✿":  [1.0, 1.0],
    "✿RESULT✿":  [2.0, 0.0]
}
```

结构与 REACT 完全同构，只是分隔符换了名字：

| 分隔符 | 权重 | 含义 |
|--------|------|------|
| `✿FUNCTION✿` | [2.0, 2.0] | 工具名 |
| `✿ARGS✿` | [2.0, 2.0] | 工具参数 |
| `✿RETURN✿` | [1.0, 1.0] | 模型向用户提问（如需澄清参数） |
| `✿RESULT✿` | [2.0, 0.0] | 工具返回结果 |

## 17.6 AlphaUmiLossScale：多值元组

`config/alpha_umi.json` 展示了另一种扩展形态：

```json
{
    "Action:": [2.0, 2.0],
    "Action Input:": [2.0, 2.0],
    "Thought:": [1.0, 1.0],
    "Final Answer:": [1.0, 1.0],
    "Observation:": [2.0, 0.0],
    "Next:": [2, 0, 2.0]
}
```

`"Next:": [2, 0, 2.0]` 是一个**三元素元组**：第一个 2.0 是分隔符权重，**中间的 0 会被原样写进 weights 列表**（第 49 行只有 `assert len(loss_scale) in {1, 2}`，不阻止长度为 3），而 `agent_content` 只追加了 2 段。

这会导致 `weights` 与 `agent_content` **长度不匹配**，从而破坏 `LossScale.__call__` 的一一对应契约。这是一个上游配置的缺陷——实践上请不要照抄这种三元素写法，把它改成 `[2.0, 2.0]` 即可。

## 17.7 AgentFlanLossScale：唯一支持 query 侧的 Agent Scale

前面所有 Agent Scale 都只看 `response`。AgentFlan 是例外（`swift/loss_scale/agent.py:9-16`）：

```python
class AgentFlanLossScale(ConfigLossScale):
    is_binary = False
    loss_scale_config = 'agentflan.json'

    def get_loss_scale(self, context: str, *, query: Optional[str] = None, **kwargs):
        if isinstance(context, str):
            return calculate_loss_scale(query, context, self.loss_scale_map['response'], self.loss_scale_map['query'])
        return super().get_loss_scale(context)
```

`config/agentflan.json` 是**嵌套结构**，分成 `response` 和 `query` 两张表：

```json
{
    "response": {
        "Name:":           [1.0, 3.0],
        "Action:":         [1.0, 3.0],
        "ACTION:":         [1.0, 3.0],
        "Tool:":           [1.0, 3.0],
        "Command":         [1.0, 3.0],
        "Arguments:":      [1.0, 3.0],
        "action input":    [1.0, 3.0],
        "ACTION_INPUT:":   [1.0, 3.0],
        "Action Input:":   [1.0, 3.0],
        "Thought:":        [1.0, 1.0],
        "Final Answer:":   [1.0, 1.0],
        "Observation:":    [2.0, 0.0]
    },
    "query": {
        "What is the tool you want to use":                    [3.0],
        "What are the required parameter names":              [3.0],
        "What is the value of":                               [3.0],
        "What are the required parameter names for this tool": [3.0]
    }
}
```

两个设计要点：

**① `response` 表极其冗长地列举了同义分隔符。** 因为这是**纯字面量匹配**（`"action input"` 和 `"Action Input:"` 在模板渲染后大小写和冒号都可能不同），配置只能穷举。这也从侧面说明为什么字面量方案在多生态 Agent 场景下维护成本高。

**② `query` 表实现的是「多轮追问放大」。** AgentFlan 是 FLAN 风格的多轮工具调用：模型的工具调用被环境**拆成若干个追问**（"What is the tool you want to use?" → "What are the required parameter names?" → "What is the value of X?"），每一轮模型只回答其中一环。

若 query 命中任一追问模板，`calculate_loss_scale` 的 ① 分支会**直接返回整段 response 用统一权重 3.0**（而不是走分段逻辑）：

```python
# 命中即返回，不再拆分
return [response], [float(loss_scale_value)]
```

放大到 3.0 的原因：这些追问轮是 Agent 能力的**瓶颈环节**——模型只需要输出一个短答案（如工具名），样本量小、token 少，用 1.0 的权重相对其他轮次信号太弱，容易学不动。

另外注意 `is_binary = False` 是**手写声明**的。因为 `AgentFlanLossScale` 的 `get_loss_scale` 覆写绕过了 `ConfigLossScale.is_binary_loss_scale` 的自动探测（后者只会遍历 `self.loss_scale_map.values()`，而这里 map 的顶层 value 是 dict 而非 list，遍历结果全为 False 恰好也得到 False，但语义是错的），显式声明更清晰。

## 17.8 IgnoreEmptyThinkLossScale：不是 Agent，但常与 Agent 组合

`swift/loss_scale/other.py` 全文 6 行：

```python
class IgnoreEmptyThinkLossScale(ConfigLossScale):
    loss_scale_config = 'ignore_empty_think.json'
```

`config/ignore_empty_think.json` 全部是**正则**（`len(v) == 1`），锚定在行首：

```json
{
    "^<think>\\s*</think>\\s*": [0.0],
    "^<seed:think><seed:cot_budget_reflect>The current thinking budget is 0, so I will directly start answering the question.</seed:cot_budget_reflect>\\n</seed:think>\\s*": [0.0],
    "^</think>\\s*": [0.0],
    "^<\\|channel>thought\\n<channel\\|>": [0.0],
    "^</mm:think>\\s*": [0.0]
}
```

处理的五种「空思考」形态：

| 形态 | 说明 |
|------|------|
| `<think></think>` | 配对的空 think 标签 |
| `<think>The current thinking budget is 0...</seed:think>` | **budget-reflect 机制**：Qwen3.5+ 在思考预算耗尽时会写入一段固定文案再退出思考；这段是模板产物而非模型推理，必须屏蔽 |
| `</think>` | 未配对的闭合标签 |
| `\|channel>thought\n\|channel\|>` | gpt-oss 的 harmony 通道标记` | MiniMax 的 think 闭合标签 |

注意它们都带 `^` 锚点和 `\s*` 后缀——只屏蔽**开头的**空 think 段，不影响正文中可能出现的同名字符串。

## 17.9 `is_binary` 的自动探测机制

`ConfigLossScale` 有一个容易被忽略但影响性能的特性（第 16 章提过 `is_binary`，这里补全语义）：

```python
@property
def is_binary_loss_scale(self):
    if self.is_binary is not None:
        return self.is_binary
    if self.loss_scale_map is None:
        return True
    return all(scale in {0.0, 1.0} for lst in self.loss_scale_map.values() for scale in lst)
```

**为什么重要**：`is_binary=True` 时，权重 0 的 token 会被直接替换成 `labels = -100`（label masking），于是**可以走 liger_kernel 等融合算子**，性能最好。权重非 0/1 时（如 2.0、3.0），则必须在 loss 计算时额外传入 `loss_scale` key，走自定义 loss 函数路径，无法享受 CE 融合算子。

按配置逐个对照：

| Loss Scale | 权重集合 | is_binary | 走 label masking？ |
|-----------|---------|-----------|------------------|
| `REACTLossScale` | {0.0, 1.0, 2.0} | **False** | 否 |
| `HermesLossScale` | {2.0} | **False** | 否 |
| `QwenLossScale` | {0.0, 1.0, 2.0} | **False** | 否 |
| `AlphaUmiLossScale` | {0.0, 1.0, 2} | **False** | 否 |
| `AgentFlanLossScale` | {0.0, 1.0, 3.0} | False（显式） | 否 |
| `IgnoreEmptyThinkLossScale` | {0.0} | **True** | **是** ✅ |

也就是说：**只有 `ignore_empty_think` 是"免费"的**——它只是把 token mask 掉，不需要任何自定义 loss 算子。这就是它最常被串联在其他 Loss Scale 后面使用的原因。

## 17.10 组合使用

所有 Agent Loss Scale 都通过 `ConcatLossScale`（第 16 章 16.5 节）串联，权重**相乘**：

```bash
# Qwen-Agent + 屏蔽空思考
swift sft --loss_scale qwen+ignore_empty_think ...

# 指定基础策略（只训练最后一轮）
swift sft --loss_scale last_round+qwen+ignore_empty_think ...

# 配合自定义 loss 函数（因为 is_binary=False，必须显式指定）
swift sft --loss_type cross_entropy --loss_scale react ...
```

多 Loss Scale 串联时有个实现细节（`swift/loss_scale/mapping.py:53-56`）：

```python
# The base_strategy is owned by the outer ConcatLossScale; sub loss scales only
# contribute their `get_loss_scale` (which does not reference base_strategy), so
# any valid placeholder ('default') is fine here.
sub_loss_scales = [loss_scale_map[name]('default') for name in ls_names]
return ConcatLossScale(sub_loss_scales, base_strategy)
```

子 Loss Scale 一律用 `'default'` 构造，真正的 `base_strategy` 由外层 `ConcatLossScale` 持有——因为 `base_strategy` 的判断发生在 `LossScale.__call__` / `_inner_call` 里，而串联时调用的只有 `get_loss_scale`，它不感知策略。

## 17.11 数据格式：直接用 `messages` 指定权重

除了走 `--loss_scale` 全局策略，`LossScale.__call__` 还支持在**单条样本**上直接覆盖（第 16 章 16.2 节的 `_inner_call`）：

```python
# swift/loss_scale/base.py:98-99
loss = messages[2 * i + 1].get('loss')          # 显式 loss 值（0/1）
loss_scale = messages[2 * i + 1].get('loss_scale')  # 显式权重
```

```json
[
  {"role": "user", "content": "北京今天天气如何？"},
  {"role": "assistant", "content": "...", "loss_scale": [1.0, 0.0, 3.0]}
]
```

还有一个更精细的入口：当 context 是 dict 且含 `loss_scale` 键时，直接按 token 粒度展开（`base.py:115-117`）：

```python
if isinstance(context, dict) and 'loss_scale' in context:
    new_context = [[token] for token in context['token_ids']]
    loss_scale = context['loss_scale']
```

即 `token_ids` 的长度必须与 `loss_scale` 的长度严格相等，每个 token 单独成段——这是做 token 级精细加权（例如只对某个工具名加权）的入口。

## 小结

- **要点 1**：Agent Loss Scale 的本质是「按语义片段分配权重」，核心目标是**保留 `Observation` 的结构信号但屏蔽其内容**，防止模型学会幻觉外部工具返回
- **要点 2**：`calculate_loss_scale` 通过「元组长度隐式决定字面量/正则模式」的约定，把 6 种 Agent 格式统一到一套极简的 JSON 配置里
- **要点 3**：`is_binary` 决定走 label masking（可用 liger_kernel 融合算子，性能最优）还是自定义 loss 路径——**除 `ignore_empty_think` 外所有 Agent Loss Scale 都是非二值的**，这是选型时容易踩的性能坑
- **要点 4**：多 Loss Scale 通过 `ConcatLossScale` 串联，权重相乘；`base_strategy` 由最外层持有
- **要点 5**：除全局策略外，还可通过 messages 的 `loss` / `loss_scale` 字段做样本级、甚至 token 级覆盖
