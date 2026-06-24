# 第 17 章 Agent Loss Scale

## 17.1 Agent 训练的特殊性

Agent 模型的输出包含多个部分，重要性不同：

- **Thought/Final Answer**：模型自主生成，应重点训练（weight=1.0）
- **Action/Action Input**：工具调用格式，需要精确学习（weight=2.0）
- **Observation**：外部环境返回，不应训练（weight=0.0）

示例输出：

    Thought: 我需要查询天气
    Action: get_weather
    Action Input: {city: Beijing}
    Observation: 北京今天晴，25C
    Thought: 我知道答案了
    Final Answer: 北京今天是晴天，25C

## 17.2 REACTLossScale

配置文件 react.json：

    {
        Action: [2.0, 2.0],
        Action Input: [2.0, 2.0],
        Thought: [1.0, 1.0],
        Final Answer: [1.0, 1.0],
        Observation: [2.0, 0.0]
    }

格式说明：[delimiter_weight, content_weight]
- 第一个值：分隔符本身的权重
- 第二个值：内容的权重

因此 Observation: [2.0, 0.0] 表示 Observation 后面内容的权重 = 0.0（不参与训练）。

## 17.3 HermesLossScale

适用于 Hermes 格式的 Agent，配置 hermes.json。

## 17.4 QwenLossScale

适用于 Qwen Agent 格式，配置 qwen.json。

## 17.5 AgentFlanLossScale

AgentFlan 支持 query 和 response 分别设置权重映射。

## 17.6 IgnoreEmptyThinkLossScale

用于忽略空的思维链（empty thinking）标签。

## 17.7 组合使用

Loss Scale 支持串联组合：

    swift sft --loss_scale react+ignore_empty_think ...

## 小结

- **要点 1**：Agent Loss Scale 通过配置实现细粒度的损失控制
- **要点 2**：不同 Agent 格式有对应的 Loss Scale 实现
- **要点 3**：支持串联组合，灵活性极强
