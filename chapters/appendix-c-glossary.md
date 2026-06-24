# 附录 C：术语表

## C

**Checkpoint**
- 训练过程中保存的模型状态，包含模型权重、优化器状态、训练进度等
- 可用于恢复训练或推理部署

**Contrastive Loss**
- 对比损失，用于嵌入学习
- 通过拉近正样本、推开负样本来学习表示

**CrossEntropy Loss**
- 交叉熵损失，语言模型的标准损失函数
- 衡量预测分布与真实分布的差异

## D

**DeepSpeed**
- 微软开源的分布式训练库
- 提供 ZeRO 优化器，通过切分优化器状态、梯度和参数减少显存占用

**DPO (Direct Preference Optimization)**
- 直接偏好优化，直接从偏好数据训练模型
- 无需训练奖励模型，训练更稳定

**DP (Data Parallelism)**
- 数据并行，每个 GPU 处理不同的数据批次
- 最常见的分布式训练策略

## E

**Embedding**
- 嵌入，将离散的 token 映射为连续的向量表示
- 用于语义相似度、检索等任务

**Epoch**
- 训练轮数，整个数据集被遍历一次的完整过程

## F

**FSDP (Fully Sharded Data Parallel)**
- PyTorch 原生的分布式训练方案
- 类似于 DeepSpeed ZeRO-3

**Full Fine-tuning**
- 全参数微调，更新模型的所有参数
- 效果最好，但显存占用大

## G

**GaLore (Gradient Low-Rank Projection)**
- 梯度低秩投影优化器
- 通过低秩近似减少优化器状态的显存占用

**GKD (Generalized Knowledge Distillation)**
- 广义知识蒸馏
- 通过 JSD 损失让学生模型学习教师模型的分布

**GRPO (Group Relative Policy Optimization)**
- 组相对策略优化
- 在每个 prompt 的多个响应内计算相对优势，无需奖励模型

**GPTQ**
- 基于 GPU 的 Post-Training 量化方法
- 4-bit 量化，压缩模型大小

## K

**KL Divergence**
- KL 散度，衡量两个概率分布的差异
- 在 RLHF 中用于约束策略更新幅度

**KTO (Kahneman-Tversky Optimization)**
- 基于前景理论的优化方法
- 只需要二元反馈（好/坏），不需要成对偏好数据

## L

**LoRA (Low-Rank Adaptation)**
- 低秩适配，通过低秩矩阵分解实现参数高效微调
- 只训练少量参数，显存占用小

**LoRA+**
- LoRA 的改进版本
- 为 A 和 B 矩阵使用不同的学习率

**Loss Scale**
- 损失缩放，控制哪些 token 参与损失计算
- 用于 Agent 训练等场景

## M

**Megatron-LM**
- NVIDIA 开发的大规模语言模型训练框架
- 支持张量并行、流水线并行、序列并行

**MRL (Matryoshka Representation Learning)**
- 嵌套表示学习
- 在不同维度截断上计算损失，支持灵活的嵌入维度

## O

**ORPO (Odds Ratio Preference Optimization)**
- 优势比偏好优化
- 无需参考模型，单阶段完成 SFT 和对齐

**ORM (Outcome Reward Model)**
- 结果奖励模型
- 基于最终结果给出奖励分数

## P

**Packing**
- 将多个短样本拼接为一个长序列
- 提高训练效率，减少 padding 开销

**PPO (Proximal Policy Optimization)**
- 近端策略优化，经典的强化学习算法
- 需要策略模型、参考模型、奖励模型、价值模型

**PP (Pipeline Parallelism)**
- 流水线并行，将模型层切分到多个 GPU
- 适合超大规模模型

**PRM (Process Reward Model)**
- 过程奖励模型
- 基于推理过程的每一步给出奖励分数

## Q

**QLoRA**
- 量化 LoRA
- 在 4-bit 量化模型上使用 LoRA 微调
- 显存占用最小

## R

**Reward Model**
- 奖励模型，将人类偏好建模为标量分数
- 用于 PPO 训练

**RLHF (Reinforcement Learning from Human Feedback)**
- 基于人类反馈的强化学习
- 通过奖励模型和 PPO 算法对齐模型行为

**Reranker**
- 重排模型，对候选文档进行排序
- 用于检索增强生成（RAG）

## S

**SFT (Supervised Fine-Tuning)**
- 监督微调
- 使用标注数据训练模型

**SimPO (Simple Preference Optimization)**
- 简化版 CPO
- 使用平均 log 概率，无需参考模型

**Sequence Parallel**
- 序列并行，将长序列切分到多个 GPU
- 适合超长文本训练

**SwanLab**
- 实验追踪平台
- 记录训练指标、超参数、模型性能

## T

**Template**
- 模板系统，统一不同模型的对话格式
- 处理 system prompt、user query、assistant response

**Tensor Parallel**
- 张量并行，将模型参数切分到多个 GPU
- 适合大规模模型的单层计算

**Tokenizer**
- 分词器，将文本转换为 token ID
- 不同模型使用不同的分词策略

**TRL (Transformer Reinforcement Learning)**
- HuggingFace 的强化学习训练库
- 提供 DPO、PPO、GRPO 等算法的实现

## W

**WandB (Weights & Biases)**
- 实验追踪平台
- 记录训练指标、可视化训练过程

## Z

**ZeRO (Zero Redundancy Optimizer)**
- DeepSpeed 的核心优化技术
- 通过切分优化器状态、梯度和参数减少显存占用
- ZeRO-1：切分优化器状态
- ZeRO-2：切分优化器状态和梯度
- ZeRO-3：切分优化器状态、梯度和参数
