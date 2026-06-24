# 目录

本指南系统性地剖析了 ms-swift 框架的所有核心模块，从基础设施到高级特性，帮助读者深入理解大模型微调的完整技术栈。

## 第一部分：宏观认知

- [第 1 章 ms-swift 是什么](/chapters/01-what-is-ms-swift) — 项目定位、核心特性与适用场景
- [第 2 章 仓库全景与技术栈](/chapters/02-repo-overview) — 目录结构、模块划分、依赖分析
- [第 3 章 快速入门](/chapters/03-quick-start) — 安装配置、基础使用、典型训练流程

## 第二部分：基础设施层

- [第 4 章 参数系统](/chapters/04-arguments) — Dataclass 参数架构、继承体系、校验逻辑
- [第 5 章 模型注册系统](/chapters/05-model-registry) — MODEL_MAPPING、ModelMeta、模型加载流程
- [第 6 章 模板系统](/chapters/06-template-system) — Template 基类、对话格式、多模态支持
- [第 7 章 数据集系统](/chapters/07-dataset-system) — 数据集加载、预处理、格式化
- [第 8 章 Tuner 微调方法](/chapters/08-tuners) — LoRA、QLoRA、全参数微调实现

## 第三部分：训练核心

- [第 9 章 Trainer 架构](/chapters/09-trainer) — Trainer 基类、SwiftMixin、训练循环
- [第 10 章 训练流水线](/chapters/10-pipelines) — SFT/RLHF Pipeline、任务编排
- [第 11 章 推理引擎](/chapters/11-infer-engine) — TransformersEngine、vLLM 集成

## 第四部分：损失函数体系 ⭐

- [第 12 章 BaseLoss 基础架构](/chapters/12-loss-base) — 损失函数抽象、BaseLoss 设计
- [第 13 章 CrossEntropyLoss 详解](/chapters/13-loss-cross-entropy) — 自定义交叉熵、token 级别损失
- [第 14 章 嵌入学习损失函数](/chapters/14-loss-embedding) — CosineSimilarity、Contrastive、InfoNCE
- [第 15 章 重排损失函数](/chapters/15-loss-reranker) — Pointwise、Listwise Reranker Loss
- [第 16 章 Loss Scale 体系](/chapters/16-loss-scale) — LossScale 基类、策略模式、动态权重
- [第 17 章 Agent Loss Scale](/chapters/17-loss-scale-agent) — REACT、Qwen、Hermes、AgentFlan

## 第五部分：RLHF 对齐训练 ⭐

- [第 18 章 RLHF 概论与算法对比](/chapters/18-rlhf-overview) — 对齐技术演进、算法选择指南
- [第 19 章 DPO 及其 14 种变体](/chapters/19-dpo) — sigmoid、hinge、ipo、exo_pair、bco_pair 等
- [第 20 章 CPO / SimPO](/chapters/20-cpo-simpo) — 无参考模型的对齐方法
- [第 21 章 ORPO](/chapters/21-orpo) —  Odds Ratio Preference Optimization
- [第 22 章 KTO](/chapters/22-kto) — Kahneman-Tversky Optimization
- [第 23 章 PPO](/chapters/23-ppo) — Proximal Policy Optimization 实现
- [第 24 章 GRPO](/chapters/24-grpo) — Group Relative Policy Optimization
- [第 25 章 GKD 知识蒸馏](/chapters/25-gkd) — Generalized Knowledge Distillation
- [第 26 章 Reward Model 训练](/chapters/26-rm) — 奖励模型训练流程
- [第 27 章 奖励函数体系](/chapters/27-rewards) — MathAccuracy、Format、CosineReward 等

## 第六部分：超参数详解

- [第 28 章 学习率与优化器](/chapters/28-learning-rate-optimizer) — 学习率、优化器、调度器
- [第 29 章 RLHF 超参数详解](/chapters/29-rlhf-hyperparams) — beta、temperature、loss_type
- [第 30 章 GRPO / PPO 超参数](/chapters/30-grpo-ppo-hyperparams) — num_generations、reward_funcs
- [第 31 章 部署与推理超参数](/chapters/31-deploy-hyperparams) — vLLM、TensorRT-LLM 配置

## 第七部分：高级主题

- [第 32 章 DeepSpeed / FSDP](/chapters/32-deepspeed-fsdp) — ZeRO 优化、分布式训练
- [第 33 章 Megatron 集成](/chapters/33-megatron) — 大规模并行训练
- [第 34 章 模型导出与部署](/chapters/34-export-deploy) — 模型导出、API 服务
- [第 35 章 评估系统](/chapters/35-evaluation) — 指标计算、评估流程

## 附录

- [附录 A 超参数参考手册](/chapters/appendix-a-hyperparams) — 完整参数列表与说明
- [附录 B 损失函数速查表](/chapters/appendix-b-loss-cheatsheet) — 公式、适用场景、代码示例
- [附录 C 术语表](/chapters/appendix-c-glossary) — 核心概念解释
- [附录 D 阅读路线图](/chapters/appendix-d-reading-path) — 按角色推荐阅读顺序
