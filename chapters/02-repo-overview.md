# 第 2 章 仓库全景与技术栈

## 2.1 目录结构总览

```
ms-swift/
├── swift/                    # 核心源码目录
│   ├── agent_template/       # Agent 模板
│   ├── arguments/            # 参数定义系统 ⭐
│   ├── callbacks/            # 训练回调
│   ├── cli/                  # 命令行入口
│   ├── config/               # 配置文件
│   ├── dataloader/           # 数据加载器
│   ├── dataset/              # 数据集处理 ⭐
│   ├── hub/                  # 模型仓库
│   ├── infer_engine/         # 推理引擎 ⭐
│   ├── loss/                 # 损失函数 ⭐
│   ├── loss_scale/           # 损失缩放 ⭐
│   ├── megatron/             # Megatron 集成
│   ├── metrics/              # 评估指标
│   ├── model/                # 模型注册 ⭐
│   ├── optimizers/           # 优化器
│   ├── pipelines/            # 训练流水线 ⭐
│   ├── ray/                  # Ray 分布式
│   ├── rewards/              # 奖励函数 ⭐
│   ├── rl_core/              # RL 核心模块
│   ├── rlhf_trainers/        # RLHF 训练器 ⭐
│   ├── rollout/              # Rollout 模块
│   ├── sequence_parallel/    # 序列并行
│   ├── template/             # 模板系统 ⭐
│   ├── trainers/             # 训练器基类
│   ├── tuner_plugin/         # Tuner 插件
│   ├── tuners/               # 微调方法 ⭐
│   ├── ui/                   # Web UI
│   └── utils/                # 工具函数
├── docs/                     # 文档
├── examples/                 # 示例代码
├── tests/                    # 测试用例
└── requirements/             # 依赖配置
```

## 2.2 核心模块关系

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                            │
│              (swift/cli/, swift/ui/)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Pipeline Layer                           │
│         (swift/pipelines/ - SFT/RLHF/Eval/Export)           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Trainer Layer                            │
│    (swift/trainers/, swift/rlhf_trainers/)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Core Components                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Model   │  │ Template │  │ Dataset  │  │  Loss    │    │
│  │ Registry │  │  System  │  │  System  │  │  System  │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Arguments Layer                           │
│              (swift/arguments/)                              │
└─────────────────────────────────────────────────────────────┘
```

## 2.3 技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 训练框架 | PyTorch + Transformers | 基础训练能力 |
| 分布式 | DeepSpeed + FSDP + Ray | 大规模训练 |
| 推理加速 | vLLM + LMDeploy + SGLang | 高性能推理 |
| 微调方法 | PEFT (LoRA/QLoRA) | 参数高效微调 |
| 量化 | bitsandbytes + GPTQ + AWQ | 模型压缩 |
| 评估 | OpenCompass + lm-eval | 模型评测 |
| 监控 | WandB + SwanLab | 实验追踪 |

## 2.4 代码规模统计

通过 `find` 命令统计：

- Python 文件：约 400+ 个
- 核心模块：30+ 个
- 损失函数实现：7 种
- RLHF 算法：9 种
- 支持的模型：200+ 种

## 2.5 依赖分析

ms-swift 的依赖分为几个层次：

**核心依赖**：
- `torch` >= 2.0：深度学习框架
- `transformers` >= 4.37：模型库
- `peft` >= 0.7：参数高效微调

**可选依赖**：
- `deepspeed`：分布式训练
- `vllm`：推理加速
- `math_verify`：数学推理验证
- `swanlab`：实验追踪

## 小结

- **要点 1**：ms-swift 采用分层架构设计，从 CLI 到 Trainer 再到核心组件，层次清晰
- **要点 2**：核心模块包括模型注册、模板系统、数据集系统、损失函数系统
- **要点 3**：技术栈涵盖训练、推理、分布式、量化等全栈能力
