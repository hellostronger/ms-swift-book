# 第 4 章 参数系统

## 4.1 参数架构概览

ms-swift 使用 Python 的 `dataclass` 机制构建了一套完整的参数系统。所有参数通过继承组合形成层次化的配置结构。

```
BaseArguments
├── ModelArguments          # 模型相关参数
├── DataArguments           # 数据相关参数
├── GenerationArguments     # 生成相关参数
├── QuantizationArguments   # 量化相关参数
├── TemplateArguments       # 模板相关参数
└── TunerArguments          # 微调相关参数

SftArguments = SwanlabArguments + TunerArguments + BaseArguments + Seq2SeqTrainingArguments

RLHFArguments = TeacherModelArguments + GRPOArguments + PPOArguments + RewardModelArguments + SftArguments
```

## 4.2 BaseArguments 核心参数

```python
# swift/arguments/base_args.py

@dataclass
class BaseArguments:
    """基础参数类"""
    
    # 任务类型
    task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 
                       'generative_reranker', 'reward_model'] = 'causal_lm'
    
    # 模型相关
    model: Optional[str] = None           # 模型 ID 或路径
    model_type: Optional[str] = None      # 模型类型
    model_revision: Optional[str] = None  # 模型版本
    
    # 数据相关
    dataset: List[str] = field(default_factory=list)
    val_dataset: List[str] = field(default_factory=list)
    split_dataset_ratio: float = 0.0      # 验证集比例
    
    # 模板相关
    template: Optional[str] = None        # 对话模板
    system: Optional[str] = None          # 系统提示词
    
    # 训练相关
    output_dir: Optional[str] = None
    seed: int = 42
    bf16: Optional[bool] = None
    fp16: Optional[bool] = None
```

## 4.3 SftArguments 训练参数

```python
# swift/arguments/sft_args.py

@dataclass
class SftArguments(SwanlabArguments, TunerArguments, BaseArguments, Seq2SeqTrainingArguments):
    """SFT 训练参数"""
    
    # 学习率
    learning_rate: Optional[float] = None  # 默认 full:1e-5, lora:1e-4
    
    # 评估策略
    eval_strategy: Optional[str] = None    # steps, epoch
    
    # DeepSpeed
    deepspeed: Optional[str] = None        # zero0~zero3_offload
    
    # FSDP
    fsdp: Optional[str] = None             # fsdp2
    
    # 其他
    add_version: bool = True               # 添加版本目录
    max_new_tokens: int = 64               # 生成最大 token 数
    temperature: float = 0.0               # 采样温度
```

## 4.4 RLHFArguments 对齐参数

```python
# swift/arguments/rlhf_args.py

@dataclass
class RLHFArguments(TeacherModelArguments, GRPOArguments, PPOArguments, 
                    RewardModelArguments, SftArguments):
    """RLHF 训练参数"""
    
    # RLHF 类型
    rlhf_type: Literal['dpo', 'orpo', 'simpo', 'kto', 'cpo', 
                       'rm', 'ppo', 'grpo', 'gkd'] = 'dpo'
    
    # 参考模型
    ref_model: Optional[str] = None
    ref_adapters: List[str] = field(default_factory=list)
    
    # 核心超参
    beta: Optional[float] = None           # 偏离参考模型的系数
    temperature: float = 0.9               # 采样温度
    max_completion_length: int = 512       # 最大生成长度
    
    # DPO 特有
    loss_type: Optional[List[str]] = None  # sigmoid, hinge, ipo...
    rpo_alpha: Optional[float] = None      # SFT loss 权重
    ld_alpha: Optional[float] = None       # LD-DPO 参数
    label_smoothing: float = 0             # 标签平滑
    
    # GRPO 特有
    num_generations: int = 8               # 每个 prompt 生成数
    reward_funcs: List[str] = field(default_factory=list)
    reward_weights: List[float] = None
    
    # PPO 特有
    num_ppo_epochs: int = 4
    kl_coef: float = 0.05
    gamma: float = 1.0
    lam: float = 0.95
```

## 4.5 参数校验与默认值

ms-swift 在 `__post_init__` 方法中进行参数校验和默认值设置：

```python
def __post_init__(self):
    # 1. 处理模型路径
    self.model = to_abspath(self.model)
    
    # 2. 设置默认学习率
    if self.learning_rate is None:
        if self.tuner_type == 'full':
            self.learning_rate = 1e-5
        else:
            self.learning_rate = 1e-4
    
    # 3. 初始化 DeepSpeed
    self._init_deepspeed()
    
    # 4. 初始化设备
    self._init_device()
    
    # 5. 校验参数组合
    self._check_args()
```

## 4.6 参数继承链

以 GRPO 训练为例，参数继承链为：

```
RLHFArguments
├── TeacherModelArguments      # teacher_model, teacher_adapters
├── GRPOArguments              # num_generations, reward_funcs
├── PPOArguments               # num_ppo_epochs, kl_coef
├── RewardModelArguments       # reward_model
└── SftArguments
    ├── SwanlabArguments       # swanlab_token, swanlab_project
    ├── TunerArguments         # tuner_type, lora_rank
    ├── BaseArguments          # model, dataset, template
    └── Seq2SeqTrainingArguments  # 来自 transformers
```

## 小结

- **要点 1**：ms-swift 使用 dataclass 继承组合构建参数系统
- **要点 2**：SftArguments 和 RLHFArguments 是两个核心参数类
- **要点 3**：参数校验和默认值在 `__post_init__` 中处理
