# 第 20 章 CPO / SimPO

## 20.1 CPO 核心原理

CPO（Contrastive Preference Optimization）是 DPO 的改进版本，**不需要参考模型**，直接在偏好数据上训练。

**核心公式：**

$$\mathcal{L}_{CPO} = -\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}\right) + \alpha \cdot \mathcal{L}_{NLL}$$

其中：
- 第一项：对比偏好损失
- 第二项：NLL（负对数似然）损失，防止遗忘
- $\alpha$：NLL 损失权重

**ms-swift 实现：**

```python
# swift/rlhf_trainers/cpo_trainer.py

class CPOTrainer(RLHFTrainerMixin, SwiftMixin, HFCPOTrainer):
    def __init__(self, model, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'CPO/SimPO does not require a ref_model.'
        
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type
        self.cpo_alpha = args.cpo_alpha
        
        if args.loss_type == 'simpo':
            self.simpo_gamma = args.simpo_gamma
```

## 20.2 SimPO 核心原理

SimPO（Simple Preference Optimization）是 CPO 的简化版本，使用**平均 log 概率**而非序列 log 概率。

**核心公式：**

$$\mathcal{L}_{SimPO} = -\log\sigma\left(\beta\left(\frac{\log\pi_\theta(y_w|x)}{|y_w|} - \frac{\log\pi_\theta(y_l|x)}{|y_l|}\right) - \gamma\right)$$

其中：
- $|y|$：序列长度
- $\gamma$：margin 参数

**优势：**
- 无需参考模型
- 长度归一化，缓解长度偏好
- 更简单的实现

## 20.3 CPO vs SimPO vs DPO

| 维度 | DPO | CPO | SimPO |
|------|-----|-----|-------|
| 参考模型 | 需要 | 不需要 | 不需要 |
| 长度归一化 | 否 | 否 | 是 |
| 显存占用 | 2倍模型 | 1倍模型 | 1倍模型 |
| 训练速度 | 慢 | 快 | 快 |
| 效果 | 好 | 好 | 好 |
| 实现复杂度 | 中 | 低 | 低 |

## 20.4 关键超参数

### cpo_alpha

```python
cpo_alpha: float = 1.0
```

**作用：** NLL 损失权重，防止模型遗忘预训练知识。
- 0：纯 CPO/SimPO
- 1.0：默认值，平衡偏好学习和语言建模
- >1：更重视语言建模

### simpo_gamma

```python
simpo_gamma: float = 1.0
```

**作用：** SimPO 的 margin 参数，控制 chosen 和 rejected 的最小差距。
- 0.5-1.5：论文推荐范围
- 默认 1.0

### beta

```python
# CPO 默认值
beta: Optional[float] = None  # 默认 0.1

# SimPO 默认值
beta: Optional[float] = 2.0   # 更大
```

**作用：** 温度参数，控制策略更新幅度。

## 20.5 训练示例

```bash
# CPO 训练
swift rlhf \
  --rlhf_type cpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset Anthropic/hh-rlhf \
  --cpo_alpha 1.0 \
  --beta 0.1 \
  --output_dir output/cpo

# SimPO 训练
swift rlhf \
  --rlhf_type simpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset Anthropic/hh-rlhf \
  --simpo_gamma 1.0 \
  --beta 2.0 \
  --output_dir output/simpo
```

**注意：** SimPO 会自动设置 `rlhf_type = 'cpo'` 和 `loss_type = 'simpo'`。

## 20.6 内部实现细节

```python
def _init_simpo(self):
    """SimPO 初始化"""
    if self.rlhf_type != 'simpo':
        return
    
    # SimPO 内部转换为 CPO
    self.rlhf_type = 'cpo'
    if self.loss_type is None:
        self.loss_type = 'simpo'
    if self.beta is None:
        self.beta = 2.  # SimPO 使用更大的 beta
```

## 小结

- **要点 1**：CPO/SimPO 不需要参考模型，显存占用更少
- **要点 2**：SimPO 使用长度归一化，缓解长度偏好
- **要点 3**：cpo_alpha 控制 NLL 损失权重，防止遗忘
- **要点 4**：SimPO 的 beta 默认值（2.0）比 CPO（0.1）更大
