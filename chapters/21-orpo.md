# 第 21 章 ORPO

## 21.1 ORPO 核心原理

ORPO（Odds Ratio Preference Optimization）是一种**无需参考模型**的对齐方法，使用优势比（odds ratio）来优化策略。

**核心公式：**

$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} - \lambda \cdot \log\sigma\left(\frac{p(y_w|x)}{p(y_l|x)} \cdot \frac{1-p(y_l|x)}{1-p(y_w|x)}\right)$$

其中：
- 第一项：SFT 损失
- 第二项：优势比偏好损失
- $\lambda$：ORPO 权重参数

**优势：**
- 无需参考模型
- 单阶段训练（SFT + 对齐同时进行）
- 计算效率高

## 21.2 ORPO vs DPO vs CPO

| 维度 | DPO | CPO | ORPO |
|------|-----|-----|------|
| 参考模型 | 需要 | 不需要 | 不需要 |
| 训练阶段 | 两阶段 | 两阶段 | 单阶段 |
| SFT 损失 | 否 | 是（可选） | 是（必须） |
| 显存占用 | 2倍模型 | 1倍模型 | 1倍模型 |
| 训练速度 | 慢 | 快 | 快 |

## 21.3 实现细节

```python
# swift/rlhf_trainers/orpo_trainer.py

class ORPOTrainer(RLHFTrainerMixin, SwiftMixin, HFORPOTrainer):
    def __init__(self, model, *_args, **kwargs):
        ref_model = kwargs.get('ref_model')
        assert ref_model is None, 'ORPO does not require a ref_model.'
        super().__init__(model, *_args, **kwargs)
```

ORPO 直接继承自 HuggingFace TRL 的 ORPOTrainer。

## 21.4 关键超参数

### lambda (ORPO 权重)

```python
# 在 TRL ORPOConfig 中定义
lambda: float = 0.1
```

**作用：** 控制偏好损失的权重。
- 0.1：默认值
- 更大：更重视偏好对齐
- 更小：更重视 SFT

### loss_scale

```python
# ORPO 默认使用 'default' loss_scale
if self.rlhf_type == 'orpo' and not self.model_meta.is_multimodal:
    self.loss_scale = 'default'
```

**作用：** ORPO 在多轮对话中默认对所有 assistant 回复计算损失。

## 21.5 训练示例

```bash
swift rlhf \
  --rlhf_type orpo \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset Anthropic/hh-rlhf \
  --output_dir output/orpo
```

## 21.6 适用场景

**适合 ORPO 的场景：**
- 计算资源有限（无法加载参考模型）
- 希望单阶段完成 SFT 和对齐
- 数据集规模较小

**不适合 ORPO 的场景：**
- 需要精确控制对齐程度
- 数据质量不高，需要两阶段训练

## 小结

- **要点 1**：ORPO 无需参考模型，单阶段完成 SFT + 对齐
- **要点 2**：使用优势比优化，计算效率高
- **要点 3**：lambda 参数控制偏好损失权重
- **要点 4**：适合资源有限或数据规模较小的场景
