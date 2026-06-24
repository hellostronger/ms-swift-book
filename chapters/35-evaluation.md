# 第 35 章 评估系统

## 35.1 评估任务类型

ms-swift 支持多种评估任务：

```python
task_type: Literal['causal_lm', 'seq_cls', 'embedding', 'reranker', 
                   'generative_reranker', 'reward_model']
```

| 任务类型 | 评估指标 | 说明 |
|---------|---------|------|
| causal_lm | loss, perplexity, accuracy | 语言模型 |
| seq_cls | accuracy, f1 | 序列分类 |
| embedding | margin, mean_pos, mean_neg | 嵌入模型 |
| reranker | accuracy | 重排模型 |

## 35.2 语言模型评估

### 基础指标

```python
# swift/metrics/acc.py

class Accuracy(EvalMetrics):
    """Token 级别准确率"""
    
    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
        
        # 忽略 padding 和 -100 位置
        mask = labels != -100
        
        # 计算准确率
        preds = logits.argmax(dim=-1)
        acc = (preds[mask] == labels[mask]).float().mean()
        
        return {'accuracy': acc.item()}
```

### 困惑度（Perplexity）

```python
class Perplexity(EvalMetrics):
    """计算困惑度"""
    
    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        loss = eval_prediction.predictions
        
        # 困惑度 = exp(交叉熵损失)
        ppl = torch.exp(torch.tensor(loss))
        
        return {'perplexity': ppl.item()}
```

## 35.3 序列分类评估

```python
# swift/metrics/nlg.py

class SeqClassificationMetrics(EvalMetrics):
    """序列分类评估指标"""
    
    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
        
        preds = logits.argmax(dim=-1)
        
        # 准确率
        accuracy = (preds == labels).float().mean()
        
        # F1 分数
        precision = ((preds == 1) & (labels == 1)).sum() / (preds == 1).sum()
        recall = ((preds == 1) & (labels == 1)).sum() / (labels == 1).sum()
        f1 = 2 * precision * recall / (precision + recall)
        
        return {
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item()
        }
```

## 35.4 嵌入模型评估

```python
# swift/metrics/embedding.py

class EmbeddingMetrics(EvalMetrics):
    """嵌入模型评估指标"""
    
    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        embeddings1 = eval_prediction.predictions[0]
        embeddings2 = eval_prediction.predictions[1]
        labels = eval_prediction.label_ids
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(
            torch.tensor(embeddings1),
            torch.tensor(embeddings2),
            dim=-1
        )
        
        # 正负样本分离度
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        pos_mean = cos_sim[pos_mask].mean()
        neg_mean = cos_sim[neg_mask].mean()
        
        # margin = 正样本相似度 - 负样本相似度
        margin = pos_mean - neg_mean
        
        return {
            'margin': margin.item(),
            'mean_pos': pos_mean.item(),
            'mean_neg': neg_mean.item()
        }
```

## 35.5 重排模型评估

```python
# swift/metrics/reranker.py

class RerankerMetrics(EvalMetrics):
    """重排模型评估指标"""
    
    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
        
        # 二分类准确率
        preds = (logits > 0).astype(int)
        accuracy = (preds == labels).mean()
        
        return {'accuracy': accuracy.item()}
```

## 35.6 NLG 评估指标

对于生成式任务，可以使用 NLG 指标：

```python
# swift/metrics/nlg.py

class NLGMetrics(EvalMetrics):
    """自然语言生成评估指标"""
    
    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        predictions = eval_prediction.predictions
        references = eval_prediction.label_ids
        
        # 解码文本
        pred_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
        ref_texts = [tokenizer.decode(r, skip_special_tokens=True) for r in references]
        
        # BLEU 分数
        bleu = compute_bleu(pred_texts, ref_texts)
        
        # ROUGE 分数
        rouge = compute_rouge(pred_texts, ref_texts)
        
        return {
            'bleu': bleu,
            'rouge1': rouge['rouge1'],
            'rouge2': rouge['rouge2'],
            'rougeL': rouge['rougeL']
        }
```

## 35.7 评估流程

### 1. 设置评估参数

```bash
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --eval_dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --eval_strategy steps \
  --eval_steps 100 \
  --per_device_eval_batch_size 4 \
  --metric_for_best_model loss \
  --greater_is_better false
```

### 2. 评估时机

```python
eval_strategy: Literal['no', 'steps', 'epoch'] = 'steps'
eval_steps: int = 100
```

- `no`：不评估
- `steps`：每 N 步评估
- `epoch`：每个 epoch 评估

### 3. 最佳模型选择

```python
metric_for_best_model: str = 'loss'
greater_is_better: bool = False
```

## 35.8 自定义评估指标

```python
from swift.metrics import EvalMetrics

class MyCustomMetric(EvalMetrics):
    
    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
        
        # 自定义指标计算
        my_metric = compute_my_metric(logits, labels)
        
        return {'my_metric': my_metric}

# 注册
from swift.metrics import metric_map
metric_map['my_metric'] = MyCustomMetric
```

## 35.9 完整评估示例

```bash
swift sft \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --eval_dataset AI-ModelScope/alpaca-gpt4-data-zh \
  --tuner_type lora \
  --eval_strategy steps \
  --eval_steps 100 \
  --per_device_eval_batch_size 4 \
  --metric_for_best_model loss \
  --greater_is_better false \
  --load_best_model_at_end true \
  --save_total_limit 3 \
  --output_dir output/qwen2_5-7b-sft
```

## 35.10 评估结果解读

训练日志示例：

```
{'loss': 2.34, 'learning_rate': 1e-04, 'epoch': 1.0}
{'eval_loss': 2.21, 'eval_accuracy': 0.45, 'eval_runtime': 12.5}
{'loss': 2.15, 'learning_rate': 9.5e-05, 'epoch': 2.0}
{'eval_loss': 2.08, 'eval_accuracy': 0.52, 'eval_runtime': 12.3}
```

指标说明：
- `loss`：训练损失
- `eval_loss`：评估损失
- `eval_accuracy`：评估准确率
- `eval_runtime`：评估耗时

## 小结

- **要点 1**：ms-swift 支持多种评估任务（causal_lm, seq_cls, embedding, reranker）
- **要点 2**：语言模型评估包括 loss、accuracy、perplexity
- **要点 3**：嵌入模型评估使用 margin 指标
- **要点 4**：支持自定义评估指标
