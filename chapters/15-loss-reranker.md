# 第 15 章 重排损失函数

## 15.1 概述

重排（Reranker）任务的目标是对候选文档进行排序：给定一个 query 和一批候选文档，输出每个文档的相关性分数（标量），再按分数排序。ms-swift 支持两种重排损失：

| 损失函数 | 原理 | 适用场景 |
|---------|------|---------|
| PointwiseReranker | 独立打分 | 简单排序 |
| ListwiseReranker | 组内对比 | 精确排序 |

源码在 `swift/loss/reranker.py`（全文仅 109 行），注册在 `loss_map` 的 `# reranker` 段。

与 embedding 最大的区别：**reranker 损失作用在 logits（标量相关性分数）上**，而 embedding 损失作用在向量上。`RerankerTrainer` 对 `generative_reranker` 有一个专门的 logits 提取逻辑（`swift/trainers/reranker_trainer.py:17-45`）：

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # Check if we have a custom loss function
    if self.compute_loss_func is not None:
        # Get labels and compute outputs
        labels = inputs.pop('labels', None)
        outputs = model(**inputs)
        if self.task_type == 'generative_reranker':
            # For generative reranker, take logits at the last valid position
            logits = outputs.logits
            attention_mask = inputs.get('attention_mask')
            last_valid_indices = -1 if attention_mask is None else get_last_valid_indices(attention_mask)
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            outputs.logits = logits[batch_indices, last_valid_indices]
        if labels is not None:
            # Call custom loss function
            loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
        ...
```

对于**分类式 reranker**（`task_type='reranker'`），`outputs.logits` 已经是 `[B, 1]` 的标量分数；两种损失都直接 `squeeze` 掉最后一维。

## 15.2 PointwiseRerankerLoss

```python
class PointwiseRerankerLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        logits = outputs.logits.squeeze(1)
        labels = labels.to(logits.dtype)
        loss_fct = nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        return loss
```

**数学原理：**

使用二元交叉熵损失，将每个文档独立分类为相关（1）或不相关（0）：

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(\sigma(x_i)) + (1-y_i)\log(1-\sigma(x_i))\right]$$

用的是 `BCEWithLogitsLoss`（内部含 sigmoid），不是 `BCELoss`——**这意味着不需要在模型里显式加 sigmoid 激活**，数值上也更稳定。

**特点：**
- 简单直接，计算效率高
- 不考虑文档间的相对关系
- 适合二分类场景

## 15.3 ListwiseRerankerLoss ⭐

```python
class ListwiseRerankerLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs):
        """
        List-wise reranker loss function.

        This loss function groups samples by query based on the pattern where each group
        consists of 1 positive document followed by n negative documents. It treats the
        ranking task as a classification problem within each group, using cross-entropy
        loss to identify the positive document among all candidates.

        Data format expected:
        - labels: [1, 0, 0, 0, 1, 0, 0, ...] where 1 indicates positive, 0 indicates negative
        - Each 1 is followed by its corresponding negative documents until the next 1

        Environment variables for configuration:
        - LISTWISE_RERANKER_TEMPERATURE: Temperature for softmax (default: 1.0)
        - LISTWISE_RERANKER_MIN_GROUP_SIZE: Minimum group size to include (default: 2)
        """
        logits = outputs.logits.squeeze(-1)  # [batch_size]
        labels = labels.float()

        # Configuration from environment variables
        temperature = float(os.environ.get('LISTWISE_RERANKER_TEMPERATURE', '1.0'))
        min_group_size = int(os.environ.get('LISTWISE_RERANKER_MIN_GROUP_SIZE', '2'))

        # Find positive sample indices to determine group boundaries
        positive_indices = torch.nonzero(labels == 1, as_tuple=False).squeeze(-1)

        if len(positive_indices) == 0:
            # No positive samples in this batch, return zero loss
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Ensure positive_indices is 1D
        if positive_indices.dim() == 0:
            positive_indices = positive_indices.unsqueeze(0)

        total_loss = 0.0
        num_groups = 0

        for i, pos_idx in enumerate(positive_indices):
            # Determine group boundaries
            group_start = pos_idx.item()
            # Find the end of current group (start of next group or end of batch)
            if i + 1 < len(positive_indices):
                group_end = positive_indices[i + 1].item()
            else:
                group_end = len(labels)

            # Extract group logits and labels
            group_logits = logits[group_start:group_end]  # [group_size]
            group_labels = labels[group_start:group_end]  # [group_size]

            # Skip groups that are too small
            if len(group_logits) < min_group_size:
                continue

            # Verify that the first sample in the group is positive
            if group_labels[0] != 1:
                continue  # Skip malformed groups

            # Apply temperature scaling for better training dynamics
            scaled_logits = group_logits / temperature

            # The positive document is always at index 0 within the group
            target = torch.tensor(0, dtype=torch.long, device=logits.device)

            # Apply cross-entropy loss: positive document should have highest score
            loss_fct = CrossEntropyLoss()
            group_loss = loss_fct(scaled_logits.unsqueeze(0), target.unsqueeze(0))

            total_loss += group_loss
            num_groups += 1

        if num_groups == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Return average loss across all groups
        return total_loss / num_groups
```

**数学原理：**

将排序问题转化为组内多分类问题——每组（1 正 + n 负）内，正样本应获得最高分：

$$\mathcal{L} = -\log \frac{\exp(s_{pos}/\tau)}{\sum_{i=1}^{N}\exp(s_i/\tau)}$$

这与 `infonce` 的数学形式完全一致，**区别在于分母的范围**：

- `infonce`：分母是 batch 内**所有样本**的文档（跨 query 共享负例池）
- `listwise_reranker`：分母**仅限当前 query 自己的组**（不跨 query）

**关键超参数：**

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `LISTWISE_RERANKER_TEMPERATURE` | 1.0 | softmax 温度；调小会让 loss 关注最强负例 |
| `LISTWISE_RERANKER_MIN_GROUP_SIZE` | 2 | 最小组大小；小于此值的组被跳过（不计算 loss） |

**数据格式（重要）：**

```
labels: [1, 0, 0, 0, 1, 0, 0, ...]
         ↑ 第1组     ↑ 第2组
```

**每组必须以 label=1 开头**。源码里有两处防御性检查：

```python
# ① 整个 batch 都没有正样本时返回 0（但保持计算图）
if len(positive_indices) == 0:
    return torch.tensor(0.0, device=logits.device, requires_grad=True)

# ② 组内第一个样本不是正样本时跳过（malformed group）
if group_labels[0] != 1:
    continue  # Skip malformed groups
```

这个「组以正样本开头」的约定**由 collator 保证**（见 15.4 节）。

**特点：**
- 考虑文档间的相对关系
- 更符合排序任务的本质
- 计算复杂度略高（Python for 循环逐组计算）

## 15.4 数据组装：`_reranker_data_collator`

理解 Listwise 损失的关键是**组是怎么来的**。答案是 `swift/template/base.py:1789-1823` 的 `_reranker_data_collator`，它在训练时把每条原始数据展开成「1 正 + N 负」的固定大小组：

```python
def _reranker_data_collator(self,
                            batch: List[Dict[str, Any]],
                            *,
                            padding_to: Optional[int] = None) -> Dict[str, Any]:
    if self.is_training:
        if not hasattr(self, 'random_state'):
            # TODO: Move to `__init__`; kept here to avoid cache invalidation caused by template hash changes.
            self.random_state = np.random.RandomState(42)
        max_positive_samples = int(os.environ.get('MAX_POSITIVE_SAMPLES', 1))
        max_negative_samples = int(os.environ.get('MAX_NEGATIVE_SAMPLES', 7))
        labels_list = []
        new_batch = []
        for b in batch:
            labels = b.pop('labels', None)
            positive_num = sum(labels)
            negative_num = len(labels) - positive_num
            max_positive = min(positive_num, max_positive_samples)
            max_negative = min(negative_num, max_negative_samples)
            for i in self.random_state.choice(positive_num, max_positive, replace=False):
                new_batch.append(
                    {key: b[key][i]
                     for key in b.keys() if isinstance(b[key], list) and b[key][i] is not None})
                labels_list.append(1)
                for j in self.random_state.choice(negative_num, max_negative, replace=False):
                    new_batch.append({
                        key: b[key][j + positive_num]
                        for key in b.keys() if isinstance(b[key], list) and b[key][j + positive_num] is not None
                    })
                    labels_list.append(0)
        res = self._data_collator(new_batch, padding_to=padding_to)
        if labels_list:
            res['labels'] = torch.tensor(labels_list, dtype=torch.long)
    else:
        res = self._data_collator(batch, padding_to=padding_to)
    return res
```

**关键点：**

**① 双重 for 循环生成「正样本打头」的组。** 外层循环选正样本（`labels_list.append(1)`），内层循环紧跟其负样本（`labels_list.append(0)`）。这保证了 Listwise 损失要求的「每组以正样本开头」的布局。

**② 两个环境变量控制组的形状：**

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `MAX_POSITIVE_SAMPLES` | 1 | 每条数据抽取多少正样本（每抽一个生成一个独立组） |
| `MAX_NEGATIVE_SAMPLES` | 7 | 每组负样本数量 |

**③ 展开后的实际 batch size 会膨胀。** 一条数据展开成 `MAX_POSITIVE_SAMPLES × (1 + MAX_NEGATIVE_SAMPLES)` 条。默认配置下，batch size 8 实际处理 8 × 8 = 64 条 query-doc 对，显存压力是名义 batch 的 8 倍。

**④ 只在训练时展开，评估时保持原样**（`else: res = self._data_collator(batch, ...)`）。评估指标（MRR / nDCG）在原始数据上计算，不需要固定组大小。

**⑤ 随机采样而非固定取前 N 个。** 用 `self.random_state.choice(...)` 随机采样正/负样本（`replace=False` 避免重复），保证每个 epoch 看到的负例组合不同。`random_state = np.random.RandomState(42)` 固定了随机种子，保证可复现。

## 15.5 两种损失对比

| 维度 | Pointwise | Listwise |
|------|-----------|----------|
| 计算方式 | 独立打分 | 组内对比 |
| 考虑文档关系 | 否 | 是 |
| 训练效率 | 高 | 中（Python for 循环） |
| 显存占用 | 低 | 高（组内多文档并行） |
| 排序精度 | 一般 | 更高 |
| 适用场景 | 简单二分类 | 精确排序 |
| 温度参数 | 无（BCE 内部 sigmoid） | `LISTWISE_RERANKER_TEMPERATURE` |

**选型建议：**

- 数据量大、GPU 资源紧张 → `pointwise_reranker`（效率高）
- 追求最优排序质量、有充足负例 → `listwise_reranker`（组内对比更贴合排序本质）
- 消融实验时两者可复用同一份数据（collator 的组结构对两者一致），切换 `--loss_type` 即可对比

## 15.6 评估指标：MRR 与 nDCG

Reranker 的评估统一用 `RerankerMetrics`（`--eval_metric reranker`，自动在 `_init_metric` 中设定），实现在 `swift/metrics/reranker.py`。它**按 label==1 的位置切分组**，然后对每组独立计算 MRR 和 nDCG：

```python
# Step 1: Find all positive sample indices (query boundaries)
positive_indices = np.where(labels == 1)[0]

# Step 2: Split into groups (queries)
for i, pos_idx in enumerate(positive_indices):
    group_start = pos_idx
    # Group ends at the next positive index or end of data
    if i + 1 < len(positive_indices):
        group_end = positive_indices[i + 1]
    else:
        group_end = len(labels)
    query_logits = logits[group_start:group_end]
    query_labels = labels[group_start:group_end]
    query_groups.append((query_logits, query_labels))
...
    # Skip groups that are too small (need at least 1 positive + 1 negative)
    if len(query_logits) < 2:
        continue
    # Verify that the first sample is positive (data format validation)
    if query_labels[0] != 1:
        continue
```

**nDCG 计算**（考虑相关文档在排序中的位置）：

```python
def calculate_ndcg_single_query(relevance_scores, ranking):
    ...
    ndcg = calculate_ndcg_single_query(relevance_scores, ranking)
    ndcg_scores.append(ndcg)
...
return {'mrr': np.mean(mrr_scores), 'ndcg': mean_ndcg, ...}
```

两个指标都基于「每组以正样本开头」的布局切分，所以 **Pointwise 训练出来的模型同样可以用这套 MRR/nDCG 评估**——评估逻辑与损失函数解耦，只依赖数据的 label 布局。

**训练过程中的 acc 指标**由 `_compute_acc` 计算（`swift/trainers/mixin.py:1125-1132`），对 listwise 用组内 argmax 判断是否命中：

```python
elif task_type in {'generative_reranker', 'reranker'}:
    if logits.dim() == 2:
        logits = logits.squeeze(-1)
    if args.loss_type == 'listwise_reranker':
        preds, labels = self._get_listwise_reranker_preds(logits, labels)
    else:
        preds = (logits > 0).long()   # pointwise: 阈值判定
    metrics = compute_acc(preds, labels.long())
```

其中 `_get_listwise_reranker_preds` 对每组取 `argmax`，检查正样本是否排在首位（`mixin.py:1068-1078`）——这是 listwise 训练时的「组内命中率」。

## 15.7 实际应用

**Pointwise 重排（分类式）**（`examples/train/reranker/train_reranker.sh`）：

```bash
# 1*5G
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model iic/gte-reranker-modernbert-base \
    --task_type reranker \
    --loss_type pointwise_reranker \
    --tuner_type full \
    --dataset MTEB/scidocs-reranking \
    --per_device_train_batch_size 64 \
    --learning_rate 6e-6 \
    --label_names labels \
    --dataloader_drop_last true
```

**Listwise 重排（分类式）**（`examples/train/reranker/train_reranker_listwise.sh`）——把 `--loss_type` 换成 `listwise_reranker` 即可，其余参数一致。

**Generative 重排**（`examples/train/reranker/train_generative_reranker.sh`）：

```bash
# 4*47G
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model Qwen/Qwen3-Reranker-4B \
    --task_type generative_reranker \
    --loss_type pointwise_reranker \
    --tuner_type full \
    --dataset MTEB/scidocs-reranking \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 6e-6 \
    --label_names labels \
    --deepspeed zero2 \
    --dataloader_drop_last true
```

> 注意：generative reranker 是「用生成式模型的语言理解能力做重排」，走 CausalLM 架构——`generative_reranker` 的 `logits` 来自最后一个有效 token 的词表投影（取 yes/no 特定 token，见 `get_generative_reranker_logits`，`swift/utils/torch_utils.py:403-410`），而非分类头。

## 15.8 关键环境变量汇总

Reranker 的所有可调项（都通过环境变量传入）：

| 变量 | 默认 | 作用 |
|------|------|------|
| `MAX_POSITIVE_SAMPLES` | 1 | 每条数据每组正样本数（每多一个正样本多一组） |
| `MAX_NEGATIVE_SAMPLES` | 7 | 每组负样本数（影响 listwise 分母大小与显存） |
| `LISTWISE_RERANKER_TEMPERATURE` | 1.0 | listwise softmax 温度 |
| `LISTWISE_RERANKER_MIN_GROUP_SIZE` | 2 | listwise 最小有效组大小 |
| `GENERATIVE_RERANKER_POSITIVE_TOKEN` | yes | generative reranker 判为「相关」的 token |
| `GENERATIVE_RERANKER_NEGATIVE_TOKEN` | no | generative reranker 判为「不相关」的 token |

`GENERATIVE_RERANKER_POSITIVE_TOKEN` / `NEGATIVE_TOKEN` 只对 `generative_reranker` 生效——它们定义了从词表 logits 中取哪两个 token 的分数之差作为相关性得分（`swift/utils/torch_utils.py:403`）：

```python
def get_generative_reranker_logits(lm_head_weight, tokenizer, hidden_states):
    positive_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    negative_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
    positive_token_id = tokenizer.convert_tokens_to_ids(positive_token)
    negative_token_id = tokenizer.convert_tokens_to_ids(negative_token)
    weight = lm_head_weight[[positive_token_id, negative_token_id]]
    logits = F.linear(hidden_states, weight)
    return logits[..., 0:1] - logits[..., 1:2]  # P(yes) - P(no)
```

**调参建议：**
- 负例难（区分度不够）→ 加大 `MAX_NEGATIVE_SAMPLES`（如 7 → 15），增强 listwise 难度
- 显存不足（generative）→ 减小 `MAX_NEGATIVE_SAMPLES` 或减小 `per_device_train_batch_size`
- listwise 训练不稳定 → 适当调大 `LISTWISE_RERANKER_TEMPERATURE`（默认 1.0 已较平缓，可试 1.5~2.0）

## 15.9 数据格式

Reranker 数据需要显式的正/负文档列表（而非 embedding 的成对格式）：

```jsonl
# LLM
{"messages": [{"role": "user", "content": "query"}], "positive_messages": [[{"role": "assistant", "content": "relevant_doc1"}]], "negative_messages": [[{"role": "assistant", "content": "irrelevant_doc1"}], [{"role": "assistant", "content": "irrelevant_doc2"}]]}
# MLLM
{"messages": [{"role": "user", "content": "<image>query"}], "images": ["/some/images.jpg"], "positive_messages": [[{"role": "assistant", "content": "<image>relevant_doc"}]], "positive_images": [["/some/pos.jpg"]], "negative_messages": [[{"role": "assistant", "content": "<image>irrelevant_doc"}]], "negative_images": [["/some/neg.jpg"]]}
```

预处理示例（`MTEBRerankPreprocessor`，`swift/dataset/dataset/llm.py:386-397`）：

```python
class MTEBRerankPreprocessor(RowPreprocessor):

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        query = row['query']
        positives = row['positive'] if isinstance(row['positive'], list) else [row['positive']]
        negatives = row['negative'] if isinstance(row['negative'], list) else [row['negative']]

        messages = [{'role': 'user', 'content': query}]
        positive_messages = [[{'role': 'assistant', 'content': positive}] for positive in positives]
        negative_messages = [[{'role': 'assistant', 'content': negative}] for negative in negatives]

        return {'messages': messages, 'positive_messages': positive_messages, 'negative_messages': negative_messages}
```

**与 embedding 数据格式的关键差异：**

| | Embedding | Reranker |
|---|---|---|
| query 字段 | `messages`（`user` role） | `messages`（`user` role） |
| 正/负文档字段 | `positive_messages` / `negative_messages`（`user` role） | `positive_messages` / `negative_messages`（**`assistant` role**） |
| 文档与 query 关系 | 独立编码（对比两条 embedding） | **拼接**成单条序列编码 |
| 负例要求 | 仅 infonce 需要（可选） | 必需（listwise 依赖负例分母） |
| 默认负例数 | 不限（由 `INFONCE_HARD_NEGATIVES` 控制） | `MAX_NEGATIVE_SAMPLES=7` |

> **注意 role 的差异**：embedding 任务中正例文档是 `user` role（因为它就是「另一条待编码的文本」），而 reranker 任务中文档是 `assistant` role（因为 query + 文档拼成一条对话，文档扮演被回答的角色）。

## 小结

- **要点 1**：Pointwise 独立打分（BCE），Listwise 组内对比（softmax CE），后者数学形式与 `infonce` 相同但分母不跨 query
- **要点 2**：`ListwiseRerankerLoss` 依赖「组以正样本开头」的 label 布局（`[1,0,0,1,0,...]`），该布局由 `_reranker_data_collator` 用双重 for 循环保证
- **要点 3**：组的大小由 `MAX_POSITIVE_SAMPLES`（默认 1）和 `MAX_NEGATIVE_SAMPLES`（默认 7）控制，展开后实际 batch 会膨胀 8 倍，是显存占用的主要来源
- **要点 4**：评估指标 MRR/nDCG 与损失函数解耦，Pointwise 和 Listwise 训练的模型共用同一套评估逻辑
- **要点 5**：Generative reranker（Qwen3 系列）走 CausalLM 架构，从 yes/no 两个 token 的 logit 差得到相关性分数，可用 `GENERATIVE_RERANKER_POSITIVE_TOKEN` 调整正例 token
