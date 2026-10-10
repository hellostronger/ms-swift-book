# 第 14 章 嵌入学习损失函数

## 14.1 概述

ms-swift 支持 4 种嵌入学习损失函数，用于训练 Embedding 模型：

| 损失函数 | 适用场景 | 核心思想 | 标签类型 |
|---------|---------|---------|---------|
| CosineSimilarity | 句子对相似度回归 | 余弦相似度 + MSE | 浮点分数 |
| Contrastive | 对比学习 | 正负样本距离对比 | 0 / 1 |
| OnlineContrastive | 在线对比学习 | 困难样本挖掘 | 0 / 1 |
| InfoNCE | 多负样本检索 | 批内负样本对比 | 无需标签 |

它们全部注册在 `swift/loss/mapping.py`：

```python
# swift/loss/mapping.py

loss_map = {
    'cross_entropy': CustomCrossEntropyLoss,  # examples
    # embedding
    'cosine_similarity': CosineSimilarityLoss,
    'contrastive': ContrastiveLoss,
    'online_contrastive': OnlineContrastiveLoss,
    'infonce': InfonceLoss,
    # # reranker
    'pointwise_reranker': PointwiseRerankerLoss,
    'listwise_reranker': ListwiseRerankerLoss,
}
```

`--loss_type` 就是这张表的 key。`SwiftMixin.create_loss_and_eval_metric` 依据它实例化损失函数并挂到 Trainer 上（`swift/trainers/mixin.py:1048`）：

```python
if args.loss_type is not None:
    res['compute_loss_func'] = loss_map[args.loss_type](args, self)
```

`--loss_type` 与 `--task_type` 是**两个正交开关**：`--task_type embedding` 决定走哪条数据管线和评估指标，`--loss_type` 决定用哪个损失函数。

评估指标的自动选择逻辑在 `swift/arguments/sft_args.py:409-416`：

```python
def _init_metric(self):
    if self.eval_metric is None:
        if self.task_type == 'causal_lm' and self.predict_with_generate:
            self.eval_metric = 'nlg'
        elif self.task_type == 'embedding':
            self.eval_metric = 'infonce' if self.loss_type == 'infonce' else 'paired'
        elif self.task_type in {'reranker', 'generative_reranker'}:
            self.eval_metric = 'reranker'
```

即：**只有 `infonce` 用 `InfonceMetrics`，其余三种损失共用 `PairedMetrics`**（原因见 14.7 节）。

## 14.2 架构前提：Embedding 模型的前向输出

要理解这 4 个损失的区别，必须先理解它们拿到的 `outputs` 是什么。

`--task_type embedding` 的模型前向返回的不是 logits，而是一个**已归一化的向量**：

```text
{'last_hidden_state': Tensor[B, D]}   # D 是 embedding 维度
```

这个「归一化 + 取最后一个有效 token」的过程是 ms-swift 自动 patch 上去的（`swift/model/register.py:324`）：

```python
if model_info.task_type == 'embedding' and auto_model_cls.__name__ != 'AutoModel':
    from swift.model.patcher import patch_output_normalizer
    patch_output_normalizer(model, model_meta=model_meta)
```

`patch_output_normalizer` 做了两件事（`swift/model/patcher.py:75-102`）：

```python
def patch_output_normalizer(module: torch.nn.Module, model_meta):

    def lm_head_forward(self, hidden_states):
        return hidden_states   # ① 短路 lm_head，不再做 vocab 投影

    lm_heads = ['lm_head', 'output', 'embed_out', 'output_layer']
    lm_head_model = get_lm_head_model(module, model_meta=model_meta, lm_heads=lm_heads)
    ...
    def _output_embedding_hook(module, args, kwargs, output):
        attention_mask = kwargs.get('attention_mask', None)
        hidden_states = output.logits
        # ② 取每条序列的最后一个有效 token
        sequence_lengths = -1 if attention_mask is None else get_last_valid_indices(attention_mask)
        embeddings = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), sequence_lengths]
        # ③ L2 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return {'last_hidden_state': embeddings.contiguous()}

    lm_head_model.register_forward_hook(_output_embedding_hook, with_kwargs=True)
```

**这一步至关重要，它直接决定了后面所有损失函数的算法形态**：

- **没有 vocab 维** → 所有损失只能基于「向量之间的相似度」，无法做语言模型的 next-token 预测
- **已经归一化** → 所以 `F.cosine_similarity` 可以直接用，点积也等价于余弦值
- **每条样本一个向量** → batch 里是 `[B, D]` 而不是 `[B, L, D]`

> 官方文档特别提醒：自行接入新模型时，**必须自行保证 forward 返回的 embedding 已归一化**，否则所有损失函数的 scale 假设都会被破坏。

## 14.3 两条数据通路：成对 vs 多负例

4 种损失实际分成两族，由两个不同的解析函数处理。

### 通路 A：成对解析（`_parse_pair_sentence`）

服务于 `cosine_similarity` / `contrastive` / `online_contrastive`。

```python
# swift/loss/embedding.py:27

def _parse_pair_sentence(outputs):
    if isinstance(outputs, dict):
        last_hidden_state = outputs['last_hidden_state']
    else:
        last_hidden_state = outputs
    batch_size = last_hidden_state.shape[0]
    shape_len = len(last_hidden_state.shape)
    # 偶数下标是句子1，奇数下标是句子2 —— 依赖 batch 内 [s1, s2, s1, s2, ...] 的交错布局
    first_sentence = list(range(0, batch_size, 2))
    second_sentence = list(range(1, batch_size, 2))
    if shape_len == 3:
        sentence1 = last_hidden_state[first_sentence][:, 0].squeeze(dim=1)
        sentence2 = last_hidden_state[second_sentence][:, 0].squeeze(dim=1)
    else:
        sentence1 = last_hidden_state[first_sentence]
        sentence2 = last_hidden_state[second_sentence]
    return sentence1, sentence2
```

**这个函数的隐含契约是「batch 必须严格按 `[s1, s2, s1, s2, ...]` 交错排列」**。该布局由 `_embedding_data_collator` 保证（`swift/template/base.py:1757`）：它把每条样本展开成 `anchor_` + `positive_` 两行，`labels` 只有一个元素（即 `float(inputs.chosen.label)`，`base.py:522`）。

> ⚠️ **重要约束**：正因为是「奇偶配对」，`cosine_similarity` / `contrastive` / `online_contrastive` 的数据**不能带 `negative_messages`**。一旦带上，batch 布局会变成 `[anchor, positive, neg1, neg2, ...]`，配对关系立刻错位成 `(anchor, positive), (neg1, neg2), ...`。想用负例，只能走 `infonce`。

### 通路 B：多负例解析（`_parse_multi_negative_sentences`）

只服务于 `infonce`。

```python
# swift/loss/embedding.py:88

def _parse_multi_negative_sentences(sentences, labels, hard_negatives=None):
    # labels 形如 [1,0,0, 1,0,0,0]，1 的位置就是每组的起点
    split_indices = torch.nonzero(labels, as_tuple=False).squeeze().tolist()
    if isinstance(split_indices, int):
        split_indices = [split_indices]
    split_indices.append(len(labels))
    # 累积偏移，转换为可直接切片的边界
    split_indices = np.array(split_indices) + np.array(list(range(len(split_indices))))
    split_tensors = []

    for i in range(len(split_indices) - 1):
        start = split_indices[i]
        end = split_indices[i + 1]
        split_part = sentences[start:end]
        if hard_negatives is not None:
            negatives = len(split_part) - 2      # 减去 anchor 和 positive
            assert negatives > 0
            if negatives > hard_negatives:
                split_part = split_part[:hard_negatives + 2]      # 截断
            elif negatives < hard_negatives:
                # 不足则随机重复采样补齐
                selected = np.random.choice(list(range(negatives)), size=hard_negatives - negatives, replace=True)
                selected += 1  # skip positive
                split_part = torch.cat((split_part, split_part[selected]), dim=0)
        split_tensors.append(split_part)
    return split_tensors
```

这对应 `docs/source_en/BestPractices/Embedding.md` 里的布局约定：

```text
# batch_size=2, tensor anchor(1)+positive(1)+negatives(3) + anchor(1)+positive(1)+negatives(2)
# labels: [1,0,0,0, 1,0,0]
#         ↑组1: 1正3负      ↑组2: 1正2负
```

源码注释写得很清楚（`embedding.py:172-175`）：

```python
# split tensors into single sample
# for example: batch_size=2 with tensor anchor(1)+positive(1)+negatives(3) + anchor(1)+positive(1)+negatives(2)
# labels will be [1,0,0,0,1,0,0], meaning 1 positive, 3 negatives, 1 positive, 2 negatives
split_tensors = _parse_multi_negative_sentences(sentences, labels, hard_negatives)
```

而且 `labels` 的构造在 `_embedding_encode` 中是硬编码的：anchor 的标签是 1（默认）或 `float(label)`，**每个 negative 无条件追加 0.0**（`swift/template/base.py:522-537`）：

```python
labels.append(float(inputs.chosen.label) if inputs.chosen.label is not None else 1.0)

for idx, negative in enumerate(inputs.negative):
    ...
    labels.append(0.0)
```

**结论**：`infonce` 的 label 字段不表达「相似度」，它只是一个**分组分隔符**。这也解释了为什么 `infonce` 是唯一不需要 `label` 字段的损失。

## 14.4 CosineSimilarityLoss

```python
class CosineSimilarityLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        # You need to return a scalar representing the loss.
        cos_score_transformation = nn.Identity()
        loss_fct = MSELoss()
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        output = cos_score_transformation(torch.cosine_similarity(sentence1, sentence2))
        return loss_fct(output, labels.to(output.dtype).view(-1))
```

**数学公式：**

$$\mathcal{L} = \text{MSE}(\cos(\mathbf{u}, \mathbf{v}), y)$$

`nn.Identity()` 是遗留的扩展点（sentence-transformers 原版用于 CosineSimilarityLossWithScaling），ms-swift 里未启用，恒等透传。

**数据格式：**

```jsonl
# LLM
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]], "label": 0.8}
# MLLM
{"messages": [{"role": "user", "content": "<image>"}], "images": ["/some/images1.jpg"], "positive_messages": [[{"role": "user", "content": "<image>sentence"}]], "positive_images": [["/some/images2.jpg"]], "label": 0.7}
```

**应用场景：** STS（语义文本相似度）任务，标签为浮点相似度分数。典型数据集 `sentence-transformers/stsb`（0-5 分）。ms-swift 注册了 4 个 subset（`swift/dataset/dataset/llm.py:373-383`）：

```python
register_dataset(
    DatasetMeta(
        ms_dataset_id='sentence-transformers/stsb',
        hf_dataset_id='sentence-transformers/stsb',
        subsets=[
            SubsetDataset('default', preprocess_func=StsbPreprocessor()),           # embedding
            SubsetDataset('positive', preprocess_func=StsbPreprocessor(sim_threshold=0.75)),  # infonce
            SubsetDataset('generate', preprocess_func=StsbGeneratePreprocessor()),
            SubsetDataset('reg', preprocess_func=StsbRegressionPreprocessor()),
        ],
        tags=['similarity', '🔥']))
```

`:positive` 子集用 `sim_threshold=0.75` 过滤掉低分样本，**专门供 `infonce` 使用**——这是官方文档给出的「同一份 STS 数据如何分别喂给两种损失」的官方答案：

```python
class StsbPreprocessor(RowPreprocessor):

    def __init__(self, sim_threshold: Optional[float] = None):
        self.sim_threshold = sim_threshold
        super().__init__()

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            'messages': [{'role': 'user', 'content': row['sentence1']}],
            'positive_messages': [[{'role': 'user', 'content': row['sentence2']}]],
            'label': row['score'],
        }
        if self.sim_threshold is None or float(row['label']) >= self.sim_threshold:
            return row
        else:
            return None      # 低于阈值直接丢弃
```

**评估指标**（`PairedMetrics`，`swift/metrics/embedding.py:42-73`）：输出 8 个值——cosine / euclidean / manhattan / dot 四种度量各自与 label 的 Pearson 和 Spearman 相关系数。

```python
eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)
...
return {
    'pearson_cosine': eval_pearson_cosine,
    'pearson_euclidean': eval_pearson_euclidean,
    'pearson_manhattan': eval_pearson_manhattan,
    'pearson_dot_product': eval_pearson_dot,
    'spearman_cosine': eval_spearman_cosine,
    'spearman_euclidean': eval_spearman_euclidean,
    'spearman_manhattan': eval_spearman_manhattan,
    'spearman_dot_product': eval_spearman_dot,
}
```

**局限**：它只学会「给两个句子打一个相似度分」，学不到「把相关文档排在前面」的能力。**不适合直接用于向量检索**（检索需要的是可比较的向量空间，而 MSE 回归只约束了成对的相对距离，没有约束全局几何结构）。

## 14.5 ContrastiveLoss

```python
class ContrastiveLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        distances = distance_metric(sentence1, sentence2)
        margin = 0.5
        labels = labels.to(sentence1.dtype)
        losses = 0.5 * (labels * distances.pow(2) + (1 - labels) * F.relu(margin - distances).pow(2))
        return losses.mean()
```

距离度量枚举定义在 `embedding.py:19-24`，注明了来源：

```python
# Code borrowed from sentence_transformers
class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)  # noqa
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)  # noqa
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)  # noqa
```

实际代码中三种损失都硬编码了 `COSINE_DISTANCE`，枚举的其他成员保留是为了兼容扩展。

**数学公式：**

$$\mathcal{L} = \frac{1}{2}\left[y \cdot d^2 + (1-y) \cdot \max(0, m-d)^2\right]$$

其中 $d = 1 - \cos(\mathbf{u}, \mathbf{v})$ 是余弦距离，$m = 0.5$ 是 margin。

- **正样本（y=1）**：$0.5 d^2$，无最小距离要求，只要靠近即可
- **负样本（y=0）**：$0.5 \max(0, m-d)^2$，距离超过 margin 后**损失归零**，不再提供梯度

**超参数：**
- `margin`：负样本的最小距离阈值，**源码硬编码为 0.5**，无环境变量可调

**数据格式：**

```jsonl
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]], "label": 1}
{"messages": [{"role": "user", "content": "sentence1"}], "positive_messages": [[{"role": "user", "content": "sentence2"}]], "label": 0}
```

**应用场景：** 句子对分类任务，标签为 0（不相似）或 1（相似）。这是 NLI / 语义匹配类数据的标准形态。

## 14.6 OnlineContrastiveLoss

```python
class OnlineContrastiveLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        distance_matrix = distance_metric(sentence1, sentence2)
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        # 计算损失
        positive_loss = positive_pairs.pow(2).sum()
        margin = 0.5
        negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss
```

**与 `ContrastiveLoss` 的实质差异：**

| 维度 | ContrastiveLoss | OnlineContrastiveLoss |
|------|-----------------|----------------------|
| 样本筛选 | 全部样本都算 | **动态筛选困难样本** |
| 难负例判定 | — | `negs < poss.max()`：比最好正例还近的负例 |
| 难正例判定 | — | `poss > negs.min()`：比最差负例还远的正例 |
| 兜底逻辑 | 无 | 正/负样本仅 1 个时退化为 `mean()` 作阈值 |
| 归约方式 | `.mean()` | **`.sum()`** ⚠️ |
| 阈值来源 | 固定 margin=0.5 | margin 固定，但**筛选阈值随 batch 动态变化** |

**两个必须注意的实现细节：**

**① 筛选阈值是相对当前 batch 的。** `poss.max()` 和 `negs.min()` 都是 batch 内统计量，所以每个 step 参与损失的样本数不同，loss 会有额外抖动。这也意味着**单个 batch 内正负样本的绝对数量会显著影响梯度尺度**。

**② 用的是 `.sum()` 而不是 `.mean()`。** 这意味着：

$$\text{loss}_{\text{effective}} \propto \text{batch\_size}$$

batch size 翻倍，loss 也翻倍。如果要做 batch size 的缩放律实验，或者要和 `contrastive` / `infonce` 的 loss 曲线对比，需要显式乘上 $1/N$ 做归一。实际使用时建议开启梯度裁剪（`--gradient_clip_val`）以保证稳定。

**应用场景：** 数据分布不均衡时，自动聚焦困难样本。适合「大量简单负例 + 少量边界样本」的场景，如语义匹配中的细粒度区分。

## 14.7 InfoNCELoss ⭐

InfoNCE 是嵌入学习中最常用的损失函数，支持批内负样本对比。源码是 `swift/loss/embedding.py` 中最长的一个类（113-321 行，共 209 行），也是唯一支持**分布式负样本扩展**的损失。

### 14.7.1 数学公式

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{d}^+)/\tau)}{\sum_{i=0}^{N}\exp(\text{sim}(\mathbf{q}, \mathbf{d}_i)/\tau)}$$

**温度 $\tau$ 的作用**：$\tau \to 0$ 时 softmax 退化为 argmax，损失只在「正确项 vs 最强错误项」之间产生梯度；$\tau$ 变大时分布变平滑，所有负例都贡献梯度。默认 0.1 是 SimCLR / CLIP 系列验证过的经验值。

### 14.7.2 完整实现

```python
class InfonceLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.1'))  # temperature
        # calculate CE across the batch, meaning all samples will be negative except the matching positive
        use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
        hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)  # how many negative prompts kept in one sample
        # mask out fake negatives
        infonce_mask_fake_negative = strtobool(os.environ.get('INFONCE_MASK_FAKE_NEGATIVE', 'False'))
        fake_neg_margin = float(os.environ.get('INFONCE_FAKE_NEG_MARGIN', '0.1'))
        # enhanced components to align with Qwen3-Embedding denominator; controlled individually
        # defaults set to False for backward compatibility
        infonce_include_qq = strtobool(os.environ.get('INFONCE_INCLUDE_QQ', 'False'))
        infonce_include_dd = strtobool(os.environ.get('INFONCE_INCLUDE_DD', 'False'))
        if hard_negatives is not None:
            hard_negatives = int(hard_negatives)
        if self.is_megatron:
            from megatron.core import mpu
            rank, world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        else:
            rank, _, world_size, _ = get_dist_setting()
        # repeat of anchor(1)+positive(1)+negatives(n)
        sentences = outputs['last_hidden_state']
        ...
```

### 14.7.3 分布式负样本扩展

`use_batch=True` 时，负样本池会跨 GPU 合并（`embedding.py:137-170`）：

```python
if world_size > 1 and use_batch:
    if getattr(sequence_parallel, 'dp_group', None) is not None:
        all_sentences = sequence_parallel._gather_object_dp(sentences.unsqueeze(0))
        labels = sequence_parallel._gather_object_dp(labels)
        rank = sequence_parallel.dp_rank
    elif self.is_megatron:
        # Megatron 走 all_gather，先交换 shape 再交换张量
        ...
    else:
        # gather all the sentences and labels across the gpus when calculate loss across all batches of all gpus
        all_sentences = gather_object(sentences.unsqueeze(0))
        labels = gather_object(labels)
    # override the gathered one
    all_sentences[rank] = sentences
    for idx in range(len(all_sentences)):
        if idx == rank:
            continue
        # we don't calculate grad from other gpus
        all_sentences[idx] = all_sentences[idx].detach().to(sentences.device)
    sentences = torch.cat(all_sentences, dim=0)
```

**显存与速度的权衡**：8 卡 × batch 8 = 64 条样本时，负样本池达到 64 个（每条 anchor 对应 63 个候选文档）。

- **收益**：负样本数量提升 8 倍，检索任务的 Recall 通常有 1-3 个点的提升
- **代价**：`gather_object` 涉及 CPU 序列化，比单纯 GPU 通信慢；显存峰值也更高

> 官方 example 脚本里 `qwen3_emb.sh` 明确注释了 `--dataloader_drop_last must be true or eval gather will throw error`——因为 `INFONCE_USE_BATCH=True` 依赖 batch 内有足量样本，最后一个不满的 batch 会让跨卡 gather 的 shape 不一致。

### 14.7.4 两种分母构造模式

代码中有一个容易忽略的分支，由 `use_batch` 决定相似度矩阵的形状。

**`use_batch=True`（默认）——跨样本对比：**

```python
# [B, neg+2, D]
sentences = torch.stack(split_tensors, dim=0)
# base q->d similarities (includes own positive and all in-batch documents)
queries = sentences[:, 0].squeeze(1)  # [B, D]
docs_all = sentences[:, 1:].reshape(-1, sentences.size(2))  # [B*(neg+1), D]
qd_matrix = torch.matmul(queries, docs_all.T)  # [B, B*(neg+1)]
# target indices: start of each group's document block (its positive)
labels = torch.tensor(range(0,
                            sentences.size(0) * (sentences.size(1) - 1),
                            sentences.size(1) - 1)).view(-1).to(sentences.device)
```

分母包含**当前样本的所有负例 + batch 内其他样本的全部文档**。

**`use_batch=False`——仅样本内对比：**

```python
# [B, 1, D] * [B, neg+1, D]
similarity_matrix = torch.matmul(sentences[:, 0:1], sentences[:, 1:].transpose(1, 2)) / temperature
# The positive one is the first element
labels = torch.zeros(len(split_tensors), dtype=torch.int64).to(sentences.device)
loss = nn.CrossEntropyLoss()(similarity_matrix.squeeze(1), labels)
```

分母只有**当前样本自带的负例**。这时**每条样本的负例数量必须完全一致**，否则 `can_batched` 判定失败，退化为 for 循环（`embedding.py:178-180`）：

```python
can_batched = hard_negatives is not None
if hard_negatives is None and len(set([s.shape[0] for s in split_tensors])) == 1:
    # all tensors have the same batch size
    can_batched = True
```

**性能影响**：`can_batched=False` 时走 270-320 行的 for 循环分支，逐样本构造 logits。官方文档给出的优化建议是：**让数据里每条样本的负例数一致，这样即使不设 `INFONCE_HARD_NEGATIVES` 也能走批量矩阵乘法**。

### 14.7.5 增强组件：QQ / DD / 假负样本

**QQ（查询-查询）块**——把 query 之间的相似度也纳入分母：

```python
if infonce_include_qq:
    # q->q similarities; exclude self via -inf on diagonal to avoid accidental positives
    qq_matrix = torch.matmul(queries, queries.T)  # [B, B]
    qq_matrix = qq_matrix.clone()
    qq_matrix.fill_diagonal_(float('-inf'))
    logits_list.append(qq_matrix)
```

**DD（文档-文档）块**——把正例文档与所有文档的相似度纳入分母：

```python
if infonce_include_dd:
    # d+ -> d (doc-doc) similarities; exclude self-positive column per row
    pos_docs = sentences[:, 1].squeeze(1)  # [B, D]
    dd_matrix = torch.matmul(pos_docs, docs_all.T)  # [B, B*(neg+1)]
    # mask self positive per row: column index = row_idx * (neg+1)
    block = sentences.size(1) - 1  # (neg+1)
    if block > 0:
        row_idx = torch.arange(dd_matrix.size(0), device=dd_matrix.device)
        col_idx = row_idx * block
        dd_matrix[row_idx, col_idx] = float('-inf')
    logits_list.append(dd_matrix)
```

两个块都做了自排除（QQ 用 `fill_diagonal_(-inf)`，DD 用 `col_idx = row_idx * block` 逐行 mask），**避免把自己当成负样本**。源码注释说明这是为对齐 Qwen3-Embedding 的分母设计，且默认关闭以保持向后兼容。

**假负样本屏蔽**——检索数据里常有「假负例」：未被标注为正例的文档实际上也是相关的（比如同一文档的多个版本、同义改写）。强行当作负例会伤害模型：

```python
if infonce_mask_fake_negative:
    # thresholds derived from positive q->d scores per row
    row_idx = torch.arange(qd_matrix.size(0), device=qd_matrix.device)
    pos_scores = qd_matrix[row_idx, labels]
    thresholds = pos_scores.view(-1, 1).detach() + fake_neg_margin

    # qd block mask
    qd_block = qd_matrix.clone()
    qd_mask = qd_block > thresholds
    qd_block[qd_mask] = float('-inf')
    ...
    # d-d block (if present): self-positive column already masked unconditionally
    if infonce_include_dd:
        # align with Qwen3-Embedding, no threshold masking for d-d
        components.append(dd_matrix)
    similarity_matrix = torch.cat(components, dim=1)
```

逻辑是：**相似度超过「正例得分 + margin」的样本，直接从分母中剔除**（置 `-inf`，softmax 后贡献为 0）。注意 QQ 块也会被同样 mask，但 DD 块不 mask（源码注释：`align with Qwen3-Embedding, no threshold masking for d-d`）。

**启用后的整体流程：**

```python
# temperature scaling and CE
similarity_matrix = similarity_matrix / temperature
loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
```

### 14.7.6 关键超参数

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `INFONCE_TEMPERATURE` | 0.1 | 温度参数，控制分布锐度 |
| `INFONCE_USE_BATCH` | True | 是否使用批内负样本；False 时仅用样本自带负例 |
| `INFONCE_HARD_NEGATIVES` | None | 每条样本保留的负例数（统一后走批量矩阵乘法） |
| `INFONCE_MASK_FAKE_NEGATIVE` | False | 是否屏蔽假负样本 |
| `INFONCE_FAKE_NEG_MARGIN` | 0.1 | 假负样本判定阈值（叠加在正例得分之上） |
| `INFONCE_INCLUDE_QQ` | False | 是否把 q-q 相似度加入分母 |
| `INFONCE_INCLUDE_DD` | False | 是否把 d-d 相似度加入分母 |

全部通过**环境变量**而非 CLI 参数传入（这与 `LossScale` 用配置文件、InfoNCE 用环境变量的设计风格不同，但和上游 FlagEmbedding 的实现保持了一致）：

```bash
INFONCE_TEMPERATURE=0.1 \
INFONCE_USE_BATCH=True \
INFONCE_HARD_NEGATIVES=4 \
swift sft --task_type embedding --loss_type infonce ...
```

**典型设置建议：**

| 场景 | 推荐配置 |
|------|---------|
| 小 batch / 单卡调试 | `USE_BATCH=True`，不设 `HARD_NEGATIVES` |
| 追求最优效果 | `HARD_NEGATIVES=4~8` + `MASK_FAKE_NEGATIVE=True` |
| 数据无负例列 | `USE_BATCH=True`（纯靠批内负例） |
| 有高质量难负例 | `USE_BATCH=False` + 固定 `HARD_NEGATIVES` |
| 复现 Qwen3-Embedding | `INCLUDE_QQ=True` + `INCLUDE_DD=True` |

**应用场景：**
- 检索任务（如 MS MARCO）
- 嵌入模型训练（如 BGE、E5）
- Qwen3-Embedding 等最新模型

**评估指标**（`InfonceMetrics`，`swift/metrics/embedding.py:149-157`）：

```python
pos_neg_margin = np.mean(pos_scores - max_neg_scores).item()
mean_neg = np.mean(neg_scores)
mean_pos = np.mean(pos_scores)
return {'margin': pos_neg_margin, 'mean_neg': mean_neg, 'mean_pos': mean_pos}
```

- **`mean_pos`**：所有正例的平均相似度，应**越高越好**（趋近 1.0）
- **`mean_neg`**：所有负例的平均相似度，应**越低越好**（趋近 -1.0）
- **`margin`**：`mean_pos - mean_neg`，是检索任务最核心的单一指标

`margin` 长期不增长通常意味着：负例太难（标注噪声）、正例太泛（query 与正例语义漂移）、或 batch size 太小导致负例池不足。

## 14.8 损失函数对比

| 损失函数 | 正样本数 | 负样本数 | 归约方式 | 训练速度 | 效果 | 评估指标 |
|---------|---------|---------|---------|---------|------|---------|
| CosineSimilarity | 1对1 | 无 | `.mean()` | 快 | 回归任务，无法用于检索 | paired（8 指标） |
| Contrastive | 1对1 | 1对1 | `.mean()` | 中 | 基线 | paired（8 指标） |
| OnlineContrastive | 动态筛选 | 动态筛选 | **`.sum()`** | 中 | 聚焦困难样本 | paired（8 指标） |
| InfoNCE | 1对多 | 批内所有 | 批量矩阵乘 | 慢（多卡更慢） | **最佳** | infonce（margin/mean_pos/mean_neg） |

**选型决策树：**

```
你的数据是什么形态？
│
├─ 浮点相似度分数（0-5 / 0-1，如 STS）
│  └─ cosine_similarity
│
├─ 0/1 二元标签（句子对是否匹配）
│  ├─ 普通对比 → contrastive
│  └─ 想自动聚焦难例 → online_contrastive
│
└─ (query, positive) 检索对，无负例标注
   └─ infonce  ← 绝大多数实际场景
      ├─ 负例不足？→ 加大 batch_size / 多卡
      ├─ 有难负例？→ 加 negative_messages + USE_BATCH=False
      └─ 标注有噪声？→ MASK_FAKE_NEGATIVE=True
```

## 14.9 训练脚本示例

**InfoNCE + LoRA（Qwen3-Embedding-4B）**（`examples/train/embedding/qwen3/qwen3_emb.sh`）：

```bash
# 2*10GiB
# --dataloader_drop_last must be true or eval gather will throw error
CUDA_VISIBLE_DEVICES=0,1 \
INFONCE_TEMPERATURE=0.1 \
NPROC_PER_NODE=2 \
swift sft \
    --model Qwen/Qwen3-Embedding-4B \
    --task_type embedding \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --learning_rate 5e-5 \
    --target_modules all-linear \
    --dataset sentence-transformers/stsb:positive \
    --attn_impl flash_attn \
    --padding_free true \
    --torch_dtype bfloat16 \
    --max_length 8192 \
    --per_device_train_batch_size 8 \
    --num_train_epochs 5 \
    --loss_type infonce \
    --dataloader_drop_last true \
    --deepspeed zero2
```

**InfoNCE + 多模态（GME-Qwen2-VL-2B）**（`examples/train/embedding/train_gme.sh`）：

```bash
nproc_per_node=8
MAX_PIXELS=1003520 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model iic/gme-Qwen2-VL-2B-Instruct \
    --tuner_type lora \
    --dataset 'swift/TextCaps:emb' \
    --task_type embedding \
    --loss_type infonce \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --dataloader_drop_last true
```

## 小结

- **要点 1**：4 种损失函数覆盖从回归到对比学习的全部场景，唯一的注册入口是 `swift/loss/mapping.py` 的 `loss_map`
- **要点 2**：前三种走「成对」通路（`_parse_pair_sentence` 奇偶配对），InfoNCE 走「多负例」通路（`_parse_multi_negative_sentences` 按 label 切分），**因此前三种不能带 `negative_messages`，InfoNCE 不需要 `label` 字段**
- **要点 3**：InfoNCE 是唯一支持分布式负样本扩展的损失，可通过 7 个环境变量精细调控；其 QQ / DD / 假负样本屏蔽三个增强组件是与其他损失拉开差距的关键
- **要点 4**：OnlineContrastive 返回 `.sum()` 而非 `.mean()`，loss 尺度依赖 batch 大小和难例数量
- **要点 5**：`cosine_similarity` 学到的是「相似度打分能力」而非「向量检索能力」，不适用于构建向量索引；实际检索场景应选 `infonce`
