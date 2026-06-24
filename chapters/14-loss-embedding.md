# 第 14 章 嵌入学习损失函数

## 14.1 概述

ms-swift 支持 4 种嵌入学习损失函数，用于训练 Embedding 模型：

| 损失函数 | 适用场景 | 核心思想 |
|---------|---------|---------|
| CosineSimilarity | 句子对相似度回归 | 余弦相似度 + MSE |
| Contrastive | 对比学习 | 正负样本距离对比 |
| OnlineContrastive | 在线对比学习 | 困难样本挖掘 |
| InfoNCE | 多负样本检索 | 批内负样本对比 |

## 14.2 CosineSimilarityLoss

```python
class CosineSimilarityLoss(BaseLoss):
    """余弦相似度损失（回归任务）"""

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        cos_score_transformation = nn.Identity()
        loss_fct = MSELoss()
        
        # 解析句子对
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        
        # 计算余弦相似度
        output = cos_score_transformation(torch.cosine_similarity(sentence1, sentence2))
        
        # MSE 损失
        return loss_fct(output, labels.to(output.dtype).view(-1))
```

**数学公式：**

$$\mathcal{L} = \text{MSE}(\cos(\mathbf{u}, \mathbf{v}), y)$$

**应用场景：** STS（语义文本相似度）任务，标签为 0-5 的相似度分数。

## 14.3 ContrastiveLoss

```python
class ContrastiveLoss(BaseLoss):
    """对比学习损失"""

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        
        # 使用余弦距离
        distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        distances = distance_metric(sentence1, sentence2)
        
        margin = 0.5
        labels = labels.to(sentence1.dtype)
        
        # 正样本：最小化距离；负样本：最大化距离（受 margin 约束）
        losses = 0.5 * (labels * distances.pow(2) + 
                       (1 - labels) * F.relu(margin - distances).pow(2))
        return losses.mean()
```

**数学公式：**

$$\mathcal{L} = \frac{1}{2}\left[y \cdot d^2 + (1-y) \cdot \max(0, m-d)^2\right]$$

其中 $d = 1 - \cos(\mathbf{u}, \mathbf{v})$ 是余弦距离，$m$ 是 margin。

**超参数：**
- `margin`：负样本的最小距离阈值，默认 0.5

**应用场景：** 句子对分类任务，标签为 0（不相似）或 1（相似）。

## 14.4 OnlineContrastiveLoss

```python
class OnlineContrastiveLoss(BaseLoss):
    """在线对比学习损失（困难样本挖掘）"""

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        sentence1, sentence2 = _parse_pair_sentence(outputs)
        
        distance_metric = SiameseDistanceMetric.COSINE_DISTANCE
        distance_matrix = distance_metric(sentence1, sentence2)
        
        # 分离正负样本
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # 挖掘困难样本
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        # 计算损失
        positive_loss = positive_pairs.pow(2).sum()
        margin = 0.5
        negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss
```

**核心思想：**
- 困难正样本：距离 > 负样本最小距离
- 困难负样本：距离 < 正样本最大距离

**应用场景：** 数据分布不均衡时，自动聚焦困难样本。

## 14.5 InfoNCELoss ⭐

InfoNCE 是嵌入学习中最常用的损失函数，支持批内负样本对比。

```python
class InfonceLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        # 从环境变量读取配置
        temperature = float(os.environ.get('INFONCE_TEMPERATURE', '0.1'))
        use_batch = strtobool(os.environ.get('INFONCE_USE_BATCH', 'True'))
        hard_negatives = os.environ.get('INFONCE_HARD_NEGATIVES', None)
        infonce_mask_fake_negative = strtobool(os.environ.get('INFONCE_MASK_FAKE_NEGATIVE', 'False'))
        fake_neg_margin = float(os.environ.get('INFONCE_FAKE_NEG_MARGIN', '0.1'))
        infonce_include_qq = strtobool(os.environ.get('INFONCE_INCLUDE_QQ', 'False'))
        infonce_include_dd = strtobool(os.environ.get('INFONCE_INCLUDE_DD', 'False'))
        
        # ... 分布式处理 ...
        
        # 计算相似度矩阵
        if use_batch:
            # 批内负样本
            similarity_matrix = torch.matmul(queries, docs_all.T) / temperature
            
            # 可选：加入 q-q 和 d-d 相似度
            if infonce_include_qq:
                qq_matrix = torch.matmul(queries, queries.T)
                qq_matrix.fill_diagonal_(float('-inf'))  # 排除自身
                logits_list.append(qq_matrix)
            
            if infonce_include_dd:
                dd_matrix = torch.matmul(pos_docs, docs_all.T)
                # 排除正样本
                dd_matrix[row_idx, col_idx] = float('-inf')
                logits_list.append(dd_matrix)
            
            # 可选：mask 假负样本
            if infonce_mask_fake_negative:
                thresholds = pos_scores.view(-1, 1) + fake_neg_margin
                qd_mask = qd_matrix > thresholds
                qd_block[qd_mask] = float('-inf')
            
            similarity_matrix = torch.cat(logits_list, dim=1)
            loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
        
        return loss
```

**数学公式：**

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(\mathbf{q}, \mathbf{d}^+)/\tau)}{\sum_{i=0}^{N}\exp(\text{sim}(\mathbf{q}, \mathbf{d}_i)/\tau)}$$

**关键超参数：**

| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| INFONCE_TEMPERATURE | 0.1 | 温度参数，控制分布锐度 |
| INFONCE_USE_BATCH | True | 是否使用批内负样本 |
| INFONCE_HARD_NEGATIVES | None | 硬负样本数量 |
| INFONCE_MASK_FAKE_NEGATIVE | False | 是否屏蔽假负样本 |
| INFONCE_FAKE_NEG_MARGIN | 0.1 | 假负样本阈值 |
| INFONCE_INCLUDE_QQ | False | 是否加入 q-q 相似度 |
| INFONCE_INCLUDE_DD | False | 是否加入 d-d 相似度 |

**应用场景：**
- 检索任务（如 MS MARCO）
- 嵌入模型训练（如 BGE、E5）
- Qwen3-Embedding 等最新模型

## 14.6 损失函数对比

| 损失函数 | 正样本数 | 负样本数 | 训练速度 | 效果 |
|---------|---------|---------|---------|------|
| CosineSimilarity | 1对1 | 无 | 快 | 回归任务 |
| Contrastive | 1对1 | 1对1 | 中 | 基线 |
| OnlineContrastive | 动态 | 动态 | 中 | 困难样本 |
| InfoNCE | 1对多 | 批内所有 | 慢 | 最佳 |

## 小结

- **要点 1**：4 种损失函数覆盖从回归到对比学习的全部场景
- **要点 2**：InfoNCE 是最强大的损失函数，支持批内负样本和硬负样本挖掘
- **要点 3**：通过环境变量可灵活配置 InfoNCE 的行为
