# 附录 B：损失函数速查表

## B.1 语言模型损失

### CustomCrossEntropyLoss

**用途**：标准语言模型训练

**公式**：
```
loss = -sum(log_softmax(logits) * labels) / num_items_in_batch
```

**适用场景**：
- SFT 训练
- 预训练

**参数**：
- `num_items_in_batch`：批次中的 token 数

---

## B.2 嵌入学习损失

### CosineSimilarityLoss

**用途**：句子对相似度回归

**公式**：
```
loss = MSE(cosine_similarity(sent1, sent2), labels)
```

**适用场景**：
- STS（语义文本相似度）任务
- 标签为 0-5 的相似度分数

**超参数**：
- 无特殊超参数

---

### ContrastiveLoss

**用途**：对比学习

**公式**：
```
d = cosine_distance(sent1, sent2)
loss = 0.5 * (y * d^2 + (1-y) * max(0, margin-d)^2)
```

**适用场景**：
- 句子对分类任务
- 标签为 0（不相似）或 1（相似）

**超参数**：
- `margin`：0.5（硬编码）

---

### OnlineContrastiveLoss

**用途**：在线对比学习（困难样本挖掘）

**公式**：
```
negs = distances[labels == 0]
poss = distances[labels == 1]
# 筛选阈值随 batch 动态变化
hard_negs = negs[negs < (poss.max() if len(poss)>1 else negs.mean())]
hard_poss = poss[poss > (negs.min() if len(negs)>1 else poss.mean())]
loss = sum(hard_poss^2) + sum(relu(margin - hard_negs)^2)   # 注意是 sum 不是 mean
```

**适用场景**：
- 数据分布不均衡
- 自动聚焦困难样本

**超参数**：
- `margin`：0.5（硬编码）

> ⚠️ **两个坑**：
> 1. 返回值用 `.sum()` 而非 `.mean()`，**loss 尺度随 batch size 和当步难例数量线性变化**
> 2. 筛选阈值 `poss.max()` / `negs.min()` 是 batch 内统计量，每步参与的样本数不同，loss 会有额外抖动

---

### InfoNCELoss

**用途**：多负样本检索

**公式**：
```
qd_matrix = queries @ docs_all.T / temperature   # [B, B*(neg+1)]
# target 是每组自己正样本的起始列
labels = tensor(range(0, B*(neg+1), neg+1))
loss = CrossEntropy(qd_matrix, labels)
```

**适用场景**：
- 检索任务（如 MS MARCO）
- 嵌入模型训练（如 BGE、E5、Qwen3-Embedding）
- **唯一支持跨 GPU 负样本扩展**的损失

**数据格式**：`messages`（anchor）+ `positive_messages`，`negative_messages` 可选。`label` 字段仅用作**分组分隔符**，不表达相似度。

**超参数**（环境变量）：
- `INFONCE_TEMPERATURE`：0.1（温度参数）
- `INFONCE_USE_BATCH`：True（True=批内负样本 / False=仅样本自带负例）
- `INFONCE_HARD_NEGATIVES`：None（固定每条样本的负例数；设了才能走批量矩阵乘法）
- `INFONCE_MASK_FAKE_NEGATIVE`：False（屏蔽相似度 > `正例得分 + margin` 的假负例）
- `INFONCE_FAKE_NEG_MARGIN`：0.1（假负样本阈值）
- `INFONCE_INCLUDE_QQ`：False（把 q-q 相似度加入分母，对角线置 -inf）
- `INFONCE_INCLUDE_DD`：False（把 d-d 相似度加入分母，逐行屏蔽自身正例列）

> 注意：`INFONCE_USE_BATCH=True` 时需配合 `--dataloader_drop_last true`，否则评估阶段跨卡 gather 会因 batch 不齐报错。

---

## B.3 重排损失

### PointwiseRerankerLoss

**用途**：点式重排

**公式**：
```
logits = outputs.logits.squeeze(1)      # [B]
loss = BCEWithLogits(logits, labels)    # 内部含 sigmoid
```

**适用场景**：
- 简单排序任务
- 二分类场景
- 数据量大、显存受限

**超参数**：
- 无特殊超参数

---

### ListwiseRerankerLoss

**用途**：列表式重排

**公式**：
```
# 按 labels==1 的位置切分组，每组以正样本开头
for each group:
    scaled_logits = group_logits / temperature
    loss += CrossEntropy(scaled_logits, target=0)   # 正样本应在组内排第一
loss = mean(losses)
```

**适用场景**：
- 精确排序
- 考虑文档间相对关系
- 有充足负例（负例难度直接决定效果）

**数据格式**（由 `_reranker_data_collator` 生成）：
```
labels = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, ...]
         └── 组1: 1正7负 ──┘└── 组2: 1正7负 ──┘
```

**超参数**（环境变量）：
- `LISTWISE_RERANKER_TEMPERATURE`：1.0（温度参数）
- `LISTWISE_RERANKER_MIN_GROUP_SIZE`：2（小于此值的组跳过不计算）
- `MAX_POSITIVE_SAMPLES`：1（每条数据抽取的正样本数）
- `MAX_NEGATIVE_SAMPLES`：7（每组负样本数）

> ⚠️ **显存坑**：一条原始数据会被展开成 `MAX_POSITIVE_SAMPLES × (1 + MAX_NEGATIVE_SAMPLES)` 条，默认配置下 `per_device_train_batch_size` 要按 1/8 折算。

---

## B.4 DPO 损失变体

### sigmoid（默认）

**公式**：
```
losses = -log(sigmoid(beta * logits))
```

**特点**：标准 DPO，最稳定

---

### hinge

**公式**：
```
losses = relu(1 - beta * logits)
```

**特点**：更关注难样本

---

### ipo

**公式**：
```
losses = (logits - 1/(2*beta))^2
```

**特点**：避免 DPO 过拟合

---

### exo_pair

**公式**：
```
losses = sigmoid(beta*logits) * (log_sigmoid(beta*logits) - log(1-label_smoothing)) + 
         sigmoid(-beta*logits) * (log_sigmoid(-beta*logits) - log(label_smoothing))
```

**特点**：指数加权

---

### bco_pair

**公式**：
```
chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
rejected_rewards = beta * (rejected_logps - ref_rejected_logps)
delta = running_mean(rewards)
losses = -log_sigmoid(chosen_rewards - delta) - log_sigmoid(-(rejected_rewards - delta))
```

**特点**：使用运行均值

---

### sppo_hard

**公式**：
```
a = chosen_logps - ref_chosen_logps
b = rejected_logps - ref_rejected_logps
losses = (a - 0.5/beta)^2 + (b + 0.5/beta)^2
```

**特点**：自博弈优化

---

### nca_pair

**公式**：
```
chosen_rewards = (chosen_logps - ref_chosen_logps) * beta
rejected_rewards = (rejected_logps - ref_rejected_logps) * beta
losses = -log_sigmoid(chosen_rewards) - 0.5*log_sigmoid(-chosen_rewards) - 0.5*log_sigmoid(-rejected_rewards)
```

**特点**：噪声对比学习

---

### aot / aot_unpaired

**公式**：
```
sorted_delta = sort(chosen_logratios) - sort(rejected_logratios)
losses = -log_sigmoid(beta * sorted_delta)
```

**特点**：最优传输对齐

---

### apo_zero / apo_down

**公式**：
```
# apo_zero
losses = (1 - sigmoid(beta*chosen_logratios)) + sigmoid(beta*rejected_logratios)

# apo_down
losses = sigmoid(beta*chosen_logratios) + (1 - sigmoid(beta*(chosen_logratios-rejected_logratios)))
```

**特点**：自适应优化

---

### discopop

**公式**：
```
logits = logits * beta
log_ratio_modulation = sigmoid(logits / discopop_tau)
logistic_component = -log_sigmoid(logits)
exp_component = exp(-logits)
losses = logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
```

**特点**：双组件加权

---

### robust

**公式**：
```
losses = (-log_sigmoid(beta*logits) * (1-label_smoothing) + 
          log_sigmoid(-beta*logits) * label_smoothing) / (1 - 2*label_smoothing)
```

**特点**：标签平滑

---

### sft

**公式**：
```
losses = nll_loss
```

**特点**：SFT 辅助损失

---

## B.5 奖励模型损失

### Reward Model Loss

**公式**：
```
if margin is not None:
    loss = -log_sigmoid(rewards_chosen - rewards_rejected - margin)
else:
    loss = -log_sigmoid(rewards_chosen - rewards_rejected)

if center_rewards_coefficient is not None:
    loss += center_rewards_coefficient * mean((rewards_chosen + rewards_rejected)^2)
```

**适用场景**：
- 奖励模型训练
- PPO 前置步骤

**超参数**：
- `center_rewards_coefficient`：None（推荐 0.01）

---

## B.6 GKD 损失

### JSD Loss

**公式**：
```
if beta == 0:
    jsd = KL(student || teacher)  # forward KL
elif beta == 1:
    jsd = KL(teacher || student)  # reverse KL
else:
    m = beta * teacher + (1-beta) * student
    jsd = beta * KL(m || teacher) + (1-beta) * KL(m || student)

loss = jsd + sft_alpha * sft_loss
```

**适用场景**：
- 知识蒸馏
- 模型压缩

**超参数**：
- `beta`：0.5（JSD 插值参数）
- `temperature`：0.9（温度缩放）
- `sft_alpha`：0.0（SFT 损失权重）

---

## B.7 损失函数选择指南

### 按任务类型选择

| 任务类型 | 推荐损失 | 说明 |
|---------|---------|------|
| SFT | cross_entropy | 标准语言模型损失 |
| DPO | sigmoid | 最稳定的 DPO 变体 |
| CPO/SimPO | cpo/simpo | 无需参考模型 |
| 嵌入-检索 | infonce | 批内/跨卡负样本，检索首选 |
| 嵌入-STS 回归 | cosine_similarity | 只适合相似度打分，不适合建索引 |
| 嵌入-二分类 | contrastive / online_contrastive | 复现 GTE-BERT 类训练配方时用 |
| 重排 | listwise_reranker | 追求 nDCG；显存够时优于 pointwise |
| 重排（显存受限） | pointwise_reranker | 效率高，评估仍用 MRR/nDCG |
| 知识蒸馏 | JSD | GKD 标准损失 |

### 按场景选择

| 场景 | 推荐损失 | 说明 |
|------|---------|------|
| 标准对齐 | DPO sigmoid | 最稳定 |
| 避免过拟合 | DPO ipo | 平方损失 |
| 难样本 | DPO hinge | 关注困难样本 |
| 噪声数据 | DPO robust | 标签平滑 |
| 检索任务 | InfoNCE | 批内负样本 |
| 精确排序 | ListwiseReranker | 列表级对比 |
| 负例标注有噪声 | InfoNCE + `MASK_FAKE_NEGATIVE=True` | 避免把相关文档当负例 |
| 需要多档位向量 | InfoNCE + `--mrl_dims` | Matryoshka，一次训练多维度 |
| batch 太小负例不足 | InfoNCE + 加大 batch / 多卡训练 | 负例池 = batch × world_size |

### embedding 损失决策树

```
数据里有什么标签？
├─ 浮点相似度分数（STS / stsb）
│  └─ cosine_similarity        ← 学到的是打分能力，不能建向量索引
├─ 0 / 1 二元标签
│  ├─ 普通对比     → contrastive
│  └─ 聚焦难例     → online_contrastive
└─ 只有 (query, positive)，无负例标注
   └─ infonce                    ← 绝大多数实际场景
      ├─ 负例不够  → 加大 batch_size / 多卡
      ├─ 有难负例  → 加 negative_messages + USE_BATCH=False + HARD_NEGATIVES
      └─ 标注有噪声 → MASK_FAKE_NEGATIVE=True
```

### 两个高频踩坑

| 坑 | 原因 | 解决 |
|---|------|------|
| `contrastive` 系列结果异常 | `_parse_pair_sentence` 按奇偶下标配对，数据带了 `negative_messages` 会错位 | 这三种损失的数据**不要写** `negative_messages` |
| `infonce` 评估时 gather 报错 | `USE_BATCH=True` 依赖 batch 内足量样本 | 加 `--dataloader_drop_last true` |

---

## B.8 损失函数对比表

| 损失函数 | 计算复杂度 | 显存占用 | 训练速度 | 效果 |
|---------|-----------|---------|---------|------|
| CrossEntropy | 低 | 低 | 快 | 基线 |
| CosineSimilarity | 低 | 低 | 快 | 回归，无法建索引 |
| Contrastive | 中 | 低 | 中 | 基线 |
| OnlineContrastive | 中 | 低 | 中 | 困难样本（loss 尺度依赖 batch） |
| InfoNCE | 高 | 高（负例池 × world_size） | 慢 | 检索最佳 |
| PointwiseReranker | 低 | 低 | 快 | 简单排序 |
| ListwiseReranker | 中 | 中（batch × 8） | 中 | 精确排序 |
| DPO sigmoid | 中 | 中（2倍模型） | 中 | 好 |
| DPO ipo | 中 | 中 | 中 | 好 |
| JSD | 中 | 中（2倍模型） | 中 | 好 |
