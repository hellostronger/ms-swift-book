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
hard_negs = negs[negs < max(poss)]
hard_poss = poss[poss > min(negs)]
loss = sum(hard_poss^2) + sum(relu(margin - hard_negs)^2)
```

**适用场景**：
- 数据分布不均衡
- 自动聚焦困难样本

**超参数**：
- `margin`：0.5（硬编码）

---

### InfoNCELoss

**用途**：多负样本检索

**公式**：
```
sim_matrix = (sentences @ sentences.T) / temperature
loss = CrossEntropy(sim_matrix, labels)
```

**适用场景**：
- 检索任务（如 MS MARCO）
- 嵌入模型训练（如 BGE、E5）

**超参数**（环境变量）：
- `INFONCE_TEMPERATURE`：0.1（温度参数）
- `INFONCE_USE_BATCH`：True（是否使用批内负样本）
- `INFONCE_HARD_NEGATIVES`：None（硬负样本数量）
- `INFONCE_MASK_FAKE_NEGATIVE`：False（是否屏蔽假负样本）
- `INFONCE_FAKE_NEG_MARGIN`：0.1（假负样本阈值）
- `INFONCE_INCLUDE_QQ`：False（是否加入 q-q 相似度）
- `INFONCE_INCLUDE_DD`：False（是否加入 d-d 相似度）

---

## B.3 重排损失

### PointwiseRerankerLoss

**用途**：点式重排

**公式**：
```
loss = BCEWithLogits(logits, labels)
```

**适用场景**：
- 简单排序任务
- 二分类场景

**超参数**：
- 无特殊超参数

---

### ListwiseRerankerLoss

**用途**：列表式重排

**公式**：
```
for each group:
    scaled_logits = logits / temperature
    loss += CrossEntropy(scaled_logits, target=0)
loss = mean(losses)
```

**适用场景**：
- 精确排序
- 考虑文档间相对关系

**超参数**（环境变量）：
- `LISTWISE_RERANKER_TEMPERATURE`：1.0（温度参数）
- `LISTWISE_RERANKER_MIN_GROUP_SIZE`：2（最小组大小）

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
| SFT | CrossEntropy | 标准语言模型损失 |
| DPO | sigmoid | 最稳定的 DPO 变体 |
| CPO/SimPO | cpo/simpo | 无需参考模型 |
| 嵌入学习 | InfoNCE | 支持批内负样本 |
| 重排 | ListwiseReranker | 考虑文档间关系 |
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

---

## B.8 损失函数对比表

| 损失函数 | 计算复杂度 | 显存占用 | 训练速度 | 效果 |
|---------|-----------|---------|---------|------|
| CrossEntropy | 低 | 低 | 快 | 基线 |
| CosineSimilarity | 低 | 低 | 快 | 回归任务 |
| Contrastive | 中 | 中 | 中 | 基线 |
| OnlineContrastive | 中 | 中 | 中 | 困难样本 |
| InfoNCE | 高 | 高 | 慢 | 最佳 |
| PointwiseReranker | 低 | 低 | 快 | 简单排序 |
| ListwiseReranker | 中 | 中 | 中 | 精确排序 |
| DPO sigmoid | 中 | 中（2倍模型） | 中 | 好 |
| DPO ipo | 中 | 中 | 中 | 好 |
| JSD | 中 | 中（2倍模型） | 中 | 好 |
