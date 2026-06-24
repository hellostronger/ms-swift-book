# 第 7 章 数据集系统

## 7.1 数据集架构

ms-swift 的数据集系统支持从 ModelScope Hub 和本地路径加载各种格式的数据集。

```
swift/dataset/
├── dataset_meta.py     # 数据集元信息
├── dataset_syntax.py   # 数据集语法定义
├── loader.py           # 数据加载器
├── packing.py          # Packing 打包
├── preprocessor/       # 数据预处理器
├── register.py         # 数据集注册
└── utils.py            # 工具函数
```

## 7.2 数据格式规范

ms-swift 支持多种标准数据格式：

### 7.2.1 SFT 格式

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}
  ]
}
```

### 7.2.2 偏好数据格式（DPO/GRPO）

```json
{
  "messages": [
    {"role": "user", "content": "1+1等于几？"}
  ],
  "chosen": {
    "role": "assistant",
    "content": "1+1等于2。"
  },
  "rejected": {
    "role": "assistant",
    "content": "我不确定。"
  }
}
```

### 7.2.3 嵌入数据格式

```json
{
  "query": "什么是机器学习？",
  "positive": ["机器学习是人工智能的一个分支..."],
  "negative": ["今天天气很好..."]
}
```

## 7.3 数据预处理流水线

```
原始数据
   ↓
RowPreprocessor    → 行级预处理
   ↓
MessageProcessor   → 消息格式转换
   ↓
Template.encode    → 模板编码
   ↓
Tokenize           → 分词
   ↓
Batch              → 批处理
```

```python
# swift/dataset/preprocessor/

class RowPreprocessor:
    """行级数据预处理器"""
    
    def __call__(self, row: Dict) -> Dict:
        # 1. 字段映射
        row = self._map_fields(row)
        
        # 2. 格式验证
        row = self._validate(row)
        
        # 3. 数据增强（可选）
        row = self._augment(row)
        
        return row
```

## 7.4 多模态数据

ms-swift 支持图文混合数据：

```json
{
  "messages": [
    {"role": "user", "content": "<image>请描述这张图片"},
    {"role": "assistant", "content": "这是一只可爱的猫"}
  ],
  "images": ["path/to/cat.jpg"]
}
```

## 7.5 Packing 策略

Packing 将多个短样本拼接为一个长序列，提高训练效率：

```python
# swift/dataset/packing.py

def pack_dataset(samples: List[Dict], max_length: int) -> List[Dict]:
    """将多个样本打包为一个序列"""
    packed = []
    current = {'input_ids': [], 'labels': [], 'attention_mask': []}
    
    for sample in samples:
        if len(current['input_ids']) + len(sample['input_ids']) > max_length:
            # 填充当前包
            packed.append(_pad_sequence(current, max_length))
            current = {'input_ids': [], 'labels': [], 'attention_mask': []}
        
        # 追加样本
        current['input_ids'].extend(sample['input_ids'])
        current['labels'].extend(sample['labels'])
        current['attention_mask'].extend(sample['attention_mask'])
    
    return packed
```

## 7.6 数据集注册

```python
from swift.dataset import register_dataset, DatasetMeta

register_dataset(DatasetMeta(
    dataset_name='my_dataset',
    dataset_path='path/to/data.jsonl',
    columns={'messages': 'messages'},
    tags=['sft', 'chinese'],
))
```

## 小结

- **要点 1**：ms-swift 支持多种标准数据格式（SFT、偏好数据、嵌入数据）
- **要点 2**：数据预处理流水线包括字段映射、格式验证、模板编码
- **要点 3**：Packing 策略可显著提高训练效率
