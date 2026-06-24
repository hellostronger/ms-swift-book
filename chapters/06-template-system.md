# 第 6 章 模板系统

## 6.1 Template 架构

模板系统是 ms-swift 的核心抽象层，负责将不同模型的对话格式统一处理。

```python
# swift/template/base.py

class Template:
    """模板基类"""
    
    def __init__(self, ...):
        self.system = system           # 系统提示词
        self.prefix = prefix           # 前缀模板
        self.prompt = prompt           # 对话模板
        self.suffix = suffix           # 后缀模板
        
    def encode(self, messages: List[Dict]) -> Dict:
        """编码消息为模型输入"""
        # 1. 应用模板
        input_ids, labels = self._apply_template(messages)
        
        # 2. 生成 attention_mask
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
    
    def decode(self, input_ids: List[int]) -> str:
        """解码 token 为文本"""
        return self.tokenizer.decode(input_ids)
```

## 6.2 模板注册

```python
# swift/template/register.py

TEMPLATE_MAPPING: Dict[str, TemplateMeta] = {}

@dataclass
class TemplateMeta:
    template_type: str
    prefix: List[Union[str, List[int]]]
    prompt: List[Union[str, List[int]]]
    suffix: List[Union[str, List[int]]]
    system_prefix: Optional[List[Union[str, List[int]]]] = None
    
def register_template(template_meta: TemplateMeta):
    TEMPLATE_MAPPING[template_meta.template_type] = template_meta
```

## 6.3 多模态支持

ms-swift 的模板系统原生支持多模态输入：

```python
class Template:
    def encode(self, messages: List[Dict]) -> Dict:
        # 处理文本
        text = self._format_messages(messages)
        
        # 处理图像
        images = self._extract_images(messages)
        if images:
            image_features = self.vision_tower(images)
            
        # 处理音频
        audios = self._extract_audios(messages)
        if audios:
            audio_features = self.audio_tower(audios)
            
        # 融合多模态特征
        inputs = self._merge_features(text_features, image_features, audio_features)
        return inputs
```

## 6.4 Agent 模板

ms-swift 支持 ReAct、Hermes、Qwen 等 Agent 模板：

```python
# swift/template/templates/react.py

class ReACTTemplate(Template):
    """ReAct Agent 模板"""
    
    def encode(self, messages: List[Dict]) -> Dict:
        # ReAct 格式
        # Thought: xxx
        # Action: xxx
        # Action Input: xxx
        # Observation: xxx
        
        for msg in messages:
            if msg['role'] == 'assistant':
                content = msg['content']
                # 解析 Action 和 Action Input
                action, action_input = self._parse_action(content)
                # 特殊处理
                ...
```

## 6.5 Loss Scale 集成

模板系统与 Loss Scale 系统深度集成，控制哪些 token 参与损失计算：

```python
class Template:
    def encode(self, messages: List[Dict]) -> Dict:
        # 应用 loss_scale
        context_list, loss_scale_list = self.loss_scale(
            context_list, context_types, messages
        )
        
        # 根据 loss_scale 生成 labels
        labels = self._generate_labels(context_list, loss_scale_list)
        
        return {'input_ids': input_ids, 'labels': labels}
```

## 6.6 常用模板

| 模板类型 | 适用模型 | 特点 |
|----------|----------|------|
| qwen | Qwen 系列 | 支持多模态 |
| llama3 | LLaMA3 | 标准对话格式 |
| chatglm3 | ChatGLM3 | 中文优化 |
| hermes | NousResearch | 支持 function call |
| react | Agent 模型 | ReAct 格式 |

## 小结

- **要点 1**：Template 系统统一了不同模型的对话格式
- **要点 2**：原生支持多模态输入（文本、图像、音频）
- **要点 3**：与 Loss Scale 系统集成，控制训练时的损失计算范围
