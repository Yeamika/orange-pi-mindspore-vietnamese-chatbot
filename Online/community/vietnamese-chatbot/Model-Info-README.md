# Model 模块说明

本目录包含项目使用的模型文件， 已经上传至modelers社区。

## 目录结构

```
Model/
├── Qwen1.5-0.5B-Chat/      # Qwen1.5 0.5B 基础模型
├── Lora-Tach/              # Vietnamese LoRA 微调模型(已合并)
├── Vietnamese-LoRA-Only/              # Vietnamese LoRA Only(LoRA 适配器)
└── lora-train_2026-01-07-16-29-48/  # 本地训练输出(LoRA 适配器)
```

## Submodules

### Qwen1.5-0.5B-Chat
- **Git 地址**: https://modelers.cn/yaemika/Qwen1.5-0.5B-Chat.git
- **访问地址**: https://modelers.cn/models/yaemika/Qwen1.5-0.5B-Chat
- **说明**: Qwen1.5 0.5B 对话模型，作为基础模型使用
- **大小**: ~1.2 GB
- **文件**: model.safetensors, tokenizer 配置等

### Lora-Tach
- **Git 地址**: https://modelers.cn/yaemika/vietnamese-chatbot.git
- **访问地址**: https://modelers.cn/models/yaemika/vietnamese-chatbot
- **说明**: 基于 Qwen 微调的越南语对话模型
- **大小**: ~900 MB
- **文件**: model.safetensors, tokenizer 配置等


### Vietnamese-LoRA-Only
- **Git 地址**: https://modelers.cn/yaemika/Vietnamese-LoRA-Only.git
- **访问地址**: https://modelers.cn/models/yaemika/Vietnamese-LoRA-Only
- **说明**: Vietnamese LoRA 适配器模型（独立版本）
- **大小**: ~15 MB
- **文件**: adapter_model.safetensors, adapter_config.json, tokenizer 配置等

### lora-train_2026-01-07-16-29-48
- **说明**: 本地训练输出的 LoRA adapter
- **状态**: 已上传至 modelers.cn (Vietnamese-LoRA-Only)

