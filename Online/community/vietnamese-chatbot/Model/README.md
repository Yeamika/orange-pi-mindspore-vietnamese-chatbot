# Model 模块说明

本目录包含项目使用的模型文件，以 Git Submodule 形式管理。

## 目录结构

```
Model/
├── Qwen1.5-0.5B-Chat/      # Qwen1.5 0.5B 基础模型
├── Lora-Tach/              # Vietnamese LoRA 微调模型
└── lora-train_2026-01-07-16-29-48/  # 本地训练输出
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

### lora-train_2026-01-07-16-29-48
- **说明**: 本地训练输出的 LoRA adapter
- **状态**: 不纳入版本控制（在 .gitignore 中）

## 克隆项目

克隆本项目时需要使用 `--recursive` 参数来获取 submodules：

```bash
git clone --recursive https://gitee.com/yerongjiang2025/vietnamese-chatbot
```

如果已经克隆但没有 submodules，运行：

```bash
git submodule update --init --recursive
```

## 更新 Submodules

更新所有 submodules 到最新版本：

```bash
git submodule update --remote
```

更新特定 submodule：

```bash
git submodule update --remote Model/Qwen1.5-0.5B-Chat
```
