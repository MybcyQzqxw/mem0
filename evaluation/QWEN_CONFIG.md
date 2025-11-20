# 配置通义千问 API

## 修改 .env 文件

在 `.env` 文件中添加以下配置:

```bash
# 选择 LLM 提供商: openai 或 qwen
LLM_PROVIDER="qwen"

# 阿里云通义千问 API Key (从 https://dashscope.console.aliyun.com/ 获取)
DASHSCOPE_API_KEY="your-dashscope-api-key"

# 模型配置 (通义千问可用模型)
MODEL="qwen-plus"  # 或 qwen-turbo, qwen-max

# Embedding 模型
EMBEDDING_MODEL="text-embedding-v3"

# 保留原有配置
MEM0_API_KEY="your-mem0-api-key"
MEM0_PROJECT_ID="your-mem0-project-id"
MEM0_ORGANIZATION_ID="your-mem0-organization-id"
ZEP_API_KEY="your-zep-api-key"
```

## 获取通义千问 API Key

1. 访问 [阿里云百炼平台](https://dashscope.console.aliyun.com/)
2. 注册/登录阿里云账号
3. 开通 DashScope 服务
4. 在控制台获取 API Key

## 可用模型

通义千问提供以下模型:
- `qwen-turbo`: 快速响应,适合日常对话
- `qwen-plus`: 平衡性能和成本 (推荐)
- `qwen-max`: 最强性能
- `qwen-long`: 长文本处理

## 使用方式

配置完成后,直接运行原有命令即可:

```bash
make run-mem0-search
```

代码会自动根据 `LLM_PROVIDER` 选择使用 OpenAI 还是通义千问。

## 切换回 OpenAI

如需切换回 OpenAI,只需将 `LLM_PROVIDER` 改为:

```bash
LLM_PROVIDER="openai"
OPENAI_API_KEY="your-openai-api-key"
MODEL="gpt-4o-mini"
```
