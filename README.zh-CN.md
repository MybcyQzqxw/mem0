<p align="center">
  <a href="https://github.com/mem0ai/mem0">
    <img src="docs/images/banner-sm.png" width="800px" alt="Mem0 - 个性化AI的记忆层">
  </a>
</p>
<p align="center" style="display: flex; justify-content: center; gap: 20px; align-items: center;">
  <a href="https://trendshift.io/repositories/11194" target="blank">
    <img src="https://trendshift.io/api/badge/repositories/11194" alt="mem0ai%2Fmem0 | Trendshift" width="250" height="55"/>
  </a>
</p>

<p align="center">
  <a href="https://mem0.ai">了解更多</a>
  ·
  <a href="https://mem0.dev/DiG">加入 Discord</a>
  ·
  <a href="https://mem0.dev/demo">演示</a>
  ·
  <a href="https://mem0.dev/openmemory">OpenMemory</a>
</p>

<p align="center">
  <a href="https://mem0.dev/DiG">
    <img src="https://img.shields.io/badge/Discord-%235865F2.svg?&logo=discord&logoColor=white" alt="Mem0 Discord">
  </a>
  <a href="https://pepy.tech/project/mem0ai">
    <img src="https://img.shields.io/pypi/dm/mem0ai" alt="Mem0 PyPI - Downloads">
  </a>
  <a href="https://github.com/mem0ai/mem0">
    <img src="https://img.shields.io/github/commit-activity/m/mem0ai/mem0?style=flat-square" alt="GitHub commit activity">
  </a>
  <a href="https://pypi.org/project/mem0ai" target="blank">
    <img src="https://img.shields.io/pypi/v/mem0ai?color=%2334D058&label=pypi%20package" alt="Package version">
  </a>
  <a href="https://www.npmjs.com/package/mem0ai" target="blank">
    <img src="https://img.shields.io/npm/v/mem0ai" alt="Npm package">
  </a>
  <a href="https://www.ycombinator.com/companies/mem0">
    <img src="https://img.shields.io/badge/Y%20Combinator-S24-orange?style=flat-square" alt="Y Combinator S24">
  </a>
</p>

<p align="center">
  <a href="https://mem0.ai/research"><strong>📄 构建具有可扩展长期记忆的生产级AI代理 →</strong></a>
</p>
<p align="center">
  <strong>⚡ 比 OpenAI Memory 准确度高 26% • 🚀 速度快 91% • 💰 令牌使用减少 90%</strong>
</p>

> **🎉 mem0ai v1.0.0 现已发布！** 此重大版本包括 API 现代化、改进的向量存储支持和增强的 GCP 集成。[查看迁移指南 →](MIGRATION_GUIDE_v1.0.zh-CN.md)

##  🔥 研究亮点
- 在 LOCOMO 基准测试中**准确度比 OpenAI Memory 高 26%**
- **响应速度比全上下文快 91%**，确保大规模低延迟
- **令牌使用比全上下文少 90%**，在不妥协的情况下降低成本
- [阅读完整论文](https://mem0.ai/research)

# 简介

[Mem0](https://mem0.ai)（"mem-zero"）通过智能记忆层增强 AI 助手和代理，实现个性化的 AI 交互。它能记住用户偏好，适应个体需求，并随时间不断学习——非常适合客户支持聊天机器人、AI 助手和自主系统。

### 主要功能和用例

**核心能力：**
- **多级记忆**：无缝保留用户、会话和代理状态，实现自适应个性化
- **开发者友好**：直观的 API、跨平台 SDK 和完全托管的服务选项

**应用场景：**
- **AI 助手**：一致的、富含上下文的对话
- **客户支持**：回忆过往工单和用户历史以提供定制化帮助
- **医疗保健**：跟踪患者偏好和历史以提供个性化护理
- **生产力与游戏**：基于用户行为的自适应工作流程和环境

## 🚀 快速入门指南 <a name="quickstart"></a>

在我们的托管平台或自托管包之间选择：

### 托管平台

通过自动更新、分析和企业安全功能，在几分钟内启动并运行。

1. 在 [Mem0 Platform](https://app.mem0.ai) 上注册
2. 通过 SDK 或 API 密钥嵌入记忆层

### 自托管（开源）

通过 pip 安装 SDK：

```bash
pip install mem0ai
```

通过 npm 安装 SDK：
```bash
npm install mem0ai
```

### 基本用法

Mem0 需要 LLM 才能运行，默认使用 OpenAI 的 `gpt-4.1-nano-2025-04-14`。但是，它支持多种 LLM；详情请参阅我们的[支持的 LLM 文档](https://docs.mem0.ai/components/llms/overview)。

第一步是实例化记忆：

```python
from openai import OpenAI
from mem0 import Memory

openai_client = OpenAI()
memory = Memory()

def chat_with_memories(message: str, user_id: str = "default_user") -> str:
    # 检索相关记忆
    relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
    memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])

    # 生成助手响应
    system_prompt = f"你是一个有帮助的 AI。根据查询和记忆回答问题。\n用户记忆：\n{memories_str}"
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
    response = openai_client.chat.completions.create(model="gpt-4.1-nano-2025-04-14", messages=messages)
    assistant_response = response.choices[0].message.content

    # 从对话中创建新记忆
    messages.append({"role": "assistant", "content": assistant_response})
    memory.add(messages, user_id=user_id)

    return assistant_response

def main():
    print("与 AI 聊天（输入 'exit' 退出）")
    while True:
        user_input = input("你：").strip()
        if user_input.lower() == 'exit':
            print("再见！")
            break
        print(f"AI：{chat_with_memories(user_input)}")

if __name__ == "__main__":
    main()
```

有关详细的集成步骤，请参阅[快速入门](https://docs.mem0.ai/quickstart)和 [API 参考](https://docs.mem0.ai/api-reference)。

## 🔗 集成与演示

- **带记忆的 ChatGPT**：由 Mem0 提供支持的个性化聊天（[在线演示](https://mem0.dev/demo)）
- **浏览器扩展**：在 ChatGPT、Perplexity 和 Claude 中存储记忆（[Chrome 扩展](https://chromewebstore.google.com/detail/onihkkbipkfeijkadecaafbgagkhglop?utm_source=item-share-cb)）
- **Langgraph 支持**：使用 Langgraph + Mem0 构建客户机器人（[指南](https://docs.mem0.ai/integrations/langgraph)）
- **CrewAI 集成**：使用 Mem0 定制 CrewAI 输出（[示例](https://docs.mem0.ai/integrations/crewai)）

## 📚 文档与支持

- 完整文档：https://docs.mem0.ai
- 社区：[Discord](https://mem0.dev/DiG) · [Twitter](https://x.com/mem0ai)
- 联系方式：founders@mem0.ai

## 引用

我们现在有一篇可以引用的论文：

```bibtex
@article{mem0,
  title={Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory},
  author={Chhikara, Prateek and Khant, Dev and Aryan, Saket and Singh, Taranjeet and Yadav, Deshraj},
  journal={arXiv preprint arXiv:2504.19413},
  year={2025}
}
```

## ⚖️ 许可证

Apache 2.0 — 详情请参阅 [LICENSE](https://github.com/mem0ai/mem0/blob/main/LICENSE) 文件。
