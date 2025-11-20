# Mem0 - 个性化 AI 的记忆层

## 概述

Mem0（读作"mem-zero"）是一个智能记忆层，为 AI 助手和智能体提供持久化、个性化的记忆能力。它使 AI 系统能够记住用户偏好、适应个体需求，并随时间持续学习——这使其成为客户支持聊天机器人、AI 助手和自主系统的理想选择。

**核心优势：**
- 在 LOCOMO 基准测试中，准确度比 OpenAI Memory 高 26%
- 响应速度比全上下文方法快 91%
- token 使用量比全上下文方法减少 90%

## 安装

```bash
# Python
pip install mem0ai

# TypeScript/JavaScript
npm install mem0ai
```

## 快速开始

### Python - 自托管
```python
from mem0 import Memory

# 初始化记忆
memory = Memory()

# 添加记忆
memory.add([
    {"role": "user", "content": "我喜欢披萨，讨厌西兰花"},
    {"role": "assistant", "content": "我会记住您的食物偏好！"}
], user_id="user123")

# 搜索记忆
results = memory.search("食物偏好", user_id="user123")
print(results)

# 获取所有记忆
all_memories = memory.get_all(user_id="user123")
```

### Python - 托管平台
```python
from mem0 import MemoryClient

# 初始化客户端
client = MemoryClient(api_key="your-api-key")

# 添加记忆
client.add([
    {"role": "user", "content": "我叫 John，是一名开发者"}
], user_id="john")

# 搜索记忆
results = client.search("你知道我的哪些信息？", user_id="john")
```

### TypeScript - 客户端 SDK
```typescript
import { MemoryClient } from 'mem0ai';

const client = new MemoryClient({ apiKey: 'your-api-key' });

// 添加记忆
const memories = await client.add([
  { role: 'user', content: '我叫 John' }
], { user_id: 'john' });

// 搜索记忆
const results = await client.search('我叫什么名字？', { user_id: 'john' });
```

### TypeScript - OSS SDK
```typescript
import { Memory } from 'mem0ai/oss';

const memory = new Memory({
  embedder: { provider: 'openai', config: { apiKey: 'key' } },
  vectorStore: { provider: 'memory', config: { dimension: 1536 } },
  llm: { provider: 'openai', config: { apiKey: 'key' } }
});

const result = await memory.add('我叫 John', { userId: 'john' });
```

## 核心 API 参考

### Memory 类（自托管）

**导入：** `from mem0 import Memory, AsyncMemory`

#### 初始化
```python
from mem0 import Memory
from mem0.configs.base import MemoryConfig

# 基础初始化
memory = Memory()

# 使用自定义配置
config = MemoryConfig(
    vector_store={"provider": "qdrant", "config": {"host": "localhost"}},
    llm={"provider": "openai", "config": {"model": "gpt-4.1-nano-2025-04-14"}},
    embedder={"provider": "openai", "config": {"model": "text-embedding-3-small"}}
)
memory = Memory(config)
```

#### 核心方法

**add(messages, *, user_id=None, agent_id=None, run_id=None, metadata=None, infer=True, memory_type=None, prompt=None)**
- **用途**: 从消息创建新记忆
- **参数**:
  - `messages`: 字符串、字典或消息字典列表
  - `user_id/agent_id/run_id`: 会话标识符（至少需要一个）
  - `metadata`: 要存储的附加元数据
  - `infer`: 是否使用 LLM 进行事实提取（默认：True）
  - `memory_type`: "procedural_memory" 用于过程记忆
  - `prompt`: 用于记忆创建的自定义提示词
- **返回值**: 包含 "results" 键的字典，包含记忆操作

**search(query, *, user_id=None, agent_id=None, run_id=None, limit=100, filters=None, threshold=None)**
- **用途**: 语义搜索记忆
- **参数**:
  - `query`: 搜索查询字符串
  - `user_id/agent_id/run_id`: 会话过滤器（至少需要一个）
  - `limit`: 最大结果数（默认：100）
  - `filters`: 附加搜索过滤器
  - `threshold`: 最小相似度分数
- **返回值**: 包含 "results" 的字典，含评分记忆

**get(memory_id)**
- **用途**: 通过 ID 检索特定记忆
- **返回值**: 包含 id、memory、hash、timestamps、metadata 的记忆字典

**get_all(*, user_id=None, agent_id=None, run_id=None, filters=None, limit=100)**
- **用途**: 列出所有记忆（可选过滤）
- **返回值**: 包含 "results" 的字典，含记忆列表

**update(memory_id, data)**
- **用途**: 更新记忆内容或元数据
- **返回值**: 成功消息字典

**delete(memory_id)**
- **用途**: 删除特定记忆
- **返回值**: 成功消息字典

**delete_all(user_id=None, agent_id=None, run_id=None)**
- **用途**: 删除会话的所有记忆（至少需要一个 ID）
- **返回值**: 成功消息字典

**history(memory_id)**
- **用途**: 获取记忆变更历史
- **返回值**: 记忆变更历史列表

**reset()**
- **用途**: 重置整个记忆存储
- **返回值**: None

### MemoryClient 类（托管平台）

**导入：** `from mem0 import MemoryClient, AsyncMemoryClient`

#### 初始化
```python
client = MemoryClient(
    api_key="your-api-key",  # 或设置 MEM0_API_KEY 环境变量
    host="https://api.mem0.ai",  # 可选
    org_id="your-org-id",  # 可选
    project_id="your-project-id"  # 可选
)
```

#### 核心方法

**add(messages, **kwargs)**
- **用途**: 从消息对话创建记忆
- **参数**: messages（消息字典列表）、user_id、agent_id、app_id、metadata、filters
- **返回值**: 包含记忆创建结果的 API 响应字典

**search(query, version="v1", **kwargs)**
- **用途**: 根据查询搜索记忆
- **参数**: query、version（"v1"/"v2"）、user_id、agent_id、app_id、top_k、filters
- **返回值**: 搜索结果字典列表

**get(memory_id)**
- **用途**: 通过 ID 检索特定记忆
- **返回值**: 记忆数据字典

**get_all(version="v1", **kwargs)**
- **用途**: 检索所有记忆（带过滤）
- **参数**: version、user_id、agent_id、app_id、top_k、page、page_size
- **返回值**: 记忆字典列表

**update(memory_id, text=None, metadata=None)**
- **用途**: 更新记忆文本或元数据
- **返回值**: 更新后的记忆数据

**delete(memory_id)**
- **用途**: 删除特定记忆
- **返回值**: 成功响应

**delete_all(**kwargs)**
- **用途**: 删除所有记忆（带过滤）
- **返回值**: 成功消息

#### 批量操作

**batch_update(memories)**
- **用途**: 在单个请求中更新多个记忆
- **参数**: 记忆更新对象列表
- **返回值**: 批量操作结果

**batch_delete(memories)**
- **用途**: 在单个请求中删除多个记忆
- **参数**: 记忆对象列表
- **返回值**: 批量操作结果

#### 用户管理

**users()**
- **用途**: 获取所有拥有记忆的用户、智能体和会话
- **返回值**: 包含用户/智能体/会话数据的字典

**delete_users(user_id=None, agent_id=None, app_id=None, run_id=None)**
- **用途**: 删除特定实体或所有实体
- **返回值**: 成功消息

**reset()**
- **用途**: 通过删除所有用户和记忆来重置客户端
- **返回值**: 成功消息

#### 附加功能

**history(memory_id)**
- **用途**: 获取记忆变更历史
- **返回值**: 记忆变更列表

**feedback(memory_id, feedback, **kwargs)**
- **用途**: 对记忆提供反馈
- **返回值**: 反馈响应

**create_memory_export(schema, **kwargs)**
- **用途**: 使用 JSON schema 创建记忆导出
- **返回值**: 导出创建响应

**get_memory_export(**kwargs)**
- **用途**: 检索导出的记忆数据
- **返回值**: 导出的数据


## 配置系统

### MemoryConfig

```python
from mem0.configs.base import MemoryConfig

config = MemoryConfig(
    vector_store=VectorStoreConfig(provider="qdrant", config={...}),
    llm=LlmConfig(provider="openai", config={...}),
    embedder=EmbedderConfig(provider="openai", config={...}),
    graph_store=GraphStoreConfig(provider="neo4j", config={...}),  # 可选
    history_db_path="~/.mem0/history.db",
    version="v1.1",
    custom_fact_extraction_prompt="自定义提示词...",
    custom_update_memory_prompt="自定义提示词..."
)
```

### 支持的提供商

#### LLM 提供商（支持 19 种）
- **openai** - OpenAI GPT 模型（默认）
- **anthropic** - Claude 模型
- **gemini** - Google Gemini
- **groq** - Groq 推理
- **ollama** - 本地 Ollama 模型
- **together** - Together AI
- **aws_bedrock** - AWS Bedrock 模型
- **azure_openai** - Azure OpenAI
- **litellm** - LiteLLM 代理
- **deepseek** - DeepSeek 模型
- **xai** - xAI 模型
- **sarvam** - Sarvam AI
- **lmstudio** - LM Studio 本地服务器
- **vllm** - vLLM 推理服务器
- **langchain** - LangChain 集成
- **openai_structured** - OpenAI 结构化输出
- **azure_openai_structured** - Azure OpenAI 结构化输出

#### 嵌入提供商（支持 10 种）
- **openai** - OpenAI 嵌入（默认）
- **ollama** - Ollama 嵌入
- **huggingface** - HuggingFace 模型
- **azure_openai** - Azure OpenAI 嵌入
- **gemini** - Google Gemini 嵌入
- **vertexai** - Google Vertex AI
- **together** - Together AI 嵌入
- **lmstudio** - LM Studio 嵌入
- **langchain** - LangChain 嵌入
- **aws_bedrock** - AWS Bedrock 嵌入

#### 向量存储提供商（支持 19 种）
- **qdrant** - Qdrant 向量数据库（默认）
- **chroma** - ChromaDB
- **pinecone** - Pinecone 向量数据库
- **pgvector** - 带 pgvector 的 PostgreSQL
- **mongodb** - MongoDB Atlas 向量搜索
- **milvus** - Milvus 向量数据库
- **weaviate** - Weaviate
- **faiss** - Facebook AI 相似性搜索
- **redis** - Redis 向量搜索
- **elasticsearch** - Elasticsearch
- **opensearch** - OpenSearch
- **azure_ai_search** - Azure AI 搜索
- **vertex_ai_vector_search** - Google Vertex AI 向量搜索
- **upstash_vector** - Upstash Vector
- **supabase** - Supabase 向量
- **baidu** - 百度向量数据库
- **langchain** - LangChain 向量存储
- **s3_vectors** - Amazon S3 Vectors
- **databricks** - Databricks 向量存储

#### 图存储提供商（支持 4 种）
- **neo4j** - Neo4j 图数据库
- **memgraph** - Memgraph
- **neptune** - AWS Neptune Analytics
- **kuzu** - Kuzu 图数据库

### 配置示例

#### OpenAI 配置
```python
config = MemoryConfig(
    llm={
        "provider": "openai",
        "config": {
            "model": "gpt-4.1-nano-2025-04-14",
            "temperature": 0.1,
            "max_tokens": 1000
        }
    },
    embedder={
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
)
```

#### 使用 Ollama 的本地设置
```python
config = MemoryConfig(
    llm={
        "provider": "ollama",
        "config": {
            "model": "llama3.1:8b",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    embedder={
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    },
    vector_store={
        "provider": "chroma",
        "config": {
            "collection_name": "my_memories",
            "path": "./chroma_db"
        }
    }
)
```

#### 使用 Neo4j 的图记忆
```python
config = MemoryConfig(
    graph_store={
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password",
            "database": "neo4j"
        }
    }
)
```

#### 企业级设置
```python
config = MemoryConfig(
    llm={
        "provider": "azure_openai",
        "config": {
            "model": "gpt-4",
            "azure_endpoint": "https://your-resource.openai.azure.com/",
            "api_key": "your-api-key",
            "api_version": "2024-02-01"
        }
    },
    vector_store={
        "provider": "pinecone",
        "config": {
            "api_key": "your-pinecone-key",
            "index_name": "mem0-index",
            "dimension": 1536
        }
    }
)
```

#### LLM 提供商
- **OpenAI** - GPT-4、GPT-3.5-turbo 及结构化输出
- **Anthropic** - 具有高级推理能力的 Claude 模型
- **Google AI** - 用于多模态应用的 Gemini 模型
- **AWS Bedrock** - 企业级 AWS 托管模型
- **Azure OpenAI** - Microsoft Azure 托管的 OpenAI 模型
- **Groq** - 高性能 LPU 优化模型
- **Together** - 开源模型推理平台
- **Ollama** - 用于隐私保护的本地模型部署
- **vLLM** - 高性能推理框架
- **LM Studio** - 本地模型管理
- **DeepSeek** - 高级推理模型
- **Sarvam** - 印度语言模型
- **XAI** - xAI 模型
- **LiteLLM** - 统一 LLM 接口
- **LangChain** - LangChain LLM 集成

#### 向量存储提供商
- **Chroma** - AI 原生开源向量数据库
- **Qdrant** - 高性能向量相似性搜索
- **Pinecone** - 带无服务器选项的托管向量数据库
- **Weaviate** - 开源向量搜索引擎
- **PGVector** - PostgreSQL 向量搜索扩展
- **Milvus** - 可扩展的开源向量数据库
- **Redis** - 使用 Redis Stack 的实时向量存储
- **Supabase** - 开源 Firebase 替代品
- **Upstash Vector** - 无服务器向量数据库
- **Elasticsearch** - 分布式搜索和分析
- **OpenSearch** - 开源搜索和分析
- **FAISS** - Facebook AI 相似性搜索
- **MongoDB** - 带向量搜索的文档数据库
- **Azure AI Search** - Microsoft 搜索服务
- **Vertex AI Vector Search** - Google Cloud 向量搜索
- **Databricks Vector Search** - Delta Lake 集成
- **Baidu** - 百度向量数据库
- **LangChain** - LangChain 向量存储集成

#### 嵌入提供商
- **OpenAI** - 高质量文本嵌入
- **Azure OpenAI** - 企业级 Azure 托管嵌入
- **Google AI** - Gemini 嵌入模型
- **AWS Bedrock** - Amazon 嵌入模型
- **Hugging Face** - 开源嵌入模型
- **Vertex AI** - Google Cloud 企业级嵌入
- **Ollama** - 本地嵌入模型
- **Together** - 开源模型嵌入
- **LM Studio** - 本地模型嵌入
- **LangChain** - LangChain 嵌入器集成

## TypeScript/JavaScript SDK

### 客户端 SDK（托管平台）

```typescript
import { MemoryClient } from 'mem0ai';

const client = new MemoryClient({
  apiKey: 'your-api-key',
  host: 'https://api.mem0.ai',  // 可选
  organizationId: 'org-id',     // 可选
  projectId: 'project-id'       // 可选
});

// 核心操作
const memories = await client.add([
  { role: 'user', content: '我喜欢披萨' }
], { user_id: 'user123' });

const results = await client.search('食物偏好', { user_id: 'user123' });
const memory = await client.get('memory-id');
const allMemories = await client.getAll({ user_id: 'user123' });

// 管理操作
await client.update('memory-id', '更新后的内容');
await client.delete('memory-id');
await client.deleteAll({ user_id: 'user123' });

// 批量操作
await client.batchUpdate([{ id: 'mem1', text: '新文本' }]);
await client.batchDelete(['mem1', 'mem2']);

// 用户管理
const users = await client.users();
await client.deleteUsers({ user_ids: ['user1', 'user2'] });

// Webhook
const webhooks = await client.getWebhooks();
await client.createWebhook({
  url: 'https://your-webhook.com',
  name: 'My Webhook',
  eventTypes: ['memory.created', 'memory.updated']
});
```

### OSS SDK（自托管）

```typescript
import { Memory } from 'mem0ai/oss';

const memory = new Memory({
  embedder: {
    provider: 'openai',
    config: { apiKey: 'your-key' }
  },
  vectorStore: {
    provider: 'qdrant',
    config: { host: 'localhost', port: 6333 }
  },
  llm: {
    provider: 'openai',
    config: { model: 'gpt-4.1-nano' }
  }
});

// 核心操作
const result = await memory.add('我喜欢披萨', { userId: 'user123' });
const searchResult = await memory.search('食物偏好', { userId: 'user123' });
const memoryItem = await memory.get('memory-id');
const allMemories = await memory.getAll({ userId: 'user123' });

// 管理
await memory.update('memory-id', '更新后的内容');
await memory.delete('memory-id');
await memory.deleteAll({ userId: 'user123' });

// 历史和重置
const history = await memory.history('memory-id');
await memory.reset();
```

### 核心 TypeScript 类型

```typescript
interface Message {
  role: 'user' | 'assistant';
  content: string | MultiModalMessages;
}

interface Memory {
  id: string;
  memory?: string;
  user_id?: string;
  categories?: string[];
  created_at?: Date;
  updated_at?: Date;
  metadata?: any;
  score?: number;
}

interface MemoryOptions {
  user_id?: string;
  agent_id?: string;
  app_id?: string;
  run_id?: string;
  metadata?: Record<string, any>;
  filters?: Record<string, any>;
  api_version?: 'v1' | 'v2';
  infer?: boolean;
  enable_graph?: boolean;
}

interface SearchResult {
  results: Memory[];
  relations?: any[];
}
```

## 高级功能

### 图记忆

图记忆（Graph Memory）可以跟踪对话中提到的实体之间的关系。

```python
# 启用图记忆
config = MemoryConfig(
    graph_store={
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "password"
        }
    }
)
memory = Memory(config)

# 添加包含关系提取的记忆
result = memory.add(
    "John 在 OpenAI 工作，是 Sarah 的朋友",
    user_id="user123"
)

# 结果包括记忆和关系
print(result["results"])     # 记忆条目
print(result["relations"])   # 图关系
```

**支持的图数据库：**
- **Neo4j**: 带 Cypher 查询的全功能图数据库
- **Memgraph**: 高性能内存图数据库
- **Neptune**: AWS 托管图数据库服务
- **kuzu** - OSS Kuzu 图数据库

### 多模态记忆

存储和检索来自文本、图像和 PDF 的记忆。

```python
# 文本 + 图像
messages = [
    {"role": "user", "content": "这是我的旅行装备"},
    {
        "role": "user",
        "content": {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
    }
]
client.add(messages, user_id="user123")

# PDF 处理
pdf_message = {
    "role": "user",
    "content": {
        "type": "pdf_url",
        "pdf_url": {"url": "https://example.com/document.pdf"}
    }
}
client.add([pdf_message], user_id="user123")
```

### 过程记忆

存储分步骤的过程和工作流程。

```python
# 添加过程记忆
result = memory.add(
    "部署应用的步骤：1. 运行测试 2. 构建 Docker 镜像 3. 推送到仓库 4. 更新 k8s 清单",
    user_id="developer123",
    memory_type="procedural_memory"
)

# 搜索过程
procedures = memory.search(
    "如何部署？",
    user_id="developer123"
)
```

### 自定义提示词

```python
custom_extraction_prompt = """
从对话中提取关键事实，重点关注：
1. 个人偏好
2. 技术技能
3. 项目需求
4. 重要日期和截止日期

对话：{messages}
"""

config = MemoryConfig(
    custom_fact_extraction_prompt=custom_extraction_prompt
)
memory = Memory(config)
```


## 常见使用模式

### 1. 个人 AI 助手

```python
class PersonalAssistant:
    def __init__(self):
        self.memory = Memory()
        self.llm = OpenAI()  # 您的 LLM 客户端
    
    def chat(self, user_input: str, user_id: str) -> str:
        # 检索相关记忆
        memories = self.memory.search(user_input, user_id=user_id, limit=5)
        
        # 从记忆构建上下文
        context = "\n".join([f"- {m['memory']}" for m in memories['results']])
        
        # 生成带上下文的响应
        prompt = f"""
        来自之前对话的上下文：
        {context}
        
        用户：{user_input}
        助手：
        """
        
        response = self.llm.generate(prompt)
        
        # 存储对话
        self.memory.add([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": response}
        ], user_id=user_id)
        
        return response
```

### 2. 客户支持机器人

```python
class SupportBot:
    def __init__(self):
        self.memory = MemoryClient(api_key="your-key")
    
    def handle_ticket(self, customer_id: str, issue: str) -> str:
        # 获取客户历史
        history = self.memory.search(
            issue,
            user_id=customer_id,
            limit=10
        )
        
        # 检查类似的历史问题
        similar_issues = [m for m in history if m['score'] > 0.8]
        
        if similar_issues:
            context = f"之前的类似问题：{similar_issues[0]['memory']}"
        else:
            context = "未发现之前的类似问题。"
        
        # 生成响应
        response = self.generate_support_response(issue, context)
        
        # 存储交互
        self.memory.add([
            {"role": "user", "content": f"问题：{issue}"},
            {"role": "assistant", "content": response}
        ], user_id=customer_id, metadata={
            "category": "support_ticket",
            "timestamp": datetime.now().isoformat()
        })
        
        return response
```

### 3. 学习助手

```python
class StudyBuddy:
    def __init__(self):
        self.memory = Memory()
    
    def study_session(self, student_id: str, topic: str, content: str):
        # 存储学习材料
        self.memory.add(
            f"学习了 {topic}：{content}",
            user_id=student_id,
            metadata={
                "topic": topic,
                "session_date": datetime.now().isoformat(),
                "type": "study_session"
            }
        )
    
    def quiz_student(self, student_id: str, topic: str) -> list:
        # 获取相关学习材料
        materials = self.memory.search(
            f"topic:{topic}",
            user_id=student_id,
            filters={"metadata.type": "study_session"}
        )
        
        # 基于材料生成测验问题
        questions = self.generate_quiz_questions(materials)
        return questions
    
    def track_progress(self, student_id: str) -> dict:
        # 获取所有学习会话
        sessions = self.memory.get_all(
            user_id=student_id,
            filters={"metadata.type": "study_session"}
        )
        
        # 分析进度
        topics_studied = {}
        for session in sessions['results']:
            topic = session['metadata']['topic']
            topics_studied[topic] = topics_studied.get(topic, 0) + 1
        
        return {
            "total_sessions": len(sessions['results']),
            "topics_covered": len(topics_studied),
            "topic_frequency": topics_studied
        }
```

### 4. 多智能体系统

```python
class MultiAgentSystem:
    def __init__(self):
        self.shared_memory = Memory()
        self.agents = {
            "researcher": ResearchAgent(),
            "writer": WriterAgent(),
            "reviewer": ReviewAgent()
        }
    
    def collaborative_task(self, task: str, session_id: str):
        # 研究阶段
        research_results = self.agents["researcher"].research(task)
        self.shared_memory.add(
            f"研究发现：{research_results}",
            agent_id="researcher",
            run_id=session_id,
            metadata={"phase": "research"}
        )
        
        # 写作阶段
        research_context = self.shared_memory.search(
            "研究发现",
            run_id=session_id
        )
        draft = self.agents["writer"].write(task, research_context)
        self.shared_memory.add(
            f"草稿内容：{draft}",
            agent_id="writer",
            run_id=session_id,
            metadata={"phase": "writing"}
        )
        
        # 审核阶段
        all_context = self.shared_memory.get_all(run_id=session_id)
        final_output = self.agents["reviewer"].review(draft, all_context)
        
        return final_output
```

### 5. 带记忆的语音助手

```python
import speech_recognition as sr
from gtts import gTTS
import pygame

class VoiceAssistant:
    def __init__(self):
        self.memory = Memory()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
    
    def listen_and_respond(self, user_id: str):
        # 监听用户
        with self.microphone as source:
            audio = self.recognizer.listen(source)
        
        try:
            # 语音转文本
            user_input = self.recognizer.recognize_google(audio)
            print(f"用户说：{user_input}")
            
            # 获取相关记忆
            memories = self.memory.search(user_input, user_id=user_id)
            context = "\n".join([m['memory'] for m in memories['results'][:3]])
            
            # 生成响应
            response = self.generate_response(user_input, context)
            
            # 存储对话
            self.memory.add([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ], user_id=user_id)
            
            # 响应转语音
            tts = gTTS(text=response, lang='zh-cn')
            tts.save("response.mp3")
            
            # 播放响应
            pygame.mixer.init()
            pygame.mixer.music.load("response.mp3")
            pygame.mixer.music.play()
            
            return response
            
        except sr.UnknownValueError:
            return "抱歉，我没听懂。"
```

## 最佳实践

### 1. 记忆组织

```python
# 使用一致的用户/智能体/会话 ID
user_id = f"user_{user_email.replace('@', '_')}"
agent_id = f"agent_{agent_name}"
run_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 添加有意义的元数据
metadata = {
    "category": "customer_support",
    "priority": "high",
    "department": "technical",
    "timestamp": datetime.now().isoformat(),
    "source": "chat_widget"
}

# 使用描述性记忆内容
memory.add(
    "客户 John Smith 报告在移动应用上的 2FA 登录问题。通过清除应用缓存解决。",
    user_id=customer_id,
    metadata=metadata
)
```

### 2. 搜索优化

```python
# 使用具体的搜索查询
results = memory.search(
    "移动应用登录问题",  # 具体关键词
    user_id=customer_id,
    limit=5,  # 合理的限制
    threshold=0.7  # 过滤低相关性结果
)

# 组合多个搜索以获得全面结果
technical_issues = memory.search("技术问题", user_id=user_id)
recent_conversations = memory.get_all(
    user_id=user_id,
    filters={"metadata.timestamp": {"$gte": last_week}},
    limit=10
)
```

### 3. 记忆生命周期管理

```python
# 定期清理旧记忆
def cleanup_old_memories(memory_client, days_old=90):
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    all_memories = memory_client.get_all()
    for mem in all_memories:
        if datetime.fromisoformat(mem['created_at']) < cutoff_date:
            memory_client.delete(mem['id'])

# 归档重要记忆
def archive_memory(memory_client, memory_id):
    memory = memory_client.get(memory_id)
    memory_client.update(memory_id, metadata={
        **memory.get('metadata', {}),
        'archived': True,
        'archive_date': datetime.now().isoformat()
    })
```

### 4. 错误处理

```python
def safe_memory_operation(memory_client, operation, *args, **kwargs):
    try:
        return operation(*args, **kwargs)
    except Exception as e:
        logger.error(f"记忆操作失败：{e}")
        # 在没有记忆的情况下回退到基本响应
        return {"results": [], "message": "记忆暂时不可用"}

# 使用
results = safe_memory_operation(
    memory_client,
    memory_client.search,
    query,
    user_id=user_id
)
```

### 5. 性能优化

```python
# 尽可能使用批量操作
memories_to_add = [
    {"content": msg1, "user_id": user_id},
    {"content": msg2, "user_id": user_id},
    {"content": msg3, "user_id": user_id}
]

# 不要多次调用 add()，而是使用批量操作
for memory_data in memories_to_add:
    memory.add(memory_data["content"], user_id=memory_data["user_id"])

# 缓存频繁访问的记忆
from functools import lru_cache

@lru_cache(maxsize=100)
def get_user_preferences(user_id: str):
    return memory.search("偏好 设置", user_id=user_id, limit=5)
```


## 集成示例

### AutoGen 集成

```python
from cookbooks.helper.mem0_teachability import Mem0Teachability
from mem0 import Memory

# 为 AutoGen 智能体添加记忆能力
memory = Memory()
teachability = Mem0Teachability(
    verbosity=1,
    reset_db=False,
    recall_threshold=1.5,
    memory_client=memory
)

# 应用到智能体
teachability.add_to_agent(your_autogen_agent)
```

### LangChain 集成

```python
from langchain.memory import ConversationBufferMemory
from mem0 import Memory

class Mem0LangChainMemory(ConversationBufferMemory):
    def __init__(self, user_id: str, **kwargs):
        super().__init__(**kwargs)
        self.mem0 = Memory()
        self.user_id = user_id
    
    def save_context(self, inputs, outputs):
        # 同时保存到 LangChain 和 Mem0
        super().save_context(inputs, outputs)
        
        # 存储到 Mem0 以实现长期记忆
        self.mem0.add([
            {"role": "user", "content": str(inputs)},
            {"role": "assistant", "content": str(outputs)}
        ], user_id=self.user_id)
    
    def load_memory_variables(self, inputs):
        # 从 LangChain 缓冲区加载
        variables = super().load_memory_variables(inputs)
        
        # 使用相关的长期记忆增强
        relevant_memories = self.mem0.search(
            str(inputs),
            user_id=self.user_id,
            limit=3
        )
        
        if relevant_memories['results']:
            long_term_context = "\n".join([
                f"- {m['memory']}" for m in relevant_memories['results']
            ])
            variables['history'] += f"\n\n相关的历史上下文：\n{long_term_context}"
        
        return variables
```

### Streamlit 应用

```python
import streamlit as st
from mem0 import Memory

# 初始化记忆
if 'memory' not in st.session_state:
    st.session_state.memory = Memory()

# 用户输入
user_id = st.text_input("用户 ID", value="user123")
user_message = st.text_input("您的消息")

if st.button("发送"):
    # 获取相关记忆
    memories = st.session_state.memory.search(
        user_message,
        user_id=user_id,
        limit=5
    )
    
    # 显示记忆
    if memories['results']:
        st.subheader("相关记忆：")
        for memory in memories['results']:
            st.write(f"- {memory['memory']} (得分：{memory['score']:.2f})")
    
    # 生成并显示响应
    response = generate_response(user_message, memories)
    st.write(f"助手：{response}")
    
    # 存储对话
    st.session_state.memory.add([
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response}
    ], user_id=user_id)

# 显示所有记忆
if st.button("显示所有记忆"):
    all_memories = st.session_state.memory.get_all(user_id=user_id)
    for memory in all_memories['results']:
        st.write(f"- {memory['memory']}")
```

### FastAPI 后端

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mem0 import MemoryClient
from typing import List, Optional

app = FastAPI()
memory_client = MemoryClient(api_key="your-api-key")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    user_id: str
    metadata: Optional[dict] = None

class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 10

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # 将消息添加到记忆
        result = memory_client.add(
            [msg.dict() for msg in request.messages],
            user_id=request.user_id,
            metadata=request.metadata
        )
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_memories(request: SearchRequest):
    try:
        results = memory_client.search(
            request.query,
            user_id=request.user_id,
            limit=request.limit
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{user_id}")
async def get_user_memories(user_id: str, limit: int = 50):
    try:
        memories = memory_client.get_all(user_id=user_id, limit=limit)
        return {"memories": memories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/memories/{memory_id}")
async def delete_memory(memory_id: str):
    try:
        result = memory_client.delete(memory_id)
        return {"status": "deleted", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## 故障排除

### 常见问题

1. **找不到记忆**
   ```python
   # 在操作前检查记忆是否存在
   memory = memory_client.get(memory_id)
   if not memory:
       print(f"记忆 {memory_id} 未找到")
   ```

2. **搜索无结果**
   ```python
   # 降低相似度阈值
   results = memory.search(
       query,
       user_id=user_id,
       threshold=0.5  # 降低阈值
   )
   
   # 检查用户是否存在记忆
   all_memories = memory.get_all(user_id=user_id)
   if not all_memories['results']:
       print("未找到该用户的记忆")
   ```

3. **配置问题**
   ```python
   # 验证配置
   try:
       memory = Memory(config)
       # 使用简单操作测试
       memory.add("测试记忆", user_id="test")
       print("配置有效")
   except Exception as e:
       print(f"配置错误：{e}")
   ```

4. **API 速率限制**
   ```python
   import time
   from functools import wraps
   
   def rate_limit_retry(max_retries=3, delay=1):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               for attempt in range(max_retries):
                   try:
                       return func(*args, **kwargs)
                   except Exception as e:
                       if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                           time.sleep(delay * (2 ** attempt))  # 指数退避
                           continue
                       raise e
               return wrapper
           return decorator
   
   @rate_limit_retry()
   def safe_memory_add(memory, content, user_id):
       return memory.add(content, user_id=user_id)
   ```

### 性能提示

1. **优化向量存储配置**
   ```python
   # 对于 Qdrant
   config = MemoryConfig(
       vector_store={
           "provider": "qdrant",
           "config": {
               "host": "localhost",
               "port": 6333,
               "collection_name": "memories",
               "embedding_model_dims": 1536,
               "distance": "cosine"
           }
       }
   )
   ```

2. **批量处理**
   ```python
   # 高效处理多个记忆
   def batch_add_memories(memory_client, conversations, user_id, batch_size=10):
       for i in range(0, len(conversations), batch_size):
           batch = conversations[i:i+batch_size]
           for conv in batch:
               memory_client.add(conv, user_id=user_id)
           time.sleep(0.1)  # 批次之间的小延迟
   ```

3. **记忆清理**
   ```python
   # 定期清理以保持性能
   def cleanup_memories(memory_client, user_id, max_memories=1000):
       all_memories = memory_client.get_all(user_id=user_id)
       if len(all_memories) > max_memories:
           # 保留最近的记忆
           sorted_memories = sorted(
               all_memories,
               key=lambda x: x['created_at'],
               reverse=True
           )
           
           # 删除最旧的记忆
           for memory in sorted_memories[max_memories:]:
               memory_client.delete(memory['id'])
   ```

## 资源

- **文档**: https://docs.mem0.ai
- **GitHub 仓库**: https://github.com/mem0ai/mem0
- **Discord 社区**: https://mem0.dev/DiG
- **平台**: https://app.mem0.ai
- **研究论文**: https://mem0.ai/research
- **示例**: https://github.com/mem0ai/mem0/tree/main/examples

## 许可证

Mem0 采用 Apache 2.0 许可证。详情请参阅 [LICENSE](https://github.com/mem0ai/mem0/blob/main/LICENSE) 文件。
