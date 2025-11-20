# 迁移指南：升级到 mem0 1.0.0

## 简明摘要

**有什么变化？** 我们通过移除令人困惑的版本参数简化了 API。现在所有内容都返回一致的格式：`{"results": [...]}`。

**您需要做什么：**
1. 升级：`pip install mem0ai==1.0.0`
2. 从代码中移除 `version` 和 `output_format` 参数
3. 更新响应处理，使用 `result["results"]` 而不是将响应视为列表

**所需时间：** 大多数项目约 5-10 分钟

---

## 快速迁移指南

### 1. 安装更新

```bash
pip install mem0ai==1.0.0
```

### 2. 更新代码

**如果您正在使用 Memory API：**

```python
# 之前
memory = Memory(config=MemoryConfig(version="v1.1"))
result = memory.add("I like pizza")

# 之后
memory = Memory()  # 就这样 - 版本现在是自动的
result = memory.add("I like pizza")
```

**如果您正在使用 Client API：**

```python
# 之前
client.add(messages, output_format="v1.1")
client.search(query, version="v2", output_format="v1.1")

# 之后
client.add(messages)  # 只需移除这些额外的参数
client.search(query)
```

### 3. 更新响应处理方式

所有响应现在使用相同的格式：带有 `"results"` 键的字典。

```python
# 之前 - 您可能这样做过
result = memory.add("I like pizza")
for item in result:  # 将其视为列表
    print(item)

# 之后 - 改为这样做
result = memory.add("I like pizza")
for item in result["results"]:  # 访问 results 键
    print(item)

# 图关系（如果您使用它们）
if "relations" in result:
    for relation in result["relations"]:
        print(relation)
```

---

## 增强的消息处理

平台客户端（MemoryClient）现在支持与 OSS 版本相同的灵活消息格式：

```python
from mem0 import MemoryClient

client = MemoryClient(api_key="your-key")

# 现在所有三种格式都可以工作：

# 1. 单个字符串（自动转换为用户消息）
client.add("I like pizza", user_id="alice")

# 2. 单个消息字典
client.add({"role": "user", "content": "I like pizza"}, user_id="alice")

# 3. 消息列表（对话）
client.add([
    {"role": "user", "content": "I like pizza"},
    {"role": "assistant", "content": "I'll remember that!"}
], user_id="alice")
```

### 异步模式配置

`async_mode` 参数现在默认为 `True`，但可以配置：

```python
# 默认行为（async_mode=True）
client.add(messages, user_id="alice")

# 显式设置异步模式
client.add(messages, user_id="alice", async_mode=True)

# 如果需要，禁用异步模式
client.add(messages, user_id="alice", async_mode=False)
```

**注意：** `async_mode=True` 为大多数用例提供了更好的性能。只有在有特定的同步处理要求时才将其设置为 `False`。

---

## 就是这样！

对于大多数用户来说，这就是您需要知道的全部内容。更改包括：
- ✅ 不再需要 `version` 或 `output_format` 参数
- ✅ 一致的 `{"results": [...]}` 响应格式
- ✅ 更简洁、更简单的 API

---

## 常见问题

**遇到 `KeyError: 'results'` 错误？**

您的代码仍然将响应视为列表。请更新它：
```python
# 将这个：
for memory in response:

# 改为这个：
for memory in response["results"]:
```

**遇到 `TypeError: unexpected keyword argument` 错误？**

您仍在传递旧参数。请移除它们：
```python
# 将这个：
client.add(messages, output_format="v1.1")

# 改为这个：
client.add(messages)
```

**看到弃用警告？**

从配置中移除任何显式的 `version="v1.0"`：
```python
# 将这个：
memory = Memory(config=MemoryConfig(version="v1.0"))

# 改为这个：
memory = Memory()
```

---

## 1.0.0 的新特性

- **更好的向量存储**：修复了 OpenSearch 并提高了所有存储的可靠性
- **更简洁的 API**：做事情的方式只有一种，不再有令人困惑的选项
- **增强的 GCP 支持**：更好的 Vertex AI 配置选项
- **灵活的消息输入**：平台客户端现在接受字符串、字典和列表（与 OSS 对齐）
- **可配置的 async_mode**：现在默认为 `True`，但用户可以根据需要覆盖

---

## 需要帮助？

- 查看 [GitHub Issues](https://github.com/mem0ai/mem0/issues)
- 阅读[文档](https://docs.mem0.ai/)
- 如果遇到困难，请提交新的 issue

---

## 高级：配置更改

**如果您使用版本配置了向量存储：**

```python
# 之前
config = MemoryConfig(
    version="v1.1",
    vector_store=VectorStoreConfig(...)
)

# 之后
config = MemoryConfig(
    vector_store=VectorStoreConfig(...)
)
```

---

## 测试迁移

快速健全性检查：

```python
from mem0 import Memory

memory = Memory()

# add 应该返回带有 "results" 的字典
result = memory.add("I like pizza", user_id="test")
assert "results" in result

# search 应该返回带有 "results" 的字典
search = memory.search("food", user_id="test")
assert "results" in search

# get_all 应该返回带有 "results" 的字典
all_memories = memory.get_all(user_id="test")
assert "results" in all_memories

print("✅ 迁移成功！")
```
