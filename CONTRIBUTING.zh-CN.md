# 为 mem0 做贡献

让我们使贡献变得简单、协作且有趣。

## 通过 PR 提交您的贡献

要做出贡献，请按照以下步骤操作：

1. Fork 并克隆此仓库
2. 在您的 fork 上使用专用功能分支 `feature/f1` 进行更改
3. 如果您修改了代码（新功能或错误修复），请为其添加测试
4. 包含适当的文档/文档字符串和运行功能的示例
5. 确保所有测试通过
6. 提交拉取请求

有关拉取请求的更多详细信息，请阅读 [GitHub 指南](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)。


### 📦 开发环境

我们使用 `hatch` 来管理开发环境。设置方法：

```bash
# 激活特定 Python 版本的环境：
hatch shell dev_py_3_9   # Python 3.9
hatch shell dev_py_3_10  # Python 3.10  
hatch shell dev_py_3_11  # Python 3.11
hatch shell dev_py_3_12  # Python 3.12

# 环境将自动安装所有开发依赖
# 在激活的 shell 中运行测试：
make test
```

### 📌 Pre-commit

为了确保我们的标准，在开始贡献之前请确保安装 pre-commit。

```bash
pre-commit install
```

### 🧪 测试

我们使用 `pytest` 在多个 Python 版本上测试我们的代码。您可以使用以下命令运行测试：

```bash
# 使用默认 Python 版本运行测试
make test

# 测试特定 Python 版本：
make test-py-3.9   # Python 3.9 环境
make test-py-3.10  # Python 3.10 环境
make test-py-3.11  # Python 3.11 环境
make test-py-3.12  # Python 3.12 环境

# 使用 hatch shell 时，运行测试：
make test  # 使用 hatch shell test_XX 激活 shell 后
```

在提交拉取请求之前，请确保所有支持的 Python 版本上的所有测试都通过。

我们期待您的拉取请求，迫不及待想看到您的贡献！
