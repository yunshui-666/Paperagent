# PaperAgent

面向本机单用户的论文知识助手。支持批量上传 PDF、三种解析入库方案、混合检索、Cross-Encoder 重排、持久化多轮会话和引用回查。

后端模块职责、代码阅读顺序及真实评测准备见 [后端重构与评测准备](BACKEND_REFACTOR_GUIDE.md)。

## 启动

在项目根目录打开 PowerShell，使用现有虚拟环境启动。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

打开 <http://127.0.0.1:8000>。API 调试页为 <http://127.0.0.1:8000/docs>，技术文档页为 <http://127.0.0.1:8000/guide>。

新目录安装建议使用本次已验证的 Python 3.14 创建虚拟环境，复制 `.env.example` 为 `.env`，填写 `DASHSCOPE_API_KEY` 和 `DEEPSEEK_API`。依赖清单中的 NumPy 要求 Python 3.12 或更高版本，其他版本需确认全部依赖有兼容的安装包。OCR 使用本机 RapidOCR，双路解析的 VLM 使用现有通义接口。桌面原项目继续使用已有 `.env`。

```powershell
python -m venv .venv
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 使用

1. 打开“文档库”，批量选择 PDF。可搜索文本的论文先用“本地文本解析”，扫描论文可用“混合解析”。
2. 等待状态变为“已入库”。在顶部选择对应知识库。
3. 在“检索设置”选择向量、BM25、混合召回或混合加重排。
4. 可以选择单篇或多篇论文作为研读范围，然后提问和追问。
5. 点击答案的引用按钮查看证据片段，再跳到原 PDF 对应页。
6. 会话自动保存。重启后可继续追问。文档库提供重建索引与删除操作。

重排模型首次使用会下载固定版本权重。已有模型位于 `pdfprocess/data/models/mmarco-minilm`。离线运行时应预先准备完整权重并设置 `PAPERAGENT_RERANKER_MODEL_DIR`。

## 低相关性时联网搜索

默认混合重排模式使用最高 `sigmoid(Cross-Encoder logit)` 作为本地相关性分数。最高分低于 0.5，或本地候选为空时，LangGraph 先让模型生成 `web_search` 工具调用与搜索词，再由工具注册表执行 Bing RSS 搜索。模型接收压缩后的 `ToolMessage` 证据组织答案。搜索入口使用现有 requests 依赖，无需搜索 API 密钥。

联网分支会清空本轮本地 context 和 sources，模型输入采用独立检索问题和本轮搜索工具协议及网页证据，历史消息与摘要排除。回答固定提示“本地数据库未检索到足够相关的内容。”。搜索失败时返回可重试错误，搜索无结果时直接提示资料不足。网页证据可在引用面板打开来源链接。

最高分恰好等于 0.5 时使用本地资料。dense、bm25、hybrid 模式可用于原有检索对比。开发时可向 `build_graph` 注入 `web_search_tool`，传入 None 可关闭该工具。

在答案的“查看检索与耗时”中检查 `local_max_relevance`、`web_search_threshold`、`evidence_origin`、`tool_call_mode`、`tool_calls`、`tool_selection_usage` 和 `web_search_query`。原始 logit 保留在本地来源的 `cross_encoder_score`，映射后的分数为 `rerank_relevance`。sigmoid 分数用作路由阈值，质量标定需要结合人工证据评测。

工具注册、扩展方法、模型调用协议及调试说明见 [工具注册表与模型搜索调用](TOOL_REGISTRY_GUIDE.md)。

## 验证与实验

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```

详细架构、每个修改文件的职责、技术选择、指标定义、断点位置和对照实验操作均在 [PROJECT_TECHNICAL_GUIDE.md](PROJECT_TECHNICAL_GUIDE.md)。

本版本按本机单用户、单服务进程运行。界面身份沿用本地使用方式。向量库使用 Chroma，任务和会话使用 SQLite。原有 `paper_chunks` 可继续选择，历史向量未登记原 PDF 时显示证据正文。

## 上下文压缩代码阅读

新增的上下文压缩实现、默认参数、断点顺序和手动对比步骤见 [上下文压缩与代码阅读](CONTEXT_COMPRESSION_GUIDE.md)。



本地 OCR 双路解析、长对话预算与论文自动入库的当前实现及调试步骤见 [PAPER_TOOLS_GUIDE.md](PAPER_TOOLS_GUIDE.md)。
