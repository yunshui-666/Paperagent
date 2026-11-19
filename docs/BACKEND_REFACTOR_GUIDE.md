# PaperAgent 后端重构与评测准备

## 1. 当前代码组织

本次后端范围为 `main.py`、`pdfprocess/` 和 `upload_service/`，各模块具有职责说明，类与函数具有中文文档字符串。复杂的文件锁、数据库事务、布局坐标、缓存命中、Chunk ID、召回排序、Token 预算和工具协议旁补充了逐步注释。

RAG 按职责拆为四个文件。

- `rag.py` 组装依赖、连接图节点，并提供 `ask()` 与命令行入口。
- `rag_state.py` 定义状态字段及跨轮持久化语义。
- `rag_prompts.py` 定义本地回答与问题改写提示词。
- `rag_nodes.py` 实现历史整理、问题改写、检索、网页结果整理、压缩与答案生成。

HTTP 请求字段集中到 `upload_service/schemas.py`。上传落盘与任务入队集中到 `upload_service/ingestion.py`，上传和重建索引共同使用 `IngestionService`。`main.py` 提供可测试的 `main(argv=None)` 启动函数。

原有 `pdfprocess.rag` 继续导出 `prompt`、`rewrite_prompt`、`RAGState` 和已有节点函数。`upload_service.app` 继续导出请求模型，便于已有脚本导入使用。

## 2. 建议的阅读顺序

### 上传与入库

```text
main.main
  → app.create_app / lifespan
  → app.upload_documents
  → IngestionService.submit / enqueue
  → FileStorage.save / TaskStore.enqueue
  → WorkerPool._loop → TaskStore.claim → run_pipeline
  → ingest_pdf1、2、3
  → run_ingestion
  → 页解析 → prepare_chunks → TongyiEmbeddings → save_chunks
```

`WorkerPool` 在循环中调用 `TaskStore.claim()` 领取任务。PDF 解析由 `ParserPool` 的 spawn 子进程执行。业务库中任务完成后，该文档进入可检索范围。

### 提问与回答

```text
app.chat
  → ChatService.chat
  → ChatService._graph
  → rag.ask
  → prepare_history_node
      ├─ 论文操作 → run_paper_agent → 论文工具循环 → END
      └─ 普通问答 → rewrite_node（可选）
  → retrieve_node
      ├─ 本地路线 → compress_node → generate_node
      └─ 联网路线 → request_search_node → web_search_node
                                  → compress_node → generate_node
  → 业务库保存答案、证据与 metrics
```

先阅读 `rag_state.py`，再对照 `rag.py` 的边与 `rag_nodes.py` 的函数。节点返回的字典就是该步骤提交的状态增量。`messages` 使用消息 reducer，其余字段按当前节点结果更新。

## 3. 调试时关注的数据身份

| 标识 | 来源与用途 |
| --- | --- |
| 解析元数据 `paper_id` / 文档 ID | PDF 内容 SHA256，用于原文件、解析缓存与向量记录关联；论文工具的 paper_id 则是官方详情 URL |
| 解析指纹 | 配置和依赖版本的散列，区分同一 PDF 的处理方案及参数 |
| `chunk_id` | 内容身份、方案、指纹和序号组成的稳定 ID，用于向量 upsert 与证据标注 |
| `task_id` | 一次持久化入库任务的身份 |
| `job_key` | 文件、方案、配置快照和集合共同决定的活动任务去重键 |
| `thread_id` | 会话身份，官方 Checkpointer 按它隔离图状态 |
| `request_id` | 同会话内请求幂等身份，成功重复请求复用已有答案 |
| `turn_id` | 业务轮次身份，也是完整问答消息稳定 ID 的组成部分 |
| `tool_call_id` | 模型发出的调用与 ToolMessage 结果之间的关联标识 |

构建正式评测集时，使用导出语料中的真实 `chunk_id`。论文、解析方案或切块参数发生变化后，重新核对标注与语料指纹。

## 4. 状态、存储与恢复

| 存储位置 | 内容 |
| --- | --- |
| Chroma | Chunk 文本、向量、页码、论文身份和处理元数据 |
| `service.sqlite3` | 文档生命周期、任务与阶段事件、会话及每轮答案证据快照 |
| `conversations.sqlite3` | 官方 LangGraph Checkpointer 保存的完整消息、摘要与各节点状态 |
| `files/` | 按内容哈希保存的原始 PDF |
| 解析产物目录 | 页级 JSON、Markdown、页面图片、区域图片及完成标记 |

关键恢复位置可结合下列断点阅读。

1. `TaskStore._transaction()` 把领取与状态更新放在同一个写事务中。
2. `run_ingestion()` 按稳定 ID 写入，校验完成后发布完成标记。
3. `ApplicationStore.availability()` 决定当前可以检索的登记文档。
4. `DocumentManager.delete()` 先标记 deleting，再执行可继续的清理操作。
5. `ChatService._checkpoint_result()` 通过稳定消息 ID 和完成答案识别可恢复轮次。
6. `summarize_history()` 使用覆盖长度与历史前缀哈希验证增量摘要。

## 5. 开始构建评测数据

### 第一步，确定论文集与知识库

选择人工熟悉的论文，完成入库并抽查正文、表格、数值、页码和 Chunk 内容。记录解析方案、切块参数、Embedding 模型、集合名称与所选文档范围。

默认三套解析方案对应 `paper_chunks1`、`paper_chunks2` 和 `paper_chunks3`。原有历史集合为 `paper_chunks`。以下命令中的集合名需要按实际评测范围填写。

### 第二步，导出真实证据

在项目根目录执行。

```powershell
.\.venv\Scripts\python.exe -B -m pdfprocess.compare_retrieval --collection paper_chunks --export-corpus .\experiments\corpus.json
```

导出内容包含 `collection_name`、`corpus_fingerprint`、每个 Chunk 的 `id`、`text` 和 `metadata`。导出读取已有向量库，实际检索评测中的问题向量化会调用已配置的模型。

### 第三步，填写本地检索标注

以项目根目录 `retrieval_eval_template.json` 为格式参考，填写以下字段。

```json
{
  "dataset_name": "填写论文集名称",
  "annotation_status": "human_verified",
  "collection_name": "paper_chunks",
  "corpus_fingerprint": "填写导出语料中的指纹",
  "questions": [
    {
      "id": "q001",
      "question": "填写根据论文核对的问题",
      "relevant_ids": ["填写真实ChunkID"]
    }
  ]
}
```

当前检索脚本要求每题具有非空正确证据集合。需要联网或需要说明资料不足的问题，可在端到端标注表中记录 `local_answerable`、预期行为、网页参考和答案要点。

把标准答案拆成可核查事实，保存原文、页码、单位与实验条件。多证据题记录每个必要事实分别由哪些 Chunk 支持。用主题或问题族划分调参集和固定测试集。

### 第四步，运行检索对比

```powershell
.\.venv\Scripts\python.exe -B -m pdfprocess.compare_retrieval --collection paper_chunks --dataset .\experiments\retrieval_eval.json --dense-k 20 --keyword-k 20 --at-k 5 10 20 --output .\experiments\retrieval_result.json
```

该命令对同一语料与问题比较 Dense、BM25、RRF 混合和 Cross-Encoder 重排。报告包括同 K 的 Recall、Hit Rate、MRR、全部候选召回率和延迟。候选池不足会限制可达到的召回范围，调参时同时观察候选池大小与最终 K。

### 第五步，准备多轮场景并测试完整服务

项目现有 `experiments/chat_scenarios.example.json` 提供场景格式。每个场景保存有序问题列表，可指定文档范围。使用 6～10 轮追问观察近期窗口与历史摘要，逐轮标注应讨论的论文、实体、条件和答案要点。

先启动服务，再执行。

```powershell
.\.venv\Scripts\python.exe -B main.py
```

```powershell
.\.venv\Scripts\python.exe -B -m upload_service.benchmark_chat --scenarios .\experiments\chat_scenarios.example.json --collection paper_chunks --repeat 3 --output .\experiments\chat_result.json
```

基准程序会创建真实实验会话并保留结果。模型调用按服务商规则计费，评测配置应记录模型名、参数、知识库和问题集版本。

## 6. 当前指标的使用方式

| 目标 | 数据和观测 |
| --- | --- |
| 检索效果 | 人工相关 Chunk 集合，计算 Recall@K、Hit Rate@K 和 MRR |
| 重排收益 | 同一候选集、同一 K 下比较排名及正确证据命中 |
| 关键词补充 | 观察 Dense 遗漏但 BM25 命中的术语、型号和专有名词问题 |
| 路由质量 | 人工标注本地是否具有充分证据，核对本地/联网选择及候选是否已包含正确证据 |
| 答案与引用 | 人工按必要事实、单位、限定条件和引用支持情况逐题评分 |
| 压缩保真 | 比较压缩前后必要证据和历史约束是否保留 |
| 模型成本 | 累计 `rewrite_usage`、`summary_usage`、`tool_selection_usage`、`generation_usage` |
| 用户时延 | 查看客户端端到端延迟分布，以及每个阶段的耗时 |

Token 预算使用保守估算，真实模型用量使用 API 返回的 usage。现有报告汇总成功轮次保存的用量，失败调用与 SDK 重试的额外费用结合服务商账单核对。标准答案、路由标签及关键事实的评分目前通过人工核对或专门评分代码完成。

## 7. 回归验证与阅读提示

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -B main.py --help
.\.venv\Scripts\python.exe -B -m pdfprocess.compare_retrieval --help
```

本次新增的 `tests/test_backend_structure.py` 检查旧导入入口、HTTP 字段、稳定任务键、上传结果结构及旧版 SQLite 状态恢复。原有测试继续覆盖 PDF 入库、候选检索、会话隔离、上下文压缩和原生工具调用协议。

自动测试通过注入模型或 MockTransport 验证工程行为，真实模型效果由论文标注集实验获得。PDF CLI 通过统一管线执行本地 OCR 与通义 VLM。无关练习、旧 ZIP 示例和重复的演示入口已移除。

## 8. 后端模块职责索引

| 文件 | 职责 |
| --- | --- |
| `main.py` | 启动入口。main() 处理命令行、环境配置和 Uvicorn 单进程启动。 |
| `pdfprocess/compare_ingest.py` | 解析方案实验入口。使用同一批 PDF、配置和参考词统计解析结果及调用成本。 |
| `pdfprocess/compare_retrieval.py` | 检索效果实验入口。导出语料供标注，并用真实 Chunk ID 计算 Recall、Hit Rate 与 MRR。 |
| `pdfprocess/context_compression.py` | 模型输入视图层。完整历史用于恢复，短历史及证据抽取用于本轮模型输入；预算数值采用统一估算单位。 |
| `pdfprocess/cross_encoder.py` | 本地模型适配层。问题和候选正文联合编码，原始 logit 交由 retrieval.py 排序与路由。 |
| `pdfprocess/embeddings.py` | 向量模型适配层。文档和问题复用相同模型，调用方负责知识库与任务生命周期。 |
| `pdfprocess/ingest_common.py` | 入库编排层。围绕文件身份、解析缓存、页级 Chunk、向量写入和完成标记组织可重试流程。 |
| `pdfprocess/paper_chunk.py` | 文本切块层。先按 Markdown 标题组织，再按字符预算切分，返回携带元数据的 Document 列表。 |
| `pdfprocess/pdf_pipeline1.py` | 方案一入口。页处理器选用本地 Markdown 提取，其余上传、切块、Embedding 和持久化复用公共编排。 |
| `pdfprocess/pdf_pipeline2.py` | 方案二入口。纯本地 RapidOCR，调用参数与方案一、三对齐。 |
| `pdfprocess/pdf_pipeline3.py` | 方案三入口。同一页并行运行本地 OCR 和通义 VLM，按区域与阅读顺序融合。 |
| `pdfprocess/pdf_vision.py` | 页面内容提取层。负责坐标转换、页面渲染、OCR 布局、区域裁剪及视觉模型结果合并。 |
| `pdfprocess/pdfload.py` | 统一 PDF 管线 CLI，默认使用本地 OCR + VLM 双路方案，网页上传由 upload_service 组织。 |
| `pdfprocess/rag.py` | 问答组装入口。build_graph() 注入依赖并连接节点，ask() 为新问题设置临时状态。 |
| `pdfprocess/rag_nodes.py` | 问答节点层。依赖由建图函数注入，节点通过返回字典提交状态增量。 |
| `pdfprocess/rag_prompts.py` | 提示词层。模型请求与输入预算使用同一模板，从而一起计入系统文本和消息格式开销。 |
| `pdfprocess/rag_state.py` | 状态契约层。messages 与摘要跨轮保存，证据、答案和工具调用按当前轮次更新。 |
| `pdfprocess/retrieval_cli.py` | 规范的本地检索调试 CLI，支持集合、模式与 TopK。 |
| `pdfprocess/retreiver.py` | 保留旧拼写的兼容入口，委托 retrieval_cli.main。 |
| `pdfprocess/retrieval.py` | 检索算法层。统一维护候选 ID 与各阶段分数，提供 Dense、BM25、RRF 和重排组合。 |
| `pdfprocess/tools/__init__.py` | 工具包公共导出。具体默认工具由 defaults.py 在服务启动时显式登记。 |
| `pdfprocess/tools/defaults.py` | 工具目录组装。这里选择默认实现，注册表和模型节点通过依赖注入读取目录。 |
| `pdfprocess/tools/registry.py` | 工具管理与执行边界。工具自身负责具体能力及客户端重试，注册表负责发现、校验与执行结果封装。 |
| `pdfprocess/tools/search_flow.py` | 模型工具协议层。先验证模型发出的搜索调用，再将带关联 ID 的工具结果交回模型回答。 |
| `pdfprocess/vector_store.py` | Chroma 写入适配层。校验向量维度和 Chunk ID 后，分批按稳定 ID 执行 upsert。 |
| `pdfprocess/web_search.py` | 公开网页搜索适配层。固定 RSS 入口返回标题、URL 和文本摘要，下游继续校验及分配证据预算。 |
| `upload_service/__init__.py` | 应用服务包。HTTP 入口见 app.py，业务对象由应用 lifespan 组装并释放。 |
| `upload_service/app.py` | HTTP 适配层。解析请求、选择响应状态并调用业务对象，业务对象在 lifespan 中集中初始化。 |
| `upload_service/application_store.py` | 业务数据访问层。在任务存储基础上管理文档生命周期、会话、幂等问答和结果快照。 |
| `upload_service/benchmark.py` | 上传性能实验入口。客户端确认耗时、任务完成耗时和服务端资源采样采用各自定义记录。 |
| `upload_service/benchmark_chat.py` | 问答性能实验入口。每个场景创建新会话，按原定问题顺序采集答案、证据、延迟与模型用量。 |
| `upload_service/chat.py` | 问答业务层。协调会话锁、模型依赖、LangGraph 检查点、请求恢复和业务结果落盘。 |
| `upload_service/config.py` | 服务配置层。环境变量在启动时转成明确类型，PipelineConfig 保存 PDF 处理相关参数。 |
| `upload_service/database.py` | 任务数据访问层。每次操作使用独立连接，写事务统一提交或回滚，任务状态和事件同步更新。 |
| `upload_service/documents.py` | 文档业务及可见范围层。Dense 与 BM25 使用同一文档就绪状态和用户选择范围。 |
| `upload_service/errors.py` | 业务异常契约。status_code 由 HTTP 异常处理器转换为响应，message 用于用户可读反馈。 |
| `upload_service/ingestion.py` | 上传与入队业务层。生成持久化配置快照和稳定任务键，连接文件存储及后台队列。 |
| `upload_service/schemas.py` | HTTP 输入契约。字段默认值与 Pydantic 约束在这里集中定义，业务关系由服务层进一步校验。 |
| `upload_service/storage.py` | 原始 PDF 存储层。流式保存、哈希标识、PDF 校验和临时文件清理由该模块处理。 |
| `upload_service/worker.py` | 后台执行层。线程从 SQLite 领取任务，PDF 解析通过 spawn 子进程执行，进度回写业务队列。 |

## 9. 基础项目整理与校园服务（2026-11-19）

路径集中到 `pdfprocess/paths.py`，PDF 配置集中到 `ingest_config.py`，方案分发集中到 `pipelines.py`。旧 `ingest_common.PipelineConfig` 导入继续可用。CLI、网页与容器遵守同一向量目录和重排缓存环境配置。各方案保留同名数字后缀入口，便于实验。

新增模块职责：`tools/wanfang_protocol.py` 处理万方 gRPC-Web/Protobuf；`tools/campus_papers.py` 处理知网/万方搜索与受限 PDF 下载；`tools/papers.py` 提供模型工具及原队列入库；`tools/paper_agent.py` 管理模型工具协议和目录指代。下载失败不会创建入库任务，pending 不会报告为完成。

技术文档统一放在 `docs/`。完整图见 [根 README](../README.md)，[校园论文工具](PAPER_TOOLS_GUIDE.md) 解释来源与权限，[部署指南](DEPLOYMENT_GUIDE.md) 解释 Docker。当前万方元数据可用，真实全文下载仍因后端机构授权不足未通过验收；Docker 配置已准备，当前机器尚无 Docker 运行环境。
