# PaperAgent 完整项目技术说明与实验手册

## 1. 项目目标和当前实现

PaperAgent 把论文文件处理、知识库检索和多轮问答连成一个可从浏览器使用的本机应用。使用者可以上传 PDF，等待后台入库，选择论文范围提问，查看证据，并在重启后继续同一个会话。

项目已有的三套解析管线、Embedding、Chroma、BM25、RRF 和 ONNX 重排模块继续承担底层计算。本次主要补齐应用入口、网页交互、持久化会话、文档生命周期和可观测信息。

| 功能 | 操作入口 | 核心实现 |
| --- | --- | --- |
| 批量上传 | 文档库，多选或拖放 PDF | FastAPI multipart，逐文件校验和独立结果 |
| 后台入库 | 上传后自动排队 | SQLite 任务队列，线程调度，子进程解析 |
| 三种解析对比 | 解析方式下拉框 | 本地 Markdown、全页本地 OCR、本地 OCR + 通义 VLM 双路融合 |
| 任务进度 | 文档状态、任务 API | 阶段事件与持久化任务状态 |
| 检索方案对比 | 检索设置 | dense、bm25、hybrid、hybrid_rerank |
| 多轮追问 | 聊天输入框 | LangGraph 消息状态、SQLite Checkpointer |
| 会话管理 | 左侧研读记录 | 新建、继续、改名、删除，thread_id 隔离 |
| 论文范围 | 顶部论文选择 | 向量召回与 BM25 共用文档范围 |
| 引用回查 | 答案中的编号和来源按钮 | 本轮证据快照、Chunk ID、PDF 页码 |
| 文档维护 | 文档库 | 原 PDF 查看、重建索引、删除及重启续清理 |
| 实验观测 | 答案耗时详情、实验脚本 | 检索问题、各阶段耗时、Token、排序和资源采样 |

本版本使用一个 Uvicorn 服务进程，可在本机或校园网共享空间运行。thread_id 隔离图状态；文档和会话列表共享，没有账号与个人权限隔离。Docker 操作及当前验证边界见 [部署指南](DEPLOYMENT_GUIDE.md)。默认地址为 `127.0.0.1:8000`，默认一个入库任务线程、一个 PDF 解析子进程。回答以完整 JSON 返回，界面等待完成后展示。

## 2. 对原项目的评价

原项目已经具有 RAG 的核心路径，模块边界也适合继续扩展。`paper_chunk.py` 把切块与模型调用分开，`vector_store.py` 提供统一写入，`rag.py` 用图描述检索与回答，三套解析管线可以共用后处理。这些模块可以直接进入应用服务层。

从可使用性看，还需要解决文件、向量、任务、会话之间的关联。单独保存 Chroma 只能回答“有哪些向量”，还需要知道“来自哪个文件、是否入库完成、处理失败在哪一步、该回答属于哪个会话”。本次增加的 SQLite 业务表和文件 SHA256 身份承担这些关联。

从研究验证看，增加候选数量、加入 BM25 或重排后，质量变化需要由同一批问题和人工证据标注确定。项目已提供排序与指标计算入口。你提供的论文问题和正确证据将成为正式质量评测集。

## 3. 系统数据流

下面是普通上传/RAG 路线。完整 LangGraph 条件边、论文工具分支与 State 以 [根 README](../README.md) 为准；论文下载队列与普通上传共用入库服务。

```text
浏览器
  ├─ POST /documents → 文件校验 → SHA256 原件 → SQLite 任务队列
  │                                           ↓
  │                            调度线程 → PDF 解析子进程
  │                                           ↓
  │                           页级 Markdown → 标题和长度切块
  │                                           ↓
  │                       Tongyi Embedding → Chroma → 任务完成
  │
  └─ POST /chat → session/thread_id → SQLite Checkpointer
                                             ↓
                     历史摘要与可选问题改写 → 统一论文范围
                                             ↓
                    Dense + BM25 → RRF → Cross-Encoder
                                             ↓
                     按最高重排相关度选择本地或模型工具搜索
                                             ↓
                         本轮证据预算压缩 → DeepSeek
                                             ↓
                        检查点 + 问答记录 → 答案与证据
```

上传页面负责选择文件、发送 HTTP 和显示进度。后端负责文件合法性检查、保存、排队和入库。PDF 解析和 Embedding 发生在后端。向量库、业务数据库和 Checkpointer 分别承载不同的数据。

## 4. 为什么选择这些技术

### 4.1 FastAPI 和同源轻量前端

FastAPI 已用于上传服务，继续使用可以复用 Pydantic 校验、OpenAPI 文档和已有测试。网页使用 HTML、CSS 和原生 JavaScript，静态资源与 API 从同一服务提供，启动只需要一个 Python 命令。

| 可选技术 | 优点 | 代价与适用情况 |
| --- | --- | --- |
| FastAPI + 当前前端 | 延续现有 Python 模块，接口有类型校验，部署简单 | 复杂前端状态依赖自行维护，适合当前研读工作台 |
| Streamlit | 很快搭出模型演示，Python 即可编写界面 | rerun 模式、复杂异步交互和会话细节需要专门处理 |
| Gradio | 模型演示、文件输入组件成熟 | 深度定制文档管理和多页面工作流时需要额外处理 |
| FastAPI + React/Vue | 组件、路由和状态生态丰富 | 需要 Node 构建和更多前端维护，适合后续复杂产品 |
| Django | 管理后台、用户与 ORM 生态完整 | 当前轻量应用会增加框架和迁移成本 |

可比较首次启动步骤数、依赖安装时间、页面操作成功率、接口 P95、维护某个交互所需修改量。框架名称本身不能推导模型回答质量。

### 4.2 PyMuPDF Markdown、OCR 与混合解析

方案 1 使用 PyMuPDF Markdown 导出提取文本层。方案 2 使用本机 RapidOCR 识别页面文字。方案 3 将同一页并行送入本机 OCR 与现有通义 VLM，按区域融合正文、表格、公式和图片描述。

三套入口保持相同参数约定，函数末尾分别为 1、2、3，例如 `ingest_pdf1`、`ingest_pdf2`、`ingest_pdf3`。批量入口为 `ingest_pdfs1/2/3` 和 `batch_upload_pdfs1/2/3`。

| 方案或替代技术 | 主要优势 | 主要代价 | 建议比较指标 |
| --- | --- | --- | --- |
| 本地 PyMuPDF Markdown | 可搜索文本 PDF 速度快，可减少云解析调用 | 扫描、复杂双栏、图表语义需要额外处理 | 秒/页、正文遗漏、阅读顺序、标题保留率 |
| 全页 OCR + 区域 VLM | 可处理扫描页，能表达图像区域语义 | 网络、调用成本、截断和生成错误 | CER、公式/表格正确率、调用次数、元/页 |
| 混合解析 | 根据页特征分配计算 | 质量阈值会影响路由，需要抽样检查 | OCR 页占比、总耗时、成本、最终证据召回 |
| MinerU | 面向文档结构、表格、公式的专门流程 | 服务限制或本地模型部署资源、版本维护 | 同一论文集上的结构质量与总成本 |
| Docling | 统一文档转换和结构表达 | 模型安装、硬件与版面质量需验证 | 阅读顺序、表格结构、CPU/GPU 时间 |
| pdfplumber/pypdf | 轻量文本或表格规则处理 | 复杂结构、扫描和图像语义需要组合方案 | 简单文档的速度、人工修正时间 |

PyMuPDF 的许可和部署方式需要在项目发布时结合官方条款核对。当前本机工程沿用已安装版本。其原生 PDF 操作放在 `spawn` 子进程，应用线程只负责任务调度和后续处理。接口文件校验使用 pypdf。

### 4.3 Markdown 标题切块 + 递归长度切块

先按 `#`、`##`、`###` 标题拆分，再按段落、换行和句子边界限制长度。默认 `chunk_size=800`、`chunk_overlap=120`，这里的长度单位是字符。入库时按页调用切块，保存 PDF 页码。

可替代方案包括固定长度、固定 Token、语义切块和父子块检索。固定长度容易实验和复现，语义切块能适应主题边界，父子块可以返回更完整上下文。后两者增加计算、存储或索引维护复杂度。

比较时固定 Embedding 和检索模式，分别测 400/800/1200 字符，overlap 取 0/80/120。记录 Chunk 数、向量化 Token/请求量、Recall@K、答案依据完整性和回答耗时。跨页表格、很长公式和长标题是需要人工抽检的样本。

### 4.4 保留 Tongyi Embedding

入库与查询共用 `TongyiEmbeddings`，当前模型为 `tongyi-embedding-vision-flash`。沿用模型可以继续使用已有向量空间，降低迁移成本。即使只向该模型传入文本，也仍需要在实验报告中记录实际模型名。

可替代模型有 BGE-M3、multilingual-e5 等本地模型，也可使用其他云 Embedding。比较维度包括中文与英文召回、领域术语、长输入限制、维度、请求延迟、吞吐、单位成本和本机内存。

更换模型后，入库与提问需要同步更换，并创建独立 Collection 重新向量化。相同维度无法证明两个模型处于同一个语义空间。现有入库会检查模型标识与维度，实验仍需记录完整配置。

### 4.5 Chroma、FAISS 和其他向量存储

当前选择 Chroma 是为了延续已有数据和持久化接口，并利用其 Chunk 正文、元数据和文档删除能力。

| 技术 | 优势 | 代价与更合适的阶段 |
| --- | --- | --- |
| Chroma 本地持久化 | 安装和调用简单，正文与 metadata 同步管理，已有数据可复用 | 本地并发、过滤效率与运维能力需要随规模验证 |
| FAISS | 进程内向量索引选择丰富，Flat/IVF/HNSW/PQ 便于研究速度与精度 | 文件、正文、ID 映射、元数据过滤和删除一致性需要另外实现 |
| Qdrant | 向量服务、payload 过滤、并发和持久化体系完整 | 需要运行与备份独立服务 |
| PostgreSQL + pgvector | 关系数据、事务和向量可集中管理 | 需要数据库运维和索引调优 |

若研究 FAISS，先构建 `float32` 矩阵，明确 cosine 或 L2 距离，保存 `vector_id → chunk_id → 正文/metadata` 映射，再实现索引持久化。cosine 实验通常需要向量归一化，再用内积索引。切换到近似索引时应以精确 Flat 检索作为基线。

比较向量库应固定相同向量、相同距离、相同查询与相同过滤条件。记录索引构建时间、索引大小、RSS、P50/P95、QPS、删除后残留率，以及相对于精确检索的 ANN Recall@K。还要保留人工证据 Recall@K，两种 Recall 的标注来源不同。

### 4.6 SQLite 业务库和 LangGraph SQLite Checkpointer

`service.sqlite3` 保存文档、入库任务、阶段事件、会话和问答业务记录。`conversations.sqlite3` 由官方 `SqliteSaver` 保存图状态。每个会话 ID 同时作为 LangGraph 的 `configurable.thread_id`。

`messages` 使用 `add_messages` reducer。每轮成功后追加用户消息与模型消息，历史用于理解追问，每轮仍重新执行检索。当前默认向模型发送旧历史摘要和最近最多 3 轮完整问答，检查点保留完整消息与增量摘要。答案完整输入预算为 16,000 保守估算 Token，其中证据最多 8,000，来源标题、页码和提示词开销一并计算。实现说明与调试步骤见 [上下文压缩与代码阅读](CONTEXT_COMPRESSION_GUIDE.md)。

| Checkpointer | 优势 | 代价与适用情况 |
| --- | --- | --- |
| InMemorySaver | 无数据库配置，适合单元测试与临时 CLI | 进程结束后历史消失 |
| SqliteSaver，本项目网页默认 | 官方实现，磁盘持久化，备份与本机调试简单 | 写入协调与磁盘 I/O 有限制，适合当前单进程应用 |
| PostgresSaver | 适合共享服务和更强并发需求 | 需要 PostgreSQL、迁移与连接管理 |

业务记录方便界面分页、展示状态和保存证据快照。检查点方便图恢复和多轮状态传递。两库之间通过稳定 `turn_id` 关联。成功回答已写检查点而业务提交中断时，重试或启动恢复可以补写业务结果。

### 4.7 BM25、RRF 和 Cross-Encoder

默认向量召回 20 条、关键词召回 20 条，按 Chunk ID 去重后用 RRF 融合，再用 Cross-Encoder 排序，最终最多 10 条进入证据预算。两路结果有交集时，融合候选会少于 40 条。

BM25 使用 jieba 中文分词与英文术语规则，适合补充精确名称、缩写和稀有术语。向量召回适合语义相近的表述。RRF 使用排名计算 `Σ 1/(60 + rank)`，避免直接相加不同量纲的原始分数。

Cross-Encoder 将问题与候选文本联合编码，当前使用 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` 的固定 revision 和量化 ONNX 权重，CPU 执行、批量 8、最多 512 Token。为长问题保留输入预算，重排阶段问题超过最多 128 Token 时截断。Embedding 和最终回答仍接收完整问题。

可替代方案有加权分数融合、BGE reranker、云 rerank API 和 LLM rerank。权重融合需要分数校准，较大的 reranker 可能提高任务相关性判断，代价是资源与延迟。LLM rerank 增加调用成本和输出约束处理。

候选集合未变化时，重排对全候选证据覆盖保持一致，它对最终 TopK 的收益需要观察。若正确证据没有进入候选池，重排无法补充该证据。

向量原始距离通常越小越接近，BM25、RRF 和本项目 Cross-Encoder logit 越大排名越靠前。重排分数没有经过概率校准，不能直接当作“答案正确概率”。

### 4.8 DeepSeek 与历史问题改写

DeepSeek 沿用当前模型接入，`temperature=0`、请求超时 60 秒、SDK 最多重试 2 次。追问改写可在检索设置关闭，首轮直接使用原问题，有历史时额外调用模型消解指代。

替代方案包括直接拼接最近用户问题、规则式指代补全或独立的小模型改写。当前方案能利用完整历史语义，代价是多一次调用，以及查询漂移的可能性。界面耗时详情中的 `retrieval_query` 用于直接检查真实改写结果。

提示词要求事实来自本轮证据、引用编号对应本轮来源，资料不足时明确表达。提示约束需要配合证据标注和人工评价，引用存在本身不能证明引用正确。

## 5. 存储结构和身份设计

默认数据目录为 `pdfprocess/data`。

```text
data/
  chroma_db/                  向量、Chunk 正文和 metadata
  ingestion/<SHA256>/         按方案和配置指纹保存的解析、页缓存和入库清单
  models/mmarco-minilm/       固定版本 ONNX 模型
  upload_service/
    files/<SHA256>.pdf        上传原件
    service.sqlite3          documents、tasks、task_events、sessions、turns
    conversations.sqlite3    LangGraph 检查点
```

文件身份取原件 SHA256，重命名不会产生新的内容身份。任务配置包含绝对数据目录、解析方案和参数快照。活动任务通过 job_key 去重，重复上传已完成文件仍会生成可审计任务，缓存命中时跳过重复计算。

文档状态依次可能为 `active`、`deleting`、`deleted`。新网页检索只发布对应集合最新任务已完成的受管文档。索引中旧有、尚未登记到服务的文档可以继续检索。原件登记和页码齐全时，引用可以跳到 PDF。

重建索引期间，该文档暂时从新网页检索范围隐藏。失败后可重试，其他已完成文档继续可用。原有 CLI 直接读 Chroma，做对比实验时应等待所有写入和删除结束。

## 6. 请求一致性和错误处理

### 上传和入库

- 校验扩展名、PDF 文件头、可读取页、加密情况和大小，默认单文件上限 50 MiB。
- 批量上传逐文件返回 accepted 或 error，单个错误不会打断其他文件的接收。
- 返回 HTTP 202 表示任务已接收，最终结果通过 `/tasks/{task_id}` 查询。
- 服务重启后将中断的 running 入库任务重新排队，页缓存和固定 Chunk ID 支持重试。
- 同一处理锁冲突时退避排队，异常退出的 PDF 进程池会重建。
- 服务根目录文件锁限制单个服务实例，防止多个实例同时恢复同一队列。

### 问答和会话

- 业务 `session_id` 与图 `thread_id` 使用同一个 ID。不同会话分别读取消息状态。
- 同一会话同时只能运行一个问题，冲突返回 409。不同会话可以分别调用，ONNX 推理由共享实例锁协调。
- 客户端为每轮生成 `request_id`。同一请求重试返回已完成结果，同一个 ID 换问题返回 409。
- 稳定消息 ID 为 `<turn_id>:human` 与 `<turn_id>:ai`，支持重试去重。
- 较早的失败问题再次执行时，展示顺序调整到当前执行位置，保持业务历史与图消息顺序一致。已完成检查点的恢复保留原轮次顺序。
- 中断后如果检查点已有完整结果，启动时补齐业务记录。未完成的轮次标为失败，可手动重试。
- 历史业务轮次存在而检查点缺失时，使用已完成轮次恢复消息，再继续执行图。
- Markdown 禁用原始 HTML 和图片，证据正文通过 textContent 显示，链接由 Markdown 解析器检查协议。

### 删除

目标文档存在 queued/running 入库任务时返回 409。其他文档仍可查询。删除先写 `deleting`，随后删除相关 Chroma 记录、解析缓存和服务原件，再写 `deleted`。中途退出后启动续清理。目录删除前会验证 SHA256 和解析目录边界。

旧会话保留当时的回答与证据快照。删除原件后，该来源的 PDF 按钮随文件可用性更新。业务任务保留审计信息。

## 7. 文件职责和本次改动

| 文件 | 本次职责 |
| --- | --- |
| `main.py` | 统一网页/API 启动，读取 .env，Windows multiprocessing 入口保护 |
| `upload_service/app.py` | 保留上传接口，加入文档、Collection、会话、问答、静态页面与技术文档路由 |
| `upload_service/application_store.py` | 新增会话与问答表、请求幂等、文档可见性与删除生命周期 |
| `upload_service/database.py` | 为原任务数据库迁移文档生命周期列，防止删除期间重新排队 |
| `upload_service/chat.py` | SqliteSaver、图缓存、历史恢复、会话锁、证据和 Markdown 输出 |
| `upload_service/documents.py` | Chroma 客户端、两路检索统一范围、原件访问与可恢复删除 |
| `upload_service/errors.py` | 将业务错误映射到清晰 HTTP 状态 |
| `upload_service/benchmark_chat.py` | 按多轮问题场景记录端到端耗时、Token 和证据快照 |
| `pdfprocess/rag.py` | 增加可选历史改写、历史窗口、证据预算、稳定消息 ID 和阶段指标，原有调用可继续使用 |
| `pdfprocess/cross_encoder.py` | 增加长问题重排输入预算，避免 only_second 截断失败 |
| `web/index.html` | 研读工作台、文档库、设置、范围、会话和证据面板结构 |
| `web/style.css` | 桌面和手机布局，交互状态与文档页样式 |
| `web/app.js` | 调用公开 API，轮询任务、提交提问、管理会话与展示引用 |
| `tests/test_application.py` | 新增完整应用的持久化、隔离、幂等、恢复、范围和删除测试 |
| `tests/test_retrieval.py` | 增加真实 ONNX 模型的长问题回归测试 |
| `requirements.txt` | 汇总原依赖并补齐应用所需版本 |
| `.env.example`、`.gitignore` | 提供配置示例与运行数据忽略规则 |
| `experiments/chat_scenarios.example.json` | 手动多轮实验场景示例，可替换成你的论文问题 |
| `README.md`、本文件 | 启动说明、设计原理、技术替代、量化实验和调试手册 |

底层继续复用 `pdf_pipeline1/2/3.py`、`pdf_vision.py`、`ingest_common.py`、`paper_chunk.py`、`embeddings.py`、`vector_store.py`、`retrieval.py`、`compare_ingest.py`、`compare_retrieval.py`、`storage.py`、`worker.py` 和上传压测脚本。pdfload CLI 与主应用共同使用三套统一管线。

## 8. 启动与配置

```powershell
Set-Location C:\Users\Mark\Desktop\langgraph
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

新安装先 `python -m venv .venv`，复制 `.env.example` 为 `.env` 并填入密钥。现有桌面项目使用已有 .env。依赖按本机验证版本固定，实际验证为 Windows、Python 3.14.8。换系统或 Python 版本时先验证依赖安装和测试结果。

| 配置 | 默认或含义 |
| --- | --- |
| DASHSCOPE_API_KEY | 文档和问题 Embedding，图表 VLM |
| DEEPSEEK_API | 回答和可选的追问改写 |
| PAPERAGENT_OCR_THREADS | 本地 OCR 的 CPU 线程数，默认 2 |
| PAPERAGENT_CHAT_MODEL | deepseek-chat |
| PAPERAGENT_SERVICE_ROOT | 业务 SQLite、Checkpointer 和上传原件根目录 |
| PAPERAGENT_INGEST_ROOT | 页缓存与 Markdown 根目录 |
| PAPERAGENT_CHROMA_DIR | 向量库目录 |
| PAPERAGENT_RERANKER_MODEL_DIR | 本地固定版本模型目录，省略时使用项目 data/models |
| PAPERAGENT_UPLOAD_WORKERS | 1，入库调度线程数 |
| PAPERAGENT_PARSE_PROCESSES | 1，PDF 解析子进程数 |
| PAPERAGENT_UPLOAD_MAX_MB | 50，实际以 1024² 字节换算 |
| PAPERAGENT_SQLITE_JOURNAL_MODE | WAL，可与 DELETE 模式对比 |

自定义切块、DPI、路由阈值或 OCR 模型可在 `PipelineConfig` 构造时设置。当前 HTTP 上传接口只暴露 scheme，保持研究变量可控。用 Python 调用 `create_app(ServiceSettings(pipeline=PipelineConfig(...)))` 可以运行指定配置。

## 9. API 手动调试

浏览器打开 `/docs`，使用 Try it out 即可上传文件、创建会话和提问。先上传，轮询任务为 completed，随后获取 Collection 和文档 ID。

```powershell
curl.exe -F "files=@C:\papers\paper.pdf" -F "scheme=1" http://127.0.0.1:8000/documents
curl.exe http://127.0.0.1:8000/collections
curl.exe http://127.0.0.1:8000/documents
```

在 PowerShell 使用结构化 JSON 创建会话，避免引号转义干扰。

```powershell
$sessionBody = @{
  collection_name = 'paper_chunks1'
  mode = 'hybrid_rerank'
  dense_k = 20
  keyword_k = 20
  top_k = 10
  rewrite_query = $true
  document_ids = @()
} | ConvertTo-Json
$session = Invoke-RestMethod http://127.0.0.1:8000/sessions -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($sessionBody))
$requestId = [guid]::NewGuid().ToString()
$chatBody = @{
  thread_id = $session.id
  request_id = $requestId
  question = '论文用了什么方法？请标注证据。'
} | ConvertTo-Json
$answer = Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($chatBody))
$answer.metrics | ConvertTo-Json -Depth 8
```

再次发送完全相同的 body，核对返回的 turn id 相同。发送下一轮时创建新的 request_id，保持 thread_id。创建第二个 session 后，检查其历史只包含该会话的问题。

| API | 作用 |
| --- | --- |
| POST /documents | 批量上传，返回逐文件任务 ID |
| GET /tasks/{id} | 任务状态、阶段事件、阶段耗时和结果 |
| GET /documents、GET /collections | 文档状态与集合中总 Chunk/可用 Chunk 数 |
| GET /documents/{id}/file | 原 PDF，支持浏览器页码 fragment |
| POST /documents/{id}/reindex | 按 scheme 重建，refresh 控制缓存 |
| DELETE /documents/{id} | 删除该受管文档的原件、索引和缓存 |
| POST/GET /sessions | 创建/列出会话 |
| GET/PATCH/DELETE /sessions/{id} | 查看、改名、删除会话 |
| POST /chat | 使用 thread_id 和 request_id 提问 |
| GET /health | 服务、队列、RSS 与 CPU 累计采样 |

## 10. 应关注哪些量化指标

| 层次 | 指标 | 计算或采集方式 |
| --- | --- | --- |
| 上传 | 接收 P50/P95、拒绝率 | 上传脚本 accept_seconds，区分 202 与最终完成 |
| 入库 | 排队、解析、切块、Embedding、保存时间 | queue_wait_seconds、result.timings 与事件时间差 |
| 吞吐 | 成功文件/秒、页/秒 | 固定文件集总页数除以墙钟时间，同时报告失败数 |
| 解析质量 | CER/WER、遗漏率、顺序正确率 | 对人工校对的同一页面文本进行编辑距离/遗漏/顺序评分 |
| 表格质量 | 单元格正确率、结构一致性 | 对照人工表格，拆分内容和行列结构评分 |
| 运行资源 | 峰值 RSS、CPU 秒、磁盘占用 | health 轮询，任务进程树采样和目录大小 |
| 成本 | 元/页、元/成功问题 | 按实际模型账单单价乘输入/输出 Token，再加 OCR 等调用费用 |
| 缓存 | 缓存页比例、跳过入库比例 | cached_pages/页数，skipped 任务数/完成任务数 |
| 检索质量 | Recall@K、HitRate@K、MRR@K | 基于问题 → 正确 Chunk ID 的人工标注 |
| 检索排序 | 同 K 的收益、全候选覆盖 | compare_retrieval 的 metrics 与 candidate_recall |
| 回答质量 | 事实正确率、引用支持率、完整性 | 按人工评分细则逐条核对论文和证据 |
| 多轮 | 指代解析成功率、跨轮约束保留率 | 固定问题序列，人工核对追问是否指向正确对象 |
| 隔离恢复 | 串话率、重启恢复率、重复消息数 | 两会话互异事实、重启续问与重复 request_id 实验 |
| 端到端 | 客户端 P50/P95、成功率、Token | benchmark_chat 输出与页面 metrics |

`retrieve_seconds` 包含该次问题 Embedding、范围过滤、BM25、融合和重排的总时间。`generate_seconds` 是生成节点耗时。`rewrite_seconds` 是改写节点耗时。`request_seconds` 从取得请求后的图处理计时，包含首次图初始化，结束点在业务结果保存之前。客户端时间还包含 HTTP、业务落库与渲染前的数据传输。

Token 分别记录在 `rewrite_usage`、`generation_usage`、`summary_usage` 和 `tool_selection_usage`，供应商未提供时为空。`summary_usage` 是分批摘要调用的用量列表。页面摘要显示回答模型的 Token，计算总成本时应同时累计改写、摘要和工具调用生成。Embedding 的计费需结合供应商账单和模型实际计费单位。预算采用 UTF-8 字节数加消息余量估算，真实计费以接口用量和账单为准。

字符识别可按 `CER=(替换字符数+删除字符数+插入字符数)/人工原文字符数` 计算。先约定是否归一化空格、换行和 Unicode，然后对全部方案使用同样规则。正文遗漏率可以按人工标出的事实句或段落统计遗漏数，表格内容正确率可以按正确单元格数除以人工表格单元格数统计。

回答评价可以建立一张 CSV，字段为 question_id、mode、事实分、完整性分、引用支持数、可核对事实数、无依据事实数、追问指代是否正确、人工备注。事实分和完整性分各取 0/1/2，分别表示错误或缺失、部分正确、完整正确。引用支持率取“被所引证据支持的可核对事实数/带引用的可核对事实数”，无依据事实率取“当前证据无法支持的事实数/全部可核对事实数”。分母为 0 时单独记录 NA。抽样双人评分并核对分歧，可以提高评分标准的一致性。

模型费用可按 `输入 Token×对应单价/计价单位 + 输出 Token×对应单价/计价单位` 计算。改写与回答分别计入，再加 Embedding、OCR 和 VLM。供应商区分缓存命中价格时，根据返回 usage 和账单拆分计价。比较“元/成功问题”时，将失败请求的实际费用计入总费用。

## 11. 召回指标和提升率怎么算

对问题 q，记人工正确证据集合为 G，检索前 K 个 ID 为 R。

```text
Recall@K = |R ∩ G| / |G|
HitRate@K = 1（前 K 个有任意正确证据）或 0
MRR@K = 1 / 首个正确证据的排名，前 K 个无正确证据时为 0
```

数据集结果采用逐问题平均。正确证据应尽量完整标注，遗漏其他有效证据会影响指标解释。多份相互替代的证据可另外按页或证据组汇总，用于减少切块方式变化带来的偏差。

假设基线 Recall@10 为 0.60，改进后为 0.75，这个教学例子的绝对提升为 15 个百分点，相对提升为 25%。本项目真实提升率需要你的论文标注运行后得到。基线为 0 时，相对提升返回 null。

重排前后应在相同 K 下比较。将 dense@5 与 hybrid_rerank@10 放在一起会同时改变候选方式和证据预算，实验解释需要把两个变量拆开。

## 12. 手动完成一次检索对比实验

### 第一步，固定语料和标注

等待全部入库完成，记录 Collection、解析方案、切块参数、Embedding 模型和软件版本。导出所有 Chunk。

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.compare_retrieval --collection paper_chunks1 --export-corpus experiments/results/corpus.json
```

从导出的正文核对论文，填写 `retrieval_eval_template.json`，使用真实 ID。建议至少准备覆盖精确术语、同义表述、跨段证据、数字与方法比较的若干类问题，初次可从 30～50 个问题开始，再扩大样本。把导出的 corpus_fingerprint 放入标注文件，保持与语料对应。

切块方案变化时 Chunk ID 可能改变，要重新匹配证据，或使用稳定的页码/原文证据位置进行映射。

### 第二步，同一语料比较四种检索

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.compare_retrieval --dataset retrieval_eval_template.json --collection paper_chunks1 --dense-k 20 --keyword-k 20 --at-k 5 10 20 --output experiments/results/retrieval-20-20.json
.\.venv\Scripts\python.exe -m pdfprocess.compare_retrieval --dataset retrieval_eval_template.json --collection paper_chunks1 --dense-k 40 --keyword-k 40 --at-k 5 10 20 --output experiments/results/retrieval-40-40.json
```

第一个命令在同一查询上得到 dense、bm25、hybrid、hybrid_rerank。模型预热时间和 BM25 建索引时间单独记录。第二个命令研究扩大候选池的效果。

查看 `same_k_recall_improvements` 得到同 K 的绝对百分点与相对百分比。查看 `dense_topk_sweep` 研究单纯增加输出 K。查看 `candidate_recall` 判断瓶颈在候选召回还是最终排序。查看每个问题的 rankings 定位失败案例。

### 第三步，解释结果

正确证据未进入候选池时，先检查解析、Chunk 边界、问题表达和两路候选数。证据已在候选池而最终 Top10 丢失时，检查 Cross-Encoder 输入截断和领域适配。排序正确但回答缺依据时，检查证据预算和生成提示。

记录每个查询的成对差值，推荐以问题为抽样单位做 bootstrap 区间。测试集和调参集应分开，最终结果报告样本数、均值、区间、延迟和成本。

## 13. 手动比较解析与上传性能

先按文本页、扫描页、双栏、表格、公式、图像分类抽样。对同一批 PDF 运行同样参数。以下命令仅解析，避免把 Embedding 费用混入解析实验。

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.compare_ingest --pdfs C:\papers\a.pdf C:\papers\b.pdf --schemes 1 2 3 --parse-only --refresh --data-root experiments/results/parse-cold --output experiments/results/parse-cold.json
```

去掉 `--refresh` 再运行，使用同一个 data-root，测缓存效果。需要完整入库时去掉 `--parse-only` 并指定单独 `--chroma-dir`。正式库和实验库分开目录便于复现实验。

运行中的服务也可以通过真实 HTTP 压测。

```powershell
.\.venv\Scripts\python.exe -m upload_service.benchmark --url http://127.0.0.1:8000 --pdfs C:\papers\a.pdf C:\papers\b.pdf --scheme 1 --concurrency 1 --batch-size 2 --output experiments/results/upload-c1.json
```

并发取 1、2、4，保持工作线程数不变，先测排队。研究处理并发时再独立修改 workers/parse_processes 并重启。避免同时改变客户端并发与处理线程，导致无法判断因果。

压测报告提供成功、失败、未完成任务、接收延迟、排队和处理时间、缓存命中、资源采样等字段。采样峰值 RSS 是离散采样结果。统计时同时记录采样间隔和机器规格。

## 14. 手动比较多轮效果与成本

复制 `experiments/chat_scenarios.example.json`，将 questions 改成你的论文问题序列。若只研读指定论文，将 GET /documents 得到的 SHA256 填入 document_ids。

```powershell
.\.venv\Scripts\python.exe -m upload_service.benchmark_chat --scenarios experiments/chat_scenarios.example.json --collection paper_chunks1 --mode hybrid_rerank --repeat 3 --output experiments/results/chat-rewrite.json
.\.venv\Scripts\python.exe -m upload_service.benchmark_chat --scenarios experiments/chat_scenarios.example.json --collection paper_chunks1 --mode hybrid_rerank --no-rewrite --repeat 3 --output experiments/results/chat-direct.json
```

脚本每次重复创建新会话，在该会话按顺序发送问题，并将实验会话保留在界面供复查。某一轮失败时停止该场景后续追问，报告的成功率分母是实际尝试轮次。网络异常会终止脚本并显示错误，应检查服务和连接后重新运行。

用同一评分表检查“它”是否指向正确论文、是否保留前轮限定、引用是否支持答案。对照实际 retrieval_query，识别改写增加了答案内容或偏离原问题的案例。使用三次以上重复观察模型波动，性能正式报告需要足够样本。

继续对 dense、bm25、hybrid、hybrid_rerank 分别运行。控制最终 top_k、证据预算、模型和问题序列。把检索质量变化与生成质量变化分开记录。Token 和延迟需要一并报告。

## 15. 在 IDE 里怎么打断点

在 PyCharm 或 VS Code 选择项目 `.venv`，以 `main.py` 为启动文件，工作目录设为项目根目录。应用进程和 PDF 子进程分开，调试解析时启用 IDE 的子进程附加，或单独运行解析函数。

| 断点位置 | 建议观察变量 | 用途 |
| --- | --- | --- |
| `storage.py` 的 `FileStorage.save` | filename、大小、校验结果、SHA256 | 文件被拒绝或重复上传 |
| `app.py` 的 `enqueue_document` | snapshot、job_key、collection_name | 检查实际入库配置与队列去重 |
| `worker.py` 的 `_run_task` | task、stage、progress details | 判断失败发生在哪一阶段 |
| `pdf_vision.py` 的 `local_page/hybrid_page` | markdown、blocks、路由阈值 | 文字顺序、扫描页分支和图表遗漏 |
| `ingest_common.py` 的 `run_ingestion` | parsed、chunks、vectors、fingerprint | 缓存、切块数量、向量化和幂等写入 |
| `documents.py` 的 `ScopedStore.get` | known、ready、document_ids | 某篇论文被过滤或错误进入检索 |
| `retrieval.py` 的 `PaperRetriever.search` | dense、keywords、fused、重排结果 | 分阶段比较正确证据排名 |
| `cross_encoder.py` 的 `score` | query_ids、encoded、logits | 输入截断、推理批次与分数 |
| `rag.py` 的 rewrite/retrieve/generate | retrieval_query、context、history、sources | 追问漂移、上下文不足与引用 |
| `chat.py` 的 `chat/_checkpoint_result` | session、turn、检查点 message IDs | 请求重试、检查点与业务提交 |

调试时建议先 scheme1、dense、top_k=3，确认完整路径后逐步开启 BM25、RRF、重排和问题改写。断点暂停会改变计时，性能实验应在关闭断点后重跑。

## 16. 恢复、备份与常见问题

关闭服务并等待退出后，整体备份 Chroma、ingestion、upload_service 与 .env 的受控副本。SQLite WAL 模式可能存在 `-wal` 和 `-shm` 文件，在线拷贝单个主库可能得到不一致备份。当前最简单方式是停服后备份完整数据目录。

| 现象 | 排查顺序 |
| --- | --- |
| 无可用知识库 | 等待任务 completed，再选 paper_chunks1/2/3 对应集合 |
| 上传后一直排队 | 查看 health workers_alive，任务 stage/ready_at，以及是否有处理锁冲突 |
| OCR 失败 | 核对 RapidOCR、ONNX Runtime、本地权重及图片尺寸，双路方案另核对通义接口与 JSON 布局 |
| 只有旧正文、没有 PDF 按钮 | 旧 Collection 中尚未登记原件，可重新上传对应 PDF 建立管理记录 |
| 向量维度或模型不匹配 | 保持同一个 Embedding 模型，另建集合重新入库 |
| 首次重排很慢 | 检查固定版本模型下载和首次 ONNX 加载，区分冷启动与热态 |
| 回答引用不准确 | 对照本轮 sources、context 和原文，补充人工错误类型记录 |
| 409 问答冲突 | 等待同会话当前问题完成，或重试原 request_id |
| 409 文档删除冲突 | 等待该文档当前入库任务结束 |
| 删除中断 | 重启后会续清理，也可在文档库重试删除 |
| 端口占用 | 使用 `python main.py --port 8001` 并打开对应地址 |

## 17. 验证范围和结果使用

本次验证包含原回归测试、新增应用测试、真实云 Embedding/DeepSeek 与本地 ONNX 的合成 PDF 全链路，以及桌面和手机尺寸的浏览器交互。具体数量、耗时和记录见随项目交付的验证报告。

真实链路使用人工构造的单页文本 PDF，能够检查上传、页码、范围、追问、重启恢复、重建和删除的连通性。论文语料的 OCR 质量、正式 Recall 提升率和大规模并发指标需要按前面的实验步骤继续采集。

当前 BM25 会读取可见语料并比较指纹，范围向量召回采用过取候选再过滤，Collection 列表也会扫描 metadata。随着 Chunk 数增加，这些操作可能成为延迟和内存瓶颈。你可以先以 1千、1万、10万 Chunk 做阶梯测试，再决定是否改为持久化关键词索引、数据库侧过滤或服务型向量库。

长期会话检查点会累计完整历史，模型输入使用增量摘要和近期完整问答。输入压缩后仍需要观察检查点大小和长会话写入时间。摘要存在信息损失，需要用固定论文问题检查事实与约束保留率。模型接口需要网络和额度，扫描路径还受 OCR/VLM 服务可用性影响。

## 18. 技术参考

- LangGraph 的持久化和 thread 说明。[LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- SQLite Checkpointer 官方包。[langgraph-checkpoint-sqlite](https://pypi.org/project/langgraph-checkpoint-sqlite/)
- FastAPI 文件上传和接口文档。[FastAPI request files](https://fastapi.tiangolo.com/tutorial/request-files/)
- PyMuPDF 的多进程处理说明。[PyMuPDF multiprocessing](https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html)
- Chroma 本地客户端与数据操作。[Chroma documentation](https://docs.trychroma.com/)
- 当前多语言重排模型及版本来源。[mmarco-mMiniLMv2-L12-H384-v1](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)
- FAISS 索引选择和向量搜索。[FAISS wiki](https://github.com/facebookresearch/faiss/wiki)

本文件的实现细节以随附代码为准。外部链接用于继续学习和核对版本行为。

## 19. 低相关性时的联网搜索

2026 年 11 月 16 日增加联网搜索工具，随后接入工具注册表与模型原生工具调用。默认混合重排图在历史整理及问题改写后执行 `retrieve → 按最高重排相关性判断 → 本地压缩回答 或 request_search → web_search → compress → generate`。0.5 为路由阈值，最高分恰好等于阈值时进入本地回答分支。

Cross-Encoder 原始输出为 logit，新增 `RetrievalHit.rerank_relevance` 属性计算数值稳定的 sigmoid。最高相关性低于 0.5，或本地候选为空时，检索节点返回空 context、空 sources 和 web 来源状态，再进入联网工具节点。

`pdfprocess/web_search.py` 定义 LangChain `web_search` 工具，使用 Bing RSS 返回最多 5 条标题、HTTP 链接和摘要。该工具使用现有 requests，设置超时、响应大小限制和一次临时故障重试。解析后清理 HTML、去重链接并过滤无效结果。

联网分支先让模型产生带 query 的 `web_search` 调用，由注册表执行，再将压缩后的网页资料作为匹配调用 ID 的 `ToolMessage` 交给模型组织答案。调用模型及答案模型都只接收独立问题和当前任务相关内容，历史消息及摘要排除。回答固定带有“本地数据库未检索到足够相关的内容。”。完整会话仍保存在检查点，后续轮次可以继续使用。联网失败进入问答失败与重试流程，空结果直接提示资料不足。引用面板提供来源网页链接。

`pdfprocess/tools/registry.py` 定义注册表，`defaults.py` 显式登记工具，`search_flow.py` 定义模型工具调用协议。模型工具选择的实际用量保存在 `tool_selection_usage` 并纳入现有基准工具汇总。扩展与断点说明见项目根目录 `TOOL_REGISTRY_GUIDE.md`。

手动调试时选择“混合召回 + Cross-Encoder”，使用一个知识库明显缺少的公开问题。检查答案 metrics 中的 `local_max_relevance < 0.5`、`evidence_origin=web`，以及 sources 全部具有 `kind=web`。再用已有论文中的问题检查本地分支。路由阈值来自本次需求，实际质量可通过标注评测继续量化。

## 20. 后端代码阅读与真实评测准备

后端按职责拆分为依赖组装、状态、提示词、节点、HTTP 输入及入队业务等模块。`rag.py` 连接图，`rag_nodes.py` 执行节点，`rag_state.py` 定义状态，`rag_prompts.py` 定义本地提示词。HTTP 字段集中到 `upload_service/schemas.py`，上传与重建索引共同调用 `IngestionService`。

后端模块、类和函数已补充中文文档字符串，关键事务、缓存、坐标、候选排序和工具协议有逐步注释。所有模块职责、断点位置、语料导出和效果评测命令见`docs/BACKEND_REFACTOR_GUIDE.md`。


本地 OCR 双路解析、长对话预算与论文自动入库的当前实现及调试步骤见 [PAPER_TOOLS_GUIDE.md](PAPER_TOOLS_GUIDE.md)。
