# PaperAgent 上传服务与实验指南

项目目录为 `C:\Users\Mark\Desktop\langgraph`。

这份文档面向正在学习 Python Agent 工程的开发者，说明上传服务如何连接现有 RAG、为什么采用这些技术，以及怎样通过实验判断改进是否有效。

## 1. 上传服务在项目中的位置

现有 PaperAgent 已具备 PDF 解析、切块、Embedding、Chroma 保存，以及混合检索和多轮问答。本次增加 HTTP 上传入口、持久化任务队列、阶段查询和基准脚本。

```text
浏览器、脚本或前端应用
        │ multipart/form-data
        ▼
FastAPI /documents
        │ 文件分块落盘、SHA256、PDF 结构校验
        ▼
SQLite 文件记录与任务队列
        │ 返回 HTTP 202 和 task_id
        ▼
任务工作线程
        ├─ 解析子进程 → 页级 Markdown
        ├─ 原有 Chunk 切分
        ├─ 原有 DashScope Embedding
        └─ 原有 Chroma 保存
        │
        ▼
SQLite 阶段事件与完成结果
        ▲
GET /tasks/{task_id}
```

前端负责选择文件、发送请求和展示任务状态。后端上传接口负责接收文件、创建任务。入库服务执行原有 PDF 处理逻辑。

HTTP 202 表示原文件已保存，任务已经提交或复用了现有活跃任务。任务状态为 `completed` 时，表示该次入库已返回完成结果。

## 2. 为什么选择这些技术

### FastAPI 与 Uvicorn

FastAPI 定义接口、校验参数并生成 OpenAPI 文档。Uvicorn 运行 ASGI 应用，处理网络连接和请求。

| 技术 | 适合当前项目的能力 | 需要承担的工作 | 对比方法 |
| --- | --- | --- | --- |
| FastAPI | 类型校验、文件参数、OpenAPI、交互式接口文档，容易包装现有 Python 函数 | 自行组织数据库和长期任务执行层 | 固定业务函数后比较上传确认延迟、吞吐和资源 |
| Flask | 路由简洁、扩展生态成熟 | 根据需要接入参数校验和接口文档扩展 | 用相同上传协议、数据校验和任务队列建立对照 |
| Django | ORM、管理后台等业务管理能力较完整 | 选择并配置项目所需组件 | 业务管理需求增加时比较开发成本及运行指标 |

FastAPI 的异步入口适合协调网络请求。PDF 解析和同步模型调用通过后台执行层处理。框架性能需要在相同业务、硬件和并发条件下实测。[FastAPI 文件上传文档](https://fastapi.tiangolo.com/tutorial/request-files/)

### multipart 与 UploadFile

浏览器上传文件时使用 `multipart/form-data`。请求中的 `files` 可重复出现，表示一批 PDF，`scheme` 表示解析方案。

`UploadFile` 使用可转存磁盘的临时文件。服务随后按 1 MiB 分块复制、计算 SHA256 并落盘，避免接收业务代码一次读入全部文件内容。PDF 解析进程的内存开销单独测量。[UploadFile 说明](https://fastapi.tiangolo.com/tutorial/request-files/)

与 JSON 中的 Base64 文件相比，multipart 可以直接传文件。Base64 的编码体积大致是原始字节的 4/3，另有 JSON 字段开销，可通过相同文件的请求体大小和端到端传输时间验证。

当前单文件上限默认为 50 MiB。该校验发生在 multipart 解析后的业务接收阶段。若需要在传输过程中限制整批请求体，需要在部署入口设置总请求体上限。

### pypdf 校验与 PyMuPDF 解析进程

pypdf 在上传阶段读取 PDF 结构、页数和加密状态。PyMuPDF 继续承担正文、版面和页面图像处理。

PyMuPDF 官方说明其多线程调用可能导致错误。因此，服务使用独立解析进程，每个进程创建自己的文档对象。任务线程负责调度、Embedding 网络调用和保存。[PyMuPDF 并发说明](https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html)

| 执行方式 | 优点 | 代价及关注指标 |
| --- | --- | --- |
| 工作线程 | 便于复用现有同步函数和等待网络 I/O | 纯 Python CPU 工作受 GIL 影响，观察 CPU、等待时间和吞吐 |
| 解析进程 | 隔离 PyMuPDF 文档对象，可分别处理文件 | 启动、模块导入、结果序列化和额外内存，观察首次任务及总 RSS |
| 独立任务系统 | 可扩展到独立工作节点，提供更完整的队列管理 | 需要部署 Broker、Worker、结果存储和监控 |

默认任务线程数为 1、解析进程数为 1。两个参数可分别调整，便于定位网络等待与 PDF 解析瓶颈。

### SQLite 保存任务

SQLite 是本地嵌入式数据库。本次使用标准库 `sqlite3`，保存三类信息。

| 表 | 主要内容 |
| --- | --- |
| documents | 文件哈希、原始名称、路径、字节数、页数 |
| tasks | 解析方案、目标 Collection、配置快照、状态、时间、执行次数、结果和错误 |
| task_events | 排队、开始、解析、切块、Embedding、保存及结束事件 |

任务通过 SQLite 事务领取。并发工作线程领取时，同一行只会被领取一次。相同文件、方案和配置的活跃任务使用唯一约束复用。

默认启用 WAL 和 FULL 同步。WAL 允许读取和写入并行，写事务仍按顺序执行，适合当前本地服务。多机器、高写入并发及更复杂事务可以使用 PostgreSQL 评估。[SQLite WAL 文档](https://www.sqlite.org/wal.html)

| 存储方式 | 使用特点 | 对比指标 |
| --- | --- | --- |
| 本地 JSON | 适合保存每页解析产物和单文档缓存 | 文件更新冲突、查询时间、崩溃后可恢复记录数 |
| SQLite | 支持事务、查询和索引，本地部署简洁 | 查询 P95、锁等待、任务领取吞吐、磁盘增长 |
| PostgreSQL | 支持多客户端和更广的业务扩展 | 网络往返、连接池等待、事务吞吐、部署维护成本 |

Chroma 保存正文、向量和检索元数据。SQLite 保存上传业务与任务状态。原有 JSON 继续保存页级解析及入库缓存。

### 持久化队列与 BackgroundTasks、Celery

服务先提交 SQLite 任务，再由工作线程领取。进程重启后，排队任务继续处理，原先的运行中任务重新入队。

FastAPI `BackgroundTasks` 适合请求返回后的轻量工作。较重任务可采用 Celery 等独立系统。本次以已有 Python 入库函数和本地持久化队列降低接入复杂度。[FastAPI 后台任务说明](https://fastapi.tiangolo.com/tutorial/background-tasks/)

当前服务使用单实例锁协调同一个任务数据库，启动命令采用一个 Uvicorn 进程。任务并发通过工作线程和解析进程参数控制。

本地队列的部署组件较少，状态机、领取事务、重试和恢复测试由项目代码维护。独立任务系统提供更完整的队列生态，同时增加部署及运行维护工作。

## 3. 启动和使用

```powershell
Set-Location 'C:\Users\Mark\Desktop\langgraph'
.\.venv\Scripts\python.exe -m pip install -r requirements-upload.txt
.\.venv\Scripts\python.exe -m uvicorn upload_service.app:app --host 127.0.0.1 --port 8000 --workers 1
```

浏览器打开 [交互式接口文档](http://127.0.0.1:8000/docs)。展开 `POST /documents`，点击 Try it out，选择一个或多个 PDF、设置 scheme，再执行请求。复制响应中的 task_id，通过任务接口查询。

首次验证可以选择有文本层的 PDF 和方案 1。完整入库使用项目已有 `.env` 中的 `DASHSCOPE_API_KEY`。方案 2、3 的 OCR 在本机运行，方案 3 的 VLM 路径使用现有 DashScope 配置。`DEEPSEEK_API` 用于后续问答生成阶段。

| 接口 | 请求与作用 |
| --- | --- |
| POST /documents | multipart 文件列表和 scheme，接受后返回任务 ID |
| GET /tasks/{task_id} | 查询状态、当前阶段、事件、结果和错误 |
| GET /health | 查询工作线程、队列、服务参数及进程资源采样 |

`scheme=1` 使用本地 Markdown，`scheme=2` 使用全页 OCR 与图表补充，`scheme=3` 使用按页选择的混合方案。目标 Collection 默认分别为 `paper_chunks1`、`paper_chunks2`、`paper_chunks3`。

例如，方案 3 的任务完成后，可通过现有命令行问答读取对应 Collection。

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.rag --collection paper_chunks3 --mode hybrid_rerank
```

### Python 批量上传示例

```python
from contextlib import ExitStack
from pathlib import Path
import httpx

paths = sorted(Path(r'C:\Users\Mark\Desktop\langgraph\articels').glob('*.pdf'))
with ExitStack() as stack:
    files = [('files', (p.name, stack.enter_context(p.open('rb')), 'application/pdf'))
             for p in paths]
    response = httpx.post('http://127.0.0.1:8000/documents',
                          files=files, data={'scheme': '1'}, timeout=120)
print(response.status_code)
print(response.json())
```

批量请求逐个校验文件。部分文件被拒绝时，其余有效文件仍可提交。响应中的 `accepted_count` 与每项 `accepted` 用于核对文件级接收结果。

### HTTP 状态与任务状态

| 场景 | 表现 |
| --- | --- |
| 至少一个文件提交成功 | HTTP 202，每项包含 accepted 和 task_id |
| 全部文件超出大小上限 | HTTP 413 |
| 全部文件校验失败或请求参数错误 | HTTP 422，错误原因在响应中 |
| 接收或入队遇到存储故障 | 文件项包含 submission_failed，整批均失败时返回 HTTP 503 |
| 任务 ID 不存在 | HTTP 404 |
| 已接受任务的模型调用失败 | GET 返回任务信息，status 为 failed，并保存错误阶段 |

任务状态依次为 `queued → running → completed`，执行错误进入 `failed`。文件处理锁暂时被占用时会退避回队列，事件中记录 processing_busy。当前阶段包括 parsing、checking_cache、chunking、embedding 和 saving。

中断上传遗留的服务临时文件在下次启动时清理。已发布的 PDF、数据库记录和解析产物继续保留。

### 配置参数

在启动服务的 PowerShell 窗口设置环境变量，随后重新启动服务。

| 环境变量 | 默认值 | 用途 |
| --- | --- | --- |
| PAPERAGENT_SERVICE_ROOT | pdfprocess/data/upload_service | 原文件、SQLite 和单实例锁 |
| PAPERAGENT_INGEST_ROOT | pdfprocess/data/ingestion | 解析产物与入库缓存 |
| PAPERAGENT_CHROMA_DIR | pdfprocess/data/chroma_db | 向量库目录 |
| PAPERAGENT_UPLOAD_WORKERS | 1 | 任务工作线程数 |
| PAPERAGENT_PARSE_PROCESSES | 1 | PDF 解析进程数 |
| PAPERAGENT_UPLOAD_MAX_MB | 50 | 单文件 MiB 上限 |
| PAPERAGENT_SQLITE_JOURNAL_MODE | WAL | WAL 或 DELETE，便于数据库对照实验 |

原文件使用内容哈希作为内部文件名。原始名称保存在 SQLite，并用于新 Chunk 的来源字段。同一 PDF 改名后仍会识别为相同内容。

## 4. 指标应如何定义

上传确认、排队、处理、完成分别计时，可以定位瓶颈。

| 指标 | 定义 | 获取方式 |
| --- | --- | --- |
| 上传确认延迟 T_ack | 客户端发出 POST 到读完响应 | 基准报告 accept_latency_seconds |
| 上传负载吞吐 | 文件字节数除以上传请求阶段时长 | payload_mib_per_second，排除 HTTP 错误实验 |
| 排队时间 T_queue | 本次领取开始减本次排队时间 | 任务 queue_wait_seconds |
| 本次处理时间 T_process | 本次开始到完成或当前时刻 | processing_seconds |
| 服务端总时间 T_total | 任务创建到最终完成 | total_seconds，包含重启及退避等待 |
| 客户端完成时间 T_e2e | POST 开始到观察到 completed | client_end_to_end_seconds，包含轮询误差 |
| 完成吞吐 | 完成任务数除以实验总时长 | completed_tasks_per_second |
| 新入库页吞吐 | 未命中完成缓存的成功页数除以总时长 | new_pages_per_second |
| 接收成功率 | 接受文件数除以提交文件数 | file_acceptance_rate |
| 任务完成率 | completed 的唯一任务数除以唯一任务总数 | task_completion_rate |
| 采样内存峰值 | 服务及其子进程 RSS 的最大采样值 | sampled_peak_rss_bytes |
| CPU 用量 | 服务与子进程累计 CPU 秒的变化 | cpu_seconds_delta |
| 缓存与活跃去重 | 完成缓存命中任务数、复用活跃任务响应数 | cache_hit_tasks、reused_active_task_responses |
| 模型调用与 Token | 成功任务返回的 OCR/VLM 用量 | successful_task_model_usage |

每个任务的 result.timings 还包含 parse、split、embed、save。这些计时来自原有入库函数。服务 parsing 阶段还包含解析进程排队、首次启动和结果传输，统计边界需要在分析时注明。

P50 表示一半样本的延迟不超过该值，P95 表示 95% 样本不超过该值。基准脚本使用排序后的线性插值计算分位数，报告保留样本数。建议每个实验配置至少收集 30 个有效样本，稳定估计 P95 通常需要更多样本。

CPU 百分比可用 `CPU 秒变化 / 实验墙钟秒 × 100` 估算，一个 CPU 核心满载对应约 100%。进程重启或子进程异常退出会影响累计值。RSS 求和会重复计算部分共享内存，采样也可能遗漏短暂峰值，因此资源数据用于固定方法下的配置对比。

Embedding 默认每批 8 个 Chunk，新任务可用 `ceil(chunk_count / 8)` 估算应用层批次数。准确成本结合服务商用量核对，包含失败请求、SDK 重试和缓存差异。

## 5. 如何运行基准脚本

保持服务运行，在另一个 PowerShell 窗口执行。

```powershell
Set-Location 'C:\Users\Mark\Desktop\langgraph'
.\.venv\Scripts\python.exe -m upload_service.benchmark --url http://127.0.0.1:8000 --pdfs '.\articels\deeplearning.pdf' '.\articels\langchaimquestion.pdf' --scheme 1 --concurrency 2 --batch-size 1 --repeat 1 --timeout 1800 --output '.\upload_benchmark.json'
```

脚本会并发上传，轮询任务，采样 health，并输出 JSON。报告区分接受的文件数、唯一任务数、成功任务、失败任务、未确认完成任务和缓存命中。完成延迟统计使用成功任务，失败处理时间单列。

达到等待预算时，报告将尚未确认完成的任务列为 unfinished。已经提交的任务继续由服务处理，可通过 task_id 查询。

`--repeat` 会重复提交同一组文件，适合去重及缓存实验。新入库吞吐实验使用不同内容的 PDF，每个配置使用新的实验目录。

## 6. 可复现的对比实验

### 准备数据与实验记录

选取 20～50 篇论文，按页数、文件体积、文本层/扫描页、表格密度分层。保存文件 SHA256、页数和大小。单栏与双栏、中文与英文分别记录。

每组固定机器、电源模式、Python 与依赖版本、输入顺序、解析方案、切块参数、Embedding 模型、API 区域及超时。记录服务商限流和网络状态。各配置至少重复 3 轮，配置顺序可交替，降低网络时间变化带来的偏差。

建议使用以下实验记录列。

| run_id | 数据集哈希 | scheme | 工作线程 | 解析进程 | 客户端并发 | batch_size | 缓存状态 | ACK P95 | 完成 P95 | 新页/秒 | 峰值 RSS | 成功率 | 模型用量 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### 实验 A，理解后台处理的作用

先使用一个文本 PDF，比较 T_ack 与 T_e2e，并查看任务事件。

预期可以在处理尚未完成时拿到 task_id。后台执行主要缩短 HTTP 请求占用时间，入库总耗时由解析、进程启动和模型调用等阶段决定。

如果比较同步 HTTP 服务，复用同一入库函数和存储设置，在单独实验入口中等待该函数完成后返回。分别比较 ACK、完成时间和同时可服务的请求数量。

### 实验 B，任务线程数与解析进程数

固定客户端并发 4、batch_size=1，先固定解析进程为 1，比较任务线程 1、2、4。再固定任务线程为 4，比较解析进程 1、2、4。

```powershell
$env:PAPERAGENT_UPLOAD_WORKERS = '4'
$env:PAPERAGENT_PARSE_PROCESSES = '2'
```

重新启动服务后运行同一数据集。观察吞吐、排队时间、处理时间、CPU 和 RSS。Embedding 等网络等待较多时，增加任务线程可能提高流水线利用率。解析占比高时，解析进程数是主要实验变量。API 限流或内存压力可能改变结果。

### 实验 C，客户端并发与批量大小

固定服务配置，依次比较客户端 concurrency=1、2、4、8。每组使用相同文件数量和总字节量。

随后固定总文件数和客户端并发，比较 batch_size=1、5、10。观察 ACK P95、文件接收率、任务排队和 HTTP 请求数。

较大批量可以减少请求次数，一批中的上传确认会等待该批文件逐个校验和入队。任务开始时间与客户端收到整批响应的时间可以从事件中核对。

### 实验 D，WAL 与 DELETE

固定线程、进程和客户端并发，仅改变 PAPERAGENT_SQLITE_JOURNAL_MODE。每组使用独立数据库目录，保持 FULL 同步设置。

观察 ACK P95、状态查询时间、任务领取吞吐、存储错误和 CPU。单工作线程、低请求量时差异可能较小，可增加客户端请求与状态查询并发观察。

SQLite 与 PostgreSQL 的实验需要保持同样的表、索引、事务边界、持久化等级和任务领取语义，并记录数据库是否与服务同机。这样能区分网络往返与数据库本身的影响。

### 实验 E，冷缓存、热缓存与去重

冷入库使用新目录，避免上一轮的解析产物和向量记录影响结果。

```powershell
$env:PAPERAGENT_SERVICE_ROOT = 'C:\Users\Mark\Desktop\langgraph\pdfprocess\data\experiments\run01\service'
$env:PAPERAGENT_INGEST_ROOT = 'C:\Users\Mark\Desktop\langgraph\pdfprocess\data\experiments\run01\ingestion'
$env:PAPERAGENT_CHROMA_DIR = 'C:\Users\Mark\Desktop\langgraph\pdfprocess\data\experiments\run01\chroma'
```

完成第一轮后，在相同目录重复上传相同 PDF。检查 reused_file、result.skipped、模型调用及 Chroma 记录数量。

同时提交相同内容的 PDF，可以观察 reused_task。完成后重新上传会创建新的任务记录，入库层核对已有向量后可命中缓存。同一 PDF 改名仍然具有相同内容哈希。

去重实验的确定性检查包括并发活跃任务复用、重复上传后的向量增量为 0，以及完成缓存命中后跳过 Embedding。

### 实验 F，故障与恢复

使用独立实验目录，记录 task_id 后进行以下操作。

1. 提交文本 PDF，处理期间强制停止实验服务进程。
2. 保留文件、SQLite、解析产物与向量库，用相同目录重新启动。
3. 查询相同 task_id，核对 attempts、service_restart 事件及最终状态。
4. 核对向量 ID 集合、Chunk 数量和模型用量。

普通 Ctrl+C 会等待已领取任务结束，适合验证优雅关闭。强制停止用于验证异常中断恢复。解析子进程或文件锁仍在结束时，任务可能先退避回队列，等待现有处理释放。

记录恢复成功率、恢复到完成的时间、重复向量数和额外模型调用。SQLite、原文件和 Chroma 分属不同存储，处理采用可重试流程。检查点之前的外部调用在恢复时可能重做，实际账单需同时核对。

### 实验 G，解析和 RAG 质量

固定论文与标注，分别用 scheme 1、2、3 入库到各自 Collection。使用相同 Embedding、切块和检索设置，运行已有 `pdfprocess.compare_retrieval`。

关注正文证据完整率、表格单元格准确率、页码准确率、Recall@5/10、MRR 和每篇文档成本。上传服务侧关注接收与执行质量，知识助手侧关注证据召回和回答质量。

### 实验 H，框架和接收方式

框架对比可以先使用固定延迟的测试 Runner，隔离外部模型波动，并在记录中标注测试 Runner。随后用真实入库函数重复实验。

Flask 或 Django 对照保持相同的 multipart 格式、文件校验、fsync、SQLite 队列和返回语义，仅更换 HTTP 适配层。`runner_name` 保存在 health 与基准报告中。

分块复制与一次性 bytes 读取的对照使用相同文件和哈希算法，分别测服务 RSS、Python 分配峰值及接收时间。测试运行在独立进程，避免上一轮分配器缓存影响结果。Base64 对照额外记录编码时间、请求体大小和解码峰值。

## 7. 如何判断结果

```text
吞吐相对变化 = (新吞吐 - 基线吞吐) / 基线吞吐 × 100%
延迟下降比例 = (基线延迟 - 新延迟) / 基线延迟 × 100%
成功率变化（百分点） = (新成功率 - 基线成功率) × 100
缓存命中比例 = 命中完成缓存的任务数 / 成功任务数
恢复成功率 = 恢复后完成的中断任务数 / 中断任务总数
```

先检查样本与成功率，再分析延迟。大量任务快速失败会影响延迟分布，因此成功和失败任务需要分开报告。新入库与缓存命中也分别报告。

同时观察吞吐、P95、RSS 和模型费用。例如，吞吐增长伴随显著排队或内存增长时，可以按目标机器与响应要求选择参数。论文实际召回改善使用你后续提供的问题与证据标注验证。

## 8. 本次验证记录

自动化测试覆盖批量上传、文件校验、并发活跃任务去重、任务持久化、重启恢复、处理锁退避、解析池重建、真实子进程解析、原始文件名及真实 Chroma 保存。原有入库、检索和多轮问答回归测试一并运行。

真实 HTTP 联调使用两份合成文本 PDF，选择 scheme 1，完成了实际 DashScope Embedding 和 Chroma 保存。两份文件均接收并完成，重复上传命中了入库缓存。测试数据位于隔离目录。

本次两次上传确认耗时平均约 0.125 秒。首次任务处理约 4.83 秒，后续任务约 0.34 秒，体现了首次进程启动、导入及排队的影响。采样父子进程 RSS 求和峰值约 421 MiB。这些小样本用于链路验证，稳定性能结论需要按前述协议采集更多样本。

方案 2、3 的本地 OCR 质量、CPU/RAM、延迟和方案 3 的 VLM 费用需要结合论文样本评测。

## 9. 文件职责与代码阅读顺序

| 文件 | 职责 |
| --- | --- |
| upload_service/app.py | HTTP 路由、应用生命周期和后台资源协调 |
| upload_service/storage.py | 分块复制、哈希、PDF 校验及发布文件 |
| upload_service/database.py | SQLite 表、事务、任务领取、恢复和事件查询 |
| upload_service/worker.py | 工作线程、解析进程池和原有入库函数适配 |
| upload_service/config.py | 环境变量与默认配置 |
| upload_service/benchmark.py | HTTP 并发实验、状态轮询及指标汇总 |
| pdfprocess/ingest_common.py | 增加阶段回调、原始来源名、解析执行器和文件标识核对 |
| tests/test_upload_service.py | 上传服务的正确性及集成测试 |
| requirements-upload.txt | 本次新增服务依赖 |

建议先阅读 app.py 的 POST 路由，再沿 storage.save、store.enqueue、store.claim、run_pipeline 和 store.finish 跟踪一次任务。最后结合 GET 任务返回的 events 理解状态变化。

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

当前服务绑定本机地址，默认单实例运行。原文件与业务队列保存在本地磁盘，关闭服务后仍可保留。命令行中的 Ctrl+C 会触发优雅关闭，等待正在执行的任务结束。
