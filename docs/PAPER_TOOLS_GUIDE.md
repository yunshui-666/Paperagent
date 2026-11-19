# 校园论文工具、双路解析与自动入库

## 1. 这次改了什么

论文工具仅使用知网与万方。万方通过官网 gRPC-Web 原生搜索元数据；知网通过网页索引定位官方论文详情页。下载成功的 PDF 复用现有方案三入库，不再实现一套平行的切块、Embedding 或数据库代码。

```text
用户：搜索万方关于知识图谱的论文
  → prepare_history → paper_agent
  → 模型 search_papers(query=..., source=wanfang)
  → 注册表 → CampusPaperClient → 元数据与来源错误
  → ToolMessage → 模型整理论文标题与官方链接 → END

用户：下载第一篇论文并入库
  → paper_catalog 中的官方详情 URL
  → 模型 download_paper(paper_id=URL)
  → 官方详情/全文入口 → 有效 PDF
  → 原 FileStorage → 原 SQLite 任务队列
  → 方案三解析 → Chunk → Embedding → Chroma
  → 工具返回真实任务状态 → 模型回答 → END
```

`paper_agent` 内部是有界模型工具循环，每轮最多六次调用，不是新增六个图节点。普通学术问答仍走原有 RAG 分支。

## 2. 来源适配为何这样选择

| 方案 | 优点 | 代价和限制 |
| --- | --- | --- |
| 万方官网协议（当前） | 中文论文原生元数据，能区分期刊/学位/会议 | 不是稳定的第三方开发者 API，字段和接口可能变化，需维护适配与回归 |
| 知网官方页面索引（当前） | 不依赖未验证的原生检索接口，只保留官方详情页 | 受搜索引擎覆盖、排序和时效性影响，不能保证找全知网论文 |
| 学校授权的官方机构 API | 接口和授权更明确，适合稳定服务 | 需确认学校与供应商是否提供服务、费用、限额和全文范围 |
| 浏览器自动化 | 可通过页面完成交互 | 验证码需人工处理、页面变化敏感、浏览器权限不能自动转移到后台服务 |
| 手动下载 PDF 后上传 | 与现有上传能力直接衔接，适合少量验证 | 需要人工取得文件，自动化程度较低 |

用户指定不使用 arXiv，因此生产论文工具没有该来源。历史会话记录继续保留；旧目录中不支持的论文 ID 不会触发旧来源请求。既有入库 PDF 和向量不因此删除。

## 3. 真实状态与错误

| 结果 | 含义 | 界面应该如何理解 |
| --- | --- | --- |
| 论文目录 | 有标题、摘要与官方 URL | 搜到了元数据，还没有全文 |
| queued/running | PDF 已取得并提交队列 | 入库仍在后台进行 |
| completed 且 ingested=true | 任务校验与向量保存完成 | 可以查询对应知识库 |
| failed | 下载、解析或入库失败 | 检查 error，不应声称可用 |

来源部分失败时保留其他来源结果，同时在最终答案明确附上来源错误。下载工具最多等待 60 秒，未完成时返回任务 ID 和 pending 状态。验证页、401/403、非 PDF、超大响应和不支持的跳转都会阻止提交；之后的 PDF 结构校验仍由原 FileStorage 完成。只提供 CAJ 的页面不支持自动入库。

**2026-11-19 现场结果：** 万方原生搜索已成功返回“知识图谱可视化综述”等记录；本机后端机构身份检查没有机构，全文入口返回 401。知网请求曾跳到安全验证；用户报告浏览器验证完成，但独立服务会话尚未证明可下载全文。因此当前不能声称知网/万方真实 PDF 下载端到端成功。浏览器、Python 和 Docker 的授权环境分别验证，项目不读取浏览器 Cookie 或绕过验证码。

## 4. 文件职责与阅读顺序

| 文件 | 职责和建议断点 |
| --- | --- |
| `pdfprocess/tools/registry.py` | 统一工具注册、参数校验、执行结果及错误包装；断点 invoke |
| `pdfprocess/tools/defaults.py` | 登记默认 web_search 与 search_papers |
| `pdfprocess/tools/wanfang_protocol.py` | 用标准 protobuf 编解码最小消息，验证 gRPC-Web 帧和服务状态 |
| `pdfprocess/tools/campus_papers.py` | 来源 URL 校验、原生/索引搜索、同会话详情与受限流式下载 |
| `pdfprocess/tools/papers.py` | 模型可调用的工具 schema，绑定当前会话知识库、提交原任务、返回状态 |
| `pdfprocess/tools/paper_agent.py` | 操作意图、目录指代、模型工具协议、次数与输入预算、最终状态汇报 |
| `upload_service/chat.py` | 给当前会话创建工具目录副本并绑定下载目标库；共享 Checkpointer |
| `pdfprocess/context_compression.py` | 历史摘要、近期消息和证据预算，摘要提示保留官方论文 URL/DOI |
| `upload_service/ingestion.py` | 原上传与论文下载共同的去重、配置快照与任务入队 |
| `upload_service/worker.py` | 领取任务并运行既有解析/入库流程 |

论文工具的 `paper_id` 是官方详情 URL；解析产物元数据里原有的 `paper_id` 是 PDF 内容 SHA256，两者属于不同层。工具结果的 `document_id` 才是下载后原件的内容身份。

## 5. 本地 OCR + 通义 VLM 双路解析

方案 1 是 PyMuPDF Markdown 文本层提取；方案 2 是纯本地 RapidOCR；方案 3 为默认双路：

1. 主线程用 PyMuPDF 把同一页渲染一次，两个分支读取同一份 PNG。
2. RapidOCR 的 PP-OCRv6 small 权重在本地 CPU 上返回文字、坐标和置信度。
3. 通义视觉接口返回区域类型、阅读顺序、表格、公式和图片描述。
4. 统一坐标后做区域匹配与融合，保存原 OCR、原 VLM 和融合 Markdown，方便核对错误来源。
5. 标题与字符切块携带页码和来源，再调用 Embedding 并按稳定 Chunk ID 写入 Chroma。

相关实现是 `local_ocr.py`、`pdf_vision.py`、`pdf_fusion.py`、`pdf_pipeline1/2/3.py` 与 `ingest_common.py`。两个分支不会在工作线程访问 PyMuPDF 页面对象。OCR 本地运行不意味着整个管线离线：方案三会把页图发给通义，Embedding 也使用远端接口。

RapidOCR 体积与 CPU 部署适合基础项目。PaddleOCR/版面模型可做结构质量对照，Tesseract 可做纯本地轻量对照，其他专业文档解析器可比较表格/公式质量，但本次不再增加额外解析后端。

## 6. 长对话如何压缩

完整 messages 仍在 Checkpointer 中，模型只读“旧历史增量摘要 + 近期完整轮次 + 当前问题 + 本轮证据”。摘要用覆盖长度及前缀哈希校验，防止历史变化后沿用过期摘要。证据用句段抽取和预算分配，保留引用映射。论文工具结果压缩时保留官方 ID、标题、任务状态和错误，并保持 assistant/tool 的关联协议。

预算统一采用保守 UTF-8 字节估算，真实 tokens 以模型 usage 为准，不能把估算值当实际 Token 用量。见 [上下文压缩指南](CONTEXT_COMPRESSION_GUIDE.md)。

## 7. 如何测试与量化

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_campus_sources.py' -v
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_paper*.py' -v
```

自动测试覆盖协议字段、部分来源失败、官方 URL 边界、流式大小限制、权限/验证码不入队、指定库入队、真实待处理状态、目录指代和工具消息预算。服务集成测试使用真实 FastAPI、SQLite、Chroma 及队列，模型 SDK 使用 MockTransport，OCR/VLM/Embedding 可注入替身，避免测试产生云费用。这些不能替代真实站点授权验收。

手动实验请保存日期、主机网络、是否走代理、来源、查询、结果 URL、HTTP 状态、任务 ID 和知识库：

| 目标 | 实验方式与指标 |
| --- | --- |
| 论文搜索相关性 | 20～50 个固定中文主题，人工标注前 K 篇相关性，计算 Precision@K/成功率与延迟；只有完整标准论文集合才能计算真正 Recall |
| 全文下载可用性 | 选择已确认学校有 PDF 授权的论文，分开记录成功、验证码、无授权、仅 CAJ 和站点错误；成功数/可授权测试数 |
| 自动入库正确性 | 核对文件页数、内容哈希、任务状态、Chunk 数量、目标库及重复下载的去重行为 |
| 双路解析收益 | 同一批文本/扫描/双栏/表格论文运行三方案，人工统计字符错误率、阅读顺序、表格单元格和公式正确率；记录秒/页和 VLM 成本 |
| 长对话保真 | 6～20 轮固定指代场景，逐轮核对主题、条件、论文 URL 和必要事实；比较模型 usage、总时延和事实保留率 |
| 最终知识库效果 | 标注真实正确 Chunk，在固定语料上比较 Recall@K/MRR；单独看候选召回与最终排序 |

不把搜索返回条数当召回率，不把 HTTP 200 当全文成功，不把队列接收当入库完成。校园网部署细节见 [部署指南](DEPLOYMENT_GUIDE.md)。
