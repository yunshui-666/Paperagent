# PaperAgent PDF 入库方案对比

项目目录为 `C:\Users\Mark\Desktop\langgraph`。三个方案使用相同切块参数、共享 Embedding 实现及相同的结果结构。

## 三个方案

| 编号 | 解析方法 | 默认 Collection | 适合观察的指标 |
| --- | --- | --- | --- |
| 1 | PyMuPDF 本地导出页级 Markdown | paper_chunks1 | 正文完整度、阅读顺序、解析耗时 |
| 2 | 每页使用本机 RapidOCR 识别文字 | paper_chunks2 | 扫描件、图表质量、模型调用数 |
| 3 | 本地 OCR 与通义 VLM 并行，按版面区域融合文字和图表 | paper_chunks3 | 召回质量与模型调用成本的平衡 |

方案 3 每个非空页执行本地 OCR 与 VLM 双路解析。文字字符数与乱码比例用于质量审计。无文字、图片或绘图对象的空白页跳过两路处理。

OCR 返回像素文字框，VLM 默认返回 0～1000 的归一化版面框，融合前转换成像素。保留 VLM 阅读顺序，每条 OCR 文字最多分配到一个区域，未匹配文字保留在页尾供检查。

已有完整 Markdown 表格直接复用。图片说明及 VLM 转写在 Markdown 中带有模型生成标识，页级 JSON 保存真实裁剪图路径与来源。

## 文件职责

| 相对项目目录的文件 | 职责 |
| --- | --- |
| pdfprocess/pdf_pipeline1.py | 方案 1 的五个公开函数 |
| pdfprocess/pdf_pipeline2.py | 方案 2 的五个公开函数 |
| pdfprocess/pdf_pipeline3.py | 方案 3 的五个公开函数 |
| pdfprocess/pdf_vision.py | 文本层提取、OCR/VLM 双路调度与页级产物 |
| pdfprocess/ingest_config.py | 三方案共用配置和环境读取；ingest_common 继续兼容导出 PipelineConfig |
| pdfprocess/pipelines.py | 网页与 CLI 共用方案分发 |
| pdfprocess/ingest_common.py | 缓存、去重、任务锁、入库编排与批量处理 |
| pdfprocess/embeddings.py | 入库和问答共用 TongyiEmbeddings |
| pdfprocess/compare_ingest.py | 同一批文件的命令行对比与 JSON 报告 |
| pdfprocess/paper_chunk.py | 原有标题和长度切分，可传入切块参数 |
| pdfprocess/vector_store.py | 向量校验、分批 upsert、数据库绝对路径 |
| pdfprocess/rag.py | 复用共享 Embedding，并支持指定 Collection |
| tests/test_pdf_ingestion.py | 三方案、批量上传、重试和真实 Chroma 集成测试 |

## 函数命名

| 用途 | 方案 1 | 方案 2 | 方案 3 |
| --- | --- | --- | --- |
| 单个 PDF 解析 | pdf_to_markdown1 | pdf_to_markdown2 | pdf_to_markdown3 |
| 单文件完整入库 | ingest_pdf1 | ingest_pdf2 | ingest_pdf3 |
| 批量本地文件入库 | ingest_pdfs1 | ingest_pdfs2 | ingest_pdfs3 |
| 单文件上传并入库 | upload_pdf1 | upload_pdf2 | upload_pdf3 |
| 批量上传并入库 | batch_upload_pdfs1 | batch_upload_pdfs2 | batch_upload_pdfs3 |

完整入库函数支持相同的关键字参数 `config`、`embedding_model`、`collection`、`ocr_client`、`vlm_client`。后四个参数用于依赖注入，默认创建项目配置对应的服务。

## 环境

Python 3.14 已验证本地解析与离线集成。新增依赖锁定为 PyMuPDF 2.1，使用原生 `Document.to_markdown()` 接口。

```powershell
Set-Location 'C:\Users\Mark\Desktop\langgraph'
.\.venv\Scripts\python.exe -m pip install -r requirements-ingest.txt
```

在项目现有 `.env` 中配置所需服务。以下值为填写示例。

```dotenv
DASHSCOPE_API_KEY=填写你的密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
PAPERAGENT_OCR_THREADS=2
```

方案 1、2 的纯解析使用本地能力，方案 3 调用通义 VLM。三套完整入库均调用现有 DashScope Embedding。

本地 OCR 模型标识为 `rapidocr-ppocrv6-small`，使用安装包自带的 ONNX 权重。VLM 默认 `qwen3-vl-flash`，可使用 `--vlm-model` 指定。

## 先比较解析

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.compare_ingest --pdfs '.\articels\deeplearning.pdf' '.\articels\langchaimquestion.pdf' --schemes 1 2 3 --parse-only --refresh --output '.\pdfprocess\data\parse-comparison.json'
```

`--parse-only` 执行解析，保存 Markdown、页级 JSON 及解析指标。`--refresh` 重新执行解析，适合测量实际处理耗时和逻辑模型请求数。缓存运行的耗时与首次运行需分别记录。

## 比较完整入库

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.compare_ingest --pdfs '.\articels\deeplearning.pdf' '.\articels\langchaimquestion.pdf' --schemes 1 2 3 --refresh --output '.\pdfprocess\data\ingest-comparison.json'
```

默认批量任务串行执行。每个文件记录独立的成功或失败结果，结束时生成 JSON 报告。任一文件失败时命令行退出码为 1，全部成功时为 0。

## 批量上传函数

以下 Python 示例可放入项目内的调用脚本，从项目目录运行。

```python
from pathlib import Path
from pdfprocess.pdf_pipeline3 import batch_upload_pdfs3

paths = sorted(Path('articels').glob('*.pdf'))
files = ((path.name, path.read_bytes()) for path in paths)
results = batch_upload_pdfs3(files)

for result in results:
    print(result['source'], result['status'])
```

上传函数接收 `(原文件名, bytes)`，可对接前端上传组件。原 PDF 自动保存到按内容哈希组织的上传目录。相同文件名、不同内容的文件使用不同目录。

批量本地文件可直接调用 `ingest_pdfs3(paths)`。方案 1、2 使用对应编号的模块和函数。

## 调整相同参数进行实验

```python
from pathlib import Path
from pdfprocess.ingest_common import PipelineConfig
from pdfprocess.pdf_pipeline3 import ingest_pdfs3

config = PipelineConfig(
    chunk_size=800,
    chunk_overlap=120,
    min_text_chars=80,
    max_bad_char_ratio=0.05,
    use_cache=False,
    collection_name='paper_chunks3_exp_a',
)
results = ingest_pdfs3(sorted(Path('articels').glob('*.pdf')), config=config)
```

同一 Collection 中，同一 PDF、同一方案在重新入库成功后保留最新配置对应的 Chunk。并排比较多个参数组合时，分别指定实验 Collection。

Embedding 默认使用 `tongyi-embedding-vision-flash`，每批 8 条。自定义 Embedding 时需要同步指定 `embedding_model_id`，查询端使用同一模型。模型或维度变化时采用新的 Collection。

## 用现有 RAG 检索各方案

```python
from pdfprocess.rag import ask, build_graph

graph = build_graph(collection_name='paper_chunks3')
first = ask(graph, '论文提出了什么方法？', 'scheme3-study')
second = ask(graph, '该方法的实验结果如何？', 'scheme3-study')
print(second['answer'])
```

每个方案使用独立 Graph 和会话 ID，固定同一套问题与模型参数，分别记录 Recall@5、答案正确率和引用页码准确率。

## 缓存、去重与恢复

- PDF 内容哈希作为论文标识，同一文件改名后仍能识别。
- 处理配置与解析依赖版本参与任务指纹。页级成功结果可复用，失败页在下次运行时重试。
- 解析失败的页面及错误保存在页级 JSON。该次完整入库在解析阶段停止。
- 已完成任务会核对解析内容摘要和 Chunk ID。重复提交命中后跳过 Embedding 与写入。
- 所有向量校验通过后执行分批 upsert。部分写入失败后可按确定的 ID 重试。
- 文档完成状态在核对向量记录后写入。向量库写入与完成标记分属不同存储，崩溃后通过记录核对恢复。
- Chroma 分批写入期间可能存在部分记录。对比检索在对应入库任务完成后执行。

## 量化指标

| 指标 | 记录方式 |
| --- | --- |
| 解析耗时 | parse_seconds 或 timings.parse，缓存运行另列 |
| 完整入库耗时 | timings.total，以及 split/embed/save 分项 |
| OCR/VLM 请求数 | metrics.ocr_calls、metrics.vlm_calls |
| 视觉模型 Token | metrics.input_tokens、metrics.output_tokens |
| 缓存页数 | metrics.cached_pages |
| 文档成功率 | completed 文件数除以总文件数 |
| 页级解析情况 | pages 中的 status、method、error、warnings |
| 检索召回 | 固定问题集，人工标注证据，计算 Recall@5 |
| 重复入库 | skipped、向量总数和 Embedding 调用情况 |

请求数统计每次应用层 SDK 调用，SDK 内部重试由服务端账单核对。Token 指标覆盖 VLM 的服务响应，本地 OCR 记录识别次数与耗时，Embedding 成本需要结合输入量和服务端用量核对。

建议使用 10～20 篇单栏、双栏、扫描、表格与公式论文。人工标注关键段落、表格单元格和证据页码，比较正文完整率、表格准确率及 Recall@5。

## 测试与验证范围

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

测试覆盖真实 PDF 本地解析、真实 Chroma 写入/查询、三套 Collection 隔离、RAG 接入、批量上传、去重、坐标、输出截断及失败重试。本地 OCR 另有真实识别测试，VLM 与 Embedding 在集成测试中使用替身，真实服务质量由上述样本实验验证。

## 本机本地方案冒烟记录

对项目现有三份 PDF 使用方案 1 纯解析、刷新缓存执行过一次批量测试。

| 文件 | PDF 页数 | 结果 | 解析耗时 |
| --- | --- | --- | --- |
| deeplearning.pdf | 15 | 全页成功 | 4.15 秒 |
| langchaimquestion.pdf | 375 | 全页成功 | 84.70 秒 |
| zhishenshinei.pdf | 297 | 295 页成功，前 2 页需要 OCR | 60.26 秒 |

第三份 PDF 的前两页均只有图片且没有可提取文本，方案 1 将其标记为失败页。这两页可用于验证方案 2、3 的 OCR 路径。上述记录仅覆盖本地解析，本机负载和启动条件会影响耗时。

## 技术参考

PyMuPDF 2.x 将 Markdown 导出放入 Document 接口，当前实现使用这一接口。[官方 API 文档](https://pymupdf.readthedocs.io/en/latest/document.html)

PyMuPDF 采用 AGPL 或商业许可。[许可说明](https://pymupdf.readthedocs.io/en/latest/about.html)

本地 OCR 使用 RapidOCR 的 ONNX 模型与 CPU 推理，VLM 使用 JSON 版面框。[RapidOCR](https://github.com/RapidAI/RapidOCR)
