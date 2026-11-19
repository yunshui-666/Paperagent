# PaperAgent 检索改进与召回评测

项目位于 `C:\Users\Mark\Desktop\langgraph`。本次完成检索链路与评测程序，论文问题及证据标注由你后续提供。真实召回率与提升幅度将在标注到位后计算。

## 默认检索链路

```text
问题
├─ 向量相似度召回 Top20
└─ 中文分词与 BM25 关键词召回 Top20
         ↓
按 Chroma 实际记录 ID 去重，RRF 融合
         ↓
Cross-Encoder 联合编码问题和各候选文本并重排
         ↓
取前 10 条，交给原有生成节点
```

同一问题最多产生 40 个不同候选，实际数量取决于两路重叠程度和知识库规模。

| mode | 执行方式 |
| --- | --- |
| dense | 向量相似度召回 |
| bm25 | 关键词召回 |
| hybrid | 向量与关键词召回，通过 RRF 融合 |
| hybrid_rerank | 混合召回后使用 Cross-Encoder 重排，作为默认模式 |

默认 `dense_k=20`、`keyword_k=20`、`top_k=10`。原始 Top5 向量基线可显式配置为 `mode='dense', dense_k=5, top_k=5`。

## 技术选择

BM25 使用 jieba 中文分词并保留英文词、数字及术语，采用正值 IDF 的 BM25 公式。关键词索引在每次检索时核对知识库正文和元数据指纹，检测到新增、同数量更新或删除时重建，适合当前本地小型知识库。

RRF 根据两路结果的排名融合，常数默认 60。向量距离、BM25 分数和 Cross-Encoder 分数分别保留。

Cross-Encoder 采用 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` 多语言模型，运行官方 ONNX INT8 版本，权重约 118.6 MB，使用现有 ONNX Runtime 的 CPU 后端。模型版本固定为 `1427fd652930e4ba29e8149678df786c240d8825`。模型将问题与候选文本组成一对后联合编码，输出相关性分数。[官方模型与 ONNX 文件](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)

模型输入上限为 512 Token，保留问题并截断超长候选正文。重排质量需要结合你的论文和标注评估。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| pdfprocess/retrieval.py | TopK 配置、BM25、RRF、去重与统一检索接口 |
| pdfprocess/cross_encoder.py | 本地模型加载、成对编码、分批 ONNX 推理 |
| pdfprocess/rag.py | 将新检索接入原有多轮 LangGraph 工作流 |
| pdfprocess/retrieval_cli.py | 规范检索 CLI；retreiver.py 保留为兼容入口 |
| pdfprocess/compare_retrieval.py | 知识库导出、标注校验、召回及提升率评测 |
| retrieval_eval_template.json | 后续论文问题与证据的填写模板 |
| requirements-retrieval.txt | 本次检索依赖 |
| tests/test_retrieval.py | 检索、指标、真实本地重排模型及 Chroma 集成测试 |

原有两组测试显式指定 Top5 向量基线，以继续验证原 RAG 和入库行为。新增测试覆盖默认增强链路。

## 运行

```powershell
Set-Location 'C:\Users\Mark\Desktop\langgraph'
.\.venv\Scripts\python.exe -m pip install -r requirements-retrieval.txt
```

本地模型目录为 `pdfprocess\data\models\mmarco-minilm`。所需文件缺失时自动按固定版本下载，也可用 `--model-dir` 指定同一模型的本地目录。

### 原始 Top5 基线

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.retrieval_cli --mode dense --dense-k 5 --top-k 5 --query '填写论文问题'
```

### 增大向量召回数量

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.retrieval_cli --mode dense --dense-k 20 --top-k 20 --query '填写论文问题'
```

### 混合召回与重排

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.retrieval_cli --mode hybrid --query '填写论文问题'
.\.venv\Scripts\python.exe -m pdfprocess.retrieval_cli --mode hybrid_rerank --query '填写论文问题'
```

三套入库方案对应的知识库可通过 `--collection paper_chunks1`、`paper_chunks2` 或 `paper_chunks3` 选择。当前原知识库为 `paper_chunks`。

### 多轮问答

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.rag --mode hybrid_rerank --dense-k 20 --keyword-k 20 --top-k 10
```

```python
from pdfprocess.rag import ask, build_graph
from pdfprocess.retrieval import RetrievalConfig

graph = build_graph(
    collection_name='paper_chunks',
    retrieval_config=RetrievalConfig(
        mode='hybrid_rerank', dense_k=20, keyword_k=20, top_k=10
    ),
)
result = ask(graph, '填写论文问题', 'paper-study')
print(result['answer'])
```

结果来源中的 `distance` 为向量距离，越小越近。`bm25_score`、`rrf_score` 和 `cross_encoder_score` 越大排名越靠前。关键词独有候选的 `distance` 为 null。Cross-Encoder 分数采用原始 logits。

## 根据论文准备标注

先完成论文入库，再导出所选 Collection 的实际证据记录。

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.compare_retrieval --collection paper_chunks --export-corpus '.\corpus_for_labels.json'
```

导出内容包含实际记录 ID、正文、来源和可用的页码。该步骤使用本地知识库读取。

填写 `retrieval_eval_template.json`，每个问题提供以下字段。

```json
{
  "dataset_name": "论文名称与标注版本",
  "annotation_status": "reviewed",
  "collection_name": "paper_chunks",
  "questions": [
    {
      "id": "q001",
      "question": "根据论文填写问题",
      "relevant_ids": ["导出文件中的实际证据ID"]
    }
  ]
}
```

一个问题可以标注多条相关证据。可将导出文件中的 `corpus_fingerprint` 复制到评测集顶层，核对知识库版本。评测前会检查问题 ID、证据 ID 和 Collection，评测结束时再次检查知识库指纹。

## 运行召回对比

```powershell
.\.venv\Scripts\python.exe -m pdfprocess.compare_retrieval --collection paper_chunks --dataset '.\retrieval_eval_template.json' --dense-k 20 --keyword-k 20 --at-k 5 10 20 --output '.\retrieval_report.json'
```

每个问题只执行一次向量 Top20 检索，各方案复用该结果。报告包括以下内容。

- 向量 Top5、Top10、Top20 的召回变化。
- dense、bm25、hybrid、hybrid_rerank 在相同 K 下的 Recall、Hit Rate、MRR。
- 混合与重排相对向量检索的绝对百分点变化和相对提升率。
- 各候选集合的召回率、数量和每个问题的实际排名。
- 分词建索引时间、模型预热时间及检索阶段耗时。

TopK 对比基于同一次向量 Top20 的前缀。模型预热与建索引时间单列。各模式的延迟由共享阶段的耗时相加得到，可用于观察阶段成本。

## 指标定义

```text
Recall@K = TopK 中命中的相关证据数 / 已标注相关证据总数
总体 Recall@K = 所有问题的 Recall@K 的算术平均

Hit Rate@K = TopK 至少命中一条证据的问题比例
MRR@K = 首条相关证据排名倒数的平均值

绝对变化（百分点） = (改进 Recall - 基线 Recall) × 100
相对提升率（%） = (改进 Recall - 基线 Recall) / 基线 Recall × 100
```

基线 Recall 为 0 时，相对提升率记录为 null，绝对百分点变化仍会报告。指标下降时保留负值。

扩大 TopK 的效果与相同 K 下的排序效果分别报告。Cross-Encoder 对已召回候选重排，候选集合保持一致，因此重排前后的候选召回率相同，固定 K 下的 Recall 可以变化。

## 验证范围

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -v
```

已验证本地多语言 Cross-Encoder 的真实推理，以及合成数据上的真实 Chroma、BM25、重排和 LangGraph 联动。单元测试使用合成标注校验指标。你的论文召回率与提升率等待后续提供的实际标注后测量。
