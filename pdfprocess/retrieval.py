"""向量召回、中文 BM25、RRF 融合与 Cross-Encoder 重排。"""

# 检索算法层。统一维护候选 ID 与各阶段分数，提供 Dense、BM25、RRF 和重排组合。

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
from threading import RLock

from langchain_core.documents import Document

from .cross_encoder import OnnxCrossEncoder


MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
STOP_WORDS = frozenset("的 了 和 是 在 与 及 一个 什么 如何 为什么 请 我 你 这 那 以及 怎样 是否 the a an and or is are to of in".split())


@dataclass(frozen=True)
class RetrievalConfig:
    """召回候选数与最终 TopK 分开设置，方便逐项做检索对比实验。"""
    mode: str = "hybrid_rerank"
    dense_k: int = 20
    keyword_k: int = 20
    top_k: int = 10
    rrf_constant: int = 60
    reranker_model_dir: Path | None = None
    reranker_batch_size: int = 8

    def __post_init__(self):
        # 参数在创建检索器前验证，避免运行到模型调用中途才报错。
        """检查模式、正整数候选数量与最终 TopK 的理论上限。"""
        if self.mode not in MODES:
            raise ValueError(f"mode 需要为 {MODES} 中的一项")
        values = (self.dense_k, self.keyword_k, self.top_k, self.rrf_constant, self.reranker_batch_size)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in values):
            raise ValueError("TopK、RRF 常数和 batch_size 需要为正整数")
        available = self.dense_k
        # 混合检索的两路候选可能重叠，这里校验的是候选数理论上限。
        if self.mode == "bm25":
            available = self.keyword_k
        if self.mode in {"hybrid", "hybrid_rerank"}:
            available = self.dense_k + self.keyword_k
        if self.top_k > available:
            raise ValueError("最终 top_k 不能超过配置的候选数量")


@dataclass
class RetrievalHit:
    """一条候选及各阶段分数。原始分数均保留，便于定位排序变化。"""
    document: Document
    chunk_id: str
    distance: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    cross_encoder_score: float | None = None

    @property
    def rerank_relevance(self):
        """将原始 logit 映射到 0～1，供回答来源路由使用。"""
        if self.cross_encoder_score is None:
            return None
        value = float(self.cross_encoder_score)
        if not math.isfinite(value):
            raise ValueError("Cross-Encoder 分数需要为有限数值")
        if value >= 0:
            # 正数分支使用 exp(-value)，避免很大的正 logit 导致 exp 溢出。
            return 1 / (1 + math.exp(-value))
        # 负数分支代数等价，exp(value) 始终落在 0～1。
        # 此 sigmoid 值尚需标注集校准，当前按用户指定阈值进行联网路由。
        exponent = math.exp(value)
        return exponent / (1 + exponent)


def document_id(document):
    """统一候选 ID，优先数据库 ID，再兼容旧 Chunk 元数据和无 ID 文档。"""
    if document.id:
        return str(document.id)
    if document.metadata.get("chunk_id"):
        return str(document.metadata["chunk_id"])
    return hashlib.sha256(
        (str(document.metadata.get("source", "")) + "\n" + document.page_content).encode("utf-8")
    ).hexdigest()


def tokenize(text):
    """中文走 jieba 分词，英文/数字保留词内连字符与小数点。"""
    import jieba
    terms = []
    for part in re.findall(r"[\u3400-\u9fff]+|[a-z0-9_]+(?:[.-][a-z0-9_]+)*", text.lower()):
        # 中英文分别处理，避免英文术语被拆成单个字母。
        if re.fullmatch(r"[\u3400-\u9fff]+", part):
            terms.extend(jieba.lcut(part))
        else:
            terms.append(part)
    return [term for term in terms if term.strip() and term not in STOP_WORDS]


def read_corpus(vector_store):
    """读取当前检索范围内的文本，供关键词索引建库。"""
    raw = vector_store.get(include=["documents", "metadatas"])
    ids = raw["ids"]
    texts = raw["documents"]
    metadatas = raw["metadatas"]
    if len(ids) != len(texts) or len(ids) != len(metadatas) or len(ids) != len(set(ids)):
        # 防止 zip 静默截断，以及重复 ID 造成同一证据被多次计分。
        raise ValueError("知识库记录 ID、正文与元数据不一致")
    return [Document(id=str(chunk_id), page_content=text, metadata=metadata or {})
            for chunk_id, text, metadata in sorted(zip(ids, texts, metadatas))
            if isinstance(text, str) and text.strip()]


def corpus_fingerprint(documents):
    # 正文或元数据改变都会触发刷新，避免只比较文档数量而漏掉更新。
    """将稳定 ID、正文和元数据一起散列，用于检测关键词索引和评测语料的变化。"""
    data = [(document_id(doc), doc.page_content, doc.metadata) for doc in documents]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


class BM25Index:
    """采用正值 IDF 的 BM25，中文分词与英文单词共同建索引。"""
    def __init__(self, loader, *, auto_refresh=True, tokenizer=tokenize, k1=1.5, b=0.75):
        # loader 延迟读取语料。k1 控制词频饱和速度，b 控制长短文档归一化。
        """保存语料加载器、分词器与 BM25 参数，倒排表在首次刷新时构建。"""
        self.loader = loader
        self.auto_refresh = auto_refresh
        self.tokenizer = tokenizer
        self.k1 = k1
        self.b = b
        self.documents = []
        self.fingerprint = None
        self.postings = {}
        # postings 为倒排表，词项 → [(文档下标, 该词在文档中的次数), ...]。
        self.lengths = []
        self.average_length = 0
        self._lock = RLock()

    def refresh(self):
        """读取当前可见语料，指纹变化时重建倒排表、文档长度和平均长度。"""
        with self._lock:
            # 先在局部变量完成新索引构建，构建成功后再替换共享状态。
            documents = self.loader()
            fingerprint = corpus_fingerprint(documents)
            if fingerprint == self.fingerprint:
                # 语料没变就复用已有倒排表，避免每次问题都重新分词全文。
                return
            postings = defaultdict(list)
            lengths = []
            for index, document in enumerate(documents):
                tokens = self.tokenizer(document.page_content)
                lengths.append(len(tokens))
                for term, count in Counter(tokens).items():
                    # Counter 将同一文档中的词频合并成一条倒排记录。
                    postings[term].append((index, count))
            self.documents = documents
            self.fingerprint = fingerprint
            self.postings = dict(postings)
            self.lengths = lengths
            self.average_length = sum(lengths) / max(1, len(lengths))

    def search(self, question, k):
        """根据问题词项累计 BM25 得分，按分数和稳定 ID 返回前 K 个候选。"""
        with self._lock:
            # RLock 支持此处在持锁状态下再次调用持锁的 refresh()。
            if self.auto_refresh or self.fingerprint is None:
                self.refresh()
            if not self.average_length:
                return []
            scores = defaultdict(float)
            for term in sorted(set(self.tokenizer(question))):
                # 问题词项去重，避免重复提问某个词时无意放大它的权重。
                matches = self.postings.get(term, [])
                frequency = len(matches)
                if not frequency:
                    continue
                idf = math.log(1 + (len(self.documents) - frequency + 0.5) / (frequency + 0.5))
                # IDF 衡量词的稀有程度，出现文档越少，区分度通常越高。
                for index, count in matches:
                    # 文档长度与平均长度之比用于削弱长 Chunk 的词频优势。
                    normalization = self.k1 * (1 - self.b + self.b * self.lengths[index] / self.average_length)
                    # BM25 标准词频项，count 增大时收益逐渐饱和。
                    scores[index] += idf * count * (self.k1 + 1) / (count + normalization)
            # 相同分数按稳定 Chunk ID 排序，保证重复实验的结果可复现。
            ranked = sorted(scores, key=lambda index: (-scores[index], document_id(self.documents[index])))
            return [RetrievalHit(self.documents[index], document_id(self.documents[index]),
                                 bm25_score=float(scores[index])) for index in ranked[:k]]


def dense_search(vector_store, question, k):
    """通过原有向量库入口检索问题向量，保留数据库返回的原始距离。"""
    results = vector_store.similarity_search_with_score(query=question, k=k)
    hits = {}
    for document, distance in results:
        chunk_id = document_id(document)
        if not math.isfinite(float(distance)):
            raise ValueError("向量检索返回非有限距离")
        if chunk_id not in hits:
            # 同一候选只保留排名最靠前的一次，避免重复进入融合阶段。
            hits[chunk_id] = RetrievalHit(document, chunk_id, distance=float(distance))
    return list(hits.values())


def reciprocal_rank_fusion(rankings, constant=60):
    """RRF 只用排名融合两路候选，贡献公式为 1 / (constant + rank)。"""
    merged = {}
    for ranking in rankings:
        seen = set()
        # 同一路排名内的重复 ID 只能贡献一次，跨路命中可以各贡献一次。
        for rank, hit in enumerate(ranking, start=1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            if hit.chunk_id not in merged:
                merged[hit.chunk_id] = replace(hit, rrf_score=0.0)
                # replace 复制数据对象，避免在原始召回列表上修改分数。
            target = merged[hit.chunk_id]
            target.rrf_score += 1 / (constant + rank)
            if hit.distance is not None:
                target.distance = hit.distance
            if hit.bm25_score is not None:
                target.bm25_score = hit.bm25_score
    return sorted(merged.values(), key=lambda hit: (-hit.rrf_score, hit.chunk_id))


def rerank(question, candidates, cross_encoder):
    """对每个 (问题, 候选文本) 联合编码，按相关性 logit 从高到低排序。"""
    if not candidates:
        return []
    scores = cross_encoder.score(question, [hit.document.page_content for hit in candidates])
    if len(scores) != len(candidates) or any(not math.isfinite(float(score)) for score in scores):
        # 校验数量，防止 zip 把缺失分数的候选静默丢掉。
        raise RuntimeError("Cross-Encoder 分数数量或数值异常")
    hits = [replace(hit, cross_encoder_score=float(score)) for hit, score in zip(candidates, scores)]
    return sorted(hits, key=lambda hit: -hit.cross_encoder_score)


class PaperRetriever:
    """RAG 的统一检索入口，负责召回、融合、重排和最终 TopK。"""
    def __init__(self, vector_store, config=None, *, cross_encoder=None, auto_refresh=True):
        """组合向量存储、关键词索引、配置及可选的共享 Cross-Encoder。"""
        self.vector_store = vector_store
        self.config = config or RetrievalConfig()
        self.keyword_index = BM25Index(lambda: read_corpus(vector_store), auto_refresh=auto_refresh)
        self.cross_encoder = cross_encoder

    def search(self, question):
        # 该函数返回的仍是原始 Chunk，压缩发生在 rag.py 的路由节点之后。
        """按模式召回，必要时执行 RRF 和重排，返回最终 TopK 个原始 Chunk。"""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("检索问题需要为非空字符串")
        config = self.config
        dense = []
        keywords = []
        if config.mode != "bm25":
            # Dense 能找到语义相近内容，问题 Embedding 由向量库适配层调用。
            dense = dense_search(self.vector_store, question, config.dense_k)
        if config.mode != "dense":
            # BM25 补充术语、型号、专有名词等字面匹配。
            keywords = self.keyword_index.search(question, config.keyword_k)
        candidates = dense
        if config.mode == "bm25":
            candidates = keywords
        if config.mode in {"hybrid", "hybrid_rerank"}:
            # 不同检索分数的尺度不同，因此使用 RRF 的排名贡献进行融合。
            candidates = reciprocal_rank_fusion([dense, keywords], config.rrf_constant)
        if config.mode == "hybrid_rerank" and candidates:
            # 无候选时跳过模型，真实模型在首次需要时才加载。
            if self.cross_encoder is None:
                self.cross_encoder = OnnxCrossEncoder(config.reranker_model_dir,
                                                      batch_size=config.reranker_batch_size)
            candidates = rerank(question, candidates, self.cross_encoder)
        # 最终证据数限制统一放在末尾，保证重排可以看到全部召回候选。
        return candidates[:config.top_k]
