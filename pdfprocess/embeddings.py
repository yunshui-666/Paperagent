"""入库与问答共享的通义 Embedding 实现。"""

# 向量模型适配层。文档和问题复用相同模型，调用方负责知识库与任务生命周期。

import os

from dashscope import MultiModalEmbedding
from langchain_core.embeddings import Embeddings


EMBEDDING_MODEL = "tongyi-embedding-vision-flash"


class TongyiEmbeddings(Embeddings):
    """适配 LangChain Embeddings 接口，入库文本和用户问题共享向量空间。"""
    def __init__(self, api_key: str | None = None, request_timeout: float = 90):
        """接收可注入的密钥和请求超时，实际模型请求发生在 embed 方法中。"""
        self.api_key = api_key
        self.request_timeout = request_timeout

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # 分批请求服务端，避免一次向量化过多 Chunk。
        """将文本分批向量化，恢复返回顺序并校验一条输入对应一条向量。"""
        vectors = []
        for i in range(0, len(texts), 8):
            batch = texts[i:i + 8]
            response = MultiModalEmbedding.call(
                model=EMBEDDING_MODEL,
                api_key=self.api_key or os.getenv("DASHSCOPE_API_KEY"),
                request_timeout=self.request_timeout,
                input=[{"text": text} for text in batch]
            )
            if response.status_code != 200:
                # 向量化失败直接上报，让现有入库或问答重试机制处理。
                raise RuntimeError(f"{response.code}: {response.message}")
            items = response.output["embeddings"]
            for key in ("index", "text_index"):
                # 按服务端返回的原始输入下标恢复顺序，避免正文和向量错配。
                if all(key in item for item in items):
                    items = sorted(items, key=lambda item: item[key])
                    break
            if len(items) != len(batch):
                # 一条输入必须对应一条向量，数量不一致时拒绝继续入库。
                raise ValueError("Embedding 数量不匹配")
            vectors.extend(item["embedding"] for item in items)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        # 查询与文档使用同一模型。单个问题复用批量接口并取第一条向量。
        """将单个问题放入同一个文档向量化入口，取出与知识库同空间的查询向量。"""
        return self.embed_documents([text])[0]
