"""检索调试入口，支持向量、BM25、混合召回与 Cross-Encoder。"""

# 本地检索调试 CLI。可指定集合、模式与 TopK，便于人工观察问题和证据排名。

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

if __package__:
    from .embeddings import TongyiEmbeddings
    from .retrieval import MODES, PaperRetriever, RetrievalConfig
else:
    from embeddings import TongyiEmbeddings
    from retrieval import MODES, PaperRetriever, RetrievalConfig


def main(argv=None):
    """加载选定知识库并运行一种检索模式，打印各候选正文和原始阶段分数。"""
    parser = argparse.ArgumentParser(description="PaperAgent 检索对比调试")
    parser.add_argument("--query", default="Q K V 和缩放点积注意力怎样计算？")
    parser.add_argument("--mode", choices=MODES, default="hybrid_rerank")
    parser.add_argument("--collection", default="paper_chunks")
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--keyword-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--chroma-dir", type=Path, default=Path(__file__).resolve().parent / "data" / "chroma_db")
    args = parser.parse_args(argv)
    # 候选正文和分数帮助定位语料缺失、召回遗漏、排序及答案生成等环节。
    try:
        config = RetrievalConfig(mode=args.mode, dense_k=args.dense_k, keyword_k=args.keyword_k,
                                 top_k=args.top_k, reranker_model_dir=args.model_dir)
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
        embedding = None
        if args.mode != "bm25":
            # BM25 读取文本即可，其余模式需要与知识库一致的问题向量模型。
            key = os.getenv("DASHSCOPE_API_KEY")
            if not key:
                raise ValueError("未配置 DASHSCOPE_API_KEY")
            embedding = TongyiEmbeddings(api_key=key)
        store = Chroma(collection_name=args.collection, persist_directory=str(args.chroma_dir.resolve()),
                       embedding_function=embedding, create_collection_if_not_exists=False)
        hits = PaperRetriever(store, config).search(args.query)
        # 本入口打印原始候选，完整会话和联网路由由 rag.py 的图组织。
        for rank, hit in enumerate(hits, start=1):
            print(f"\n[{rank}] {hit.chunk_id} | {hit.document.metadata.get('source', '未知来源')}")
            print({"distance": hit.distance, "bm25_score": hit.bm25_score,
                   "rrf_score": hit.rrf_score, "cross_encoder_score": hit.cross_encoder_score})
            print(hit.document.page_content[:500])
        if not hits:
            print("没有召回相关文档。")
    except Exception as exc:
        print(f"检索失败：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
