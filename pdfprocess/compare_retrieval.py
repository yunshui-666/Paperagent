"""在同一问题集、知识库和候选集合上比较召回与重排。"""

# 检索效果实验入口。导出语料供标注，并用真实 Chunk ID 计算 Recall、Hit Rate 与 MRR。

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import statistics
import time

from dotenv import load_dotenv
from langchain_chroma import Chroma

from .cross_encoder import OnnxCrossEncoder
from .embeddings import TongyiEmbeddings
from .ingest_common import write_json
from .retrieval import (
    BM25Index, RetrievalConfig, corpus_fingerprint, dense_search,
    document_id, read_corpus, reciprocal_rank_fusion, rerank
)


from .paths import PROJECT_DIR, chroma_directory


def validate_dataset(dataset, documents, collection_name=None):
    """校验问题 ID、非空相关证据集、集合名称和语料指纹，标注不匹配时在检索前报错。"""
    if not isinstance(dataset, dict) or not isinstance(dataset.get("questions"), list):
        raise ValueError("评测集需要为包含 questions 列表的 JSON 对象")
    if not dataset["questions"]:
        raise ValueError("评测集为空，请先按论文填写问题与相关证据 ID")
    if collection_name and dataset.get("collection_name", collection_name) != collection_name:
        raise ValueError("评测集的 collection_name 与当前知识库不一致")
    fingerprint = corpus_fingerprint(documents)
    # 指纹把人工标注绑定到明确语料版本，重新切块后需要重新核对证据 ID。
    if dataset.get("corpus_fingerprint", fingerprint) != fingerprint:
        raise ValueError("知识库内容与标注时的指纹不一致，请核对证据标注")
    available_ids = {document_id(document) for document in documents}
    question_ids = set()
    for item in dataset["questions"]:
        if not isinstance(item, dict):
            raise ValueError("questions 中的每项需要为对象")
        query_id = item.get("id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in question_ids:
            raise ValueError("每个问题需要唯一的非空字符串 id")
        question_ids.add(query_id)
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"问题 {query_id} 缺少 question")
        relevant_ids = item.get("relevant_ids")
        # Recall 分母是相关 Chunk 数，本地资料不足题在端到端标注中评价。
        if (not isinstance(relevant_ids, list) or not relevant_ids
                or any(not isinstance(value, str) or not value for value in relevant_ids)
                or len(relevant_ids) != len(set(relevant_ids))):
            raise ValueError(f"问题 {query_id} 需要非空且无重复的 relevant_ids")
        unknown = set(relevant_ids) - available_ids
        if unknown:
            raise ValueError(f"问题 {query_id} 的证据 ID 不在当前知识库中：{sorted(unknown)}")


def ranking_metrics(ranked_ids, relevant_ids, k):
    """计算单题前 K 条结果的 Recall、Hit Rate 和 MRR，重复候选占位但只贡献一次命中。"""
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("评测 K 需要为正整数")
    gold = set(relevant_ids)
    if not gold:
        raise ValueError("相关证据集合不能为空")
    # 重复证据只计一次命中，重复条目仍占用实际检索位置。
    ranked = list(ranked_ids)[:k]
    # 先截断再去重，重复候选真实占据检索名额，评测结果反映这一损失。
    relevant_count = len(set(ranked) & gold)
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(ranked, start=1):
        # MRR 使用首条正确证据的位置，多证据覆盖情况由 Recall 描述。
        if chunk_id in gold:
            reciprocal_rank = 1 / rank
            break
    return {"recall": relevant_count / len(gold),
            "hit_rate": float(relevant_count > 0), "mrr": reciprocal_rank}


def improvement(baseline, value):
    """返回百分点变化与相对百分比变化，基线为零时相对变化使用 None 表示。"""
    relative = None
    if baseline != 0:
        relative = (value - baseline) / baseline * 100
    return {"absolute_percentage_points": (value - baseline) * 100,
            "relative_percent": relative}


def summarize(records, mode, k):
    """对指定模式和 K 汇总各题检索指标，并计算该模式平均检索延迟。"""
    metrics = [ranking_metrics(row["rankings"][mode], row["relevant_ids"], k) for row in records]
    result = {name: statistics.mean(item[name] for item in metrics) for name in ("recall", "hit_rate", "mrr")}
    result["mean_latency_ms"] = statistics.mean(row["latencies_ms"][mode] for row in records)
    return result


def evaluate_retrieval(vector_store, dataset, *, config=None, cross_encoder=None,
                       at_k=(5, 10, 20), collection_name=None):
    """用固定语料和同一组问题比较四种模式，保留逐题排名、候选召回率和耗时。
    评测前后验证语料指纹，遇到知识库变化会报错，便于复现实验。
    """
    config = config or RetrievalConfig()
    if not at_k or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in at_k):
        raise ValueError("at_k 需要包含正整数")
    cutoffs = sorted(set(at_k))
    documents = read_corpus(vector_store)
    validate_dataset(dataset, documents, collection_name)
    fingerprint = corpus_fingerprint(documents)
    index = BM25Index(lambda: documents, auto_refresh=False)
    # 评测固定内存语料，索引构建和模型预热耗时单独记录。
    started = time.perf_counter()
    index.refresh()
    index_build_seconds = time.perf_counter() - started
    if cross_encoder is None:
        cross_encoder = OnnxCrossEncoder(config.reranker_model_dir,
                                         batch_size=config.reranker_batch_size)
    started = time.perf_counter()
    cross_encoder.score("预热", ["模型预热文本"])
    warmup_seconds = time.perf_counter() - started
    records = []
    for item in dataset["questions"]:
        # Dense 和 BM25 各调用一次，混合与重排复用同一候选池。
        # at_k 控制评价截断位置，候选池大小由 dense_k 和 keyword_k 决定。
        question = item["question"]
        try:
            started = time.perf_counter()
            dense = dense_search(vector_store, question, config.dense_k)
            dense_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            keywords = index.search(question, config.keyword_k)
            keyword_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            fused = reciprocal_rank_fusion([dense, keywords], config.rrf_constant)
            fusion_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            reranked = rerank(question, fused, cross_encoder)
            rerank_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            raise RuntimeError(f"评测问题 {item['id']} 执行失败，完整评测尚未完成") from exc
        rankings = {"dense": dense, "bm25": keywords, "hybrid": fused, "hybrid_rerank": reranked}
        rows = {name: [hit.chunk_id for hit in hits] for name, hits in rankings.items()}
        if set(rows["hybrid"]) != set(rows["hybrid_rerank"]):
            # 重排只调整候选顺序，候选总集合相等时可解释同 K 指标变化。
            raise RuntimeError("重排前后候选集合发生变化")
        records.append({
            "id": item["id"], "question": question, "relevant_ids": item["relevant_ids"],
            "rankings": rows,
            "candidate_counts": {name: len(hits) for name, hits in rankings.items()},
            "latencies_ms": {"dense": dense_ms, "bm25": keyword_ms,
                             "hybrid": dense_ms + keyword_ms + fusion_ms,
                             "hybrid_rerank": dense_ms + keyword_ms + fusion_ms + rerank_ms}
        })
    # 检测同数量 upsert 和删除，避免在变化中的知识库上输出一张混合指标表。
    if corpus_fingerprint(read_corpus(vector_store)) != fingerprint:
        raise RuntimeError("评测期间知识库内容发生变化，请在入库完成后重新评测")
    methods = ("dense", "bm25", "hybrid", "hybrid_rerank")
    metrics = {mode: {str(k): summarize(records, mode, k) for k in cutoffs} for mode in methods}
    gains = {mode: {str(k): improvement(metrics["dense"][str(k)]["recall"], metrics[mode][str(k)]["recall"])
                    for k in cutoffs} for mode in methods if mode != "dense"}
    topk_sweep = {str(k): metrics["dense"][str(k)] for k in cutoffs}
    # 同 K 比较检索算法，TopK sweep 观察同一种算法扩大结果数后的变化。
    smallest_k = cutoffs[0]
    topk_gains = {str(k): improvement(topk_sweep[str(smallest_k)]["recall"], topk_sweep[str(k)]["recall"])
                 for k in cutoffs if k != smallest_k}
    candidate_recall = {
        # 使用全部候选作为截断位置，揭示正确证据是否进入重排器。
        mode: statistics.mean(ranking_metrics(row["rankings"][mode], row["relevant_ids"],
                                               max(1, len(row["rankings"][mode])))["recall"]
                              for row in records) for mode in methods
    }
    return {
        "dataset_name": dataset.get("dataset_name", "unnamed"),
        "annotation_status": dataset.get("annotation_status", "unspecified"),
        "question_count": len(records), "collection_name": collection_name,
        "corpus_size": len(documents), "corpus_fingerprint": fingerprint,
        "config": {name: str(value) if isinstance(value, Path) else value
                   for name, value in asdict(config).items() if name not in {"mode", "top_k"}},
        "evaluation_at_k": cutoffs,
        "reranker": {"implementation": type(cross_encoder).__name__,
                     "repository": getattr(cross_encoder, "repository", None),
                     "revision": getattr(cross_encoder, "revision", None)},
        "index_build_seconds": index_build_seconds, "model_warmup_seconds": warmup_seconds,
        "metrics": metrics, "same_k_recall_improvements": gains,
        "dense_topk_sweep": topk_sweep, "dense_topk_improvements": topk_gains,
        "candidate_recall": candidate_recall, "questions": records
    }


def main(argv=None):
    """导出可标注语料，或读取人工标注集运行检索评测并保存报告。"""
    parser = argparse.ArgumentParser(description="比较向量、BM25、混合召回和 Cross-Encoder 重排")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--export-corpus", type=Path, help="导出 Chunk ID、正文与页码，供证据标注")
    parser.add_argument("--collection", default="paper_chunks")
    parser.add_argument("--chroma-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--keyword-k", type=int, default=20)
    parser.add_argument("--at-k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.dataset is None and args.export_corpus is None:
        parser.error("请提供 --dataset 或 --export-corpus")
    try:
        config = RetrievalConfig(dense_k=args.dense_k, keyword_k=args.keyword_k,
                                 top_k=1, reranker_model_dir=args.model_dir)
        load_dotenv(PROJECT_DIR / ".env")
        directory = args.chroma_dir if args.chroma_dir is not None else chroma_directory()
        output = args.output or directory.parent / "retrieval_comparison.json"
        embedding = None
        if args.dataset is not None:
            # Dense 评测需要问题向量模型，导出语料只需读取已有 Chunk。
            key = os.getenv("DASHSCOPE_API_KEY")
            if not key:
                raise ValueError("未配置 DASHSCOPE_API_KEY")
            embedding = TongyiEmbeddings(api_key=key)
        store = Chroma(collection_name=args.collection, persist_directory=str(directory.resolve()),
                       embedding_function=embedding, create_collection_if_not_exists=False)
        if args.export_corpus is not None:
            # 使用实际数据库 ID 进行标注，论文身份与页码位于 metadata 中。
            documents = read_corpus(store)
            write_json(args.export_corpus, {
                "collection_name": args.collection, "corpus_fingerprint": corpus_fingerprint(documents),
                "documents": [{"id": document_id(doc), "text": doc.page_content, "metadata": doc.metadata}
                              for doc in documents]
            })
            print(f"已导出 {len(documents)} 个 Chunk：{args.export_corpus.resolve()}")
        if args.dataset is not None:
            dataset = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
            report = evaluate_retrieval(store, dataset, config=config, at_k=args.at_k,
                                        collection_name=args.collection)
            write_json(output, report)
            for mode, cutoffs in report["metrics"].items():
                print(mode, " | ".join(f"Recall@{k}={value['recall']:.2%}" for k, value in cutoffs.items()))
            print(f"完整结果：{output.resolve()}")
    except Exception as exc:
        print(f"执行失败：{exc}")
        if exc.__cause__ is not None:
            print(f"原因：{exc.__cause__}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
