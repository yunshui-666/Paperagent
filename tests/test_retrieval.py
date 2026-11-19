import contextlib
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import chromadb
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from langchain_core.documents import Document

from pdfprocess import rag
from pdfprocess.compare_retrieval import (
    evaluate_retrieval, improvement, ranking_metrics, validate_dataset, main as evaluation_main
)
from pdfprocess.retrieval_cli import main as retrieval_main
from pdfprocess.cross_encoder import DEFAULT_MODEL_DIR, MODEL_FILE, OnnxCrossEncoder
from pdfprocess.retrieval import (
    BM25Index, PaperRetriever, RetrievalConfig, RetrievalHit, corpus_fingerprint,
    dense_search, document_id, read_corpus, reciprocal_rank_fusion, rerank, tokenize
)


class MemoryStore:
    def __init__(self, documents, dense_ids=None):
        self.documents = documents
        self.dense_ids = dense_ids or []
        self.calls = []

    def get(self, **kwargs):
        return {"ids": [doc.id for doc in self.documents],
                "documents": [doc.page_content for doc in self.documents],
                "metadatas": [doc.metadata for doc in self.documents]}

    def similarity_search_with_score(self, *, query, k):
        self.calls.append((query, k))
        documents = {doc.id: doc for doc in self.documents}
        return [(documents[chunk_id], float(index)) for index, chunk_id in enumerate(self.dense_ids[:k])]


class TargetReranker:
    def __init__(self):
        self.inputs = []

    def score(self, question, texts):
        self.inputs.append((question, list(texts)))
        return [float("target" in text) for text in texts]


def make_documents():
    return [
        Document(id="a", page_content="unrelated ocean water", metadata={"source": "paper.pdf", "chunk_id": "a"}),
        Document(id="b", page_content="attention query key value", metadata={"source": "paper.pdf", "chunk_id": "b"}),
        Document(id="c", page_content="attention target normalization", metadata={"source": "paper.pdf", "chunk_id": "c"})
    ]


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.documents = make_documents()
        self.store = MemoryStore(self.documents, ["a", "b"])

    def test_default_expands_candidates_and_final_topk(self):
        config = RetrievalConfig()
        self.assertEqual((config.dense_k, config.keyword_k, config.top_k), (20, 20, 10))
        self.assertEqual(config.mode, "hybrid_rerank")

    def test_invalid_configurations_are_rejected(self):
        cases = [{"mode": "unknown"}, {"dense_k": 0}, {"top_k": True},
                 {"mode": "dense", "dense_k": 5, "top_k": 10}]
        for options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                RetrievalConfig(**options)

    def test_dense_uses_requested_candidate_k_and_final_limit(self):
        hits = PaperRetriever(self.store, RetrievalConfig(mode="dense", dense_k=20, top_k=1)).search("query")
        self.assertEqual(self.store.calls, [("query", 20)])
        self.assertEqual([hit.chunk_id for hit in hits], ["a"])

    def test_dense_deduplicates_by_record_id(self):
        self.store.dense_ids = ["a", "a", "b"]
        self.assertEqual([hit.chunk_id for hit in dense_search(self.store, "query", 20)], ["a", "b"])

    def test_keyword_only_avoids_embedding_calls(self):
        config = RetrievalConfig(mode="bm25", top_k=2)
        hits = PaperRetriever(self.store, config).search("normalization")
        self.assertEqual([hit.chunk_id for hit in hits], ["c"])
        self.assertEqual(self.store.calls, [])
        self.assertIsNone(hits[0].distance)
        self.assertGreater(hits[0].bm25_score, 0)

    def test_keyword_index_handles_common_terms_with_positive_idf(self):
        documents = [Document(id="1", page_content="common"), Document(id="2", page_content="common common")]
        index = BM25Index(lambda: documents, tokenizer=str.split)
        hits = index.search("common", 10)
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(hit.bm25_score > 0 for hit in hits))

    def test_keyword_index_refreshes_same_size_updates(self):
        retriever = PaperRetriever(self.store, RetrievalConfig(mode="bm25", top_k=1))
        self.assertEqual(retriever.search("normalization")[0].chunk_id, "c")
        self.store.documents[2] = Document(id="c", page_content="unrelated updated text")
        self.assertEqual(retriever.search("normalization"), [])

    def test_keyword_index_detects_deletion(self):
        retriever = PaperRetriever(self.store, RetrievalConfig(mode="bm25", top_k=1))
        self.assertTrue(retriever.search("normalization"))
        self.store.documents.pop()
        self.assertEqual(retriever.search("normalization"), [])

    def test_no_keyword_matches_returns_empty(self):
        hits = PaperRetriever(self.store, RetrievalConfig(mode="bm25")).search("xyzzzz")
        self.assertEqual(hits, [])

    def test_chinese_and_english_tokenization(self):
        terms = tokenize("QKV 和注意力的计算")
        self.assertIn("qkv", terms)
        self.assertIn("注意力", terms)
        self.assertNotIn("和", terms)

    def test_rrf_merges_channels_and_retains_separate_scores(self):
        dense = [RetrievalHit(self.documents[1], "b", distance=0.2)]
        keyword = [RetrievalHit(self.documents[1], "b", bm25_score=5)]
        hits = reciprocal_rank_fusion([dense, keyword])
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0].rrf_score, 2 / 61)
        self.assertEqual(hits[0].distance, 0.2)
        self.assertEqual(hits[0].bm25_score, 5)
        self.assertIsNone(dense[0].rrf_score)

    def test_rrf_duplicate_within_channel_is_counted_once(self):
        hit = RetrievalHit(self.documents[0], "a")
        fused = reciprocal_rank_fusion([[hit, hit]])
        self.assertAlmostEqual(fused[0].rrf_score, 1 / 61)

    def test_hybrid_adds_keyword_only_evidence(self):
        hits = PaperRetriever(self.store, RetrievalConfig(mode="hybrid", top_k=10)).search("attention")
        self.assertEqual({hit.chunk_id for hit in hits}, {"a", "b", "c"})

    def test_cross_encoder_reranks_actual_union_only(self):
        encoder = TargetReranker()
        hits = PaperRetriever(self.store, cross_encoder=encoder).search("attention")
        self.assertEqual(hits[0].chunk_id, "c")
        self.assertEqual(len(encoder.inputs[0][1]), 3)
        self.assertEqual({hit.chunk_id for hit in hits}, {"a", "b", "c"})
        self.assertEqual(hits[0].cross_encoder_score, 1.0)

    def test_reranker_score_errors_raise(self):
        candidates = [RetrievalHit(self.documents[0], "a")]
        for scores in ([], [float("nan")], [1, 2]):
            encoder = Mock()
            encoder.score.return_value = scores
            with self.subTest(scores=scores), self.assertRaises(RuntimeError):
                rerank("query", candidates, encoder)

    def test_empty_retrieval_does_not_load_reranker(self):
        encoder = Mock()
        hits = PaperRetriever(MemoryStore([]), cross_encoder=encoder).search("query")
        self.assertEqual(hits, [])
        encoder.score.assert_not_called()

    def test_record_id_takes_precedence_over_metadata(self):
        document = Document(id="actual", page_content="text", metadata={"chunk_id": "old"})
        self.assertEqual(document_id(document), "actual")

    def test_read_corpus_preserves_legacy_metadata(self):
        rows = read_corpus(self.store)
        self.assertEqual(rows[0].id, "a")
        self.assertEqual(rows[0].metadata["source"], "paper.pdf")

    def test_default_graph_runs_enhanced_retrieval_and_history(self):
        answer = Mock()
        answer.invoke.return_value = "论文证据 [1]"
        graph = rag.build_graph(vector_store=self.store, answer_chain=answer, cross_encoder=TargetReranker())
        first = rag.ask(graph, "attention", "thread-a")
        second = rag.ask(graph, "attention target", "thread-a")
        self.assertEqual(first["sources"][0]["chunk_id"], "c")
        self.assertIsNone(first["sources"][0]["distance"])
        self.assertIn("cross_encoder_score", first["sources"][0])
        self.assertEqual(len(second["messages"]), 4)
        self.assertEqual(len(answer.invoke.call_args.args[0]["history"]), 2)

    def test_reranker_failure_keeps_completed_history(self):
        encoder = TargetReranker()
        answer = Mock()
        answer.invoke.return_value = "回答 [1]"
        graph = rag.build_graph(vector_store=self.store, answer_chain=answer, cross_encoder=encoder)
        rag.ask(graph, "attention", "a")
        encoder.score = Mock(side_effect=RuntimeError("model failed"))
        with self.assertRaisesRegex(RuntimeError, "知识库检索失败"):
            rag.ask(graph, "attention", "a")
        state = graph.get_state({"configurable": {"thread_id": "a"}})
        self.assertEqual(len(state.values["messages"]), 2)

    def test_keyword_cli_runs_without_embedding_credentials(self):
        with patch("pdfprocess.retrieval_cli.Chroma", return_value=self.store):
            with patch("pdfprocess.retrieval_cli.load_dotenv"), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(retrieval_main(["--mode", "bm25", "--query", "attention"]), 0)
        self.assertEqual(self.store.calls, [])

    def test_export_cli_provides_corpus_for_future_annotations(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"),
                                         ignore_cleanup_errors=True) as directory:
            output = Path(directory) / "corpus.json"
            with patch("pdfprocess.compare_retrieval.Chroma", return_value=self.store):
                with patch("pdfprocess.compare_retrieval.load_dotenv"), patch.dict(os.environ, {}, clear=True):
                    self.assertEqual(evaluation_main(["--export-corpus", str(output)]), 0)
            self.assertIn('"id": "a"', output.read_text(encoding="utf-8"))
        self.assertEqual(self.store.calls, [])


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.documents = make_documents()
        self.store = MemoryStore(self.documents, ["a", "b"])
        self.dataset = {"dataset_name": "synthetic_test_only", "annotation_status": "synthetic",
                        "questions": [{"id": "q1", "question": "attention", "relevant_ids": ["c"]}]}

    def test_recall_hit_rate_and_mrr_have_distinct_definitions(self):
        metrics = ranking_metrics(["bad", "gold1"], ["gold1", "gold2"], 2)
        self.assertEqual(metrics, {"recall": 0.5, "hit_rate": 1.0, "mrr": 0.5})

    def test_duplicates_do_not_inflate_recall_or_free_positions(self):
        self.assertEqual(ranking_metrics(["x", "x", "gold"], ["gold"], 2)["recall"], 0)
        self.assertEqual(ranking_metrics(["gold", "gold"], ["gold", "other"], 2)["recall"], 0.5)

    def test_zero_baseline_relative_gain_is_undefined(self):
        self.assertEqual(improvement(0, 0.5), {"absolute_percentage_points": 50, "relative_percent": None})

    def test_negative_improvements_are_reported(self):
        self.assertEqual(improvement(0.5, 0.25), {"absolute_percentage_points": -25, "relative_percent": -50})

    def test_unknown_evidence_ids_fail_before_search(self):
        self.dataset["questions"][0]["relevant_ids"] = ["missing"]
        encoder = Mock()
        with self.assertRaisesRegex(ValueError, "不在当前知识库"):
            evaluate_retrieval(self.store, self.dataset, cross_encoder=encoder)
        self.assertEqual(self.store.calls, [])
        encoder.score.assert_not_called()

    def test_empty_dataset_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "评测集为空"):
            validate_dataset({"questions": []}, self.documents)

    def test_changed_corpus_fingerprint_is_rejected(self):
        self.dataset["corpus_fingerprint"] = "old"
        with self.assertRaisesRegex(ValueError, "指纹"):
            validate_dataset(self.dataset, self.documents)

    def test_equal_k_and_candidate_recall_are_both_reported(self):
        report = evaluate_retrieval(self.store, self.dataset, cross_encoder=TargetReranker(), at_k=(1, 2))
        self.assertEqual(report["metrics"]["dense"]["1"]["recall"], 0)
        self.assertEqual(report["metrics"]["hybrid_rerank"]["1"]["recall"], 1)
        self.assertEqual(report["candidate_recall"]["hybrid"], report["candidate_recall"]["hybrid_rerank"])
        self.assertEqual(len(self.store.calls), 1)
        self.assertEqual(report["reranker"]["implementation"], "TargetReranker")
        self.assertIsNone(report["reranker"]["repository"])

    def test_relevant_document_is_never_injected_into_candidates(self):
        self.store.documents[2] = Document(id="c", page_content="hidden evidence")
        encoder = TargetReranker()
        report = evaluate_retrieval(self.store, self.dataset, cross_encoder=encoder, at_k=(1,))
        self.assertEqual(report["metrics"]["hybrid_rerank"]["1"]["recall"], 0)
        self.assertNotIn("hidden evidence", encoder.inputs[-1][1])

    def test_changed_corpus_during_evaluation_invalidates_report(self):
        class UpdatingReranker(TargetReranker):
            def score(inner, question, texts):
                if question != "预热":
                    self.store.documents[0].page_content = "updated content"
                return super().score(question, texts)

        with self.assertRaisesRegex(RuntimeError, "知识库内容发生变化"):
            evaluate_retrieval(self.store, self.dataset, cross_encoder=UpdatingReranker())


LOCAL_MODEL_DIR = Path(os.environ.get("PAPERAGENT_RERANKER_MODEL_DIR", str(DEFAULT_MODEL_DIR)))


@unittest.skipUnless((LOCAL_MODEL_DIR / MODEL_FILE).is_file(), "本地 Cross-Encoder 权重尚未下载")
class RealCrossEncoderTests(unittest.TestCase):
    def test_long_question_fits_model_input_budget(self):
        model = OnnxCrossEncoder(LOCAL_MODEL_DIR)
        scores = model.score("这个问题很长，请比较论文的研究方法和实验结果。" * 150,
                             ["论文研究方法和实验结果。"])
        self.assertEqual(len(scores), 1)
        self.assertTrue(math.isfinite(scores[0]))

    def test_local_multilingual_cross_encoder_scores_query_document_pairs(self):
        model = OnnxCrossEncoder(LOCAL_MODEL_DIR, batch_size=1)
        scores = model.score("训练时为什么要清零梯度？", [
            "每次反向传播前清理累积梯度，避免把前一步的梯度重复累加。",
            "卷积的步长决定输出特征图的空间分辨率。"
        ])
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(math.isfinite(score) for score in scores))
        self.assertGreater(scores[0], scores[1])

    def test_real_chroma_hybrid_reranker_and_graph_work_together(self):
        class TestEmbeddings(Embeddings):
            def embed_documents(self, texts):
                return [[1.0, 0.5, 0.1] for text in texts]

            def embed_query(self, text):
                return [1.0, 0.5, 0.1]

        client = chromadb.EphemeralClient()
        name = "retrieval_test_" + uuid4().hex
        client.create_collection(name, embedding_function=None)
        self.addCleanup(client.delete_collection, name)
        store = Chroma(client=client, collection_name=name, embedding_function=TestEmbeddings())
        store.add_documents([
            Document(page_content="每次反向传播前清理累积梯度，避免把前一步的梯度重复累加。",
                     metadata={"source": "synthetic.pdf", "page": 1}),
            Document(page_content="卷积的步长决定输出特征图的空间分辨率。",
                     metadata={"source": "synthetic.pdf", "page": 2})
        ], ids=["gradient", "convolution"])
        answer = Mock()
        answer.invoke.return_value = "先清理累积梯度 [1]"
        # 此用例验证本地模型与 Chroma 的联动，联网分支由独立路由测试覆盖。
        graph = rag.build_graph(vector_store=store, answer_chain=answer,
                                cross_encoder=OnnxCrossEncoder(LOCAL_MODEL_DIR), web_search_tool=None)
        result = rag.ask(graph, "训练时为什么要清零梯度？", "real-integration")
        self.assertEqual(result["sources"][0]["chunk_id"], "gradient")
        self.assertEqual(result["sources"][0]["page"], 1)
        self.assertIn("cross_encoder_score", result["sources"][0])


if __name__ == "__main__":
    unittest.main()
