"""统一配置与基础项目入口的回归，防止容器和 CLI 意外读写另一份数据。"""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from pdfprocess import compare_retrieval, rag, retrieval_cli, retreiver
from pdfprocess.cross_encoder import OnnxCrossEncoder
from pdfprocess.ingest_config import PipelineConfig
from pdfprocess.ingest_common import PipelineConfig as LegacyPipelineConfig
from pdfprocess.pipelines import get_pipeline
from pdfprocess.retrieval import RetrievalConfig
from pdfprocess.vector_store import save_chunks
from upload_service.config import ServiceSettings


class FoundationConfigurationTests(unittest.TestCase):
    def temporary(self):
        value = tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True)
        self.addCleanup(value.cleanup)
        return Path(value.name)

    def test_web_and_pdf_cli_share_environment_configuration(self):
        with patch.dict(os.environ, {"PAPERAGENT_INGEST_ROOT": "fixture-ingestion", "PAPERAGENT_CHROMA_DIR": "fixture-chroma",
                                     "PAPERAGENT_OCR_THREADS": "3", "PAPERAGENT_VLM_MODEL": "fixture-vlm"}):
            web = ServiceSettings.from_environment().pipeline
            cli = PipelineConfig.from_environment()
            explicit = PipelineConfig.from_environment(ocr_threads=1)
        self.assertEqual(web, cli)
        self.assertEqual(cli.ocr_threads, 3)
        self.assertEqual(cli.vlm_model, "fixture-vlm")
        self.assertEqual(cli.chroma_dir, Path("fixture-chroma"))
        self.assertEqual(explicit.ocr_threads, 1)
        self.assertIs(LegacyPipelineConfig, PipelineConfig)

    def test_dispatch_and_legacy_cli_keep_public_contract(self):
        self.assertIs(retreiver.main, retrieval_cli.main)
        for scheme in (1, 2, 3):
            module = get_pipeline(scheme)
            for name in ("pdf_to_markdown", "ingest_pdf", "ingest_pdfs", "upload_pdf", "batch_upload_pdfs"):
                self.assertTrue(callable(getattr(module, f"{name}{scheme}")))
        for invalid in (True, 0, 4, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                get_pipeline(invalid)
        with patch.dict(os.environ, {"PAPERAGENT_RERANKER_MODEL_DIR": "fixture-models"}):
            self.assertEqual(OnnxCrossEncoder().model_dir, Path("fixture-models"))
            self.assertEqual(OnnxCrossEncoder("explicit-models").model_dir, Path("explicit-models"))

    def test_vector_write_retrieval_export_and_rag_use_same_environment_store(self):
        directory = self.temporary()
        name = "configuration_test"
        document = Document(page_content="configuration evidence", metadata={"chunk_id": "c1", "source": "fixture.pdf"})
        with patch.dict(os.environ, {"PAPERAGENT_CHROMA_DIR": str(directory / "chroma")}):
            with contextlib.redirect_stdout(io.StringIO()):
                save_chunks([document], [[0.1, 0.2, 0.3]], collection_name=name)
                # BM25 不发问题向量请求，验证的是真实共享目录而非网络模型。
                self.assertEqual(retrieval_cli.main(["--collection", name, "--mode", "bm25", "--query", "configuration"]), 0)
                exported = directory / "corpus.json"
                self.assertEqual(compare_retrieval.main(["--collection", name, "--export-corpus", str(exported)]), 0)
                answer = Mock()
                answer.invoke.return_value = "evidence [1]"
                graph = rag.build_graph(collection_name=name, answer_chain=answer, retrieval_config=RetrievalConfig(mode="bm25"))
                result = rag.ask(graph, "configuration", "configuration-session")
        self.assertEqual(json.loads(exported.read_text(encoding="utf-8"))["documents"][0]["id"], "c1")
        self.assertEqual(result["sources"][0]["chunk_id"], "c1")

    def test_readme_default_graph_matches_compiled_nodes_and_edges(self):
        from pdfprocess.context_compression import CompressionConfig
        retriever = Mock()
        retriever.config = RetrievalConfig()
        graph = rag.build_graph(retriever=retriever, answer_chain=Mock(), query_rewriter=Mock(),
            compression_config=CompressionConfig(), summary_chain=Mock(), paper_model=Mock(),
            tool_call_chain=Mock(), web_answer_chain=Mock())
        drawable = graph.get_graph()
        self.assertEqual(set(drawable.nodes), {"__start__", "__end__", "prepare_history", "rewrite", "retrieve",
                                              "request_search", "web_search", "compress", "generate", "paper_agent"})
        self.assertEqual({(edge.source, edge.target) for edge in drawable.edges}, {
            ("__start__", "prepare_history"), ("prepare_history", "paper_agent"), ("prepare_history", "rewrite"),
            ("rewrite", "retrieve"), ("retrieve", "compress"), ("retrieve", "request_search"),
            ("request_search", "web_search"), ("web_search", "compress"), ("compress", "generate"),
            ("generate", "__end__"), ("paper_agent", "__end__")})


if __name__ == "__main__":
    unittest.main()
