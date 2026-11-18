"""模块拆分的定向回归，验证入口、稳定任务键和既有检查点恢复。"""

import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Annotated, TypedDict
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

import main as application_main
from pdfprocess import rag, rag_nodes, rag_prompts, rag_state
from pdfprocess.context_compression import CompressionConfig, history_fingerprint
from pdfprocess.ingest_common import PipelineConfig
from pdfprocess.retrieval import RetrievalConfig, RetrievalHit
from upload_service import app as http_app, schemas
from upload_service.ingestion import IngestionService


class LegacyState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_summary: str
    summary_covered: int
    summary_fingerprint: str


class BackendStructureTests(unittest.TestCase):
    def test_rag_public_exports_point_to_extracted_modules(self):
        self.assertIs(rag.RAGState, rag_state.RAGState)
        self.assertIs(rag.prompt, rag_prompts.prompt)
        self.assertIs(rag.rewrite_prompt, rag_prompts.rewrite_prompt)
        for name in ("retrieve_node", "web_search_node", "compress_node", "generate_node"):
            self.assertIs(getattr(rag, name), getattr(rag_nodes, name))

    def test_http_schema_imports_and_defaults(self):
        for name in ("SessionRequest", "ChatRequest", "RenameRequest", "ReindexRequest"):
            self.assertIs(getattr(http_app, name), getattr(schemas, name))
        request = schemas.SessionRequest(collection_name="paper_chunks")
        self.assertEqual((request.mode, request.dense_k, request.keyword_k, request.top_k), ("hybrid_rerank", 20, 20, 10))
        self.assertEqual(schemas.ReindexRequest().model_dump(), {"scheme": 3, "refresh": True})

    def test_main_function_forwards_startup_arguments(self):
        with patch.object(application_main.multiprocessing, "freeze_support") as freeze:
            with patch.object(application_main, "load_dotenv"), patch.object(application_main.uvicorn, "run") as run:
                self.assertEqual(application_main.main(["--host", "127.0.0.1", "--port", "8765"]), 0)
        freeze.assert_called_once()
        run.assert_called_once_with("upload_service.app:app", host="127.0.0.1", port=8765, workers=1)

    def ingestion(self):
        config = PipelineConfig(data_root=Path("fixture-ingest"), chroma_dir=Path("fixture-chroma"))
        store, storage, notify = Mock(), Mock(), Mock()
        store.enqueue.return_value = {"task_id": "t", "status": "queued", "reused_task": False}
        service = IngestionService(store, storage, SimpleNamespace(pipeline=config), notify)
        document = {"id": "a" * 64, "name": "paper.pdf", "path": "fixture.pdf", "size_bytes": 100,
                    "page_count": 2, "reused_file": False, "copy_seconds": 0.01, "validation_seconds": 0.02}
        return service, document

    def test_ingestion_task_key_matches_existing_snapshot_contract(self):
        service, document = self.ingestion()
        with patch("upload_service.ingestion.file_identity", return_value=(Path("fixture.pdf"), document["id"], "fp", Path("cache"))):
            result = service.enqueue(document, 3)
        snapshot = asdict(service.settings.pipeline)
        snapshot["data_root"] = str(Path(snapshot["data_root"]).resolve())
        snapshot["chroma_dir"] = str(Path(snapshot["chroma_dir"]).resolve())
        expected = hashlib.sha256(json.dumps([document["id"], 3, "fp", snapshot, "paper_chunks3"],
                                            sort_keys=True).encode("utf-8")).hexdigest()
        service.store.enqueue.assert_called_once_with(document, 3, "paper_chunks3", snapshot, expected)
        service.notify.assert_called_once()
        self.assertEqual(result["task_id"], "t")

    def test_refresh_is_recorded_in_task_configuration(self):
        service, document = self.ingestion()
        with patch("upload_service.ingestion.file_identity", return_value=(Path("fixture.pdf"), document["id"], "fp", Path("cache"))):
            service.enqueue(document, 2, refresh=True)
        self.assertFalse(service.store.enqueue.call_args.args[3]["use_cache"])
        self.assertTrue(service.settings.pipeline.use_cache)

    def test_file_identity_mismatch_never_enqueues_or_notifies(self):
        service, document = self.ingestion()
        with patch("upload_service.ingestion.file_identity", return_value=(Path("fixture.pdf"), "changed", "fp", Path("cache"))):
            with self.assertRaisesRegex(RuntimeError, "哈希"):
                service.enqueue(document, 1)
        service.store.enqueue.assert_not_called()
        service.notify.assert_not_called()

    def test_submit_preserves_upload_response_fields(self):
        service, document = self.ingestion()
        service.storage.save.return_value = document
        with patch.object(service, "enqueue", return_value={"task_id": "t", "status": "queued"}):
            result = service.submit("paper.pdf", object(), 1)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["document_id"], document["id"])
        self.assertEqual((result["page_count"], result["size_bytes"], result["task_id"]), (2, 100, "t"))

    def test_legacy_sqlite_state_restores_into_extracted_graph(self):
        history = [HumanMessage(content="old first", id="1:human"), AIMessage(content="old answer", id="1:ai"),
                   HumanMessage(content="old second", id="2:human"), AIMessage(content="old answer 2", id="2:ai")]
        config = {"configurable": {"thread_id": "legacy"}}
        with tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True) as directory:
            path = str(Path(directory) / "legacy.sqlite3")
            with SqliteSaver.from_conn_string(path) as checkpoint:
                builder = StateGraph(LegacyState)
                builder.add_node("generate", lambda state: {})
                builder.add_edge(START, "generate")
                builder.add_edge("generate", END)
                builder.compile(checkpointer=checkpoint).update_state(config, {
                    "messages": history, "conversation_summary": "old summary", "summary_covered": 2,
                    "summary_fingerprint": history_fingerprint(history[:2])}, as_node="generate")
            with SqliteSaver.from_conn_string(path) as checkpoint:
                retriever = Mock()
                retriever.config = RetrievalConfig(mode="dense")
                retriever.search.return_value = [RetrievalHit(Document(page_content="evidence", metadata={"source": "p.pdf"}), "c")]
                answer, summary = Mock(), Mock()
                answer.invoke.return_value = "new answer [1]"
                graph = rag.build_graph(retriever=retriever, answer_chain=answer, checkpointer=checkpoint,
                                        compression_config=CompressionConfig(recent_turns=1), summary_chain=summary)
                result = rag.ask(graph, "new question", "legacy", turn_id="3")
                self.assertEqual(len(result["messages"]), 6)
                self.assertEqual(result["messages"][-1].id, "3:ai")
                self.assertEqual(result["conversation_summary"], "old summary")
                summary.invoke.assert_not_called()

    def test_all_backend_modules_and_definitions_have_documentation(self):
        root = Path(__file__).resolve().parent.parent
        paths = [root / "main.py", *sorted((root / "pdfprocess").rglob("*.py")),
                 *sorted((root / "upload_service").rglob("*.py"))]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"))
            self.assertTrue(ast.get_docstring(tree), str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self.assertTrue(ast.get_docstring(node), f"{path.name} {node.name}")


if __name__ == "__main__":
    unittest.main()
