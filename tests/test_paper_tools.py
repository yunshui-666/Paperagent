"""论文客户端、工具路由及长对话预算测试，外部服务使用本地替身。"""

from contextlib import contextmanager
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from pdfprocess import rag
from pdfprocess.context_compression import CompressionConfig
from pdfprocess.retrieval import RetrievalConfig
from pdfprocess.tools.paper_agent import paper_request_kind, run_paper_agent
from pdfprocess.tools.papers import ArxivClient, canonical_id, create_download_tool, search_papers
from pdfprocess.tools.registry import ToolRegistry


PAPERS = [{"arxiv_id": "1706.03762", "title": "Attention Is All You Need", "authors": ["Fixture author"],
           "summary": "Synthetic metadata for a public paper.", "published": "2017-06-12",
           "url": "https://arxiv.org/abs/1706.03762", "pdf_url": "https://arxiv.org/pdf/1706.03762"}]
ATOM_XML = b'''<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/1706.03762v1</id>
<title> Attention Is All You Need </title><summary>Public metadata.</summary><author><name>Author</name></author>
<published>2017-06-12T00:00:00Z</published></entry></feed>'''


def response(payload, *, status=200, headers=None):
    result = MagicMock()
    result.__enter__.return_value = result
    result.status_code = status
    result.headers = headers or {}
    result.iter_content.return_value = [payload]
    return result


def call(name, arguments, identifier):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments, "id": identifier, "type": "tool_call"}],
                     usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})


class SequenceModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.inputs = []
        self.bindings = []

    def bind_tools(self, tools, **kwargs):
        self.bindings.append({"names": [tool.name for tool in tools], **kwargs})
        return self

    def bind(self, **kwargs):
        return self

    def invoke(self, messages):
        self.inputs.append(deepcopy(messages))
        reply = self.replies.pop(0)
        return reply(messages) if callable(reply) else reply


class ArxivClientTests(unittest.TestCase):
    def test_identifier_validation(self):
        for identifier in ("1706.03762", "1706.03762v2", "hep-th/9901001", "math.AG/9901001"):
            self.assertEqual(canonical_id(identifier), identifier)
        for identifier in (None, "", "https://localhost/a.pdf", "../../secret", "1706.03762?x=1"):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                canonical_id(identifier)

    def test_search_atom_metadata_and_canonical_urls(self):
        with patch("pdfprocess.tools.papers.requests.get", return_value=response(ATOM_XML)) as get:
            result = ArxivClient().search("attention", 2)
        self.assertEqual(result[0]["arxiv_id"], "1706.03762v1")
        self.assertEqual(result[0]["pdf_url"], "https://arxiv.org/pdf/1706.03762v1")
        self.assertEqual(get.call_args.args[0], "https://export.arxiv.org/api/query")

    def test_search_errors_and_response_size(self):
        for payload in (b"<html/>", b"bad xml", b"x" * (2 * 1024 * 1024 + 1)):
            with self.subTest(size=len(payload)), patch("pdfprocess.tools.papers.requests.get", return_value=response(payload)):
                with self.assertRaises(Exception):
                    ArxivClient().search("attention")

    def test_pdf_stream_size_and_redirect_boundary(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True) as directory:
            with patch("pdfprocess.tools.papers.requests.get", return_value=response(b"%PDF fixture")):
                with ArxivClient().open_pdf("1706.03762", max_bytes=100, directory=directory) as stream:
                    self.assertEqual(stream.read(), b"%PDF fixture")
            with patch("pdfprocess.tools.papers.requests.get", return_value=response(b"x" * 101)):
                with self.assertRaises(ValueError):
                    with ArxivClient().open_pdf("1706.03762", max_bytes=100, directory=directory):
                        pass
            with patch("pdfprocess.tools.papers.requests.get", return_value=response(b"", status=302,
                       headers={"Location": "http://127.0.0.1/private"})) as get:
                with self.assertRaises(ValueError):
                    with ArxivClient().open_pdf("1706.03762", max_bytes=100, directory=directory):
                        pass
                self.assertEqual(get.call_count, 1)

    def test_download_submits_to_selected_collection_and_reports_pending(self):
        @contextmanager
        def open_pdf(*args, **kwargs):
            yield io.BytesIO(b"fixture bytes")
        ingestion = Mock()
        ingestion.storage.root = Path("fixture")
        ingestion.settings.max_file_bytes = 100
        ingestion.submit.return_value = {"task_id": "task", "document_id": "document"}
        store = Mock()
        store.get.return_value = {"id": "task", "status": "running", "result": None, "error": None}
        tool = create_download_tool(ingestion, store, "selected_library", client=SimpleNamespace(open_pdf=open_pdf), wait_seconds=0)
        result = tool.invoke({"paper_id": "1706.03762"})
        self.assertFalse(result["ingested"])
        self.assertEqual(result["status"], "running")
        self.assertEqual(ingestion.submit.call_args.kwargs["collection_name"], "selected_library")

    def test_download_completed_and_failed_states_are_explicit(self):
        @contextmanager
        def open_pdf(*args, **kwargs):
            yield io.BytesIO(b"fixture")
        ingestion = Mock()
        ingestion.storage.root = Path("fixture")
        ingestion.settings.max_file_bytes = 100
        ingestion.submit.return_value = {"task_id": "task", "document_id": "document"}
        for status in ("completed", "failed"):
            store = Mock()
            store.get.return_value = {"id": "task", "status": status, "result": {"chunk_count": 3}, "error": None}
            tool = create_download_tool(ingestion, store, "library", client=SimpleNamespace(open_pdf=open_pdf), wait_seconds=0)
            result = tool.invoke({"paper_id": "1706.03762"})
            self.assertEqual(result["ingested"], status == "completed")


class PaperAgentTests(unittest.TestCase):
    def setUp(self):
        self.search = Mock(return_value=PAPERS)
        self.download = Mock(return_value={"arxiv_id": "1706.03762", "task_id": "task", "document_id": "doc",
                                          "collection_name": "library", "status": "completed", "ingested": True, "chunk_count": 3})
        self.registry = ToolRegistry()
        self.registry.register(StructuredTool.from_function(self.search, name="search_papers", description="Search papers",
                                                            args_schema=search_papers.args_schema))
        @contextmanager
        def unused(*args, **kwargs):
            yield io.BytesIO()
        template = create_download_tool(Mock(), Mock(), "library", client=SimpleNamespace(open_pdf=unused))
        self.registry.register(StructuredTool.from_function(self.download, name="download_paper", description="Download paper",
                                                            args_schema=template.args_schema))

    def test_operation_intents_and_explanatory_questions(self):
        self.assertEqual(paper_request_kind("帮我搜索论文 retrieval augmented generation"), "search")
        self.assertEqual(paper_request_kind("下载刚才第一篇并入库"), "download")
        for question in ("如何下载论文？", "这个方法准确率是多少？", "如何构建搜索论文工具？"):
            self.assertIsNone(paper_request_kind(question))

    def test_search_then_download_and_context_protocol(self):
        model = SequenceModel([call("search_papers", {"query": "attention"}, "s"),
                               call("download_paper", {"paper_id": "1706.03762"}, "d"), AIMessage(content="论文已处理。")])
        result = run_paper_agent({"question": "搜索论文并下载第一篇入库", "turn_id": "one"}, model=model,
                                 registry=self.registry, collection_name="library")
        self.assertTrue(result["metrics"]["paper_downloads"][0]["ingested"])
        self.assertIn("completed", result["answer"])
        self.assertEqual([message.id for message in result["messages"]], ["one:human", "one:ai"])
        self.assertEqual(result["paper_catalog"][0]["arxiv_id"], "1706.03762")
        self.assertTrue(all(size <= 16000 for size in result["metrics"]["paper_agent_input_tokens"]))
        self.assertIsInstance(model.inputs[-1][-1], ToolMessage)

    def test_search_request_does_not_offer_download_tool(self):
        model = SequenceModel([call("search_papers", {"query": "attention"}, "s"), AIMessage(content="Found paper.")])
        run_paper_agent({"question": "搜索论文 attention"}, model=model, registry=self.registry, collection_name="library")
        self.assertEqual(model.bindings[0]["names"], ["search_papers"])
        self.download.assert_not_called()

    def test_unknown_download_identifier_is_rejected(self):
        model = SequenceModel([call("download_paper", {"paper_id": "2401.12345"}, "d")])
        with self.assertRaisesRegex(RuntimeError, "ID 需要来自"):
            run_paper_agent({"question": "下载 1706.03762"}, model=model, registry=self.registry, collection_name="library")
        self.download.assert_not_called()

    def test_ignored_required_download_terminates(self):
        model = SequenceModel([call("search_papers", {"query": "attention"}, "s"),
                               AIMessage(content="I'll download"), AIMessage(content="I'll download")])
        with self.assertRaisesRegex(RuntimeError, "未执行所请求的论文下载"):
            run_paper_agent({"question": "搜索论文并下载第一篇"}, model=model, registry=self.registry, collection_name="library")

    def test_tool_count_is_bounded(self):
        replies = [call("search_papers", {"query": "attention"}, f"s{i}") for i in range(6)] + [AIMessage(content="Results.")]
        model = SequenceModel(replies)
        result = run_paper_agent({"question": "搜索论文 attention"}, model=model, registry=self.registry, collection_name="library")
        self.assertEqual(len(result["metrics"]["tool_calls"]), 6)
        self.assertEqual(len(model.inputs), 7)

    def test_network_download_error_is_reported_without_forced_repeat(self):
        self.download.side_effect = RuntimeError("PDF network timeout")
        model = SequenceModel([call("download_paper", {"paper_id": "1706.03762"}, "d"),
                               AIMessage(content="下载暂时失败，请稍后重试。")])
        result = run_paper_agent({"question": "下载 1706.03762 并入库"}, model=model,
                                 registry=self.registry, collection_name="library")
        self.download.assert_called_once()
        self.assertFalse(result["metrics"]["paper_downloads"][0]["ingested"])
        self.assertIn("PDF network timeout", result["answer"])
        self.assertNotIn("`None`", result["answer"])

    def test_truncated_tool_response_does_not_execute_download(self):
        reply = call("download_paper", {"paper_id": "1706.03762"}, "d")
        reply.response_metadata = {"finish_reason": "length"}
        with self.assertRaisesRegex(RuntimeError, "无效响应"):
            run_paper_agent({"question": "下载 1706.03762 并入库"}, model=SequenceModel([reply]),
                             registry=self.registry, collection_name="library")
        self.download.assert_not_called()

    def test_large_results_fit_budget_and_keep_entire_identifier_catalog(self):
        self.search.return_value = [{**PAPERS[0], "arxiv_id": f"2401.{i:05d}", "title": "长标题" * 200,
                                     "summary": "长摘要" * 1000} for i in range(10)]
        model = SequenceModel([call("search_papers", {"query": "attention"}, "s"), AIMessage(content="Results.")])
        result = run_paper_agent({"question": "搜索论文 attention", "model_history": [HumanMessage(content="历史" * 2000)]},
            model=model, registry=self.registry, collection_name="library", compression_config=CompressionConfig())
        self.assertTrue(all(size <= 16000 for size in result["metrics"]["paper_agent_input_tokens"]))
        self.assertEqual(len(result["paper_catalog"]), 10)
        self.assertIn("2401.00009", str(model.inputs[-1]))

    def test_graph_routes_before_retrieval_and_preserves_paper_catalog(self):
        retriever = Mock()
        retriever.config = RetrievalConfig(mode="dense")
        answer, summary = Mock(), Mock()
        summary.invoke.return_value = "Summary with paper ID 1706.03762"
        model = SequenceModel([call("search_papers", {"query": "attention"}, "s"), AIMessage(content="1706.03762 found"),
                               call("download_paper", {"paper_id": "1706.03762"}, "d"), AIMessage(content="Done")])
        graph = rag.build_graph(retriever=retriever, answer_chain=answer, paper_model=model, tool_registry=self.registry,
                                compression_config=CompressionConfig(), summary_chain=summary, collection_name="library")
        rag.ask(graph, "搜索论文 attention", "a")
        result = rag.ask(graph, "下载第一篇并入库", "a")
        self.assertEqual(len(result["messages"]), 4)
        retriever.search.assert_not_called()
        answer.invoke.assert_not_called()

    def test_context_configuration_is_loaded_and_validated(self):
        with patch.dict(os.environ, {"PAPERAGENT_RECENT_TURNS": "2", "PAPERAGENT_CONTEXT_INPUT_TOKENS": "12000"}):
            config = CompressionConfig.from_environment()
        self.assertEqual((config.recent_turns, config.max_input_tokens), (2, 12000))

    def test_registry_copies_isolate_session_bindings(self):
        copied = self.registry.copy()
        self.assertIsNot(copied, self.registry)
        copied._tools.pop("download_paper")
        self.assertIsNotNone(self.registry.get("download_paper"))


if __name__ == "__main__":
    unittest.main()
