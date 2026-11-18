"""正常服务入口接线测试，真实 DeepSeek SDK 通过 MockTransport 在进程内返回。

这些测试检查真实请求的 tools、tool_choice、assistant/tool 消息协议。
HTTP 请求被测试传输层完整截获，不访问 DeepSeek、Bing 或 Embedding 服务。
"""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chromadb
from fastapi.testclient import TestClient
import httpx
from langchain_core.tools import StructuredTool
from langchain_deepseek import ChatDeepSeek

from pdfprocess.ingest_common import PipelineConfig
from pdfprocess.tools.registry import ToolRegistry
from pdfprocess.tools.search_flow import LOCAL_KNOWLEDGE_NOTICE
from pdfprocess.web_search import web_search
from upload_service.app import create_app
from upload_service.benchmark_chat import run_chat_benchmark
from upload_service.config import ServiceSettings


class FixtureEmbedding:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class FixtureEncoder:
    value = -4

    def score(self, question, documents):
        return [self.value] * len(documents)


class ToolCallingServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.settings = ServiceSettings(service_root=root / "service", pipeline=PipelineConfig(
            data_root=root / "ingest", chroma_dir=root / "chroma"))
        client = chromadb.PersistentClient(path=str(root / "chroma"))
        client.get_or_create_collection("tool_calling_test").add(ids=["local"],
            documents=["PRIVATE_LOCAL_EVIDENCE about public research topic."],
            metadatas=[{"source": "PRIVATE_PAPER.pdf", "page": 1}], embeddings=[[1.0, 0.0, 0.0]])
        self.encoder = FixtureEncoder()
        self.requests = []
        self.searches = []
        self.bad_call = False
        self.fail_search = False
        self.results = [{"title": "Public evidence", "url": "https://example.org/public",
                         "content": "Public research temperature is 25 C."}]

        def search(query: str):
            self.searches.append(query)
            if self.fail_search:
                raise RuntimeError("fixture search unavailable")
            return self.results

        self.registry = ToolRegistry()
        self.registry.register(StructuredTool.from_function(search, name="web_search", description="Search public references",
                                                           args_schema=web_search.args_schema), tags=("search",))

        def respond(request):
            payload = json.loads(request.content)
            self.requests.append(payload)
            system = payload["messages"][0]["content"]
            finish_reason = "stop"
            if payload.get("tools"):
                if self.bad_call:
                    message = {"role": "assistant", "content": "Skipping search"}
                else:
                    message = {"role": "assistant", "content": None, "tool_calls": [
                        {"id": "fixture-call", "type": "function", "function": {"name": "web_search",
                            "arguments": json.dumps({"query": "Model organized public query"})}}]}
                    finish_reason = "tool_calls"
            elif payload["messages"][-1]["role"] == "tool":
                message = {"role": "assistant", "content": "Public research uses 25 C. [1]"}
            elif "你负责压缩历史对话" in system:
                message = {"role": "assistant", "content": "PRIVATE_SUMMARY"}
            elif "你只负责检索问题改写" in system:
                message = {"role": "assistant", "content": "Public independent research question"}
            else:
                message = {"role": "assistant", "content": "PRIVATE_LOCAL_ANSWER [1]"}
            return httpx.Response(200, json={"id": "fixture-completion", "object": "chat.completion", "created": 1,
                "model": "deepseek-chat", "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

        self.http = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(self.http.close)

        # 预先构造真实 SDK 客户端，基准工具替换 httpx.Client 时也能复用。
        model = ChatDeepSeek(model="deepseek-chat", api_key="fixture-key", http_client=self.http,
                             max_retries=0, temperature=0, timeout=60)
        patchers = [patch("upload_service.chat.ChatDeepSeek", return_value=model),
                    patch("upload_service.chat.load_dotenv"),
                    patch("upload_service.chat.TongyiEmbeddings", return_value=FixtureEmbedding()),
                    patch("upload_service.chat.OnnxCrossEncoder", return_value=self.encoder),
                    patch("upload_service.chat.create_default_registry", return_value=self.registry),
                    patch.dict(os.environ, {"DEEPSEEK_API": "fixture-key", "DASHSCOPE_API_KEY": "fixture-key",
                                            "PAPERAGENT_CHAT_MODEL": "deepseek-chat"})]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def app(self):
        return create_app(self.settings, runner=lambda task, progress: {"status": "completed", "chunk_count": 0})

    def session(self, client):
        response = client.post("/sessions", json={"collection_name": "tool_calling_test", "mode": "hybrid_rerank",
                                                 "top_k": 1, "rewrite_query": True})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def ask(self, client, session, index=0):
        return client.post("/chat", json={"thread_id": session, "request_id": f"r{index}",
                                           "question": f"Public research question {index}"})

    def test_native_sdk_function_call_protocol_and_idempotency(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            response = self.ask(client, session)
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertTrue(result["answer"].startswith(LOCAL_KNOWLEDGE_NOTICE))
            self.assertEqual(self.searches, ["Model organized public query"])
            self.assertEqual(len(self.requests), 2)
            planning, answer = self.requests
            self.assertEqual([entry["function"]["name"] for entry in planning["tools"]], ["web_search"])
            self.assertEqual(planning["tool_choice"], {"type": "function", "function": {"name": "web_search"}})
            self.assertEqual([message["role"] for message in answer["messages"]], ["system", "user", "assistant", "tool"])
            self.assertEqual(answer["messages"][-1]["tool_call_id"], "fixture-call")
            self.assertIn("25 C", answer["messages"][-1]["content"])
            self.assertNotIn("tools", answer)
            self.assertNotIn("PRIVATE_", json.dumps(self.requests))
            self.assertEqual(result["metrics"]["tool_selection_usage"]["total_tokens"], 15)
            self.assertEqual(result["metrics"]["generation_usage"]["total_tokens"], 15)
            self.assertEqual(self.ask(client, session).status_code, 200)
            self.assertEqual(len(self.requests), 2)

    def test_service_restart_preserves_web_answer_sources_and_full_messages(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            self.assertEqual(self.ask(client, session).status_code, 200)
        with TestClient(self.app()) as client:
            previous = client.get("/sessions/" + session).json()["turns"][0]
            self.assertTrue(previous["answer"].startswith(LOCAL_KNOWLEDGE_NOTICE))
            self.assertEqual(previous["sources"][0]["kind"], "web")
            self.assertEqual(self.ask(client, session, 1).status_code, 200)
            values = client.app.state.chat.checkpointer.get_tuple({"configurable": {"thread_id": session}}).checkpoint["channel_values"]
            self.assertEqual(len(values["messages"]), 4)
            other = self.session(client)
            self.assertEqual(self.ask(client, other).status_code, 200)
            other_values = client.app.state.chat.checkpointer.get_tuple({"configurable": {"thread_id": other}}).checkpoint["channel_values"]
            self.assertEqual(len(other_values["messages"]), 2)

    def test_summary_and_local_evidence_excluded_after_local_conversation(self):
        self.encoder.value = 3
        with TestClient(self.app()) as client:
            session = self.session(client)
            for index in range(4):
                self.assertEqual(self.ask(client, session, index).status_code, 200)
            self.encoder.value = -4
            response = self.ask(client, session, 4)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["metrics"]["summary_calls"], 1)
            planning, answer = self.requests[-2:]
            self.assertNotIn("PRIVATE_", json.dumps([planning, answer]))
            self.assertEqual(response.json()["metrics"]["evidence_origin"], "web")

    def test_invalid_model_call_returns_retryable_error_with_local_notice(self):
        self.bad_call = True
        with TestClient(self.app()) as client:
            session = self.session(client)
            response = self.ask(client, session)
            self.assertEqual(response.status_code, 502)
            self.assertIn(LOCAL_KNOWLEDGE_NOTICE, response.text)
            self.assertEqual(self.searches, [])
            self.bad_call = False
            self.assertEqual(self.ask(client, session).status_code, 200)
            self.assertEqual(len(client.get("/sessions/" + session).json()["turns"]), 1)

    def test_search_failure_and_empty_results_show_local_notice(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            self.fail_search = True
            response = self.ask(client, session)
            self.assertEqual(response.status_code, 502)
            self.assertIn(LOCAL_KNOWLEDGE_NOTICE, response.text)
            self.fail_search = False
            self.results = []
            response = self.ask(client, session)
            self.assertEqual(response.status_code, 200)
            self.assertIn(LOCAL_KNOWLEDGE_NOTICE, response.json()["answer"])
            self.assertIn("联网搜索未找到", response.json()["answer"])

    def test_benchmark_includes_tool_selection_model_cost(self):
        # 只替换基准模块的 httpx 引用，保留真实 SDK 自己使用的 httpx 类型。
        transport = SimpleNamespace(Client=lambda **kwargs: TestClient(self.app()))
        with patch("upload_service.benchmark_chat.httpx", transport):
            report = run_chat_benchmark("http://testserver", [{"questions": ["Public research"]}],
                                        collection="tool_calling_test", top_k=1)
        self.assertEqual(report["completed_turns"], 1, report["records"])
        self.assertEqual(report["tokens"]["tool_selection_usage_total_tokens"], 15)
        self.assertEqual(report["tokens"]["generation_usage_total_tokens"], 15)


if __name__ == "__main__":
    unittest.main()
