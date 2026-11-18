"""验证正常上传服务问答入口的压缩接线，所有模型调用由本地函数代替。"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chromadb
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pdfprocess.ingest_common import PipelineConfig
from upload_service.app import create_app
from upload_service.benchmark_chat import run_chat_benchmark
from upload_service.config import ServiceSettings


class CompressionServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.settings = ServiceSettings(service_root=root / "service", pipeline=PipelineConfig(
            data_root=root / "ingest", chroma_dir=root / "chroma"))
        # 使用真实 Chroma 与 BM25，省去外部 Embedding 和答案服务。
        client = chromadb.PersistentClient(path=str(root / "chroma"))
        collection = client.get_or_create_collection("compression_test")
        collection.add(ids=["sample"], documents=["Alpha research uses 25 C. " * 30],
                       metadatas=[{"source": "sample.pdf", "page": 1}], embeddings=[[1.0, 0.0]])
        self.inputs = []
        self.summary_calls = 0
        self.fail_summary = False

        def model(value, **kwargs):
            received = value.to_messages()
            if "你负责压缩历史对话" in received[0].content:
                if self.fail_summary:
                    raise RuntimeError("controlled summary failure")
                self.summary_calls += 1
                content = "Alpha research, temperature 25 C, user asks study conditions."
            elif "你只负责检索问题改写" in received[0].content:
                content = "What are the Alpha research conditions?"
            else:
                self.inputs.append(received)
                content = "Alpha research uses 25 C. [1]"
            return AIMessage(content=content, usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})

        # 走 ChatService._graph 的生产分支，仅替换模型客户端，不注入 graph_factory。
        patchers = [patch("upload_service.chat.ChatDeepSeek", return_value=RunnableLambda(model)),
                    patch("upload_service.chat.load_dotenv"),
                    patch.dict(os.environ, {"DEEPSEEK_API": "fixture-key"})]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def app(self):
        return create_app(self.settings, runner=lambda task, progress: {"status": "completed", "chunk_count": 0})

    def session(self, client):
        response = client.post("/sessions", json={"collection_name": "compression_test", "mode": "bm25",
                                                "top_k": 1, "rewrite_query": True})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def ask(self, client, session, index, question=None):
        return client.post("/chat", json={"thread_id": session, "request_id": f"r{index}",
                                           "question": question or f"Alpha research question {index}"})

    def test_normal_entry_persists_summary_metrics_and_full_history(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            for index in range(5):
                response = self.ask(client, session, index)
                self.assertEqual(response.status_code, 200, response.text)
            metrics = response.json()["metrics"]
            self.assertEqual(metrics["summary_calls"], 1)
            self.assertEqual(metrics["summary_usage"][0]["total_tokens"], 15)
            self.assertEqual(metrics["generation_usage"]["input_tokens"], 10)
            self.assertGreater(metrics["evidence_duplicate_units"], 0)
            self.assertLessEqual(metrics["input_tokens_after_compression"], 16000)
            self.assertIn("历史对话摘要", self.inputs[-1][1].content)
            self.assertEqual(len(client.get("/sessions/" + session).json()["turns"]), 5)
            count = len(self.inputs)
            self.assertEqual(self.ask(client, session, 4).status_code, 200)
            self.assertEqual(len(self.inputs), count)
        with TestClient(self.app()) as client:
            response = self.ask(client, session, 5)
            self.assertEqual(response.status_code, 200, response.text)
            values = client.app.state.chat.checkpointer.get_tuple({"configurable": {"thread_id": session}}).checkpoint["channel_values"]
            self.assertEqual(len(values["messages"]), 12)
            self.assertEqual(values["summary_covered"], 4)

    def test_summary_failure_retries_through_api(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            for index in range(4):
                self.assertEqual(self.ask(client, session, index).status_code, 200)
            self.fail_summary = True
            self.assertEqual(self.ask(client, session, 4).status_code, 502)
            self.fail_summary = False
            self.assertEqual(self.ask(client, session, 4).status_code, 200)
            turns = client.get("/sessions/" + session).json()["turns"]
            self.assertEqual(len(turns), 5)
            self.assertTrue(all(turn["status"] == "completed" for turn in turns))

    def test_long_question_returns_actionable_422(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            response = self.ask(client, session, 0, "长" * 1000)
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn("缩短", response.text)
            self.assertEqual(self.inputs, [])

    def test_benchmark_counts_summary_rewrite_and_generation_usage(self):
        with patch("upload_service.benchmark_chat.httpx.Client", return_value=TestClient(self.app())):
            report = run_chat_benchmark("http://testserver", [{"questions": [f"Alpha research {i}" for i in range(5)]}],
                                        collection="compression_test", mode="bm25", top_k=1)
        self.assertEqual(report["completed_turns"], 5)
        self.assertEqual(report["tokens"]["summary_usage_total_tokens"], 15)
        self.assertEqual(report["tokens"]["generation_usage_total_tokens"], 75)
        self.assertEqual(report["tokens"]["rewrite_usage_total_tokens"], 60)


if __name__ == "__main__":
    unittest.main()
