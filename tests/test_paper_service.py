"""论文工具通过真实服务进入 PDF 管线与 Chroma，所有外部响应均在本地提供。"""

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chromadb
from fastapi.testclient import TestClient
import httpx
from langchain_deepseek import ChatDeepSeek
import pymupdf

from pdfprocess.embeddings import EMBEDDING_MODEL
from pdfprocess.ingest_common import PipelineConfig
from pdfprocess.pdf_pipeline3 import ingest_pdf3
from pdfprocess.tools.papers import CAMPUS_CLIENT
from upload_service.app import create_app
from upload_service.config import ServiceSettings
from test_pdf_ingestion import local_ocr, vision_client
from test_paper_tools import PAPERS


class FixtureEmbedding:
    def embed_query(self, text):
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]


class PaperServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = ServiceSettings(service_root=self.root / "service", pipeline=PipelineConfig(
            data_root=self.root / "ingest", chroma_dir=self.root / "chroma"))
        self.client_db = chromadb.PersistentClient(path=str(self.root / "chroma"))
        self.collection = self.client_db.get_or_create_collection("papers_test", metadata={"embedding_model": EMBEDDING_MODEL})
        self.collection.add(ids=["legacy"], documents=["Original local evidence."],
                            metadatas=[{"source": "legacy.pdf", "paper_id": "legacy"}], embeddings=[[0.0, 1.0, 0.0]])
        with pymupdf.open() as document:
            page = document.new_page(width=400, height=600)
            page.insert_text((30, 80), "Synthetic downloaded scientific evidence. Temperature is 25 C.")
            self.pdf_bytes = document.tobytes()
        self.requests = []
        self.fail_ingestion = False

        @contextmanager
        def open_pdf(*args, **kwargs):
            yield io.BytesIO(self.pdf_bytes)

        def respond(request):
            payload = json.loads(request.content)
            self.requests.append(payload)
            system = payload["messages"][0]["content"]
            finish = "stop"
            if "你负责执行论文搜索与下载入库任务" in system and payload.get("tools"):
                human = next(item["content"] for item in reversed(payload["messages"]) if item["role"] == "user")
                last = payload["messages"][-1]
                choice = payload.get("tool_choice")
                forced = choice.get("function", {}).get("name") if isinstance(choice, dict) else None
                if forced == "download_paper" or (last["role"] == "tool" and last.get("name") == "search_papers" and "下载" in human):
                    name, args = "download_paper", {"paper_id": "https://d.wanfangdata.com.cn/periodical/fixture001"}
                elif last["role"] != "tool":
                    name, args = "search_papers", {"query": "attention", "max_results": 1}
                else:
                    name, args = None, None
                if name:
                    message = {"role": "assistant", "content": None, "tool_calls": [{"id": f"call-{len(self.requests)}",
                        "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
                    finish = "tool_calls"
                else:
                    message = {"role": "assistant", "content": "论文 https://d.wanfangdata.com.cn/periodical/fixture001 的工具结果如上。"}
            elif "你负责压缩历史对话" in system:
                message = {"role": "assistant", "content": "论文目录包含 https://d.wanfangdata.com.cn/periodical/fixture001，用户在比较 attention。"}
            elif "你只负责检索问题改写" in system:
                message = {"role": "assistant", "content": "What is the temperature in the downloaded scientific evidence?"}
            else:
                message = {"role": "assistant", "content": "Temperature is 25 C. [1]"}
            return httpx.Response(200, json={"id": "fixture-completion", "object": "chat.completion", "created": 1,
                "model": "deepseek-chat", "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})

        self.http = httpx.Client(transport=httpx.MockTransport(respond))
        self.addCleanup(self.http.close)
        model = ChatDeepSeek(model="deepseek-chat", api_key="fixture", http_client=self.http, max_retries=0)
        patchers = [patch("upload_service.chat.ChatDeepSeek", return_value=model), patch("upload_service.chat.load_dotenv"),
                    patch("upload_service.chat.TongyiEmbeddings", return_value=FixtureEmbedding()),
                    patch.object(CAMPUS_CLIENT, "search", return_value=PAPERS),
                    patch.object(CAMPUS_CLIENT, "open_pdf", side_effect=open_pdf),
                    patch.dict(os.environ, {"DEEPSEEK_API": "fixture", "DASHSCOPE_API_KEY": "fixture"})]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def runner(self, task, progress):
        if self.fail_ingestion:
            raise RuntimeError("controlled embedding failure")
        return ingest_pdf3(task["path"], config=PipelineConfig(**task["config"]), source_name=task["document_name"],
                           embedding_model=FixtureEmbedding(), ocr_client=local_ocr("Synthetic downloaded scientific evidence. Temperature is 25 C."),
                           vlm_client=vision_client(), progress_callback=progress, expected_paper_id=task["document_id"])

    def app(self):
        return create_app(self.settings, runner=self.runner)

    def session(self, client):
        response = client.post("/sessions", json={"collection_name": "papers_test", "mode": "dense", "top_k": 1})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def ask(self, client, session, question, request_id):
        return client.post("/chat", json={"thread_id": session, "question": question, "request_id": request_id})

    def test_download_automatically_chunks_embeds_and_becomes_retrievable(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            response = self.ask(client, session, "搜索论文 attention 并下载第一篇入库", "download")
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            downloaded = result["metrics"]["paper_downloads"][0]
            self.assertTrue(downloaded["ingested"])
            self.assertEqual(downloaded["collection_name"], "papers_test")
            self.assertGreater(downloaded["chunk_count"], 0)
            task = client.get("/tasks/" + downloaded["task_id"]).json()
            self.assertEqual(task["status"], "completed")
            rows = self.collection.get(where={"paper_id": downloaded["document_id"]})
            self.assertTrue(rows["ids"])
            self.assertEqual(rows["metadatas"][0]["parse_method"], "local_ocr_vlm_dual")
            self.assertIn("Temperature is 25 C", rows["documents"][0])
            count = len(self.requests)
            self.assertEqual(self.ask(client, session, "搜索论文 attention 并下载第一篇入库", "download").status_code, 200)
            self.assertEqual(len(self.requests), count)
            answer = self.ask(client, session, "What is the temperature?", "question")
            self.assertEqual(answer.status_code, 200, answer.text)
            self.assertEqual(answer.json()["sources"][0]["document_id"], downloaded["document_id"])

    def test_paper_catalog_survives_service_restart_for_followup_download(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            self.assertEqual(self.ask(client, session, "搜索论文 attention", "search").status_code, 200)
        with TestClient(self.app()) as client:
            result = self.ask(client, session, "下载刚才第一篇并入库", "download")
            self.assertEqual(result.status_code, 200, result.text)
            self.assertTrue(result.json()["metrics"]["paper_downloads"][0]["ingested"])

    def test_failed_ingestion_is_reported_as_failed(self):
        self.fail_ingestion = True
        with TestClient(self.app()) as client:
            session = self.session(client)
            response = self.ask(client, session, "下载 https://d.wanfangdata.com.cn/periodical/fixture001 并入库", "download")
            self.assertEqual(response.status_code, 200, response.text)
            item = response.json()["metrics"]["paper_downloads"][0]
            self.assertEqual(item["status"], "failed")
            self.assertFalse(item["ingested"])
            self.assertIn("failed", response.json()["answer"])

    def test_long_dialogue_is_summarized_and_model_inputs_stay_bounded(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            for index in range(7):
                response = self.ask(client, session, f"搜索论文 attention topic {index}", str(index))
                self.assertEqual(response.status_code, 200, response.text)
            metrics = response.json()["metrics"]
            self.assertGreater(metrics["summary_calls"], 0)
            self.assertTrue(all(value <= metrics["input_token_budget"] for value in metrics["paper_agent_input_tokens"]))
            values = client.app.state.chat.checkpointer.get_tuple({"configurable": {"thread_id": session}}).checkpoint["channel_values"]
            self.assertEqual(len(values["messages"]), 14)
            self.assertTrue(values["conversation_summary"])
            other = self.session(client)
            self.assertEqual(self.ask(client, other, "搜索论文 attention", "new").status_code, 200)
            other_values = client.app.state.chat.checkpointer.get_tuple({"configurable": {"thread_id": other}}).checkpoint["channel_values"]
            self.assertEqual(len(other_values["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
