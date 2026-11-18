from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import io
import os
from pathlib import Path
import sqlite3
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import patch

import chromadb
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pypdf import PdfWriter

from pdfprocess.ingest_common import PipelineConfig
from pdfprocess.rag import build_graph
from pdfprocess.retrieval import PaperRetriever, RetrievalConfig
from upload_service.app import create_app
from upload_service.application_store import ApplicationStore
from upload_service.chat import MARKDOWN
from upload_service.config import ServiceSettings
from upload_service.documents import ScopedStore
from upload_service.benchmark_chat import run_chat_benchmark


def fixture_pdf():
    document = PdfWriter()
    document.add_blank_page(width=300, height=400)
    stream = io.BytesIO()
    document.write(stream)
    return stream.getvalue()


class FixtureEmbedding:
    def embed_query(self, question):
        return [1.0, 0.0, 0.0]


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = ServiceSettings(service_root=self.root / "service", pipeline=PipelineConfig(
            data_root=self.root / "ingest", chroma_dir=self.root / "chroma"))
        self.client_db = chromadb.PersistentClient(path=str(self.root / "chroma"))
        self.collection = self.client_db.get_or_create_collection("paper_chunks1")
        self.collection.add(ids=["legacy"], documents=["Legacy evidence about solar energy."],
                            metadatas=[{"source": "legacy.pdf", "paper_id": "legacy", "page": 1}],
                            embeddings=[[1.0, 0.0, 0.0]])
        self.calls = []
        self.release = None
        self.entered = Event()
        self.fail_next = False

    def graph_factory(self, session, checkpointer):
        def answer(values):
            self.calls.append(values)
            self.entered.set()
            if self.release:
                self.release.wait(10)
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("controlled model failure")
            history = " | ".join(message.content for message in values["history"])
            return AIMessage(content=f"Evidence [1]. History {history}",
                             usage_metadata={"input_tokens": 8, "output_tokens": 4, "total_tokens": 12})
        raw = self.collection
        from langchain_chroma import Chroma
        vector = Chroma(client=self.client_db, collection_name=raw.name, embedding_function=FixtureEmbedding())
        database = ApplicationStore(self.settings.service_root / "service.sqlite3")
        scoped = ScopedStore(vector, FixtureEmbedding(), database, raw.name, session["options"]["document_ids"])
        return build_graph(retriever=PaperRetriever(scoped, RetrievalConfig(mode="dense", top_k=1)),
                           answer_chain=RunnableLambda(answer), checkpointer=checkpointer,
                           include_source_text=True, history_limit=12, max_context_chars=16000)

    def app(self, runner=None):
        return create_app(self.settings, graph_factory=self.graph_factory, runner=runner or self.runner)

    def runner(self, task, progress):
        progress("saving")
        chunk_id = task["document_id"] + ":0"
        self.collection.upsert(ids=[chunk_id], documents=["Registered paper evidence about batteries."],
                               metadatas=[{"paper_id": task["document_id"], "page": 1, "source": "paper.pdf"}],
                               embeddings=[[1.0, 0.0, 0.0]])
        return {"status": "completed", "chunk_count": 1}

    def upload(self, client):
        response = client.post("/documents", data={"scheme": "1"}, files={"files": ("paper.pdf", fixture_pdf())})
        self.assertEqual(response.status_code, 202, response.text)
        item = response.json()["items"][0]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            task = client.get("/tasks/" + item["task_id"]).json()
            if task["status"] in {"completed", "failed"}:
                self.assertEqual(task["status"], "completed", task)
                return item
            time.sleep(0.02)
        self.fail("fixture task timeout")

    def session(self, client, **options):
        response = client.post("/sessions", json={"collection_name": "paper_chunks1", "mode": "dense", "top_k": 1, **options})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def ask(self, client, session_id, question="What is the evidence?", request_id="request-1"):
        return client.post("/chat", json={"thread_id": session_id, "request_id": request_id, "question": question})

    def test_home_static_and_schema(self):
        with TestClient(self.app()) as client:
            self.assertIn("PaperAgent", client.get("/").text)
            self.assertEqual(client.get("/assets/app.js").status_code, 200)
            self.assertIn("/chat", client.get("/openapi.json").json()["paths"])

    def test_checkpoint_restart_and_thread_isolation(self):
        with TestClient(self.app()) as client:
            first = self.session(client)
            second = self.session(client)
            self.assertEqual(self.ask(client, first, "Remember Alpha").status_code, 200)
            self.assertEqual(self.ask(client, second, "Remember Beta").status_code, 200)
        with TestClient(self.app()) as client:
            response = self.ask(client, first, "What did I ask?", "request-2")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("Remember Alpha", response.json()["answer"])
            self.assertNotIn("Remember Beta", response.json()["answer"])
            self.assertEqual(len(client.get("/sessions/" + first).json()["turns"]), 2)

    def test_idempotent_request_and_question_conflict(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            first = self.ask(client, session).json()
            second = self.ask(client, session).json()
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(len(self.calls), 1)
            self.assertEqual(self.ask(client, session, "Different question").status_code, 409)

    def test_failed_turn_can_retry_without_duplicate_history(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            self.fail_next = True
            self.assertEqual(self.ask(client, session).status_code, 502)
            self.assertEqual(self.ask(client, session).status_code, 200)
            checkpoint = client.app.state.chat.checkpointer.get_tuple({"configurable": {"thread_id": session}})
            self.assertEqual(len(checkpoint.checkpoint["channel_values"]["messages"]), 2)
            self.assertEqual(len(client.get("/sessions/" + session).json()["turns"]), 1)

    def test_older_failed_retry_follows_actual_conversation_order(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            self.fail_next = True
            self.assertEqual(self.ask(client, session, "Failed first").status_code, 502)
            self.assertEqual(self.ask(client, session, "Succeeded second", "second").status_code, 200)
            self.assertEqual(self.ask(client, session, "Failed first").status_code, 200)
            questions = [turn["question"] for turn in client.get("/sessions/" + session).json()["turns"]]
            self.assertEqual(questions, ["Succeeded second", "Failed first"])

    def test_same_session_concurrent_request_is_rejected(self):
        self.release = Event()
        with TestClient(self.app()) as client, ThreadPoolExecutor(max_workers=1) as executor:
            session = self.session(client)
            running = executor.submit(self.ask, client, session)
            try:
                self.assertTrue(self.entered.wait(10))
                self.assertEqual(self.ask(client, session, "Other", "other").status_code, 409)
            finally:
                self.release.set()
            self.assertEqual(running.result().status_code, 200)

    def test_checkpoint_completion_recovers_business_commit_failure(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            store = client.app.state.task_store
            with patch.object(store, "finish_turn", side_effect=sqlite3.OperationalError("controlled commit failure")):
                self.assertEqual(self.ask(client, session).status_code, 502)
            self.assertEqual(self.ask(client, session).status_code, 200)
            self.assertEqual(len(self.calls), 1)

    def test_startup_recovers_orphan_completed_checkpoint(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            self.ask(client, session)
            with client.app.state.task_store._transaction() as connection:
                connection.execute("UPDATE turns SET status='running',answer=NULL,sources_json=NULL")
        with TestClient(self.app()) as client:
            turn = client.get("/sessions/" + session).json()["turns"][0]
            self.assertEqual(turn["status"], "completed")
            self.assertEqual(len(self.calls), 1)

    def test_old_checkpoint_result_recovers_after_a_later_turn(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            store = client.app.state.task_store
            with patch.object(store, "finish_turn", side_effect=sqlite3.OperationalError("controlled failure")):
                self.assertEqual(self.ask(client, session, "Alpha").status_code, 502)
            self.assertEqual(self.ask(client, session, "Beta", "second").status_code, 200)
            self.assertEqual(self.ask(client, session, "Alpha").status_code, 200)
            self.assertEqual(len(self.calls), 2)

    def test_different_sessions_can_generate_concurrently(self):
        self.release = Event()
        with TestClient(self.app()) as client, ThreadPoolExecutor(max_workers=2) as executor:
            first, second = self.session(client), self.session(client)
            futures = [executor.submit(self.ask, client, session) for session in (first, second)]
            try:
                deadline = time.monotonic() + 10
                while len(self.calls) < 2 and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertEqual(len(self.calls), 2)
            finally:
                self.release.set()
            self.assertEqual([future.result().status_code for future in futures], [200, 200])

    def test_startup_marks_incomplete_turn_retryable(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            client.app.state.task_store.reserve_turn(session, "orphan", "Unfinished")
        with TestClient(self.app()) as client:
            self.assertEqual(client.get("/sessions/" + session).json()["turns"][0]["status"], "failed")
            self.assertEqual(self.ask(client, session, "Unfinished", "orphan").status_code, 200)

    def test_missing_checkpoint_rebuilds_from_business_history(self):
        with TestClient(self.app()) as client:
            session = self.session(client)
            self.ask(client, session, "Alpha")
            client.app.state.chat.checkpointer.delete_thread(session)
        with TestClient(self.app()) as client:
            response = self.ask(client, session, "Follow up", "second")
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("Alpha", response.json()["answer"])

    def test_document_scope_pdf_link_and_metrics(self):
        with TestClient(self.app()) as client:
            document = self.upload(client)
            session = self.session(client, document_ids=[document["document_id"]])
            response = self.ask(client, session).json()
            self.assertEqual(len(response["sources"]), 1)
            self.assertEqual(response["sources"][0]["document_id"], document["document_id"])
            self.assertEqual(response["metrics"]["generation_usage"]["total_tokens"], 12)
            url = response["sources"][0]["file_url"]
            self.assertEqual(client.get(url).headers["content-type"], "application/pdf")

    def test_delete_removes_vectors_original_cache_and_can_revive(self):
        with TestClient(self.app()) as client:
            document = self.upload(client)
            document_id = document["document_id"]
            cache = self.settings.pipeline.data_root / document_id
            cache.mkdir(parents=True)
            (cache / "test.json").write_text("{}")
            response = client.delete("/documents/" + document_id)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertFalse(cache.exists())
            self.assertEqual(self.collection.count(), 1)
            self.assertEqual(client.get("/documents").json()["items"], [])
            self.assertEqual(client.get(f"/documents/{document_id}/file").status_code, 404)
            self.assertEqual(self.upload(client)["document_id"], document_id)

    def test_active_target_cannot_be_deleted_other_document_stays_readable(self):
        release = Event()
        def blocked(task, progress):
            release.wait(10)
            return self.runner(task, progress)
        with TestClient(self.app(runner=blocked)) as client:
            try:
                item = client.post("/documents", files={"files": ("pending.pdf", fixture_pdf())}).json()["items"][0]
                self.assertEqual(client.delete("/documents/" + item["document_id"]).status_code, 409)
                session = self.session(client)
                self.assertEqual(self.ask(client, session).json()["sources"][0]["chunk_id"], "legacy")
            finally:
                release.set()

    def test_deleting_doc_is_hidden_and_cleanup_resumes_after_restart(self):
        with TestClient(self.app()) as client:
            item = self.upload(client)
            client.app.state.task_store.begin_delete(item["document_id"])
            counts = client.get("/collections").json()["items"][0]
            self.assertEqual(counts["ready_chunk_count"], 1)
            self.assertEqual(counts["chunk_count"], 2)
        with TestClient(self.app()) as client:
            self.assertEqual(client.get("/documents").json()["items"], [])
            self.assertEqual(self.collection.count(), 1)

    def test_reindex_endpoint_and_rename_delete_session(self):
        with TestClient(self.app()) as client:
            item = self.upload(client)
            response = client.post(f"/documents/{item['document_id']}/reindex", json={"scheme": 1, "refresh": True})
            self.assertEqual(response.status_code, 202)
            session = self.session(client)
            self.ask(client, session)
            self.assertEqual(client.patch("/sessions/" + session, json={"title": "Study"}).json()["title"], "Study")
            self.assertEqual(client.delete("/sessions/" + session).status_code, 200)
            self.assertEqual(client.get("/sessions/" + session).status_code, 404)
            self.assertIsNone(client.app.state.chat.checkpointer.get_tuple({"configurable": {"thread_id": session}}))

    def test_scope_validation_empty_question_and_unknown_resources(self):
        with TestClient(self.app()) as client:
            self.assertEqual(client.post("/sessions", json={"collection_name": "absent"}).status_code, 404)
            self.assertEqual(client.post("/sessions", json={"collection_name": "paper_chunks1", "document_ids": ["absent"]}).status_code, 422)
            session = self.session(client)
            self.assertEqual(self.ask(client, session, "  ").status_code, 422)
            self.assertEqual(self.ask(client, "unknown").status_code, 404)

    def test_schema_migrates_preexisting_task_database(self):
        root = self.root / "migration.sqlite3"
        with closing(sqlite3.connect(root)) as connection:
            connection.execute("CREATE TABLE documents(id TEXT PRIMARY KEY,name TEXT NOT NULL,path TEXT NOT NULL,size_bytes INTEGER NOT NULL,page_count INTEGER NOT NULL,created_at REAL NOT NULL)")
            connection.execute("INSERT INTO documents VALUES ('legacy','old.pdf','old.pdf',10,1,0)")
            connection.commit()
        database = ApplicationStore(root)
        database.initialize()
        database.initialize()
        self.assertEqual(database.document("legacy")["lifecycle"], "active")

    def test_markdown_disables_raw_html_script_links_and_images(self):
        rendered = MARKDOWN.render('<script>alert(1)</script>\n\n[bad](javascript:alert(1))\n\n![image](https://example.com/a.png)')
        self.assertNotIn("<script>", rendered)
        self.assertNotIn('href="javascript:', rendered)
        self.assertNotIn("<img", rendered)

    def test_bm25_scope_excludes_unready_managed_chunks(self):
        with TestClient(self.app()) as client:
            item = self.upload(client)
            from langchain_chroma import Chroma
            vector = Chroma(client=self.client_db, collection_name=self.collection.name, embedding_function=None)
            scoped = ScopedStore(vector, None, client.app.state.task_store, self.collection.name, [item["document_id"]])
            retriever = PaperRetriever(scoped, RetrievalConfig(mode="bm25", top_k=1))
            self.assertEqual(retriever.search("batteries")[0].document.metadata["paper_id"], item["document_id"])
            client.app.state.task_store.begin_delete(item["document_id"])
            self.assertEqual(retriever.search("batteries"), [])

    def test_chat_benchmark_records_latency_usage_and_ordered_history(self):
        with patch("upload_service.benchmark_chat.httpx.Client", return_value=TestClient(self.app())):
            report = run_chat_benchmark("http://testserver", [{"name": "study", "questions": ["Alpha", "Follow up"]}],
                                        collection="paper_chunks1", mode="dense", top_k=1)
        self.assertEqual(report["completed_turns"], 2)
        self.assertEqual(report["tokens"]["generation_usage_total_tokens"], 24)
        self.assertEqual(report["client_latency_seconds"]["count"], 2)
        self.assertIn("Alpha", report["records"][1]["result"]["answer"])

    def test_chat_benchmark_rejects_invalid_questions(self):
        with self.assertRaises(ValueError):
            run_chat_benchmark("http://testserver", [{"questions": [""]}], collection="paper_chunks1")

    def test_web_sources_are_returned_by_api_and_survive_restart(self):
        from test_web_search import FixtureRetriever, local_hit, WEB_RESULTS
        def factory(session, checkpoint):
            return build_graph(retriever=FixtureRetriever([local_hit(-3)]),
                answer_chain=RunnableLambda(lambda values: "Web evidence [1]"),
                web_search_tool=RunnableLambda(lambda values: WEB_RESULTS), checkpointer=checkpoint)
        with TestClient(create_app(self.settings, runner=self.runner, graph_factory=factory)) as client:
            session = self.session(client, mode="hybrid_rerank")
            answer = self.ask(client, session).json()
            self.assertEqual(answer["metrics"]["evidence_origin"], "web")
            self.assertEqual(answer["sources"][0]["url"], WEB_RESULTS[0]["url"])
            self.assertIsNone(answer["sources"][0]["file_url"])
        with TestClient(create_app(self.settings, runner=self.runner, graph_factory=factory)) as client:
            turn = client.get("/sessions/" + session).json()["turns"][0]
            self.assertEqual(turn["sources"][0]["kind"], "web")
            self.assertEqual(len(client.get("/sessions/" + session).json()["turns"]), 1)

    def test_web_failure_is_retryable_through_chat_api(self):
        from test_web_search import FixtureRetriever, local_hit, WEB_RESULTS
        attempts = []
        def search(values):
            attempts.append(values)
            if len(attempts) == 1:
                raise RuntimeError("controlled search failure")
            return WEB_RESULTS
        def factory(session, checkpoint):
            return build_graph(retriever=FixtureRetriever([local_hit(-3)]),
                answer_chain=RunnableLambda(lambda values: "Web evidence [1]"),
                web_search_tool=RunnableLambda(search), checkpointer=checkpoint)
        with TestClient(create_app(self.settings, runner=self.runner, graph_factory=factory)) as client:
            session = self.session(client, mode="hybrid_rerank")
            self.assertEqual(self.ask(client, session).status_code, 502)
            self.assertEqual(self.ask(client, session).status_code, 200)
            turns = client.get("/sessions/" + session).json()["turns"]
            self.assertEqual(len(turns), 1)
            self.assertEqual(turns[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
