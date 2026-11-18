from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, replace
import hashlib
import io
from pathlib import Path
import os
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import Mock, patch

import chromadb
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from pypdf import PdfWriter

from pdfprocess import ingest_common
from pdfprocess.ingest_common import PipelineConfig, ProcessingBusyError
from pdfprocess.pdf_pipeline1 import ingest_pdf1
from upload_service.app import create_app
from upload_service.benchmark import distribution, percentile
from upload_service.config import ServiceSettings
from upload_service.database import TaskStore
from upload_service.storage import FileStorage, UploadRejected
from upload_service.worker import ParserPool


def pdf_bytes(pages=1):
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=300, height=400)
    stream = io.BytesIO()
    writer.write(stream)
    return stream.getvalue()


def successful_runner(task, progress):
    for stage in ("parsing", "checking_cache", "chunking", "embedding", "saving"):
        progress(stage, {"test": True})
    return {"status": "completed", "source": task["document_name"],
            "paper_id": task["document_id"], "skipped": False, "chunk_count": 1, "metrics": {}}


def wait_task(client, task_id, timeout=10):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        response = client.get(f"/tasks/{task_id}")
        response.raise_for_status()
        task = response.json()
        if task["status"] in {"completed", "failed"}:
            return task
        time.sleep(0.02)
    raise AssertionError("任务未在测试预算内结束")


class UploadServiceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = ServiceSettings(
            service_root=self.root / "service",
            pipeline=PipelineConfig(data_root=self.root / "ingest", chroma_dir=self.root / "chroma")
        )
        self.data = pdf_bytes()

    def test_batch_upload_continues_after_invalid_file(self):
        app = create_app(self.settings, runner=successful_runner)
        with TestClient(app) as client:
            response = client.post("/documents", data={"scheme": 1}, files=[
                ("files", ("a.pdf", self.data, "application/pdf")),
                ("files", ("bad.pdf", b"invalid", "application/pdf")),
                ("files", ("b.pdf", pdf_bytes(2), "application/pdf"))
            ])
            self.assertEqual(response.status_code, 202)
            body = response.json()
            self.assertEqual((body["accepted_count"], body["rejected_count"]), (2, 1))
            for item in body["items"]:
                if item["accepted"]:
                    task = wait_task(client, item["task_id"])
                    self.assertEqual(task["status"], "completed")
                    self.assertEqual(task["scheme"], 1)
                    self.assertEqual(task["collection_name"], "paper_chunks1")

    def test_http_acceptance_precedes_pipeline_completion(self):
        release = Event()

        def blocked_runner(task, progress):
            progress("parsing")
            release.wait(10)
            return successful_runner(task, progress)

        with TestClient(create_app(self.settings, runner=blocked_runner)) as client:
            try:
                response = client.post("/documents", files={"files": ("paper.pdf", self.data)})
                self.assertEqual(response.status_code, 202)
                task_id = response.json()["items"][0]["task_id"]
                task = client.get(f"/tasks/{task_id}").json()
                self.assertIn(task["status"], {"queued", "running"})
            finally:
                release.set()
            self.assertEqual(wait_task(client, task_id)["status"], "completed")

    def test_concurrent_same_content_reuses_active_task(self):
        release = Event()

        def blocked_runner(task, progress):
            release.wait(10)
            return successful_runner(task, progress)

        with TestClient(create_app(replace(self.settings, worker_count=2), runner=blocked_runner)) as client:
            try:
                def submit(index):
                    return client.post("/documents", files={"files": (f"alias-{index}.pdf", self.data)}).json()

                with ThreadPoolExecutor(max_workers=4) as executor:
                    responses = list(executor.map(submit, range(4)))
                items = [response["items"][0] for response in responses]
                self.assertTrue(all(item["accepted"] for item in items))
                self.assertEqual(len({item["task_id"] for item in items}), 1)
                self.assertEqual(sum(item["reused_task"] for item in items), 3)
            finally:
                release.set()
            wait_task(client, items[0]["task_id"])

    def test_finished_file_can_be_resubmitted_as_new_auditable_task(self):
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            first = client.post("/documents", files={"files": ("paper.pdf", self.data)}).json()["items"][0]
            wait_task(client, first["task_id"])
            second = client.post("/documents", files={"files": ("paper.pdf", self.data)}).json()["items"][0]
            wait_task(client, second["task_id"])
            self.assertNotEqual(first["task_id"], second["task_id"])
            self.assertEqual(first["document_id"], second["document_id"])
            self.assertTrue(second["reused_file"])

    def test_invalid_file_is_rejected_without_task(self):
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            response = client.post("/documents", files={"files": ("wrong.txt", self.data)})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["items"][0]["error"]["code"], "invalid_extension")
            self.assertEqual(sum(client.get("/health").json()["tasks"].values()), 0)

    def test_oversized_file_has_413_status(self):
        settings = replace(self.settings, max_file_bytes=10)
        with TestClient(create_app(settings, runner=successful_runner)) as client:
            response = client.post("/documents", files={"files": ("paper.pdf", self.data)})
            self.assertEqual(response.status_code, 413)
            self.assertEqual(response.json()["items"][0]["error"]["code"], "file_too_large")
        self.assertEqual(list((settings.service_root / "files" / "staging").glob("*.part")), [])

    def test_encrypted_pdf_is_rejected(self):
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=400)
        writer.encrypt("password")
        stream = io.BytesIO()
        writer.write(stream)
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            response = client.post("/documents", files={"files": ("encrypted.pdf", stream.getvalue())})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["items"][0]["error"]["code"], "invalid_pdf")

    def test_invalid_scheme_and_missing_files_use_validation_errors(self):
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            self.assertEqual(client.post("/documents", data={"scheme": 9},
                                         files={"files": ("paper.pdf", self.data)}).status_code, 422)
            self.assertEqual(client.post("/documents", data={"scheme": 1}).status_code, 422)

    def test_original_filename_cannot_escape_storage_directory(self):
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            item = client.post("/documents", files={"files": ("../../outside.pdf", self.data)}).json()["items"][0]
            task = wait_task(client, item["task_id"])
            self.assertEqual(task["document_name"], "outside.pdf")
            saved = list((self.settings.service_root / "files").glob("*.pdf"))
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0].stem, hashlib.sha256(self.data).hexdigest())

    def test_pipeline_error_keeps_stage_and_does_not_mark_completed(self):
        def failed_runner(task, progress):
            progress("embedding")
            raise RuntimeError("simulated embedding failure")

        with TestClient(create_app(self.settings, runner=failed_runner)) as client:
            item = client.post("/documents", files={"files": ("paper.pdf", self.data)}).json()["items"][0]
            task = wait_task(client, item["task_id"])
            self.assertEqual(task["status"], "failed")
            self.assertEqual(task["error"]["stage"], "embedding")
            self.assertIsNone(task["result"])

    def test_stage_events_and_time_fields_are_available(self):
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            item = client.post("/documents", files={"files": ("paper.pdf", self.data)}).json()["items"][0]
            task = wait_task(client, item["task_id"])
            self.assertEqual([event["stage"] for event in task["events"]], [
                "queued", "starting", "parsing", "checking_cache", "chunking", "embedding", "saving", "completed"
            ])
            self.assertGreaterEqual(task["queue_wait_seconds"], 0)
            self.assertGreaterEqual(task["processing_seconds"], 0)
            self.assertTrue(task["created_at"].endswith("+00:00"))

    def test_unknown_task_returns_404(self):
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            self.assertEqual(client.get("/tasks/unknown").status_code, 404)

    def test_busy_processing_lock_requeues_then_completes(self):
        calls = []

        def busy_once(task, progress):
            calls.append(task["id"])
            if len(calls) == 1:
                raise ProcessingBusyError("simulated existing parser")
            return successful_runner(task, progress)

        with TestClient(create_app(self.settings, runner=busy_once)) as client:
            item = client.post("/documents", files={"files": ("paper.pdf", self.data)}).json()["items"][0]
            task = wait_task(client, item["task_id"])
            self.assertEqual(task["status"], "completed")
            self.assertEqual(task["attempts"], 2)
            self.assertTrue(any(event["details"].get("reason") == "processing_busy" for event in task["events"]))

    def test_changed_stored_pdf_is_rejected_before_embedding(self):
        path = self.root / "changed.pdf"
        path.write_bytes(self.data)
        with self.assertRaisesRegex(RuntimeError, "文件标识不一致"):
            ingest_pdf1(path, config=self.settings.pipeline, expected_paper_id="wrong-id")

    def test_swagger_and_openapi_expose_multipart_upload(self):
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            self.assertEqual(client.get("/docs").status_code, 200)
            operation = client.get("/openapi.json").json()["paths"]["/documents"]["post"]
            self.assertIn("multipart/form-data", operation["requestBody"]["content"])

    def test_restart_cleans_only_generated_upload_temporary_files(self):
        staging = self.settings.service_root / "files" / "staging"
        staging.mkdir(parents=True)
        stale = staging / ("a" * 32 + ".part")
        stale.write_bytes(b"interrupted upload")
        unrelated = staging / "notes.txt"
        unrelated.write_text("keep", encoding="utf-8")
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            self.assertEqual(client.get("/health").json()["staging_files_cleaned"], 1)
        self.assertFalse(stale.exists())
        self.assertTrue(unrelated.exists())

    def test_restart_requeues_interrupted_task(self):
        store = TaskStore(self.settings.service_root / "service.sqlite3")
        store.initialize()
        document = FileStorage(self.settings.service_root / "files", 100000).save("paper.pdf", io.BytesIO(self.data))
        created = store.enqueue(document, 1, "paper_chunks1", asdict(self.settings.pipeline), "interrupted")
        store.claim()
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            task = wait_task(client, created["task_id"])
            self.assertEqual(task["attempts"], 2)
            self.assertEqual(task["status"], "completed")
            self.assertEqual(client.get("/health").json()["recovered_tasks"], 1)
            self.assertTrue(any(event["details"].get("reason") == "service_restart" for event in task["events"]))

    def test_task_records_survive_normal_restart(self):
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            item = client.post("/documents", files={"files": ("paper.pdf", self.data)}).json()["items"][0]
            wait_task(client, item["task_id"])
        with TestClient(create_app(self.settings, runner=successful_runner)) as client:
            self.assertEqual(client.get(f"/tasks/{item['task_id']}").json()["status"], "completed")

    def test_second_service_instance_cannot_recover_an_active_instance(self):
        with TestClient(create_app(self.settings, runner=successful_runner)):
            with self.assertRaises(RuntimeError):
                with TestClient(create_app(self.settings, runner=successful_runner)):
                    pass

    def test_wal_and_delete_modes_can_be_selected(self):
        for mode in ("WAL", "DELETE"):
            settings = replace(self.settings, service_root=self.root / mode, sqlite_journal_mode=mode)
            with TestClient(create_app(settings, runner=successful_runner)) as client:
                self.assertEqual(client.get("/health").json()["sqlite_journal_mode"], mode)

    def test_atomic_claims_do_not_dispatch_a_task_twice(self):
        store = TaskStore(self.root / "claims.sqlite3")
        store.initialize()
        document = FileStorage(self.root / "files", 100000).save("paper.pdf", io.BytesIO(self.data))
        for index in range(10):
            store.enqueue(document, 1, "test", asdict(self.settings.pipeline), f"job-{index}")
        with ThreadPoolExecutor(max_workers=4) as executor:
            tasks = list(executor.map(lambda _: store.claim(), range(10)))
        self.assertEqual(len({task["id"] for task in tasks}), 10)
        self.assertTrue(all(task["status"] == "running" for task in tasks))

    def test_default_runner_uses_real_child_parser_and_chroma(self):
        import pymupdf

        with pymupdf.open() as document:
            page = document.new_page()
            page.insert_text((72, 72), "PaperAgent upload service integration test.")
            data = document.tobytes()

        class TestEmbeddings(Embeddings):
            def __init__(self):
                self.calls = 0

            def embed_documents(self, texts):
                self.calls += 1
                return [[1.0, 0.5, 0.2] for _ in texts]

            def embed_query(self, text):
                return [1.0, 0.5, 0.2]

        embedding = TestEmbeddings()
        with patch.object(ingest_common, "create_embedding_model", return_value=embedding):
            with TestClient(create_app(self.settings)) as client:
                response = client.post("/documents", data={"scheme": 1}, files={"files": ("original-name.pdf", data)})
                self.assertEqual(response.status_code, 202)
                first_item = response.json()["items"][0]
                first = wait_task(client, first_item["task_id"], timeout=60)
                self.assertEqual(first["status"], "completed", first.get("error"))
                self.assertEqual(first["result"]["source"], "original-name.pdf")
                second_item = client.post("/documents", data={"scheme": 1},
                                          files={"files": ("alias.pdf", data)}).json()["items"][0]
                second = wait_task(client, second_item["task_id"], timeout=60)
                self.assertTrue(second["result"]["skipped"])
                self.assertEqual(embedding.calls, 1)
        collection = chromadb.PersistentClient(path=str(self.settings.pipeline.chroma_dir)).get_collection("paper_chunks1")
        self.assertTrue(all(metadata["source"] == "original-name.pdf" for metadata in collection.get()["metadatas"]))


class UploadUtilityTests(unittest.TestCase):
    def test_percentiles_are_linear_interpolated(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 95), 3.85)
        self.assertIsNone(distribution([])["p95"])

    def test_invalid_settings_are_rejected(self):
        for settings in ({"worker_count": 0}, {"parse_processes": 0},
                         {"max_file_bytes": 0}, {"sqlite_journal_mode": "unknown"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                ServiceSettings(**settings)

    def test_broken_parser_pool_is_replaced_for_next_task(self):
        broken = Mock()
        broken.submit.side_effect = BrokenProcessPool("simulated parser crash")
        healthy = Mock()
        healthy.submit.return_value.result.return_value = {"status": "completed"}
        with patch("upload_service.worker.ProcessPoolExecutor", side_effect=[broken, healthy]):
            pool = ParserPool(1)
            with self.assertRaisesRegex(RuntimeError, "进程池已重置"):
                pool.parse(Mock(), "file", PipelineConfig())
            self.assertEqual(pool.parse(Mock(), "file", PipelineConfig())["status"], "completed")
            pool.stop()
        broken.shutdown.assert_called_once_with(wait=False, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
