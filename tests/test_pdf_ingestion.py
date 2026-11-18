import contextlib
from dataclasses import replace
import importlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import uuid4

import chromadb
import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_chroma import Chroma

from pdfprocess import embeddings, ingest_common, pdf_vision, rag
from pdfprocess.retrieval import RetrievalConfig
from pdfprocess.compare_ingest import main as compare_main
from pdfprocess.ingest_common import PipelineConfig
from pdfprocess.pdf_pipeline1 import (
    batch_upload_pdfs1, ingest_pdf1, ingest_pdfs1, pdf_to_markdown1
)
from pdfprocess.pdf_pipeline2 import pdf_to_markdown2, ingest_pdf2
from pdfprocess.pdf_pipeline3 import pdf_to_markdown3, ingest_pdf3
from pdfprocess.pdf_fusion import parse_layout
from pdfprocess.vector_store import save_chunks


TEXT = "PaperAgent evaluates retrieval augmented generation with evidence. " * 8
OCR_TEXT = json.dumps({"blocks": [{"type": "text", "bbox": [0, 0, 1000, 1000], "content": ""}]})


def local_ocr(text="Recognized heading. Recognized paper text."):
    return SimpleNamespace(recognize=Mock(return_value=[{"text": text, "bbox": [20, 20, 600, 100], "confidence": 0.99}]))


def vision_client(text=OCR_TEXT, finish_reason="stop"):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=20, completion_tokens=30)
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))))
    return client


class FakeEmbeddings(Embeddings):
    def __init__(self):
        self.calls = 0

    def embed_documents(self, texts):
        self.calls += 1
        return [[1.0, float(len(text) % 13 + 1), 0.5] for text in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class IngestionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(
            dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.config = PipelineConfig(data_root=self.root / "artifacts", chroma_dir=self.root / "chroma")
        self.pdf = self.make_pdf("paper.pdf")
        self.embedding = FakeEmbeddings()
        self.client = chromadb.EphemeralClient()
        name = "test_" + uuid4().hex
        self.collection = self.client.create_collection(
            name, metadata={"embedding_model": embeddings.EMBEDDING_MODEL}, embedding_function=None
        )
        self.addCleanup(self.client.delete_collection, name)

    def make_pdf(self, filename, pages=1, text=TEXT):
        path = self.root / filename
        with pymupdf.open() as document:
            for index in range(pages):
                page = document.new_page(width=400, height=600)
                page.insert_textbox(pymupdf.Rect(30, 30, 370, 570), f"Page {index + 1}\n" + text)
            document.save(path)
        return path

    def kwargs(self):
        return {"config": self.config, "embedding_model": self.embedding, "collection": self.collection}

    def test_all_scheme_entrypoints_exist(self):
        for scheme in (1, 2, 3):
            module = importlib.import_module(f"pdfprocess.pdf_pipeline{scheme}")
            for name in ("pdf_to_markdown", "ingest_pdf", "ingest_pdfs", "upload_pdf", "batch_upload_pdfs"):
                self.assertTrue(callable(getattr(module, f"{name}{scheme}")))

    def test_local_parser_uses_real_pdf_and_cache(self):
        first = pdf_to_markdown1(self.pdf, config=self.config)
        second = pdf_to_markdown1(self.pdf, config=self.config)
        self.assertEqual(first["status"], "completed")
        self.assertIn("PaperAgent", first["pages"][0]["markdown"])
        self.assertEqual(first["metrics"]["ocr_calls"], 0)
        self.assertEqual(second["metrics"]["cached_pages"], 1)
        self.assertTrue(Path(first["markdown_path"]).is_file())

    def test_same_file_renamed_has_same_identity(self):
        renamed = self.root / "renamed.pdf"
        renamed.write_bytes(self.pdf.read_bytes())
        first = ingest_pdf1(self.pdf, **self.kwargs())
        second = ingest_pdf1(renamed, **self.kwargs())
        self.assertEqual(first["paper_id"], second["paper_id"])
        self.assertTrue(second["skipped"])
        self.assertEqual(self.embedding.calls, 1)

    def test_full_ingestion_uses_real_chroma_and_retains_metadata(self):
        result = ingest_pdf1(self.pdf, **self.kwargs())
        rows = self.collection.get(include=["metadatas", "documents"])
        self.assertEqual(result["chunk_count"], self.collection.count())
        self.assertEqual(rows["metadatas"][0]["page"], 1)
        self.assertEqual(rows["metadatas"][0]["source"], "paper.pdf")
        self.assertEqual(rows["metadatas"][0]["scheme"], 1)
        self.assertIn("paper_id", rows["metadatas"][0])
        query = self.collection.query(query_embeddings=[[1.0, 2.0, 0.5]], n_results=1)
        self.assertIn("PaperAgent", query["documents"][0][0])

    def test_repeat_ingestion_skips_embedding_and_writes(self):
        first = ingest_pdf1(self.pdf, **self.kwargs())
        second = ingest_pdf1(self.pdf, **self.kwargs())
        self.assertFalse(first["skipped"])
        self.assertTrue(second["skipped"])
        self.assertEqual(self.embedding.calls, 1)
        self.assertEqual(self.collection.count(), first["chunk_count"])

    def test_new_split_configuration_replaces_old_chunks_for_same_paper(self):
        first = ingest_pdf1(self.pdf, **self.kwargs())
        config = replace(self.config, chunk_size=60, chunk_overlap=10)
        second = ingest_pdf1(self.pdf, **{**self.kwargs(), "config": config})
        self.assertGreater(second["chunk_count"], first["chunk_count"])
        self.assertEqual(self.collection.count(), second["chunk_count"])
        self.assertFalse(set(first["chunk_ids"]) & set(self.collection.get()["ids"]))

    def test_real_ingestion_connects_to_existing_rag_workflow(self):
        ingest_pdf1(self.pdf, **self.kwargs())
        vector_store = Chroma(client=self.client, collection_name=self.collection.name,
                              embedding_function=self.embedding, create_collection_if_not_exists=False)
        inputs = []

        def answer(values):
            inputs.append(values)
            return "PaperAgent evidence [1]"

        graph = rag.build_graph(vector_store=vector_store, answer_chain=RunnableLambda(answer),
                                retrieval_config=RetrievalConfig(mode="dense", dense_k=5, top_k=5))
        result = rag.ask(graph, "What does PaperAgent evaluate?", "ingestion-test")
        self.assertIn("PaperAgent", inputs[0]["context"])
        self.assertEqual(result["sources"][0]["source"], "paper.pdf")
        self.assertEqual(len(result["messages"]), 2)

    def test_default_collections_are_isolated_in_persistent_chroma(self):
        output = self.root / "all-schemes.json"
        with patch.object(ingest_common, "create_embedding_model", return_value=self.embedding):
            with patch.object(pdf_vision, "get_client", return_value=vision_client()), patch.object(pdf_vision, "_ocr_client", return_value=local_ocr()):
                status = compare_main([
                    "--pdfs", str(self.pdf), "--schemes", "1", "2", "3",
                    "--data-root", str(self.config.data_root),
                    "--chroma-dir", str(self.config.chroma_dir), "--output", str(output)
                ])
        self.assertEqual(status, 0)
        results = json.loads(output.read_text(encoding="utf-8"))["results"]
        self.assertEqual([item["collection_name"] for item in results], [
            "paper_chunks1", "paper_chunks2", "paper_chunks3"
        ])
        client = chromadb.PersistentClient(path=str(self.config.chroma_dir))
        for index in (1, 2, 3):
            rows = client.get_collection(f"paper_chunks{index}").get()
            self.assertTrue(all(item["scheme"] == index for item in rows["metadatas"]))

    def test_missing_vector_is_repaired(self):
        first = ingest_pdf1(self.pdf, **self.kwargs())
        self.collection.delete(ids=[first["chunk_ids"][0]])
        repaired = ingest_pdf1(self.pdf, **self.kwargs())
        self.assertFalse(repaired["skipped"])
        self.assertEqual(self.collection.count(), first["chunk_count"])

    def test_batch_paths_continue_after_bad_file(self):
        results = ingest_pdfs1([self.root / "missing.pdf", self.pdf], **self.kwargs())
        self.assertEqual([item["status"] for item in results], ["failed", "completed"])

    def test_batch_uploads_accept_same_filename_and_different_contents(self):
        other = self.make_pdf("other.pdf", text=TEXT + "Different paper.")
        results = batch_upload_pdfs1([
            ("paper.pdf", b"invalid PDF"),
            ("paper.pdf", self.pdf.read_bytes()),
            ("paper.pdf", other.read_bytes())
        ], **self.kwargs())
        self.assertEqual([item["status"] for item in results], ["failed", "completed", "completed"])
        self.assertNotEqual(results[1]["paper_id"], results[2]["paper_id"])

    def test_upload_basename_stays_in_data_root(self):
        path = ingest_common.save_uploaded_pdf("../../outside.pdf", self.pdf.read_bytes(), config=self.config)
        self.assertTrue(path.is_relative_to(self.config.data_root.resolve()))
        self.assertEqual(path.name, "outside.pdf")

    def test_batch_api_rejects_single_string(self):
        with self.assertRaises(TypeError):
            ingest_pdfs1(str(self.pdf), **self.kwargs())
        with self.assertRaises(TypeError):
            batch_upload_pdfs1("file.pdf", **self.kwargs())

    def test_local_ocr_scheme_recognizes_every_page(self):
        path = self.make_pdf("two-pages.pdf", pages=2)
        client = local_ocr()
        result = pdf_to_markdown2(path, config=self.config, ocr_client=client)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"]["ocr_calls"], 2)
        self.assertEqual(client.recognize.call_count, 2)
        self.assertEqual(result["metrics"]["vlm_calls"], 0)

    def test_dual_text_page_executes_both_routes(self):
        ocr, vlm = local_ocr(), vision_client()
        result = pdf_to_markdown3(self.pdf, config=self.config, ocr_client=ocr, vlm_client=vlm)
        self.assertEqual(result["status"], "completed")
        ocr.recognize.assert_called_once()
        vlm.chat.completions.create.assert_called_once()
        self.assertEqual(result["pages"][0]["method"], "local_ocr_vlm_dual")
        self.assertIn("Recognized paper text", result["pages"][0]["markdown"])

    def test_dual_scanned_page_uses_local_ocr_and_vlm(self):
        path = self.root / "scan.pdf"
        with pymupdf.open(self.pdf) as source:
            image_bytes = source[0].get_pixmap().tobytes("png")
        with pymupdf.open() as document:
            page = document.new_page(width=400, height=600)
            page.insert_image(page.rect, stream=image_bytes)
            document.save(path)
        result = pdf_to_markdown3(path, config=self.config, ocr_client=local_ocr(), vlm_client=vision_client())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["metrics"]["ocr_calls"], 1)
        self.assertEqual(result["metrics"]["vlm_calls"], 1)

    def test_dual_image_region_keeps_pixels_and_provenance(self):
        text = json.dumps({"blocks": [{"type": "image", "bbox": [100, 100, 500, 500], "content": "Figure description."}]})
        result = pdf_to_markdown3(self.pdf, config=self.config, ocr_client=local_ocr(), vlm_client=vision_client(text))
        self.assertEqual(result["status"], "completed")
        block = result["pages"][0]["blocks"][0]
        self.assertTrue(block["generated"])
        self.assertEqual(block["bbox"], [80, 120, 400, 600])
        self.assertEqual(block["provenance"], "vlm")
        self.assertTrue(Path(result["pages"][0]["image_path"]).is_file())
        self.assertTrue(Path(result["pages"][0]["ocr_path"]).is_file())
        self.assertTrue(Path(result["pages"][0]["vlm_path"]).is_file())

    def test_local_ocr_does_not_call_vlm(self):
        table = "| Metric | Value |\n| --- | --- |\n| Recall | 0.8 |"
        vlm = vision_client()
        result = pdf_to_markdown2(self.pdf, config=self.config, ocr_client=local_ocr(table), vlm_client=vlm)
        self.assertIn(table, result["pages"][0]["markdown"])
        vlm.chat.completions.create.assert_not_called()

    def test_dual_table_structure_keeps_ocr_audit_text(self):
        table = "| Metric | Value |\n| --- | --- |\n| Recall | 0.8 |"
        client = local_ocr()
        client.recognize.return_value = [
            {"text": "Before", "bbox": [20, 20, 180, 80], "confidence": 0.99},
            {"text": "Recall 0.8", "bbox": [20, 130, 300, 180], "confidence": 0.98},
            {"text": "After", "bbox": [20, 300, 300, 360], "confidence": 0.99}]
        layout = {"blocks": [{"type": "text", "bbox": [0, 0, 1000, 100], "content": "Wrong text"},
            {"type": "table", "bbox": [0, 100, 1000, 200], "content": table},
            {"type": "text", "bbox": [0, 200, 1000, 1000], "content": ""}]}
        result = pdf_to_markdown3(self.pdf, config=self.config, ocr_client=client, vlm_client=vision_client(json.dumps(layout)))
        markdown = result["pages"][0]["markdown"]
        self.assertEqual(result["status"], "completed")
        self.assertIn(table, markdown)
        self.assertNotIn("Wrong text", markdown)
        self.assertLess(markdown.index("Before"), markdown.index(table))
        self.assertLess(markdown.index(table), markdown.index("After"))
        self.assertIn("Recall 0.8", result["pages"][0]["blocks"][1]["ocr_text"])

    def test_truncated_vlm_fails_before_embedding(self):
        with self.assertRaisesRegex(RuntimeError, "PDF 解析失败"):
            ingest_pdf3(self.pdf, **self.kwargs(), ocr_client=local_ocr(), vlm_client=vision_client(finish_reason="length"))
        self.assertEqual(self.embedding.calls, 0)
        self.assertEqual(self.collection.count(), 0)

    def test_successful_pages_are_cached_when_failed_page_is_retried(self):
        path = self.make_pdf("retry.pdf", pages=2)
        client = local_ocr()
        good = client.recognize.return_value
        client.recognize.side_effect = [good, TimeoutError("simulated timeout")]
        failed = pdf_to_markdown2(path, config=self.config, ocr_client=client)
        self.assertEqual(failed["status"], "failed")
        client.recognize.side_effect = None
        repaired = pdf_to_markdown2(path, config=self.config, ocr_client=client)
        self.assertEqual(repaired["status"], "completed")
        self.assertEqual(repaired["metrics"]["cached_pages"], 1)
        self.assertEqual(repaired["metrics"]["ocr_calls"], 1)

    def test_embedding_error_writes_no_vectors(self):
        self.embedding.embed_documents = Mock(side_effect=TimeoutError("simulated timeout"))
        with self.assertRaises(TimeoutError):
            ingest_pdf1(self.pdf, **self.kwargs())
        self.assertEqual(self.collection.count(), 0)

    def test_partial_write_failure_is_idempotent_on_retry(self):
        config = replace(self.config, chunk_size=40, chunk_overlap=5)
        kwargs = {**self.kwargs(), "config": config}
        real_save = save_chunks

        def fail_after_first(chunks, vectors, **options):
            real_save(chunks[:1], vectors[:1], **options)
            raise OSError("simulated write failure")

        with patch.object(ingest_common, "save_chunks", side_effect=fail_after_first):
            with self.assertRaises(OSError):
                ingest_pdf1(self.pdf, **kwargs)
        self.assertEqual(self.collection.count(), 1)
        result = ingest_pdf1(self.pdf, **kwargs)
        self.assertEqual(self.collection.count(), result["chunk_count"])
        self.assertGreater(result["chunk_count"], 1)

    def test_invalid_vectors_fail_before_writing(self):
        chunks = [Document(page_content="test", metadata={"chunk_id": "test"})]
        for vectors in ([[float("nan")]], [[]], []):
            with self.subTest(vectors=vectors):
                with self.assertRaises(ValueError):
                    save_chunks(chunks, vectors, collection=self.collection)
        self.assertEqual(self.collection.count(), 0)

    def test_different_embedding_model_is_rejected(self):
        config = replace(self.config, embedding_model_id="different-model")
        with self.assertRaisesRegex(ValueError, "模型"):
            ingest_pdf1(self.pdf, **{**self.kwargs(), "config": config})

    def test_custom_model_identifier_requires_matching_embedding_object(self):
        config = replace(self.config, embedding_model_id="different-model")
        with self.assertRaisesRegex(ValueError, "embedding_model"):
            ingest_pdf1(self.pdf, config=config, collection=self.collection)

    def test_parse_refresh_invalidates_old_completed_marker(self):
        first = ingest_pdf2(self.pdf, **self.kwargs(), ocr_client=local_ocr())
        changed = local_ocr("Updated paper evidence.")
        refreshed = pdf_to_markdown2(self.pdf, config=replace(self.config, use_cache=False), ocr_client=changed)
        second = ingest_pdf2(self.pdf, **self.kwargs(), ocr_client=changed)
        self.assertNotEqual(first["content_hash"], refreshed["content_hash"])
        self.assertFalse(second["skipped"])
        self.assertIn("Updated paper evidence", self.collection.get()["documents"][0])

    def test_processing_lock_rejects_concurrent_same_task(self):
        lock = self.root / ".lock"
        with ingest_common.processing_lock(lock):
            with self.assertRaises(RuntimeError):
                with ingest_common.processing_lock(lock):
                    self.fail("second lock was acquired")

    def test_comparison_cli_writes_machine_readable_report(self):
        output = self.root / "report.json"
        status = compare_main([
            "--pdfs", str(self.pdf), "--schemes", "1", "--parse-only",
            "--data-root", str(self.config.data_root), "--output", str(output)
        ])
        self.assertEqual(status, 0)
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["results"][0]["status"], "completed")
        self.assertTrue(report["parse_only"])

    def test_rag_can_select_comparison_collection(self):
        with patch.object(rag, "load_dotenv"), patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test"}):
            with patch.object(rag, "Chroma") as chroma:
                rag.build_graph(answer_chain=Mock(), collection_name="paper_chunks3")
        self.assertEqual(chroma.call_args.kwargs["collection_name"], "paper_chunks3")


class LayoutAndEmbeddingTests(unittest.TestCase):
    def test_normalized_coordinates_are_scaled(self):
        text = json.dumps({"blocks": [{"type": "image", "bbox": [100, 200, 500, 600], "content": ""}]})
        blocks = parse_layout(text, 2000, 3000)
        self.assertEqual(blocks[0]["bbox"], [200, 600, 1000, 1800])

    def test_pixel_layout_coordinates_are_supported(self):
        text = json.dumps({"blocks": [{"type": "text", "bbox": [10, 20, 50, 60], "content": "Text"}]})
        self.assertEqual(parse_layout(text, 200, 300, "pixels")[0]["bbox"], [10, 20, 50, 60])

    def test_model_reading_order_is_preserved(self):
        text = json.dumps({"blocks": [{"type": "text", "bbox": [0, 500, 400, 800], "content": "Left column end"},
                                    {"type": "text", "bbox": [500, 0, 1000, 300], "content": "Right column start"}]})
        self.assertEqual(parse_layout(text, 1000, 1000)[0]["content"], "Left column end")

    def test_invalid_layout_fails(self):
        for coordinates in ([5, 5, 1, 1], [0, 0], [True, 0, 5, 5]):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                parse_layout(json.dumps({"blocks": [{"type": "image", "bbox": coordinates, "content": ""}]}), 1000, 1000)

    def test_shared_embedding_batches_and_reorders_results(self):
        def response(**kwargs):
            size = len(kwargs["input"])
            return SimpleNamespace(status_code=200, output={"embeddings": [
                {"index": index, "embedding": [float(index)]} for index in reversed(range(size))
            ]})

        with patch.object(embeddings.MultiModalEmbedding, "call", side_effect=response) as call:
            result = embeddings.TongyiEmbeddings(api_key="test").embed_documents(["text"] * 9)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(result[:8], [[float(index)] for index in range(8)])


if __name__ == "__main__":
    unittest.main()
