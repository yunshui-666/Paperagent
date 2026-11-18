"""本地 OCR、双路融合和真实 ONNX 推理测试，VLM 使用可控响应。"""

import json
from pathlib import Path
from threading import Barrier
import unittest
from unittest.mock import patch

import pymupdf

from pdfprocess.ingest_common import PipelineConfig
from pdfprocess.local_ocr import LocalOCR, validate_lines
from pdfprocess.pdf_fusion import fuse_page, parse_layout
from pdfprocess.pdf_pipeline3 import pdf_to_markdown3
import test_pdf_ingestion as fixture_cases
from test_pdf_ingestion import local_ocr, vision_client


class FusionTests(unittest.TestCase):
    def test_overlapping_regions_assign_each_line_once(self):
        lines = [{"text": "Value 25", "bbox": [10, 10, 60, 30], "confidence": 0.9}]
        layout = [{"type": "text", "bbox": [0, 0, 100, 100], "content": ""},
                  {"type": "title", "bbox": [5, 5, 70, 40], "content": ""}]
        result = fuse_page(lines, layout)
        assigned = [index for block in result["blocks"] for index in block["ocr_line_ids"]]
        self.assertEqual(assigned, [0])
        self.assertEqual(result["blocks"][0]["type"], "title")

    def test_unmatched_text_is_retained(self):
        result = fuse_page([{"text": "Uncovered condition", "bbox": [10, 10, 50, 30], "confidence": 0.9}], [])
        self.assertIn("Uncovered condition", result["markdown"])
        self.assertEqual(result["unmatched_ocr_lines"], 1)
        self.assertFalse(result["blocks"][0]["generated"])

    def test_body_uses_ocr_and_visual_math_keeps_audit_text(self):
        lines = [{"text": "Not above 25 C", "bbox": [0, 0, 90, 30], "confidence": 0.98},
                 {"text": "x = 2", "bbox": [0, 40, 90, 60], "confidence": 0.92}]
        layout = [{"type": "text", "bbox": [0, 0, 100, 35], "content": "Above 35 C"},
                  {"type": "formula", "bbox": [0, 35, 100, 80], "content": "$x = 2$"}]
        result = fuse_page(lines, layout)
        self.assertIn("Not above 25 C", result["markdown"])
        self.assertNotIn("Above 35 C", result["markdown"])
        self.assertIn("$x = 2$", result["markdown"])
        self.assertEqual(result["blocks"][1]["ocr_text"], "x = 2")

    def test_layout_validation_and_exact_deduplication(self):
        block = {"type": "text", "bbox": [0, 0, 1000, 1000], "content": ""}
        self.assertEqual(len(parse_layout(json.dumps({"blocks": [block, block]}), 100, 100)), 1)
        for value in ({}, {"blocks": "bad"}, {"blocks": [{**block, "type": "unknown"}]},
                      {"blocks": [{**block, "bbox": [0, 0, float("nan"), 3]}]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_layout(json.dumps(value), 100, 100)

    def test_ocr_output_validation(self):
        for lines in (None, [{"text": "x", "bbox": [0, 0, 2, 2], "confidence": float("nan")}],
                      [{"text": "x", "bbox": [2, 2, 1, 1], "confidence": 0.8}]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                validate_lines(lines, 100, 100)

    def test_invalid_local_configuration(self):
        for kwargs in ({"threads": 0}, {"threads": True}, {"min_confidence": float("nan")}, {"model_id": "remote-model"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LocalOCR(**kwargs)
        for kwargs in ({"ocr_threads": 0}, {"ocr_match_threshold": 2}, {"ocr_min_confidence": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PipelineConfig(**kwargs)


class DualPipelineCases(unittest.TestCase):
    setUp = fixture_cases.IngestionTests.setUp
    make_pdf = fixture_cases.IngestionTests.make_pdf

    def test_both_routes_enter_concurrently(self):
        barrier = Barrier(2)
        ocr = local_ocr()
        recognized = ocr.recognize.return_value
        def recognize(image):
            barrier.wait(timeout=5)
            return recognized
        ocr.recognize.side_effect = recognize
        vlm = vision_client()
        response = vlm.chat.completions.create.return_value
        def analyze(**kwargs):
            barrier.wait(timeout=5)
            return response
        vlm.chat.completions.create.side_effect = analyze
        result = pdf_to_markdown3(self.pdf, config=self.config, ocr_client=ocr, vlm_client=vlm)
        self.assertEqual(result["status"], "completed")
        page = result["pages"][0]
        self.assertGreaterEqual(page["dual_seconds"], max(page["ocr_seconds"], page["vlm_seconds"]))

    def test_invalid_visual_json_is_saved_and_can_retry(self):
        result = pdf_to_markdown3(self.pdf, config=self.config, ocr_client=local_ocr(), vlm_client=vision_client("bad json"))
        self.assertEqual(result["status"], "failed")
        raw = list(Path(result["artifacts_dir"]).glob("vlm/*.json"))
        self.assertEqual(json.loads(raw[0].read_text(encoding="utf-8"))["raw_response"], "bad json")
        repaired = pdf_to_markdown3(self.pdf, config=self.config, ocr_client=local_ocr(), vlm_client=vision_client())
        self.assertEqual(repaired["status"], "completed")

    def test_blank_page_skips_both_routes(self):
        path = self.root / "blank.pdf"
        with pymupdf.open() as document:
            document.new_page()
            document.save(path)
        ocr, vlm = local_ocr(), vision_client()
        result = pdf_to_markdown3(path, config=self.config, ocr_client=ocr, vlm_client=vlm)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["pages"][0]["blank"])
        ocr.recognize.assert_not_called()
        vlm.chat.completions.create.assert_not_called()

    def test_real_local_ocr_runs_with_outbound_requests_blocked(self):
        with pymupdf.open() as document:
            page = document.new_page(width=600, height=350)
            page.insert_text((40, 100), "PaperAgent Local OCR 12345", fontsize=24)
            image = page.get_pixmap(dpi=144, alpha=False).tobytes("png")
        with patch("requests.get", side_effect=AssertionError("OCR must stay local")):
            lines = LocalOCR().recognize(image)
        text = " ".join(line["text"] for line in lines)
        self.assertIn("PaperAgent", text)
        self.assertIn("12345", text)
        self.assertTrue(all(0 <= line["confidence"] <= 1 for line in lines))


if __name__ == "__main__":
    unittest.main()
