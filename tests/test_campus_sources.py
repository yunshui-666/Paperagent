"""校园论文来源的协议、官网边界、下载权限和自动入队回归。"""
from contextlib import contextmanager
import io
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import requests

from pdfprocess.tools.campus_papers import CampusPaperClient, PaperAccessError, canonical_id, provider_of
from pdfprocess.tools.paper_agent import clean_catalog, run_paper_agent
from pdfprocess.tools.papers import create_download_tool
from pdfprocess.tools.wanfang_protocol import MESSAGES, decode_search, encode_search
import test_paper_tools as agent_fixtures
from langchain_core.messages import AIMessage


PAPER_URL = "https://d.wanfangdata.com.cn/periodical/fixture001"
PAPERS = agent_fixtures.PAPERS
SequenceModel = agent_fixtures.SequenceModel
call = agent_fixtures.call


def response(payload, *, status=200, headers=None):
    result = requests.Response()
    result.status_code = status
    result._content = payload
    result._content_consumed = True
    result.headers.update(headers or {})
    result.encoding = "utf-8"
    result.url = PAPER_URL
    return result


def session(*responses):
    result = MagicMock()
    result.__enter__.return_value = result
    result.cookies = requests.cookies.RequestsCookieJar()
    result.headers = {}
    result.get.side_effect = responses
    return result


def search_payload():
    result = MESSAGES["SearchResponse"]()
    result.status = True
    result.count = 1
    resource = result.resources.add()
    resource.type = "Periodical"
    resource.periodical.id = "fixture001"
    resource.periodical.titles.append("<span>知识图谱</span>测试论文")
    resource.periodical.abstracts.append("Synthetic scientific evidence.")
    resource.periodical.year = 2026
    payload = result.SerializeToString()
    return b"\x00"+struct.pack(">I",len(payload))+payload


class CampusClientTests(unittest.TestCase):
    def temporary(self):
        value = tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"),ignore_cleanup_errors=True)
        self.addCleanup(value.cleanup)
        return Path(value.name)

    def test_only_cnki_and_wanfang_article_urls_are_accepted(self):
        cnki = "https://kns.cnki.net/kcms2/article/abstract?filename=TEST&dbcode=CJFD"
        self.assertEqual(canonical_id(cnki+"#page=2"),cnki)
        self.assertEqual(canonical_id(PAPER_URL),PAPER_URL)
        self.assertEqual(provider_of(cnki),"cnki")
        for value in (None,"","https://arxiv.org/abs/1706.03762","https://cnki.net.evil.test/kcms/x",
                      "https://localhost/a.pdf","https://s.wanfangdata.com.cn/paper","https://x:y@kns.cnki.net/kcms/x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_id(value)

    def test_wanfang_request_and_response_use_real_wire_shape(self):
        encoded = encode_search("知识图谱",3)
        request = MESSAGES["SearchRequest"]()
        request.ParseFromString(encoded[5:])
        self.assertEqual(request.common.search_word,"知识图谱")
        self.assertEqual(request.common.page_size,3)
        fake = session()
        fake.post.return_value = response(search_payload())
        client = CampusPaperClient()
        with patch.object(client,"_session",return_value=fake):
            result = client.search("知识图谱",3,source="wanfang")
        self.assertEqual(result["papers"][0]["paper_id"],PAPER_URL)
        self.assertEqual(result["papers"][0]["title"],"知识图谱测试论文")
        self.assertEqual(result["errors"],[])

    def test_incomplete_compressed_and_failed_grpc_responses_are_rejected(self):
        for payload in (b"",b"\x00\0\0\0\x05x",b"\x01\0\0\0\x00"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                decode_search(payload)
        trailer=b"grpc-status: 7\r\n"
        with self.assertRaises(RuntimeError):
            decode_search(search_payload()+b"\x80"+struct.pack(">I",len(trailer))+trailer)

    def test_partial_provider_failure_preserves_other_results(self):
        client = CampusPaperClient()
        with patch.object(client,"_search_wanfang",return_value=PAPERS), patch.object(client,"_search_cnki",side_effect=RuntimeError("verification required")):
            result=client.search("知识图谱",5)
        self.assertEqual(result["papers"],PAPERS)
        self.assertEqual(result["errors"][0]["source"],"cnki")

    def test_cnki_index_results_are_restricted_to_official_paper_pages(self):
        cnki="https://www.cnki.com.cn/Article/CJFDTotal-TEST202601001.htm"
        records=[{"title":"论文","url":cnki,"content":"摘要"},{"title":"其他","url":"https://example.com/Article/x","content":"x"}]
        with patch("pdfprocess.tools.campus_papers.BingWebSearch") as search:
            search.return_value.search.return_value=records
            result=CampusPaperClient().search("检索",3,source="cnki")
        self.assertEqual(len(result["papers"]),1)
        self.assertEqual(result["papers"][0]["paper_id"],cnki)

    def test_official_pdf_download_is_streamed_and_bounded(self):
        fake=session(response(b'<meta name="citation_pdf_url" content="/download/a.pdf">'),response(b"%PDF-1.7 fixture"))
        client=CampusPaperClient()
        with patch.object(client,"_session",return_value=fake):
            with client.open_pdf(PAPER_URL,max_bytes=1000,directory=self.temporary()/"downloads") as stream:
                self.assertTrue(stream.read().startswith(b"%PDF-"))
        large=session(response(b"%PDF-"+b"x"*1001))
        with patch.object(client,"_session",return_value=large), self.assertRaises(PaperAccessError):
            with client.open_pdf(PAPER_URL,max_bytes=1000,directory=self.temporary()):
                pass

    def test_redirect_to_other_sources_is_rejected_before_request(self):
        fake=session(response(b"",status=302,headers={"Location":"https://example.com/a.pdf"}))
        client=CampusPaperClient()
        with patch.object(client,"_session",return_value=fake), self.assertRaises(PaperAccessError):
            with client.open_pdf(PAPER_URL,max_bytes=1000,directory=self.temporary()):
                pass
        self.assertEqual(fake.get.call_count,1)

    def test_permission_and_captcha_errors_never_submit_ingestion(self):
        for body,status in ((b"",401),("安全验证 captchaType".encode(),200)):
            client=CampusPaperClient()
            fake=session(response(body,status=status))
            ingestion=Mock()
            ingestion.storage.root=self.temporary()
            ingestion.settings.max_file_bytes=1000
            tool=create_download_tool(ingestion,Mock(),"library",client=client,wait_seconds=0)
            with patch.object(client,"_session",return_value=fake), self.assertRaises(PaperAccessError):
                tool.invoke({"paper_id":PAPER_URL})
            ingestion.submit.assert_not_called()

    def test_download_submits_to_selected_library_and_reports_pending(self):
        @contextmanager
        def open_pdf(*args,**kwargs):
            yield io.BytesIO(b"fixture")
        ingestion=Mock()
        ingestion.storage.root=self.temporary()
        ingestion.settings.max_file_bytes=1000
        ingestion.submit.return_value={"task_id":"task","document_id":"doc"}
        store=Mock()
        store.get.return_value={"id":"task","status":"running","result":None,"error":None}
        tool=create_download_tool(ingestion,store,"selected",client=SimpleNamespace(open_pdf=open_pdf),wait_seconds=0)
        result=tool.invoke({"paper_id":PAPER_URL})
        self.assertFalse(result["ingested"])
        self.assertEqual(result["status"],"running")
        self.assertEqual(ingestion.submit.call_args.kwargs["collection_name"],"selected")

    def test_legacy_unsupported_catalog_is_skipped_without_network(self):
        self.assertEqual(clean_catalog([{"title":"old","url":"https://example.com/paper"}]),[])


class CampusAgentRegressionTests(unittest.TestCase):
    def setUp(self):
        agent_fixtures.PaperAgentTests.setUp(self)

    def test_new_search_and_download_does_not_use_previous_topic(self):
        model=SequenceModel([call("search_papers",{"query":"new topic"},"s"),
                             call("download_paper",{"paper_id":PAPER_URL},"d"),AIMessage(content="完成")])
        run_paper_agent({"question":"搜索新主题论文并下载第一篇","paper_catalog":PAPERS},
                        model=model,registry=self.registry,collection_name="library")
        self.assertEqual(model.bindings[0]["tool_choice"],"search_papers")

    def test_partial_source_errors_are_visible_in_final_answer(self):
        self.search.return_value={"papers":PAPERS,"errors":[{"source":"cnki","message":"需要站点验证"}]}
        model=SequenceModel([call("search_papers",{"query":"知识图谱"},"s"),AIMessage(content="已找到论文")])
        result=run_paper_agent({"question":"搜索知识图谱论文"},model=model,registry=self.registry,collection_name="library")
        self.assertIn("需要站点验证",result["answer"])
        self.assertEqual(result["metrics"]["paper_search_errors"][0]["source"],"cnki")
