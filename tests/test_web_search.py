import contextlib
import io
import math
import unittest
from unittest.mock import MagicMock, Mock, patch

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
import requests

from pdfprocess import rag
from pdfprocess.retrieval import RetrievalConfig, RetrievalHit
from pdfprocess.web_search import BingWebSearch, WebSearchError, clean_results, web_search


WEB_RESULTS = [{"title": "Public reference", "url": "https://example.org/reference",
                "content": "WEB_ONLY_EVIDENCE"}]
RSS = b'''<?xml version="1.0"?><rss version="2.0"><channel><item>
<title>Public reference</title><link>https://example.org/reference</link>
<description>WEB_ONLY_EVIDENCE</description></item></channel></rss>'''


def local_hit(logit=-2):
    return RetrievalHit(Document(id="local", page_content="LOCAL_ONLY_EVIDENCE",
                                 metadata={"source": "private.pdf", "paper_id": "local"}),
                        "local", cross_encoder_score=logit)


class FixtureRetriever:
    def __init__(self, hits, mode="hybrid_rerank"):
        self.hits = hits
        self.config = RetrievalConfig(mode=mode)
        self.queries = []

    def search(self, question):
        self.queries.append(question)
        return self.hits


def response(payload=RSS):
    result = MagicMock()
    result.__enter__.return_value = result
    result.iter_content.return_value = [payload]
    return result


class WebSearchToolTests(unittest.TestCase):
    def test_rss_search_and_langchain_tool_entrypoint(self):
        with patch("pdfprocess.web_search.requests.get", return_value=response()) as request:
            result = web_search.invoke({"query": "independent question"})
        self.assertEqual(result, WEB_RESULTS)
        self.assertEqual(request.call_args.kwargs["params"]["q"], "independent question")
        self.assertEqual(request.call_args.args[0], "https://www.bing.com/search")

    def test_html_deduplication_and_invalid_urls(self):
        results = [{"title": "<b>Page</b>", "url": "https://example.org/a", "content": "<p>Evidence &amp; text</p>"},
                   {"title": "Duplicate", "url": "https://example.org/a#part", "content": "duplicate"},
                   {"title": "Invalid", "url": "javascript:alert(1)", "content": "bad"},
                   {"title": "Credentials", "url": "https://user:password@example.org/", "content": "bad"},
                   {"url": "file:///secret", "content": "bad"},
                   {"url": "https://example.org/empty", "content": " "}, None]
        self.assertEqual(clean_results(results), [{"title": "Page", "url": "https://example.org/a", "content": "Evidence & text"}])

    def test_query_validation(self):
        for query in (None, 123, "", "  "):
            with self.subTest(query=query), self.assertRaises(ValueError):
                BingWebSearch().search(query)

    def test_invalid_rss_and_non_rss_xml_are_errors(self):
        for payload in (b"<html>captcha", b"<html/>"):
            with self.subTest(payload=payload), patch("pdfprocess.web_search.requests.get", return_value=response(payload)):
                with self.assertRaises(WebSearchError):
                    BingWebSearch().search("query")

    def test_response_size_is_bounded(self):
        with patch("pdfprocess.web_search.requests.get", return_value=response(b"x" * (1024 * 1024 + 1))):
            with self.assertRaisesRegex(WebSearchError, "大小"):
                BingWebSearch().search("query")

    def test_temporary_connection_failure_retries_once(self):
        with patch("pdfprocess.web_search.requests.get", side_effect=[requests.Timeout(), response()]) as request:
            with patch("pdfprocess.web_search.time.sleep"):
                self.assertEqual(BingWebSearch().search("query"), WEB_RESULTS)
        self.assertEqual(request.call_count, 2)

    def test_exhausted_timeout_is_an_error(self):
        with patch("pdfprocess.web_search.requests.get", side_effect=requests.Timeout()) as request:
            with patch("pdfprocess.web_search.time.sleep"), self.assertRaises(WebSearchError):
                BingWebSearch().search("query")
        self.assertEqual(request.call_count, 2)

    def test_permanent_http_error_does_not_retry(self):
        value = response()
        value.status_code = 403
        value.raise_for_status.side_effect = requests.HTTPError(response=value)
        with patch("pdfprocess.web_search.requests.get", return_value=value) as request:
            with self.assertRaises(WebSearchError):
                BingWebSearch().search("query")
        self.assertEqual(request.call_count, 1)

    def test_empty_rss_and_result_limit(self):
        with patch("pdfprocess.web_search.requests.get", return_value=response(b"<rss><channel/></rss>")):
            self.assertEqual(BingWebSearch().search("query"), [])
        results = [{"title": str(index), "url": f"https://example.org/{index}", "content": "text"} for index in range(10)]
        self.assertEqual(len(clean_results(results)), 5)


class WebFallbackGraphTests(unittest.TestCase):
    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.retriever = FixtureRetriever([local_hit()])
        self.search = Mock()
        self.search.invoke.return_value = WEB_RESULTS
        self.answer = Mock()
        self.answer.invoke.return_value = "Answer from current evidence [1]"

    def graph(self, **kwargs):
        return rag.build_graph(retriever=self.retriever, answer_chain=self.answer,
                               web_search_tool=self.search, include_source_text=True, **kwargs)

    def test_sigmoid_is_stable_and_retains_raw_logits(self):
        for value, expected in ((-1000, 0.0), (0, 0.5), (1000, 1.0)):
            hit = local_hit(value)
            self.assertEqual(hit.rerank_relevance, expected)
            self.assertEqual(hit.cross_encoder_score, value)
        with self.assertRaises(ValueError):
            _ = local_hit(float("nan")).rerank_relevance

    def test_below_threshold_uses_web_context_and_sources_only(self):
        result = rag.ask(self.graph(), "Question", "one")
        payload = self.answer.invoke.call_args.args[0]
        self.assertNotIn("LOCAL_ONLY_EVIDENCE", payload["context"])
        self.assertIn("WEB_ONLY_EVIDENCE", payload["context"])
        self.assertEqual(payload["history"], [])
        self.assertTrue(all(source["kind"] == "web" for source in result["sources"]))
        self.assertEqual(result["metrics"]["evidence_origin"], "web")
        self.assertLess(result["metrics"]["local_max_relevance"], 0.5)
        self.assertEqual(result["metrics"]["web_fallback_reason"], "low_relevance")

    def test_exact_threshold_and_high_scores_use_local(self):
        for score in (0.0, 3.0):
            self.retriever.hits = [local_hit(score)]
            result = rag.ask(self.graph(), "Question", str(score))
            self.assertEqual(result["metrics"]["evidence_origin"], "local")
            self.assertIn("LOCAL_ONLY_EVIDENCE", result["context"])
            self.assertGreaterEqual(result["sources"][0]["rerank_relevance"], 0.5)
        self.search.invoke.assert_not_called()

    def test_gate_uses_maximum_score(self):
        self.retriever.hits = [local_hit(-3), local_hit(2)]
        result = rag.ask(self.graph(), "Question", "one")
        self.assertEqual(result["evidence_origin"], "local")
        self.search.invoke.assert_not_called()

    def test_no_local_candidates_searches_web(self):
        self.retriever.hits = []
        result = rag.ask(self.graph(), "Question", "one")
        self.assertEqual(result["metrics"]["web_fallback_reason"], "no_local_results")
        self.assertEqual(result["sources"][0]["url"], WEB_RESULTS[0]["url"])

    def test_web_failure_leaves_checkpoint_without_local_evidence(self):
        self.search.invoke.side_effect = WebSearchError("controlled network failure")
        graph = self.graph()
        with self.assertRaisesRegex(RuntimeError, "联网搜索失败"):
            rag.ask(graph, "Question", "one")
        state = graph.get_state({"configurable": {"thread_id": "one"}}).values
        self.assertEqual(state["context"], "")
        self.assertEqual(state["sources"], [])
        self.assertEqual(state["answer"], "")
        self.answer.invoke.assert_not_called()
        self.search.invoke.side_effect = None
        result = rag.ask(graph, "Question", "one")
        self.assertEqual(len(result["messages"]), 2)

    def test_empty_web_results_skip_generation(self):
        self.search.invoke.return_value = []
        result = rag.ask(self.graph(), "Question", "one")
        self.assertEqual(result["sources"], [])
        self.assertIn("联网搜索未找到", result["answer"])
        self.answer.invoke.assert_not_called()

    def test_previous_local_answer_is_excluded_from_web_generation(self):
        graph = self.graph()
        self.retriever.hits = [local_hit(4)]
        self.answer.invoke.return_value = "HISTORICAL_LOCAL_ANSWER"
        rag.ask(graph, "First question", "one")
        self.retriever.hits = [local_hit(-4)]
        self.answer.invoke.return_value = "Web answer [1]"
        result = rag.ask(graph, "Second question", "one")
        payload = self.answer.invoke.call_args.args[0]
        self.assertEqual(payload["history"], [])
        self.assertNotIn("LOCAL_", str(payload))
        self.assertEqual(len(result["messages"]), 4)

    def test_independent_rewritten_question_is_used_for_search_and_answer(self):
        rewriter = Mock()
        rewriter.invoke.return_value = "Independent public question"
        graph = self.graph(query_rewriter=rewriter)
        self.retriever.hits = [local_hit(4)]
        rag.ask(graph, "Topic", "one")
        self.retriever.hits = [local_hit(-4)]
        rag.ask(graph, "What about it?", "one")
        self.search.invoke.assert_called_once_with({"query": "Independent public question"})
        self.assertEqual(self.answer.invoke.call_args.args[0]["question"], "Independent public question")

    def test_next_local_turn_resets_route(self):
        graph = self.graph()
        rag.ask(graph, "Web question", "one")
        self.retriever.hits = [local_hit(3)]
        result = rag.ask(graph, "Local question", "one")
        self.assertEqual(result["evidence_origin"], "local")
        self.assertNotIn("web_search_seconds", result["metrics"])
        self.assertNotIn("WEB_ONLY_EVIDENCE", result["context"])

    def test_checkpointer_rebuild_and_thread_isolation(self):
        checkpoint = InMemorySaver()
        first = self.graph(checkpointer=checkpoint)
        rag.ask(first, "Thread A", "a")
        second = self.graph(checkpointer=checkpoint)
        self.retriever.hits = [local_hit(4)]
        result = rag.ask(second, "Followup", "a")
        self.assertEqual(len(result["messages"]), 4)
        result = rag.ask(second, "Thread B", "b")
        self.assertEqual(len(result["messages"]), 2)

    def test_context_budget_and_unsafe_results(self):
        self.search.invoke.return_value = [
            {"title": "Unsafe", "url": "javascript:alert(1)", "content": "bad"},
            {"title": "Safe", "url": "https://example.org/reference", "content": "a" * 300}]
        result = rag.ask(self.graph(max_context_chars=50), "Question", "one")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(len(result["sources"][0]["text"]), 50)

    def test_invalid_thresholds_are_rejected(self):
        for threshold in (-0.1, 1.1, True, "0.5", math.nan, math.inf):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.graph(web_search_threshold=threshold)

    def test_missing_rerank_scores_fail_before_answer(self):
        self.retriever.hits = [local_hit(None)]
        with self.assertRaisesRegex(RuntimeError, "知识库检索失败"):
            rag.ask(self.graph(), "Question", "one")
        self.answer.invoke.assert_not_called()
        self.search.invoke.assert_not_called()

    def test_plain_retrieval_modes_keep_local_route(self):
        for mode in ("dense", "bm25", "hybrid"):
            self.retriever.config = RetrievalConfig(mode=mode)
            result = rag.ask(self.graph(), "Question", mode)
            self.assertEqual(result["evidence_origin"], "local")
        self.search.invoke.assert_not_called()

    def test_explicit_tool_disable_supports_local_experiments(self):
        graph = rag.build_graph(retriever=self.retriever, answer_chain=self.answer, web_search_tool=None)
        result = rag.ask(graph, "Question", "one")
        self.assertEqual(result["evidence_origin"], "local")

    def test_default_graph_uses_real_tool_with_mock_http(self):
        graph = rag.build_graph(retriever=self.retriever, answer_chain=self.answer)
        with patch("pdfprocess.web_search.requests.get", return_value=response()):
            result = rag.ask(graph, "Question", "one")
        self.assertEqual(result["sources"][0]["kind"], "web")


if __name__ == "__main__":
    unittest.main()
