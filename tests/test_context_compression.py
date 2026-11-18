"""压缩回归测试。检索、摘要和答案均可控，无需联网或 API 密钥。

阅读测试时可把每个 test_ 函数当作一个输入/期望输出示例。
"""

from dataclasses import replace
from pathlib import Path
import contextlib
import io
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from pdfprocess import rag
from pdfprocess.context_compression import (
    CompressionConfig, clip_text, compress_evidence, estimate_tokens,
    history_fingerprint, message_tokens, recent_history, render_source,
    summarize_history, summary_prompt,
)
from pdfprocess.retrieval import RetrievalConfig, RetrievalHit


def source(text, chunk="c1", document="paper-a", page=1):
    """构造与真实 retrieve_node 输出一致的证据字典。"""
    return {"id": 1, "source": document + ".pdf", "document_id": document,
            "page": page, "chunk_id": chunk, "distance": 0.2, "text": text}


def messages(count=4, prefix="topic"):
    """count 代表问答轮次数，原始历史含 count * 2 条消息。"""
    result = []
    for index in range(count):
        result.extend([HumanMessage(content=f"{prefix} question {index}", id=f"{prefix}{index}:human"),
                       AIMessage(content=f"{prefix} answer {index}", id=f"{prefix}{index}:ai")])
    return result


class FixtureRetriever:
    def __init__(self, *, web=False, text="实验温度为 25 °C。结论不适用于高温条件。"):
        self.config = RetrievalConfig(mode="hybrid_rerank" if web else "dense")
        self.hit = RetrievalHit(Document(page_content=text, metadata={"source": "paper.pdf", "page": 1}),
                                "chunk-a", cross_encoder_score=-3 if web else 3)
        self.queries = []

    def search(self, question):
        self.queries.append(question)
        return [self.hit]


class CompressionRuleTests(unittest.TestCase):
    def test_utf8_measurement_and_safe_clipping(self):
        self.assertEqual(estimate_tokens("中A"), 4)
        self.assertEqual(clip_text("中A", 2), "")
        self.assertEqual(clip_text("中A", 3), "中")
        self.assertEqual(clip_text("abc", -1), "")

    def test_invalid_settings(self):
        for value in (0, -1, True, 1.5, "100"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CompressionConfig(max_input_tokens=value)
        with self.assertRaises(ValueError):
            CompressionConfig(history_tokens=1000)
        with self.assertRaises(ValueError):
            CompressionConfig(summary_input_tokens=2000)

    def test_same_page_overlap_removed_and_chunk_ids_retained(self):
        context, kept, metric = compress_evidence([
            source("方法使用注意力。温度为 25 °C。", "a"),
            source("温度为 25 °C。结论不适用于高温。", "b"),
            source("方法使用注意力。", "c"),
        ], "温度", 3000)
        self.assertEqual(context.count("温度为 25 °C。"), 1)
        self.assertIn("结论不适用于高温。", context)
        self.assertEqual([item["id"] for item in kept], [1, 2])
        self.assertEqual(kept[0]["overlap_chunk_ids"], ["b", "c"])
        self.assertEqual(metric["evidence_duplicate_units"], 2)
        self.assertIn("页码：1", context)

    def test_independent_papers_and_pages_keep_attribution(self):
        _, kept, _ = compress_evidence([
            source("The result is 25.", "a"), source("The result is 25.", "b", document="paper-b"),
            source("The result is 25.", "c", page=2),
        ], "result", 3000)
        self.assertEqual(len(kept), 3)
        self.assertEqual([item["id"] for item in kept], [1, 2, 3])

    def test_spaces_inside_words_do_not_create_false_duplicates(self):
        _, kept, _ = compress_evidence([source("not able.", "a"), source("notable.", "b")], "able", 3000)
        self.assertEqual(len(kept), 2)

    def test_extractive_selection_prefers_query_terms_and_keeps_negation(self):
        item = source("Background filler material. Temperature is not above 25 C. Other unrelated discussion.")
        budget = estimate_tokens(render_source(item, "")) + 33
        context, kept, _ = compress_evidence([item], "Temperature", budget)
        self.assertIn("Temperature is not above 25 C.", context)
        self.assertNotIn("Background", context)
        self.assertLessEqual(estimate_tokens(context), budget)
        self.assertEqual(kept[0]["text"], "Temperature is not above 25 C.")

    def test_large_header_and_single_oversized_sentence_are_skipped(self):
        context, kept, _ = compress_evidence([source("x" * 1000)], "x", 100)
        self.assertEqual((context, kept), ("", []))
        context, kept, _ = compress_evidence([source("ok.", document="x" * 1000)], "x", 100)
        self.assertEqual((context, kept), ("", []))

    def test_web_snapshot_keeps_url_and_budget_includes_it(self):
        item = {**source("Public evidence. " * 100), "kind": "web", "url": "https://example.org/paper"}
        context, kept, _ = compress_evidence([item], "Public", 300)
        self.assertLessEqual(estimate_tokens(context), 300)
        self.assertEqual(kept[0]["url"], item["url"])
        self.assertIn("链接：https://example.org/paper", context)

    def test_recent_history_keeps_complete_turns(self):
        recent, boundary = recent_history(messages(5), CompressionConfig(recent_turns=2))
        self.assertEqual(len(recent), 4)
        self.assertEqual(boundary, 6)
        self.assertEqual([item.type for item in recent], ["human", "ai", "human", "ai"])

    def test_summary_preserves_full_history_and_records_usage(self):
        original = messages(5)
        chain = Mock()
        chain.invoke.return_value = AIMessage(content="topic and constraints [1]", usage_metadata={
            "input_tokens": 20, "output_tokens": 5, "total_tokens": 25})
        result = summarize_history({"messages": original}, config=CompressionConfig(recent_turns=2), summary_chain=chain)
        self.assertEqual(len(original), 10)
        self.assertNotIn("messages", result)
        self.assertEqual(result["summary_covered"], 6)
        self.assertNotIn("[1]", result["conversation_summary"])
        self.assertEqual(result["summary_fingerprint"], history_fingerprint(original[:6]))
        self.assertEqual(result["metrics"]["summary_usage"][0]["input_tokens"], 20)
        again = summarize_history({**result, "messages": original}, config=CompressionConfig(recent_turns=2), summary_chain=chain)
        self.assertEqual(again["metrics"]["summary_calls"], 0)
        self.assertEqual(chain.invoke.call_count, 1)

    def test_modified_history_invalidates_old_summary(self):
        chain = Mock()
        chain.invoke.return_value = "rebuilt"
        original = messages(4)
        result = summarize_history({"messages": original}, config=CompressionConfig(recent_turns=1), summary_chain=chain)
        original[0] = HumanMessage(content="changed", id=original[0].id)
        summarize_history({**result, "messages": original}, config=CompressionConfig(recent_turns=1), summary_chain=chain)
        self.assertEqual(chain.invoke.call_count, 2)
        self.assertEqual(chain.invoke.call_args.args[0]["summary"], "")

    def test_oversized_old_messages_have_bounded_summary_requests(self):
        chain = Mock()
        chain.invoke.return_value = "retained topic"
        original = [HumanMessage(content="A" * 30000, id="a"), AIMessage(content="B" * 30000, id="b")]
        result = summarize_history({"messages": original}, config=CompressionConfig(), summary_chain=chain)
        self.assertEqual(result["summary_covered"], 2)
        self.assertEqual(result["metrics"]["summary_clipped_messages"], 2)
        for call in chain.invoke.call_args_list:
            self.assertLessEqual(message_tokens(summary_prompt.format_messages(**call.args[0])), 8000)
        self.assertEqual(len(original[0].content), 30000)

    def test_summary_output_is_bounded(self):
        chain = Mock()
        chain.invoke.return_value = "摘要" * 3000
        result = summarize_history({"messages": messages()}, config=CompressionConfig(recent_turns=1), summary_chain=chain)
        self.assertLessEqual(estimate_tokens(result["conversation_summary"]), 1200)
        self.assertLessEqual(message_tokens(result["model_history"]), 4000)


class CompressionGraphTests(unittest.TestCase):
    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.retriever = FixtureRetriever()
        self.answer = Mock()
        self.answer.invoke.return_value = "25 °C，且不适用于高温。[1]"
        self.summary = Mock()
        self.summary.invoke.side_effect = lambda payload: "topic " + payload["transcript"][:100]
        self.config = CompressionConfig(recent_turns=1)

    def graph(self, **kwargs):
        return rag.build_graph(retriever=self.retriever, answer_chain=self.answer, summary_chain=self.summary,
                               compression_config=self.config, **kwargs)

    def test_short_history_avoids_summary_calls(self):
        graph = self.graph()
        rag.ask(graph, "温度？", "a")
        result = rag.ask(graph, "适用条件？", "a")
        self.summary.invoke.assert_not_called()
        self.assertEqual(len(self.answer.invoke.call_args.args[0]["history"]), 2)
        self.assertEqual(len(result["messages"]), 4)

    def test_long_conversation_uses_incremental_summary(self):
        graph = self.graph()
        for index in range(5):
            result = rag.ask(graph, f"question {index}", "a", turn_id=f"turn{index}")
        self.assertEqual(len(result["messages"]), 10)
        self.assertEqual(result["summary_covered"], 6)
        self.assertEqual(self.summary.invoke.call_count, 3)
        payload = self.answer.invoke.call_args.args[0]
        self.assertEqual(len(payload["history"]), 3)
        self.assertIn("历史对话摘要", payload["history"][0].content)
        self.assertEqual(payload["history"][1].content, "question 3")
        self.assertLessEqual(result["metrics"]["input_tokens_after_compression"], 16000)

    def test_real_prompt_uses_summary_and_evidence_without_new_variables(self):
        inputs = []
        chain = rag.prompt | RunnableLambda(lambda value: inputs.append(value.to_messages()) or AIMessage(content="ok [1]"))
        graph = rag.build_graph(retriever=self.retriever, answer_chain=chain, summary_chain=self.summary,
                                compression_config=self.config)
        for index in range(3):
            rag.ask(graph, str(index), "a")
        self.assertEqual([m.type for m in inputs[-1]], ["system", "human", "human", "ai", "human"])
        self.assertIn("历史对话摘要", inputs[-1][1].content)
        self.assertIn("温度", inputs[-1][-1].content)

    def test_checkpoint_rebuild_and_threads_are_isolated(self):
        checkpoint = InMemorySaver()
        graph = self.graph(checkpointer=checkpoint)
        for index in range(3):
            rag.ask(graph, f"alpha {index}", "a")
        graph = self.graph(checkpointer=checkpoint)
        result = rag.ask(graph, "beta", "b")
        self.assertEqual(result["conversation_summary"], "")
        self.assertEqual(self.answer.invoke.call_args.args[0]["history"], [])
        result = rag.ask(graph, "alpha followup", "a")
        self.assertEqual(len(result["messages"]), 8)
        self.assertIn("alpha", result["conversation_summary"])
        self.assertNotIn("beta", result["conversation_summary"])

    def test_sqlite_close_reopen_preserves_summary(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("PAPERAGENT_TEST_ROOT"), ignore_cleanup_errors=True) as directory:
            path = str(Path(directory) / "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(path) as checkpoint:
                graph = self.graph(checkpointer=checkpoint)
                for index in range(3):
                    rag.ask(graph, f"alpha {index}", "a")
                saved = graph.get_state({"configurable": {"thread_id": "a"}}).values["conversation_summary"]
            with SqliteSaver.from_conn_string(path) as checkpoint:
                graph = self.graph(checkpointer=checkpoint)
                self.assertEqual(graph.get_state({"configurable": {"thread_id": "a"}}).values["conversation_summary"], saved)
                result = rag.ask(graph, "continue", "a")
                self.assertEqual(len(result["messages"]), 8)

    def test_legacy_checkpoint_builds_summary_lazily(self):
        checkpoint = InMemorySaver()
        graph = self.graph(checkpointer=checkpoint)
        graph.update_state({"configurable": {"thread_id": "a"}}, {"messages": messages()}, as_node="generate")
        result = rag.ask(graph, "followup", "a")
        self.assertEqual(len(result["messages"]), 10)
        self.assertEqual(result["summary_covered"], 6)

    def test_summary_failure_does_not_append_half_turn_and_retry_works(self):
        graph = self.graph()
        rag.ask(graph, "first", "a", turn_id="1")
        rag.ask(graph, "second", "a", turn_id="2")
        self.summary.invoke.side_effect = RuntimeError("fixture error")
        with self.assertRaisesRegex(RuntimeError, "历史摘要生成失败"):
            rag.ask(graph, "third", "a", turn_id="3")
        values = graph.get_state({"configurable": {"thread_id": "a"}}).values
        self.assertEqual(len(values["messages"]), 4)
        self.summary.invoke.side_effect = None
        self.summary.invoke.return_value = "first topic"
        result = rag.ask(graph, "third", "a", turn_id="3")
        self.assertEqual(len(result["messages"]), 6)
        self.assertEqual(result["messages"][-1].id, "3:ai")

    def test_answer_retry_reuses_checkpointed_summary(self):
        graph = self.graph()
        rag.ask(graph, "first", "a")
        rag.ask(graph, "second", "a")
        self.answer.invoke.side_effect = RuntimeError("answer failed")
        with self.assertRaisesRegex(RuntimeError, "答案生成失败"):
            rag.ask(graph, "third", "a", turn_id="third")
        calls = self.summary.invoke.call_count
        self.answer.invoke.side_effect = None
        result = rag.ask(graph, "third", "a", turn_id="third")
        self.assertEqual(self.summary.invoke.call_count, calls)
        self.assertEqual(len(result["messages"]), 6)

    def test_invalid_summary_response_is_retryable(self):
        for answer in ("", [], "[1]"):
            self.summary.invoke.side_effect = None
            self.summary.invoke.return_value = answer
            with self.subTest(answer=answer), self.assertRaises(RuntimeError):
                summarize_history({"messages": messages()}, config=self.config, summary_chain=self.summary)

    def test_web_answer_excludes_summary_and_local_history(self):
        graph = self.graph()
        for index in range(3):
            rag.ask(graph, f"LOCAL_SECRET {index}", "a")
        self.retriever.config = RetrievalConfig(mode="hybrid_rerank")
        self.retriever.hit.cross_encoder_score = -4
        search = Mock()
        search.invoke.return_value = [{"title": "public", "url": "https://example.org/paper", "content": "WEB_EVIDENCE."}]
        rewriter = Mock()
        rewriter.invoke.return_value = "Independent public question"
        # 重建图后沿用同一检查点，模拟本地对话转入联网分支。
        graph = self.graph(checkpointer=graph.checkpointer, web_search_tool=search, query_rewriter=rewriter)
        result = rag.ask(graph, "What about it?", "a")
        payload = self.answer.invoke.call_args.args[0]
        self.assertEqual(payload["history"], [])
        self.assertNotIn("LOCAL_SECRET", str(payload))
        self.assertIn("WEB_EVIDENCE", payload["context"])
        self.assertEqual(payload["question"], "Independent public question")
        self.assertEqual(result["evidence_origin"], "web")
        self.assertTrue(all(item["kind"] == "web" for item in result["sources"]))

    def test_budget_covers_system_question_history_and_source_headers(self):
        self.config = replace(self.config, max_input_tokens=2400, evidence_tokens=2200)
        self.retriever.hit.document.page_content = "温度为 25 °C。" * 200
        graph = self.graph()
        result = rag.ask(graph, "温度是多少？", "a")
        payload = self.answer.invoke.call_args.args[0]
        total = message_tokens(rag.prompt.format_messages(**payload))
        self.assertLessEqual(total, 2400)
        self.assertEqual(result["metrics"]["input_tokens_after_compression"], total)
        self.assertGreater(result["metrics"]["evidence_tokens_before"], result["metrics"]["evidence_tokens_after"])
        self.assertEqual(result["sources"][0]["text"], "温度为 25 °C。")

    def test_oversized_question_fails_before_retrieval_or_model(self):
        graph = self.graph()
        with self.assertRaisesRegex(ValueError, "问题超过输入预算"):
            rag.ask(graph, "长" * 1000, "a")
        self.assertEqual(self.retriever.queries, [])
        self.answer.invoke.assert_not_called()
        self.summary.invoke.assert_not_called()

    def test_over_budget_rewrite_is_rejected(self):
        rewriter = Mock()
        rewriter.invoke.return_value = "x" * 2500
        graph = self.graph(query_rewriter=rewriter)
        rag.ask(graph, "first", "a")
        with self.assertRaisesRegex(ValueError, "改写问题超过输入预算"):
            rag.ask(graph, "second", "a")
        self.assertEqual(len(self.retriever.queries), 1)

    def test_rewrite_input_is_bounded(self):
        self.config = replace(self.config, rewrite_input_tokens=1400)
        rewriter = Mock()
        rewriter.invoke.return_value = "independent question"
        graph = self.graph(query_rewriter=rewriter)
        rag.ask(graph, "q" * 900, "a")
        rag.ask(graph, "followup", "a")
        # 空间不足时可以完全去掉历史并跳过改写，仍保持调用输入受限。
        for call in rewriter.invoke.call_args_list:
            self.assertLessEqual(message_tokens(rag.rewrite_prompt.format_messages(**call.args[0])), 1400)

    def test_missing_summary_dependency_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "summary_chain"):
            rag.build_graph(retriever=self.retriever, answer_chain=self.answer, compression_config=self.config)

    def test_no_complete_evidence_sentence_skips_answer_model(self):
        self.retriever.hit.document.page_content = "x" * 20000
        result = rag.ask(self.graph(), "topic", "a")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["context"], "")
        self.assertIn("没有可用的完整证据", result["answer"])
        self.answer.invoke.assert_not_called()

    def test_system_prompt_over_budget_is_reported(self):
        self.config = replace(self.config, max_input_tokens=100)
        with self.assertRaisesRegex(ValueError, "问题和历史已占满输入预算"):
            rag.ask(self.graph(), "topic", "a")
        self.answer.invoke.assert_not_called()

    def test_default_model_entry_enables_compression_without_cloud_calls(self):
        inputs = []
        def fake_model(value, **kwargs):
            inputs.append((value.to_messages(), kwargs))
            return AIMessage(content="Answer [1]", usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7})
        with patch.dict(os.environ, {"DEEPSEEK_API": "fixture"}), patch.object(rag, "load_dotenv"):
            with patch.object(rag, "ChatDeepSeek", return_value=RunnableLambda(fake_model)):
                graph = rag.build_graph(retriever=self.retriever)
        result = rag.ask(graph, "topic", "a")
        self.assertIn("input_tokens_after_compression", result["metrics"])
        self.assertEqual(result["metrics"]["generation_usage"]["input_tokens"], 5)
        self.assertEqual(inputs[0][1]["max_tokens"], 2048)


if __name__ == "__main__":
    unittest.main()
