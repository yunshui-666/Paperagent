"""原生工具调用协议、证据隔离、预算和会话恢复回归测试。"""

import contextlib
import io
import unittest
from unittest.mock import Mock

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from pdfprocess import rag
from pdfprocess.context_compression import CompressionConfig, message_tokens
from pdfprocess.retrieval import RetrievalConfig, RetrievalHit
from pdfprocess.tools.registry import ToolRegistry
from pdfprocess.tools.search_flow import LOCAL_KNOWLEDGE_NOTICE, web_answer_prompt


WEB_RESULTS = [{"title": "Public paper", "url": "https://example.org/paper", "content": "WEB_EVIDENCE temperature is 25 C."}]


class FixtureRetriever:
    def __init__(self):
        self.config = RetrievalConfig()
        self.score = -3
        self.empty = False

    def search(self, query):
        if self.empty:
            return []
        return [RetrievalHit(Document(page_content="PRIVATE_LOCAL_EVIDENCE", metadata={"source": "private.pdf"}),
                             "private", cross_encoder_score=self.score)]


def call_message(*, name="web_search", query="Model generated query", call_id="call-one"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {"query": query}, "id": call_id, "type": "tool_call"}],
                     usage_metadata={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15})


class AgentSearchTests(unittest.TestCase):
    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.retriever = FixtureRetriever()
        self.caller = Mock()
        self.caller.invoke.return_value = call_message()
        self.answer = Mock()
        self.answer.invoke.return_value = "LOCAL_ANSWER"
        self.web_answer = Mock()
        self.web_answer.invoke.return_value = AIMessage(content="Web evidence supports 25 C. [1]",
            usage_metadata={"input_tokens": 12, "output_tokens": 6, "total_tokens": 18})
        self.summary = Mock()
        self.summary.invoke.return_value = "LOCAL_HISTORY_SUMMARY"
        self.search = Mock()
        self.search.return_value = WEB_RESULTS
        self.registry = ToolRegistry()
        self.registry.register(StructuredTool.from_function(self.search, name="web_search",
            description="Search public references", args_schema=rag.web_search.args_schema), tags=("search",))

    def graph(self, **kwargs):
        return rag.build_graph(retriever=self.retriever, answer_chain=self.answer, tool_registry=self.registry,
                               tool_call_chain=self.caller, web_answer_chain=self.web_answer,
                               include_source_text=True, **kwargs)

    def test_low_relevance_model_calls_registry_then_reads_tool_message(self):
        result = rag.ask(self.graph(), "Original question", "a")
        self.caller.invoke.assert_called_once_with({"question": "Original question"})
        self.search.assert_called_once_with(query="Model generated query")
        self.answer.invoke.assert_not_called()
        payload = self.web_answer.invoke.call_args.args[0]
        self.assertEqual(set(payload), {"question", "tool_messages"})
        ai, reply = payload["tool_messages"]
        self.assertIsInstance(ai, AIMessage)
        self.assertIsInstance(reply, ToolMessage)
        self.assertEqual(reply.tool_call_id, ai.tool_calls[0]["id"])
        self.assertIn("WEB_EVIDENCE", reply.content)
        self.assertNotIn("PRIVATE_LOCAL", str(payload))
        self.assertTrue(result["answer"].startswith(LOCAL_KNOWLEDGE_NOTICE))
        self.assertEqual(result["metrics"]["tool_call_mode"], "model")
        self.assertEqual(result["metrics"]["tool_selection_usage"]["input_tokens"], 11)
        self.assertTrue(result["metrics"]["tool_calls"][0]["success"])
        self.assertEqual(result["sources"][0]["url"], WEB_RESULTS[0]["url"])

    def test_threshold_boundary_and_high_scores_keep_local(self):
        graph = self.graph()
        for score in (0, 3):
            self.retriever.score = score
            result = rag.ask(graph, "question", str(score))
            self.assertEqual(result["answer"], "LOCAL_ANSWER")
            self.assertEqual(result["evidence_origin"], "local")
        self.caller.invoke.assert_not_called()
        self.search.assert_not_called()

    def test_empty_local_candidates_use_model_search(self):
        self.retriever.empty = True
        result = rag.ask(self.graph(), "question", "a")
        self.assertEqual(result["metrics"]["web_fallback_reason"], "no_local_results")
        self.search.assert_called_once()

    def test_empty_web_results_keep_notice_and_skip_answer(self):
        self.search.return_value = []
        result = rag.ask(self.graph(), "question", "a")
        self.assertTrue(result["answer"].startswith(LOCAL_KNOWLEDGE_NOTICE))
        self.assertIn("联网搜索未找到", result["answer"])
        self.web_answer.invoke.assert_not_called()

    def test_notice_is_not_duplicated_if_model_already_includes_it(self):
        self.web_answer.invoke.return_value = LOCAL_KNOWLEDGE_NOTICE + "\n25 C. [1]"
        result = rag.ask(self.graph(), "question", "a")
        self.assertEqual(result["answer"].count(LOCAL_KNOWLEDGE_NOTICE), 1)

    def test_bad_model_calls_never_execute_tool_or_use_local_data(self):
        invalid = [AIMessage(content="I can answer directly"), call_message(name="delete_files"),
                   call_message(query=""), call_message(query="x" * 2001), call_message(call_id=""),
                   AIMessage(content="", tool_calls=call_message().tool_calls * 2),
                   AIMessage(content="", tool_calls=[{"name": "web_search", "args": {}, "id": "c", "type": "tool_call"}]),
                   AIMessage(content="", tool_calls=[{"name": "web_search", "args": {"query": 123}, "id": "c", "type": "tool_call"}])]
        for index, reply in enumerate(invalid):
            self.caller.invoke.return_value = reply
            graph = self.graph()
            with self.subTest(index=index), self.assertRaisesRegex(RuntimeError, "搜索工具调用生成失败"):
                rag.ask(graph, "question", str(index))
            values = graph.get_state({"configurable": {"thread_id": str(index)}}).values
            self.assertEqual(values["context"], "")
            self.assertEqual(values["sources"], [])
            self.assertEqual(values.get("messages", []), [])
        self.search.assert_not_called()
        self.web_answer.invoke.assert_not_called()

    def test_invalid_call_can_retry_without_duplicate_turns(self):
        graph = self.graph()
        self.caller.invoke.return_value = AIMessage(content="no call")
        with self.assertRaises(RuntimeError):
            rag.ask(graph, "question", "a", turn_id="one")
        self.caller.invoke.return_value = call_message()
        result = rag.ask(graph, "question", "a", turn_id="one")
        self.assertEqual([message.id for message in result["messages"]], ["one:human", "one:ai"])

    def test_search_exception_preserves_notice_and_empty_local_context(self):
        self.search.side_effect = RuntimeError("controlled search error")
        graph = self.graph()
        with self.assertRaisesRegex(RuntimeError, LOCAL_KNOWLEDGE_NOTICE):
            rag.ask(graph, "question", "a")
        values = graph.get_state({"configurable": {"thread_id": "a"}}).values
        self.assertEqual(values["sources"], [])
        self.assertEqual(values["context"], "")
        self.web_answer.invoke.assert_not_called()

    def test_answer_stage_rejects_unexpected_tool_call(self):
        self.web_answer.invoke.return_value = call_message()
        with self.assertRaisesRegex(RuntimeError, "答案生成失败"):
            rag.ask(self.graph(), "question", "a")

    def test_compression_budget_covers_full_tool_protocol(self):
        config = CompressionConfig(max_input_tokens=2400, evidence_tokens=2200)
        self.search.return_value = [{**WEB_RESULTS[0], "content": "WEB_EVIDENCE temperature is 25 C. " * 100}]
        result = rag.ask(self.graph(compression_config=config, summary_chain=self.summary), "temperature", "a")
        payload = self.web_answer.invoke.call_args.args[0]
        actual_input = message_tokens(web_answer_prompt.format_messages(**payload))
        self.assertLessEqual(actual_input, 2400)
        self.assertEqual(result["metrics"]["input_tokens_after_compression"], actual_input)
        self.assertEqual(payload["tool_messages"][1].content, result["context"])
        self.assertGreater(result["metrics"]["evidence_duplicate_units"], 0)

    def test_tool_schema_and_selection_prompt_have_input_budget(self):
        config = CompressionConfig(tool_input_tokens=100)
        with self.assertRaisesRegex(ValueError, "搜索工具调用提示词超过预算"):
            rag.ask(self.graph(compression_config=config, summary_chain=self.summary), "question", "a")
        self.caller.invoke.assert_not_called()

    def test_local_summary_and_full_history_never_enter_web_protocol(self):
        self.retriever.score = 3
        graph = self.graph(compression_config=CompressionConfig(recent_turns=1), summary_chain=self.summary)
        for index in range(3):
            rag.ask(graph, f"PRIVATE_HISTORY {index}", "a")
        self.retriever.score = -3
        result = rag.ask(graph, "Public standalone question", "a")
        self.assertIn("LOCAL_HISTORY_SUMMARY", result["conversation_summary"])
        self.assertNotIn("PRIVATE_", str(self.caller.invoke.call_args.args[0]))
        self.assertNotIn("LOCAL_", str(self.web_answer.invoke.call_args.args[0]))
        self.assertEqual(len(result["messages"]), 8)

    def test_checkpoint_rebuild_thread_isolation_and_next_turn_reset(self):
        checkpoint = InMemorySaver()
        graph = self.graph(checkpointer=checkpoint)
        rag.ask(graph, "first", "a")
        graph = self.graph(checkpointer=checkpoint)
        result = rag.ask(graph, "other", "b")
        self.assertEqual(len(result["messages"]), 2)
        self.retriever.score = 3
        result = rag.ask(graph, "local followup", "a")
        self.assertIsNone(result["search_call"])
        self.assertNotIn("tool_selection_usage", result["metrics"])
        self.assertEqual(len(result["messages"]), 4)

    def test_explicit_local_baseline_disables_model_tool_calling(self):
        result = rag.ask(self.graph(web_search_tool=None), "question", "a")
        self.assertEqual(result["evidence_origin"], "local")
        self.caller.invoke.assert_not_called()

    def test_incomplete_chain_configuration_fails_early(self):
        with self.assertRaisesRegex(ValueError, "一起提供"):
            rag.build_graph(retriever=self.retriever, answer_chain=self.answer, tool_call_chain=self.caller)


if __name__ == "__main__":
    unittest.main()
