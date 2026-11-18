import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver

from pdfprocess import rag
from pdfprocess.retrieval import RetrievalConfig


BASELINE_RETRIEVAL = RetrievalConfig(mode="dense", dense_k=5, top_k=5)


class ConversationTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        self.store = Mock()
        self.store.similarity_search_with_score.return_value = [
            (
                Document(
                    page_content="缩放点积注意力将 QK^T 除以根号 d_k。",
                    metadata={"source": "attention.pdf", "chunk_id": "paper001_0"}
                ),
                0.25
            )
        ]
        self.chain = Mock()
        self.chain.invoke.return_value = "缩放因子为根号 d_k。[1]"
        self.graph = rag.build_graph(vector_store=self.store, answer_chain=self.chain,
                                     retrieval_config=BASELINE_RETRIEVAL)

    def test_single_turn_preserves_retrieval_and_sources(self):
        result = rag.ask(self.graph, "缩放点积注意力怎样计算？", "a")

        self.store.similarity_search_with_score.assert_called_once_with(
            query="缩放点积注意力怎样计算？", k=5
        )
        self.chain.invoke.assert_called_once()
        inputs = self.chain.invoke.call_args.args[0]
        self.assertEqual(inputs["history"], [])
        self.assertIn("[1] 来源：attention.pdf", inputs["context"])
        self.assertEqual(result["answer"], "缩放因子为根号 d_k。[1]")
        self.assertEqual(result["sources"], [{
            "id": 1,
            "source": "attention.pdf",
            "chunk_id": "paper001_0",
            "distance": 0.25
        }])
        self.assertEqual([message.type for message in result["messages"]], ["human", "ai"])

    def test_history_accumulates_without_repeating_old_messages(self):
        questions = ["Q 是什么？", "它如何参与计算？", "为什么需要缩放？"]
        for index, question in enumerate(questions):
            result = rag.ask(self.graph, question, "a")
            history = self.chain.invoke.call_args.args[0]["history"]
            self.assertEqual(len(history), index * 2)
            self.assertEqual(len(result["messages"]), (index + 1) * 2)

        self.assertEqual(
            [message.content for message in result["messages"][::2]], questions
        )
        self.assertEqual(self.store.similarity_search_with_score.call_count, 3)
        self.assertEqual(self.chain.invoke.call_count, 3)

    def test_interleaved_threads_are_isolated(self):
        for index in range(10):
            for thread_id in ("a", "b"):
                question = f"{thread_id} 的问题 {index}"
                result = rag.ask(self.graph, question, thread_id)
                history = self.chain.invoke.call_args.args[0]["history"]
                self.assertEqual(len(history), index * 2)
                self.assertTrue(all(
                    message.content.startswith(f"{thread_id} 的问题")
                    for message in history if isinstance(message, HumanMessage)
                ))
                self.assertEqual(len(result["messages"]), (index + 1) * 2)

        for thread_id in ("a", "b"):
            state = self.graph.get_state({"configurable": {"thread_id": thread_id}})
            self.assertEqual(len(state.values["messages"]), 20)
            self.assertEqual(state.values["question"], f"{thread_id} 的问题 9")

    def test_real_prompt_chain_receives_history_in_order(self):
        model_inputs = []

        def fake_model(prompt_value):
            model_inputs.append(prompt_value.to_messages())
            return AIMessage(content="测试回答 [1]")

        chain = rag.prompt | RunnableLambda(fake_model) | StrOutputParser()
        graph = rag.build_graph(vector_store=self.store, answer_chain=chain,
                                retrieval_config=BASELINE_RETRIEVAL)
        rag.ask(graph, "第一问", "a")
        result = rag.ask(graph, "第二问", "a")

        messages = model_inputs[1]
        self.assertEqual([message.type for message in messages], [
            "system", "human", "ai", "human"
        ])
        self.assertEqual(messages[1].content, "第一问")
        self.assertEqual(messages[2].content, "测试回答 [1]")
        self.assertIn("第二问", messages[3].content)
        self.assertIn("缩放点积注意力", messages[3].content)
        self.assertEqual(result["answer"], "测试回答 [1]")

    def test_no_results_clears_previous_context_and_records_turn(self):
        rag.ask(self.graph, "第一问", "a")
        self.store.similarity_search_with_score.return_value = []
        result = rag.ask(self.graph, "不存在的内容", "a")

        self.assertEqual(result["context"], "")
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["answer"], "知识库中未检索到相关资料。")
        self.assertEqual(len(result["messages"]), 4)
        self.assertEqual(result["messages"][-1].content, result["answer"])
        self.assertEqual(self.chain.invoke.call_count, 1)

    def test_invalid_inputs_do_not_write_checkpoints(self):
        cases = [
            (None, "a", TypeError),
            (123, "a", TypeError),
            ("", "a", ValueError),
            (" \n ", "a", ValueError),
            ("问题", None, TypeError),
            ("问题", 123, TypeError),
            ("问题", "", ValueError),
            ("问题", " \t ", ValueError)
        ]
        for question, thread_id, error_type in cases:
            with self.subTest(question=question, thread_id=thread_id):
                with self.assertRaises(error_type):
                    rag.ask(self.graph, question, thread_id)

        self.assertEqual(list(self.graph.checkpointer.list(None)), [])
        self.store.similarity_search_with_score.assert_not_called()
        self.chain.invoke.assert_not_called()

    def test_retrieval_failure_preserves_history_and_can_be_resubmitted(self):
        self.check_failure_and_resubmission("retrieve")

    def test_generation_failure_preserves_history_and_can_be_resubmitted(self):
        self.check_failure_and_resubmission("generate")

    def check_failure_and_resubmission(self, node):
        first = rag.ask(self.graph, "已完成的问题", "a")
        previous_ids = [message.id for message in first["messages"]]
        operation = self.store.similarity_search_with_score
        if node == "generate":
            operation = self.chain.invoke
        cause = TimeoutError("模拟服务超时")
        operation.side_effect = cause

        with self.assertRaises(RuntimeError) as caught:
            rag.ask(self.graph, "待重试的问题", "a")
        self.assertIs(caught.exception.__cause__, cause)
        failed = self.graph.get_state({"configurable": {"thread_id": "a"}})
        self.assertEqual([message.id for message in failed.values["messages"]], previous_ids)
        self.assertEqual(failed.values["answer"], "")
        if node == "retrieve":
            self.assertEqual(failed.values["context"], "")
            self.assertEqual(failed.values["sources"], [])

        operation.side_effect = None
        result = rag.ask(self.graph, "待重试的问题", "a")
        self.assertEqual(len(result["messages"]), 4)
        self.assertEqual([message.id for message in result["messages"][:2]], previous_ids)
        self.assertEqual(result["messages"][-2].content, "待重试的问题")

    def test_invalid_model_output_does_not_append_history(self):
        for output in ("", " \n ", None, {"answer": "invalid"}):
            with self.subTest(output=output):
                self.chain.invoke.return_value = output
                with self.assertRaisesRegex(RuntimeError, "答案生成失败"):
                    rag.ask(self.graph, "问题", "a")
                state = self.graph.get_state({"configurable": {"thread_id": "a"}})
                self.assertEqual(state.values.get("messages", []), [])

    def test_raw_graph_retains_question_input_interface(self):
        result = self.graph.invoke(
            {"question": "问题"},
            config={"configurable": {"thread_id": "raw"}}
        )
        self.assertEqual(result["question"], "问题")
        self.assertEqual(len(result["messages"]), 2)

    def test_explicit_checkpointer_restores_history_in_rebuilt_graph(self):
        checkpointer = InMemorySaver()
        graph = rag.build_graph(
            vector_store=self.store, answer_chain=self.chain, checkpointer=checkpointer,
            retrieval_config=BASELINE_RETRIEVAL
        )
        rag.ask(graph, "第一问", "a")
        graph = rag.build_graph(
            vector_store=self.store, answer_chain=self.chain, checkpointer=checkpointer,
            retrieval_config=BASELINE_RETRIEVAL
        )
        result = rag.ask(graph, "第二问", "a")
        self.assertIs(graph.checkpointer, checkpointer)
        self.assertEqual(len(result["messages"]), 4)

    def test_injected_dependencies_need_no_credentials(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(rag, "load_dotenv") as load:
            graph = rag.build_graph(vector_store=self.store, answer_chain=self.chain)
        load.assert_not_called()
        self.assertIsInstance(graph.checkpointer, InMemorySaver)

    def test_missing_embedding_key(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(rag, "load_dotenv"):
            with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
                rag.build_graph(answer_chain=self.chain)

    def test_missing_deepseek_key(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(rag, "load_dotenv"):
            with self.assertRaisesRegex(ValueError, "DEEPSEEK_API"):
                rag.build_graph(vector_store=self.store)

    def test_chroma_initialization_error_has_database_path(self):
        cause = OSError("模拟数据库读取失败")
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
            with patch.object(rag, "load_dotenv"), patch.object(rag, "Chroma", side_effect=cause):
                with self.assertRaises(RuntimeError) as caught:
                    rag.build_graph(answer_chain=self.chain)
        self.assertIn(str(rag.CHROMA_DIR), str(caught.exception))
        self.assertIs(caught.exception.__cause__, cause)

    def test_chroma_path_is_relative_to_source_and_existing_parameters_are_preserved(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
            with patch.object(rag, "load_dotenv"), patch.object(rag, "Chroma") as chroma:
                rag.build_graph(answer_chain=self.chain)
        arguments = chroma.call_args.kwargs
        self.assertEqual(arguments["persist_directory"], str(
            Path(rag.__file__).resolve().parent / "data" / "chroma_db"
        ))
        self.assertEqual(arguments["collection_name"], "paper_chunks")
        self.assertFalse(arguments["create_collection_if_not_exists"])

    def test_deepseek_initialization_error_preserves_cause(self):
        cause = ValueError("模拟模型配置错误")
        with patch.dict(os.environ, {"DEEPSEEK_API": "test-key"}):
            with patch.object(rag, "load_dotenv"):
                with patch.object(rag, "ChatDeepSeek", side_effect=cause):
                    with self.assertRaisesRegex(RuntimeError, "DeepSeek 初始化失败") as caught:
                        rag.build_graph(vector_store=self.store)
        self.assertIs(caught.exception.__cause__, cause)

    def test_cli_reuses_graph_and_thread_after_failed_turn(self):
        with patch.object(rag, "build_graph", return_value=self.graph) as build:
            with patch("builtins.input", side_effect=["cli", "", "第一问", "第二问", "exit"]):
                self.assertEqual(rag.main(), 0)
        build.assert_called_once_with(retrieval_config=None, collection_name="paper_chunks")
        state = self.graph.get_state({"configurable": {"thread_id": "cli"}})
        self.assertEqual(len(state.values["messages"]), 4)
        self.assertEqual(self.chain.invoke.call_count, 2)

    def test_cli_initialization_failure_returns_error_status(self):
        with patch.object(rag, "build_graph", side_effect=ValueError("缺少密钥")):
            self.assertEqual(rag.main(), 1)

    def test_import_does_not_load_environment_or_initialize_services(self):
        code = """
from unittest.mock import patch
with patch('dotenv.load_dotenv') as load:
    with patch('langchain_chroma.Chroma') as chroma:
        with patch('langchain_deepseek.ChatDeepSeek') as deepseek:
            from pdfprocess import rag
load.assert_not_called()
chroma.assert_not_called()
deepseek.assert_not_called()
"""
        environment = os.environ.copy()
        environment.pop("DASHSCOPE_API_KEY", None)
        environment.pop("DEEPSEEK_API", None)
        environment["PYTHONPATH"] = str(Path(rag.__file__).resolve().parent.parent)
        result = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=Path(__file__).resolve().parent,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
