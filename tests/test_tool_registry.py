"""注册表单元测试，工具全部使用本地函数，不产生外部请求。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import unittest
from unittest.mock import Mock

from langchain_core.tools import tool

from pdfprocess.tools.defaults import create_default_registry
from pdfprocess.tools.registry import ToolRegistry, ToolRegistryError


@tool
def add_numbers(a: int, b: int) -> int:
    """计算两个整数之和。"""
    return a + b


class ToolRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.registry.register(add_numbers, tags=("math", "local"))

    def test_describe_and_list_reuse_base_tool_schema(self):
        description = self.registry.describe("add_numbers")
        self.assertEqual(description["name"], "add_numbers")
        self.assertIn("a", description["args_schema"]["properties"])
        self.assertEqual(description["tags"], ["math", "local"])
        self.assertEqual(self.registry.list_tools(tag="search"), [])
        self.assertEqual(self.registry.get_langchain_tools(), [add_numbers])
        self.assertIs(self.registry.get("add_numbers"), add_numbers)

    def test_duplicate_registration_fails_early(self):
        with self.assertRaisesRegex(ToolRegistryError, "重复"):
            self.registry.register(add_numbers)

    def test_valid_invocation_returns_result_and_serializable_record(self):
        result = self.registry.invoke("add_numbers", {"a": 2, "b": 3})
        self.assertTrue(result.success)
        self.assertEqual(result.data, 5)
        self.assertIsNone(result.error)
        self.assertGreaterEqual(result.elapsed_seconds, 0)
        self.assertEqual(result.record({"a": 2, "b": 3})["tool_name"], "add_numbers")

    def test_unknown_tool_has_structured_error(self):
        result = self.registry.invoke("missing", {})
        self.assertFalse(result.success)
        self.assertEqual(result.error.code, "unavailable")
        with self.assertRaises(ToolRegistryError):
            self.registry.get_langchain_tools(names=("missing",))

    def test_disabled_tool_is_listed_on_request_and_cannot_execute(self):
        registry = ToolRegistry()
        registry.register(add_numbers, enabled=False)
        self.assertEqual(registry.list_tools(), [])
        self.assertFalse(registry.list_tools(enabled_only=False)[0]["enabled"])
        self.assertEqual(registry.get_langchain_tools(), [])
        self.assertEqual(registry.invoke("add_numbers", {"a": 1, "b": 2}).error.code, "unavailable")

    def test_invalid_arguments_are_rejected_before_execution(self):
        for arguments in ({"a": 1}, {"a": "bad", "b": 2}, {"a": 1, "b": 2, "typo": 3}, "wrong"):
            with self.subTest(arguments=arguments):
                result = self.registry.invoke("add_numbers", arguments)
                self.assertFalse(result.success)
                self.assertEqual(result.error.code, "invalid_arguments")

    def test_tool_exception_is_returned_as_execution_error(self):
        @tool
        def failing(query: str) -> str:
            """用于验证错误传播的本地函数。"""
            raise RuntimeError("controlled error")
        self.registry.register(failing)
        result = self.registry.invoke("failing", {"query": "q"})
        self.assertEqual(result.error.code, "execution_failed")
        self.assertIn("controlled", result.error.message)

    def test_non_json_output_is_explicitly_reported(self):
        @tool
        def unsupported(query: str):
            """模拟工具返回未转换的特殊对象。"""
            return object()
        self.registry.register(unsupported)
        result = self.registry.invoke("unsupported", {"query": "q"})
        self.assertEqual(result.error.code, "execution_failed")
        self.assertIsNone(result.data)

    def test_registration_validates_types(self):
        with self.assertRaises(TypeError):
            self.registry.register(lambda: None)
        for options in ({"enabled": "yes"}, {"tags": "search"}, {"tags": ("",)}):
            with self.subTest(options=options), self.assertRaises(TypeError):
                ToolRegistry().register(add_numbers, **options)

    def test_generator_tags_are_materialized_once(self):
        registry = ToolRegistry()
        registry.register(add_numbers, tags=(tag for tag in ["math", "math", "local"]))
        self.assertEqual(registry.describe("add_numbers")["tags"], ["math", "local"])

    def test_execution_does_not_hold_registry_lock(self):
        barrier = Barrier(2)
        @tool
        def parallel(query: str):
            """两个调用须同时进入，验证执行过程不会占用目录锁。"""
            barrier.wait(timeout=5)
            return query
        self.registry.register(parallel)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(self.registry.invoke, "parallel", {"query": str(i)}) for i in range(2)]
            self.assertTrue(all(future.result().success for future in futures))

    def test_default_registry_adapts_existing_injected_search(self):
        search = Mock()
        search.invoke.return_value = [{"title": "fixture"}]
        registry = create_default_registry(search)
        result = registry.invoke("web_search", {"query": "question"})
        self.assertTrue(result.success)
        search.invoke.assert_called_once_with({"query": "question"})
        self.assertEqual(create_default_registry(None).list_tools(), [])

    def test_default_registry_aliases_legacy_custom_tool_names(self):
        @tool
        def custom_search(query: str):
            """兼容此前使用不同名称注入的搜索实现。"""
            return [{"query": query}]
        registry = create_default_registry(custom_search)
        self.assertEqual(registry.describe("web_search")["name"], "web_search")
        self.assertEqual(registry.invoke("web_search", {"query": "q"}).data, [{"query": "q"}])


if __name__ == "__main__":
    unittest.main()
