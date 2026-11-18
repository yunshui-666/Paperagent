"""默认工具目录。新增工具后在 create_default_registry 中添加一行注册。"""

# 工具目录组装。这里选择默认实现，注册表和模型节点通过依赖注入读取目录。

from langchain_core.tools import BaseTool, StructuredTool

# 相对导入兼容 python -m pdfprocess.rag 及现有脚本入口。
if __package__.startswith("pdfprocess."):
    from ..web_search import web_search
else:
    from web_search import web_search

from .registry import ToolRegistry
from .papers import search_papers


def create_default_registry(search_tool=web_search):
    """创建默认工具目录，将已有搜索实现登记为 web_search，支持原有注入调用约定。"""
    registry = ToolRegistry()
    if search_tool is None:
        return registry
    if not isinstance(search_tool, BaseTool) or search_tool.name != "web_search":
        # 兼容原有测试/实验注入的 Runnable，它们已经提供 .invoke(dict) 接口。
        def invoke_search(query: str):
            """把模型统一使用的 query 参数转交给注入搜索对象的 invoke 方法。"""
            return search_tool.invoke({"query": query})
        selected = StructuredTool.from_function(invoke_search, name="web_search",
                                               description=web_search.description,
                                               args_schema=web_search.args_schema)
    else:
        selected = search_tool
    registry.register(selected, tags=("search", "web"))
    registry.register(search_papers, tags=("search", "paper"))
    return registry
