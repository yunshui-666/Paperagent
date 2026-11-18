"""PaperAgent 工具注册与调用接口，工具实例在服务启动时显式登记。"""

# 工具包公共导出。具体默认工具由 defaults.py 在服务启动时显式登记。

from .registry import ToolRegistry, ToolRegistryError, ToolResult

__all__ = ["ToolRegistry", "ToolRegistryError", "ToolResult"]
