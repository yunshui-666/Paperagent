"""轻量工具注册表，复用 LangChain BaseTool 的名称、说明和参数 Schema。

学习时先读 register()，再读 invoke()。注册表只执行调用，工作流决定调用时机。
注册表对象保存在进程内，返回的执行记录可以写入 LangGraph 检查点。
"""

# 工具管理与执行边界。工具自身负责具体能力及客户端重试，注册表负责发现、校验与执行结果封装。

from dataclasses import asdict, dataclass
import json
from threading import RLock
import time
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import ValidationError


class ToolRegistryError(ValueError):
    """注册冲突、工具缺失或工具禁用等配置错误。"""


@dataclass(frozen=True)
class ToolSpec:
    # 工具名称、描述和参数继续以 BaseTool 为准，避免维护两份不一致的定义。
    """保存原生 BaseTool、启用状态和用途标签，供目录查询与模型工具绑定使用。"""
    tool: BaseTool
    enabled: bool
    tags: tuple[str, ...]


@dataclass(frozen=True)
class ToolError:
    """执行错误的简洁表示，code 说明错误阶段，message 提供可读原因。"""
    code: str
    message: str


@dataclass(frozen=True)
class ToolResult:
    """统一保存工具名称、成功状态、正文数据、错误信息和本次执行耗时。"""
    tool_name: str
    success: bool
    data: Any
    error: ToolError | None
    elapsed_seconds: float

    def record(self, arguments):
        """只保存调用信息，正文结果由 sources 保留，避免 metrics 重复存大段文本。"""
        return {"tool_name": self.tool_name, "arguments": arguments, "success": self.success,
                "error": asdict(self.error) if self.error else None,
                "elapsed_seconds": self.elapsed_seconds}


class ToolRegistry:
    """线程安全的进程内工具目录，负责登记、发现、输入校验和同步调用。"""
    def __init__(self):
        """创建工具映射与可重入目录锁，工具执行阶段使用独立调用上下文。"""
        self._tools: dict[str, ToolSpec] = {}
        self._lock = RLock()

    def register(self, tool: BaseTool, *, enabled=True, tags=()):
        """显式注册一个工具，重复名称直接报错，方便启动时发现配置问题。"""
        if not isinstance(tool, BaseTool):
            raise TypeError("请注册 LangChain BaseTool 或 @tool 创建的工具")
        if not isinstance(tool.name, str) or not tool.name.strip():
            raise ToolRegistryError("工具名称不能为空")
        if not isinstance(enabled, bool):
            raise TypeError("enabled 必须为布尔值")
        if isinstance(tags, str):
            raise TypeError("tags 需要由非空字符串组成")
        tags = tuple(tags)
        if any(not isinstance(tag, str) or not tag.strip() for tag in tags):
            raise TypeError("tags 需要由非空字符串组成")
        with self._lock:
            if tool.name in self._tools:
                raise ToolRegistryError(f"工具名称重复：{tool.name}")
            self._tools[tool.name] = ToolSpec(tool, enabled, tuple(dict.fromkeys(tags)))

    def copy(self):
        """复制目录映射，为每个会话绑定独立目标库，同时共享无状态工具对象。"""
        registry = ToolRegistry()
        with self._lock:
            registry._tools = dict(self._tools)
        return registry

    def get(self, name: str) -> BaseTool:
        """获取已启用的原生工具，供 LangChain bind_tools 使用。"""
        with self._lock:
            spec = self._tools.get(name)
            if spec is None:
                raise ToolRegistryError(f"工具不存在：{name}")
            if not spec.enabled:
                raise ToolRegistryError(f"工具已禁用：{name}")
            return spec.tool

    def describe(self, name: str):
        """返回指定工具的名称、说明、JSON 参数 Schema、标签和启用状态。"""
        with self._lock:
            spec = self._tools.get(name)
            if spec is None:
                raise ToolRegistryError(f"工具不存在：{name}")
            return {"name": spec.tool.name, "description": spec.tool.description,
                    "args_schema": spec.tool.get_input_schema().model_json_schema(),
                    "enabled": spec.enabled, "tags": list(spec.tags)}

    def list_tools(self, *, enabled_only=True, tag=None):
        """返回工具目录快照，界面或调试代码可以直接打印查看。"""
        with self._lock:
            return [self.describe(name) for name, spec in self._tools.items()
                    if (not enabled_only or spec.enabled) and (tag is None or tag in spec.tags)]

    def get_langchain_tools(self, *, names=None):
        # 工作流可以只向模型暴露当前节点需要的工具集合。
        """返回指定名称或全部已启用工具的原生对象，供模型 bind_tools 使用。"""
        with self._lock:
            selected = names if names is not None else [name for name, spec in self._tools.items() if spec.enabled]
            return [self.get(name) for name in selected]

    def invoke(self, name: str, arguments: dict) -> ToolResult:
        """统一校验和执行，失败也返回带错误类别的 ToolResult。

        超时和重试由每个工具的 HTTP/SDK 实现设置，避免重复重试有副作用的工具。
        执行期间不持注册表锁，不同会话的耗时工具可以独立执行。
        """
        started = time.perf_counter()
        data, error = None, None
        try:
            tool = self.get(name)
            schema = tool.get_input_schema()
            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须为字典")
            # Pydantic 的部分模型默认忽略多余字段，这里主动报告拼错的参数名。
            unknown = set(arguments) - set(schema.model_fields)
            if unknown:
                raise ValueError(f"未知工具参数：{', '.join(sorted(unknown))}")
            schema.model_validate(arguments)
        except ToolRegistryError as exc:
            error = ToolError("unavailable", str(exc))
        except (ValidationError, TypeError, ValueError) as exc:
            error = ToolError("invalid_arguments", str(exc)[:500])
        if error is None:
            try:
                data = tool.invoke(arguments)
                # 检查点和 HTTP 使用 JSON 可表示的数据，工具需自行将特殊对象转成字典。
                json.dumps(data, ensure_ascii=False, allow_nan=False)
            except Exception as exc:
                data = None
                error = ToolError("execution_failed", str(exc)[:500])
        return ToolResult(name, error is None, data, error, time.perf_counter() - started)
