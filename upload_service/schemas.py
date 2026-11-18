"""HTTP 请求模型，只定义输入字段、默认值与基础校验约束。"""

# HTTP 输入契约。字段默认值与 Pydantic 约束在这里集中定义，业务关系由服务层进一步校验。

from pydantic import BaseModel, Field


class SessionRequest(BaseModel):
    """会话创建输入，记录知识库、文档范围、检索模式、候选数量及问题改写选项。"""
    title: str = Field(default="新对话", min_length=1, max_length=120)
    collection_name: str = Field(min_length=1, max_length=512)
    document_ids: list[str] = Field(default_factory=list)
    mode: str = "hybrid_rerank"
    dense_k: int = Field(default=20, ge=1)
    keyword_k: int = Field(default=20, ge=1)
    top_k: int = Field(default=10, ge=1)
    rewrite_query: bool = True


class ChatRequest(BaseModel):
    """问答输入，thread_id 标识会话，request_id 标识同会话内的幂等请求。"""
    thread_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=3000)


class RenameRequest(BaseModel):
    """会话重命名输入，标题长度在模型层验证，空白内容由 HTTP 入口验证。"""
    title: str = Field(min_length=1, max_length=120)


class ReindexRequest(BaseModel):
    """重建索引输入，scheme 选择解析方案，refresh 决定是否重新生成缓存结果。"""
    scheme: int = Field(default=3, ge=1, le=3)
    refresh: bool = True
