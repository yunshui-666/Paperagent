"""LangGraph 状态契约，集中说明消息、摘要与当前轮次结果。"""

# 状态契约层。messages 与摘要跨轮保存，证据、答案和工具调用按当前轮次更新。

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langgraph.graph.message import add_messages


class RAGState(TypedDict, total=False):
    """图节点共享的状态契约。messages 使用 reducer，其他字段按节点结果覆盖更新。"""
    # 临时字段在 ask() 中逐轮重置，避免失败重试沿用上轮证据。
    question: str  # 用户本轮原始提问，成功后写入完整消息历史。
    context: str  # 本轮实际使用的证据文本，经过来源选择和预算处理。
    sources: list[dict]  # 来源编号、Chunk ID、页码或 URL，以及证据文本快照。
    answer: str  # 本轮最终可展示答案，失败轮次保持空字符串。
    messages: Annotated[list[AnyMessage], add_messages]
    # messages 的 reducer 按消息 ID 追加/替换，完整历史不被摘要删减。
    # 以下三个摘要字段与 messages 一起由 Checkpointer 按 thread_id 持久化。
    conversation_summary: str  # 已压缩的较早历史，仅辅助理解本地问答问题。
    summary_covered: int  # 摘要覆盖的原始消息前缀长度。
    summary_fingerprint: str  # 前缀 ID、角色和正文的哈希，用于识别旧摘要失效。
    # model_history 是本轮选出的短历史视图，可以与完整 messages 长度不同。
    model_history: list[AnyMessage]
    turn_id: str | None  # 业务轮次身份，组合为稳定的 human/ai 消息 ID。
    retrieval_query: str  # 问题改写后的独立查询，未改写时使用原问题。
    metrics: dict  # 各阶段耗时、原始模型 usage、路由原因和工具执行记录。
    evidence_origin: str  # local/web 由检索判定，papers 由论文工具节点输出。
    # 本轮模型产生的工具调用单独保存，完整对话仍只追加成功的用户/答案消息对。
    search_call: AIMessage | None
    paper_catalog: list[dict]  # 最近一次论文检索目录，支持跨轮下载指代，最多十条。
