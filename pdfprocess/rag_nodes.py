"""LangGraph 节点实现，每个节点只返回本次需要更新的状态字段。

阅读顺序为 prepare_history_node → rewrite_node → retrieve_node →
web_search_node（联网路线）→ compress_node → generate_node。
依赖通过关键字参数注入，节点本身不创建模型或打开数据库。
"""

# 问答节点层。依赖由建图函数注入，节点通过返回字典提交状态增量。

import hashlib
import time

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from .rag_state import RAGState
from .rag_prompts import prompt, rewrite_prompt
from .retrieval import PaperRetriever
from .web_search import clean_results
from .context_compression import (CompressionConfig, QuestionBudgetError, compress_evidence,
                                  estimate_tokens, message_tokens, summarize_history)
from .tools.search_flow import LOCAL_KNOWLEDGE_NOTICE, web_answer_inputs, web_answer_prompt


def prepare_history_node(state: RAGState, *, compression_config, summary_chain) -> dict:
    """先校验新问题预算，再生成增量摘要与近期模型历史视图。"""
    # 在任何模型或检索调用前拒绝过长的新问题，便于用户直接修正。
    if estimate_tokens(state["question"]) > compression_config.question_tokens:
        raise QuestionBudgetError("当前问题超过输入预算，请缩短后重新提交")
    return summarize_history(state, config=compression_config, summary_chain=summary_chain)


def rewrite_node(state: RAGState, *, query_rewriter, history_limit=None, compression_config=None) -> dict:
    """使用预算内短历史消解指代，返回独立检索问题、模型用量与耗时。"""
    # 改写也采用短历史视图，旧摘要帮助解析较远轮次中的指代。
    started = time.perf_counter()
    query = state["question"]
    history = state.get("messages", [])[-(history_limit or 12):]
    if compression_config:
        history = state.get("model_history", [])
    usage = {}
    if compression_config:
        # 自定义较小预算时逐个移除摘要或完整轮次，避免孤立 AI 消息。
        rewrite_size = message_tokens(rewrite_prompt.format_messages(question=query, history=history))
        while history and rewrite_size > compression_config.rewrite_input_tokens:
            # 摘要是单条 human 消息，近期问答则以 human/ai 成对移除。
            drop = 1
            if len(history) >= 2 and [m.type for m in history[:2]] == ["human", "ai"]:
                drop = 2
            history = history[drop:]
            rewrite_size = message_tokens(rewrite_prompt.format_messages(question=query, history=history))
        if rewrite_size > compression_config.rewrite_input_tokens:
            raise ValueError("改写提示词超过预算，请缩短问题或增大 rewrite_input_tokens")
    if history:
        rewritten = query_rewriter.invoke({"question": query, "history": history})
        if isinstance(rewritten, BaseMessage):
            usage = getattr(rewritten, "usage_metadata", None) or {}
            rewritten = rewritten.content
        if not isinstance(rewritten, str) or not rewritten.strip():
            raise RuntimeError("追问改写结果为空")
        query = rewritten.strip()
        if compression_config and estimate_tokens(query) > compression_config.question_tokens:
            raise ValueError("改写问题超过输入预算，请重新提交")
    return {"retrieval_query": query, "metrics": {**state.get("metrics", {}),
        "rewrite_seconds": time.perf_counter() - started, "rewrite_usage": usage}}


def retrieve_node(state: RAGState, *, retriever: PaperRetriever, include_source_text=False,
                  max_context_chars=None, web_search_threshold=None) -> dict:
    """召回及重排后先判断本地/联网路由，再由下游节点压缩对应证据。"""
    print("\n[Retrieve] 正在检索知识库...")

    try:
        started = time.perf_counter()
        results = retriever.search(state.get("retrieval_query") or state["question"])
        # 阈值使用原始候选的最高重排相关度，后续压缩不会改变该路由依据。
        scores = [hit.rerank_relevance for hit in results if hit.rerank_relevance is not None]
        best_score = max(scores, default=None)
        metrics = {**state.get("metrics", {}),
                   "retrieval_query": state.get("retrieval_query") or state["question"],
                   "local_max_relevance": best_score}
        if web_search_threshold is not None:
            # 只有启用 Cross-Encoder 的混合模式才会传入阈值。
            # 任一候选缺少分数属于配置/模型异常，需要明确报错。
            if results and len(scores) != len(results):
                raise ValueError("联网路由需要每条候选的 Cross-Encoder 分数")
            metrics["web_search_threshold"] = web_search_threshold
            if not results or best_score < web_search_threshold:
                # 严格使用 <，恰好为 0.5 时继续本地路线。
                metrics.update({"retrieve_seconds": time.perf_counter() - started,
                                "source_count": 0, "evidence_origin": "web",
                                "web_fallback_reason": "no_local_results" if not results else "low_relevance"})
                # 本地候选只用于阈值判断，联网分支的检查点从这里开始清空证据。
                return {"context": "", "sources": [], "evidence_origin": "web", "metrics": metrics}
        context_parts = []
        sources = []
        remaining = max_context_chars
        for i, hit in enumerate(results, start=1):
            # 保留 chunk_id 和页码，压缩后的来源仍可追溯到原始 PDF。
            doc = hit.document
            if remaining is not None and remaining <= 0:
                break
            text = doc.page_content
            if remaining is not None:
                # 兼容旧调用的字符截断。新压缩流程在建图时将该参数设为 None。
                text = text[:remaining]
                remaining -= len(text)
            source = doc.metadata.get("source", "未知来源")
            chunk_id = hit.chunk_id
            context_parts.append(
                f"[{i}] 来源：{source}\n"
                f"内容：{text}"
            )
            entry = {
                "id": i,
                "source": source,
                "chunk_id": chunk_id,
                "distance": hit.distance
            }
            for name in ("bm25_score", "rrf_score", "cross_encoder_score", "rerank_relevance"):
                # 只记录当前检索模式真正产生的分数，不伪造缺失分数。
                value = getattr(hit, name)
                if value is not None:
                    entry[name] = value
            if "page" in doc.metadata:
                entry["page"] = doc.metadata["page"]
            if include_source_text:
                # 压缩节点直接读取此文本，前端展示压缩后真正使用的证据。
                entry["text"] = text
                entry["document_id"] = doc.metadata.get("paper_id")
            sources.append(entry)
            print(f"[{i}] {source} | chunk_id={chunk_id}")
    except Exception as exc:
        # 包装业务错误并保留 __cause__，服务层可显示底层原因。
        raise RuntimeError("知识库检索失败，请检查 Embedding、Chroma 和重排模型配置。") from exc

    metrics.update({"retrieve_seconds": time.perf_counter() - started,
                    "source_count": len(sources), "evidence_origin": "local"})
    return {"context": "\n\n".join(context_parts), "sources": sources,
            "evidence_origin": "local", "metrics": metrics}


def web_search_node(state: RAGState, *, tool_registry, max_context_chars=None) -> dict:
    """执行注册表中的搜索调用，构造本轮网页证据供压缩节点处理。"""
    started = time.perf_counter()
    query = state.get("retrieval_query") or state["question"]
    call = state.get("search_call")
    if call is not None:
        # 在生产分支，这里的 query 来自模型返回的 function-call 参数。
        query = call.tool_calls[0]["args"]["query"]
    try:
        execution = tool_registry.invoke("web_search", {"query": query})
        if not execution.success:
            raise RuntimeError(execution.error.message)
        results = clean_results(execution.data)
        # clean_results 统一验证 URL、摘要和去重，后续只处理规范字典。
    except Exception as exc:
        raise RuntimeError(LOCAL_KNOWLEDGE_NOTICE + " 联网搜索失败，请检查网络后重试。") from exc
    context_parts = []
    sources = []
    remaining = max_context_chars
    for result in results:
        if remaining is not None and remaining <= 0:
            break
        text = result["content"]
        if remaining is not None:
            text = text[:remaining]
            remaining -= len(text)
        index = len(sources) + 1
        context_parts.append(f"[{index}] 网页：{result['title']}\n链接：{result['url']}\n摘要：{text}")
        sources.append({"id": index, "source": result["title"], "url": result["url"],
                        "text": text, "kind": "web", "distance": None,
                        "chunk_id": "web_" + hashlib.sha256(result["url"].encode("utf-8")).hexdigest()[:16]})
    return {"context": "\n\n".join(context_parts), "sources": sources, "evidence_origin": "web",
            "metrics": {**state.get("metrics", {}), "evidence_origin": "web",
                        "local_knowledge_notice": LOCAL_KNOWLEDGE_NOTICE,
                        "tool_call_mode": "model" if call is not None else "direct",
                        "tool_calls": [execution.record({"query": query})],
                        "web_search_seconds": time.perf_counter() - started,
                        "web_search_query": query, "source_count": len(sources)}}


def compress_node(state: RAGState, *, compression_config: CompressionConfig) -> dict:
    """计算整个答案提示词预算，再分配证据可用空间。"""
    started = time.perf_counter()
    config = compression_config
    use_web = state.get("evidence_origin") == "web"
    # 摘要也属于历史。联网回答一律排除，延续原有的来源隔离约束。
    history = [] if use_web else state.get("model_history", [])
    question = (state.get("retrieval_query") or state["question"]) if use_web else state["question"]
    if estimate_tokens(question) > config.question_tokens:
        raise QuestionBudgetError("当前问题超过输入预算，请缩短后重新提交")
    # 系统提示词、问题、历史和消息边界先占用额度，余量才用于证据。
    uses_tool_protocol = use_web and state.get("search_call") is not None
    if uses_tool_protocol:
        # 工具结果也走相同预算，计算时包含 assistant 调用和 ToolMessage 的开销。
        base_tokens = message_tokens(web_answer_prompt.format_messages(**web_answer_inputs(state, "")))
    else:
        base_tokens = message_tokens(prompt.format_messages(question=question, context="", history=history))
    available = min(config.evidence_tokens, config.max_input_tokens - base_tokens)
    if available <= 0:
        raise ValueError("问题和历史已占满输入预算，请增大 max_input_tokens")
    context, sources, metrics = compress_evidence(state.get("sources", []),
                                                 state.get("retrieval_query") or question, available)
    if uses_tool_protocol:
        before = message_tokens(web_answer_prompt.format_messages(**web_answer_inputs(state, state.get("context", ""))))
        after = message_tokens(web_answer_prompt.format_messages(**web_answer_inputs(state, context)))
    else:
        before = message_tokens(prompt.format_messages(question=question, context=state.get("context", ""), history=history))
        after = message_tokens(prompt.format_messages(question=question, context=context, history=history))
    if after > config.max_input_tokens:
        raise ValueError("压缩后的完整提示词仍超过输入预算")
    return {"context": context, "sources": sources, "model_history": history,
            "metrics": {**state.get("metrics", {}), **metrics,
                        "input_tokens_before_evidence_compression": before,
                        "input_tokens_after_compression": after,
                        "input_token_budget": config.max_input_tokens,
                        "history_tokens_used": message_tokens(history),
                        "output_token_reserve": config.max_output_tokens,
                        "compression_seconds": time.perf_counter() - started}}


def generate_node(state: RAGState, *, answer_chain: Runnable, history_limit=None,
                  compression_config=None, web_answer_chain=None) -> dict:
    """使用本轮证据与历史视图回答，成功后才追加一个完整问答轮次。"""
    started = time.perf_counter()
    usage = {}
    use_web = state.get("evidence_origin") == "web"
    if not state.get("context"):
        # 没有可用证据时直接给出固定反馈，避免模型凭历史或常识补出事实。
        answer = "联网搜索未找到可用资料，暂时无法回答。" if use_web else "知识库中未检索到相关资料。"
        if compression_config and state.get("metrics", {}).get("source_count_before_compression", 0):
            answer = "本轮资料在上下文预算内没有可用的完整证据，请缩小问题范围或增大证据预算。"
    else:
        print("\n[Generate] DeepSeek 正在生成答案...")
        try:
            history = state.get("model_history", []) if compression_config else state.get("messages", [])
            if compression_config is None and history_limit is not None:
                history = history[-history_limit:]
            # 联网回答只接收独立问题和本轮网页证据，排除历史本地回答。
            if use_web:
                history = []
            if use_web and state.get("search_call") is not None:
                # 原生工具协议继续给模型，回答仅依据当前压缩后的工具结果。
                answer = web_answer_chain.invoke(web_answer_inputs(state, state["context"]))
            else:
                # 原有注入 answer_chain 的实验接口保留 question/context/history 约定。
                answer = answer_chain.invoke({
                    "question": (state.get("retrieval_query") or state["question"]) if use_web else state["question"],
                    "context": state["context"],
                    "history": history
                })
            if isinstance(answer, BaseMessage):
                if getattr(answer, "tool_calls", None) or getattr(answer, "invalid_tool_calls", None):
                    raise ValueError("答案阶段应返回文本，收到意外的工具调用")
                # 保留真实 API 用量，包括可能存在的缓存 Token 明细。
                usage = getattr(answer, "usage_metadata", None) or {}
                answer = answer.content
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("模型返回了空回答或非文本回答")
        except Exception as exc:
            notice = LOCAL_KNOWLEDGE_NOTICE + " " if use_web else ""
            raise RuntimeError(notice + "答案生成失败，请检查 DeepSeek 服务后重新提交问题。") from exc

    if use_web and not answer.lstrip().startswith(LOCAL_KNOWLEDGE_NOTICE):
        # 提示词要求与程序兜底共同确保搜索成功、空结果等回答都会说明本地不足。
        answer = LOCAL_KNOWLEDGE_NOTICE + "\n\n" + answer

    # 完整轮次一起写入消息历史，失败轮次保留在检查点的中间状态中。
    turn_id = state.get("turn_id")
    # 稳定 ID 使 add_messages 在重试相同轮次时替换对应消息。
    return {
        "answer": answer,
        "metrics": {**state.get("metrics", {}), "generate_seconds": time.perf_counter() - started,
                    "generation_usage": usage},
        "messages": [
            HumanMessage(content=state["question"], id=f"{turn_id}:human" if turn_id else None),
            AIMessage(content=answer, id=f"{turn_id}:ai" if turn_id else None)
        ]
    }
