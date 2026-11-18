"""RAG 的依赖组装、LangGraph 建图、单轮调用和命令行入口。

提示词见 rag_prompts.py，状态字段见 rag_state.py，节点逻辑见 rag_nodes.py。
本模块继续导出原有 prompt、RAGState 与节点函数，便于已有调用入口使用。
"""

# 问答组装入口。build_graph() 注入依赖并连接节点，ask() 为新问题设置临时状态。

import math
import os
from functools import partial
from pathlib import Path
from uuid import uuid4

import dashscope
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

if __package__:
    from .embeddings import TongyiEmbeddings
    from .retrieval import PaperRetriever, RetrievalConfig
    from .web_search import web_search
    from .context_compression import CompressionConfig, summary_prompt
    from .rag_state import RAGState
    from .rag_prompts import prompt, rewrite_prompt
    from .rag_nodes import (prepare_history_node, rewrite_node, retrieve_node, web_search_node,
                            compress_node, generate_node)
    from .tools.defaults import create_default_registry
    from .tools.search_flow import create_search_chains, request_search_node
    from .tools.paper_agent import paper_request_kind, run_paper_agent
else:
    from embeddings import TongyiEmbeddings
    from retrieval import PaperRetriever, RetrievalConfig
    from web_search import web_search
    from context_compression import CompressionConfig, summary_prompt
    from rag_state import RAGState
    from rag_prompts import prompt, rewrite_prompt
    from rag_nodes import (prepare_history_node, rewrite_node, retrieve_node, web_search_node,
                           compress_node, generate_node)
    from tools.defaults import create_default_registry
    from tools.search_flow import create_search_chains, request_search_node
    from tools.paper_agent import paper_request_kind, run_paper_agent


PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = Path(__file__).resolve().parent / "data" / "chroma_db"


def build_graph(
    *,
    vector_store: VectorStore | None = None,
    answer_chain: Runnable | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    collection_name: str = "paper_chunks",
    retrieval_config: RetrievalConfig | None = None,
    retriever: PaperRetriever | None = None,
    cross_encoder=None,
    query_rewriter=None,
    history_limit=None,
    max_context_chars=None,
    include_source_text=False,
    web_search_tool=web_search,
    web_search_threshold=0.5,
    compression_config: CompressionConfig | None = None,
    summary_chain=None,
    tool_registry=None,
    tool_call_chain=None,
    web_answer_chain=None,
    paper_model=None,
) -> CompiledStateGraph:
    """构建一次并复用，测试可注入检索器、模型和 Checkpointer。

    注入 answer_chain 的既有调用保持原行为。需要压缩时同时传入
    compression_config 和 summary_chain。默认 CLI 的真实模型会自动启用压缩。
    模型搜索分支同时注入 tool_call_chain 和 web_answer_chain，工具经注册表执行。
    """
    retrieval_config = retrieval_config or RetrievalConfig()
    registry = tool_registry if tool_registry is not None else create_default_registry(web_search_tool)
    # 调用方可注入已经登记测试工具或自定义工具的目录，生产默认目录在服务启动时创建。
    if (vector_store is None and retriever is None) or answer_chain is None:
        load_dotenv(PROJECT_DIR / ".env")

    if vector_store is None and retriever is None:
        # 有现成检索器时直接复用。仅默认运行路径创建 Chroma 与 Embedding。
        embedding_function = None
        if retrieval_config.mode != "bm25":
            dashscope_key = os.getenv("DASHSCOPE_API_KEY")
            if not dashscope_key:
                raise ValueError("未配置 DASHSCOPE_API_KEY")
            dashscope.api_key = dashscope_key
            embedding_function = TongyiEmbeddings()
        try:
            vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=str(CHROMA_DIR),
                embedding_function=embedding_function,
                create_collection_if_not_exists=False
            )
        except Exception as exc:
            raise RuntimeError(f"无法加载 Chroma 数据库：{CHROMA_DIR}") from exc

    if retriever is None:
        retriever = PaperRetriever(vector_store, retrieval_config, cross_encoder=cross_encoder)

    enable_web = web_search_tool is not None and retriever.config.mode == "hybrid_rerank"
    # 当前路由只在存在 Cross-Encoder 分数的混合重排模式下评估阈值。
    # 显式关闭工具可用于本地基线实验，其他检索模式沿用本地路线。
    if (isinstance(web_search_threshold, bool) or not isinstance(web_search_threshold, (int, float))
            or not math.isfinite(web_search_threshold) or not 0 <= web_search_threshold <= 1):
        raise ValueError("联网搜索阈值需要在 0 到 1 之间")

    if answer_chain is None:
        deepseek_key = os.getenv("DEEPSEEK_API")
        if not deepseek_key:
            raise ValueError("未配置 DEEPSEEK_API")
        try:
            llm = ChatDeepSeek(
                model="deepseek-chat",
                api_key=deepseek_key,
                temperature=0,
                timeout=60,
                max_retries=2
            )
            compression_config = compression_config or CompressionConfig.from_environment()
            paper_model = paper_model or llm
            # API 的 max_tokens 控制实际输出，压缩模块另外限制输入。
            answer_chain = prompt | llm.bind(max_tokens=compression_config.max_output_tokens)
            summary_chain = summary_chain or summary_prompt | llm.bind(max_tokens=512)
            if enable_web and tool_call_chain is None:
                tool_call_chain, web_answer_chain = create_search_chains(
                    llm, registry, max_output_tokens=compression_config.max_output_tokens)
        except Exception as exc:
            raise RuntimeError("DeepSeek 初始化失败，请检查模型配置。") from exc

    if compression_config is not None and summary_chain is None:
        raise ValueError("启用上下文压缩时需要提供 summary_chain")
    if (tool_call_chain is None) != (web_answer_chain is None):
        raise ValueError("tool_call_chain 与 web_answer_chain 需要一起提供")

    builder = StateGraph(RAGState)
    # partial 预先绑定检索器、模型和预算，LangGraph 运行节点时只需传入 state。
    # 压缩启用后读取完整候选文本，避免旧的字符截断先丢掉相关句子。
    # 老调用的字符限制仍保留，以兼容此前的对比实验。
    builder.add_node("retrieve", partial(
        retrieve_node,
        retriever=retriever,
        include_source_text=include_source_text or compression_config is not None,
        max_context_chars=None if compression_config else max_context_chars,
        web_search_threshold=web_search_threshold if enable_web else None,
    ))
    builder.add_node("generate", partial(
        generate_node,
        answer_chain=answer_chain,
        history_limit=history_limit,
        compression_config=compression_config,
        web_answer_chain=web_answer_chain,
    ))
    first_node = "retrieve"
    if query_rewriter is not None:
        builder.add_node("rewrite", partial(rewrite_node, query_rewriter=query_rewriter,
                                            history_limit=history_limit, compression_config=compression_config))
        builder.add_edge("rewrite", "retrieve")
        first_node = "rewrite"
    if compression_config is not None:
        builder.add_node("prepare_history", partial(prepare_history_node,
                                                    compression_config=compression_config, summary_chain=summary_chain))
        builder.add_edge(START, "prepare_history")
        builder.add_node("compress", partial(compress_node, compression_config=compression_config))
        builder.add_edge("compress", "generate")
    route_from = "prepare_history" if compression_config is not None else START
    if paper_model is not None and any(item["name"] in {"search_papers", "download_paper"} for item in registry.list_tools()):
        builder.add_node("paper_agent", partial(run_paper_agent, model=paper_model, registry=registry,
                                                 collection_name=collection_name, compression_config=compression_config))
        builder.add_conditional_edges(route_from,
            lambda state: "papers" if paper_request_kind(state["question"]) else "rag",
            {"papers": "paper_agent", "rag": first_node})
        builder.add_edge("paper_agent", END)
    else:
        builder.add_edge(route_from, first_node)
    # 两条证据路线汇入同一个压缩入口，阈值判定始终在它之前。
    answer_entry = "compress" if compression_config else "generate"
    if enable_web:
        # request_search 只生成标准工具调用，web_search 节点负责实际执行并整理网页证据。
        builder.add_node("web_search", partial(web_search_node, tool_registry=registry,
                                               max_context_chars=None if compression_config else max_context_chars))
        search_entry = "web_search"
        if tool_call_chain is not None:
            builder.add_node("request_search", partial(request_search_node, tool_call_chain=tool_call_chain,
                                                        registry=registry, compression_config=compression_config))
            builder.add_edge("request_search", "web_search")
            search_entry = "request_search"
        builder.add_conditional_edges("retrieve", lambda state: state["evidence_origin"],
                                      {"local": answer_entry, "web": search_entry})
        builder.add_edge("web_search", answer_entry)
    else:
        builder.add_edge("retrieve", answer_entry)
    builder.add_edge("generate", END)
    # 每条路线最终都由 generate 追加完整问答对，再进入 END 形成可恢复的完成状态。

    if checkpointer is None:
        # CLI 默认在进程内持久化，服务入口传入官方 SqliteSaver 跨重启保存。
        checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)


def ask(graph: CompiledStateGraph, question: str, thread_id: str, *, turn_id=None) -> RAGState:
    """校验新一轮输入，并返回含 answer、sources 和 messages 的状态。"""
    if not isinstance(question, str):
        raise TypeError("question 必须是字符串")
    if not question.strip():
        raise ValueError("question 不能为空")
    if not isinstance(thread_id, str):
        raise TypeError("thread_id 必须是字符串")
    if not thread_id.strip():
        raise ValueError("thread_id 不能为空")

    # 每轮重置临时结果，避免失败检查点中的结果沿用上一轮内容。
    # 这里省略 messages 和三个摘要字段，让 Checkpointer 继续保留它们。
    return graph.invoke(
        {"question": question, "context": "", "sources": [], "answer": "",
         "turn_id": turn_id, "retrieval_query": question, "metrics": {}, "evidence_origin": "local",
         "model_history": [], "search_call": None},
        config={"configurable": {"thread_id": thread_id}}
    )


def main(*, retrieval_config=None, collection_name="paper_chunks") -> int:
    """构建并复用问答图，接收会话 ID 与多轮 CLI 问题，返回程序退出码。"""
    try:
        graph = build_graph(retrieval_config=retrieval_config, collection_name=collection_name)
    except (ValueError, RuntimeError) as exc:
        print(f"初始化失败：{exc}")
        return 1

    thread_id = input("会话 ID（直接回车自动生成）：").strip() or str(uuid4())
    print(f"当前会话：{thread_id}，输入 exit 结束对话。")
    while True:
        question = input("\n问题：")
        if question.strip().lower() == "exit":
            return 0
        try:
            result = ask(graph, question, thread_id)
        except (ValueError, RuntimeError) as exc:
            print(f"本轮失败：{exc}")
            continue

        print("\n========== RAG 最终回答 ==========")
        print(result["answer"])
        print("\n========== 检索来源 ==========")
        for source in result["sources"]:
            print(source)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PaperAgent 多轮 RAG")
    parser.add_argument("--collection", default="paper_chunks")
    parser.add_argument("--mode", choices=("dense", "bm25", "hybrid", "hybrid_rerank"), default="hybrid_rerank")
    parser.add_argument("--dense-k", type=int, default=20)
    parser.add_argument("--keyword-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    try:
        config = RetrievalConfig(mode=args.mode, dense_k=args.dense_k, keyword_k=args.keyword_k,
                                 top_k=args.top_k, reranker_model_dir=args.model_dir)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        raise SystemExit(main(retrieval_config=config, collection_name=args.collection))
    except (EOFError, KeyboardInterrupt):
        print("\n对话结束。")
