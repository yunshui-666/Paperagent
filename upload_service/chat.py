"""问答业务层。协调会话锁、模型依赖、LangGraph 检查点、请求恢复和业务结果落盘。"""

from collections import OrderedDict
from contextlib import contextmanager
import os
from pathlib import Path
from threading import Lock
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.sqlite import SqliteSaver
from markdown_it import MarkdownIt

from pdfprocess.cross_encoder import OnnxCrossEncoder
from pdfprocess.embeddings import TongyiEmbeddings
from pdfprocess.rag import ask, build_graph, prompt, rewrite_prompt
from pdfprocess.context_compression import CompressionConfig, QuestionBudgetError, summary_prompt
from pdfprocess.retrieval import PaperRetriever, RetrievalConfig
from pdfprocess.tools.defaults import create_default_registry
from pdfprocess.tools.search_flow import LOCAL_KNOWLEDGE_NOTICE, create_search_chains
from pdfprocess.tools.papers import create_download_tool
from .errors import ProjectError


MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"]).disable("image")


class ChatService:
    """连接 HTTP 业务记录、RAG 工作流和 SQLite 检查点。

    业务库保存用户可见的会话/问答/证据，Checkpointer 保存图状态及摘要。
    阅读问答入口可从 chat() → _graph() → rag.ask() 开始。
    """

    def __init__(self, store, settings, documents, graph_factory=None, *, ingestion=None):
        # 依赖由应用入口注入，测试可以用 graph_factory 替换真实模型调用。
        """保存业务依赖，创建工具目录和 SQLite Checkpointer，并恢复中断的业务轮次。"""
        self.store = store
        self.settings = settings
        self.documents = documents
        self.graph_factory = graph_factory
        self.ingestion = ingestion
        # 工具集中在 defaults.py 登记，各会话复用同一目录，执行结果按会话独立保存。
        self.tool_registry = create_default_registry()
        self._checkpoint_context = SqliteSaver.from_conn_string(str(Path(settings.service_root).resolve() / "conversations.sqlite3"))
        # SqliteSaver 是上下文管理器。连接保持到服务关闭，支持进程重启恢复。
        self.checkpointer = self._checkpoint_context.__enter__()
        try:
            self.checkpointer.setup()
            # 重排模型由各个会话共享，避免重复加载 ONNX 权重。
            self.cross_encoder = OnnxCrossEncoder(os.getenv("PAPERAGENT_RERANKER_MODEL_DIR"))
            self._locks = {}
            self._graphs = OrderedDict()
            self._guard = Lock()
            # 服务上次可能在“模型已返回、业务库尚未落盘”之间中断。
            # 优先从已完成的图检查点补齐业务记录，减少重复生成。
            for turn in self.store.interrupted_turns():
                recovered = self._checkpoint_result(turn)
                if recovered:
                    self.store.finish_turn(turn["id"], recovered["answer"], recovered["sources"], recovered.get("metrics", {}))
                else:
                    self.store.fail_turn(turn["id"], "服务中断了本次问答，可以重试该问题")
        except BaseException:
            self.close()
            raise

    def close(self):
        """由应用生命周期调用，释放检查点数据库连接。"""
        self._checkpoint_context.__exit__(None, None, None)

    @contextmanager
    def session_lock(self, session_id):
        # 同会话只允许一个问题执行，防止两个请求基于同一旧历史并发更新摘要。
        # 不同 session_id 各持有一把锁，可以各自推进。
        """为同会话的问答与删除提供非阻塞互斥，冲突时返回可读业务错误。"""
        with self._guard:
            lock = self._locks.setdefault(session_id, Lock())
        if not lock.acquire(blocking=False):
            raise ProjectError("该会话正在回答，请等待当前问题完成")
        try:
            yield
        finally:
            lock.release()

    def create_session(self, title, collection_name, options):
        # 验证知识库和文档就绪状态，检索器稍后复用这些已验证的参数。
        """校验知识库、所选文档就绪状态及检索参数，再保存新会话。"""
        if collection_name not in {item["name"] for item in self.documents.collections()}:
            raise ProjectError("知识库尚未创建，请先完成文档入库", 404)
        selected = options.get("document_ids", [])
        _, ready = self.store.availability(collection_name)
        if any(document_id not in ready for document_id in selected):
            raise ProjectError("所选文档尚未在该知识库中完成入库", 422)
        RetrievalConfig(mode=options["mode"], dense_k=options["dense_k"], keyword_k=options["keyword_k"], top_k=options["top_k"])
        return self.store.create_session(title, collection_name, options)

    def _graph(self, session):
        """按会话懒加载并缓存图，缓存淘汰后从持久化检查点恢复。"""
        with self._guard:
            # OrderedDict 实现最近使用缓存，上限 16 只约束内存中的图对象。
            # 原始消息和摘要保存在 SQLite，重新建图后仍然可用。
            if session["id"] in self._graphs:
                self._graphs.move_to_end(session["id"])
                return self._graphs[session["id"]]
            if self.graph_factory is not None:
                graph = self.graph_factory(session, self.checkpointer)
            else:
                # 真实调用才读取配置，单元测试注入依赖时无需任何 API 密钥。
                load_dotenv(Path(__file__).resolve().parent.parent / ".env")
                options = session["options"]
                config = RetrievalConfig(mode=options["mode"], dense_k=options["dense_k"],
                                         keyword_k=options["keyword_k"], top_k=options["top_k"])
                key = os.getenv("DEEPSEEK_API")
                if not key:
                    raise ProjectError("请在项目 .env 配置 DEEPSEEK_API", 503)
                embedding = None
                # BM25 直接对文本做关键词检索，其他模式需要问题向量。
                if config.mode != "bm25":
                    embedding_key = os.getenv("DASHSCOPE_API_KEY")
                    if not embedding_key:
                        raise ProjectError("请在项目 .env 配置 DASHSCOPE_API_KEY", 503)
                    embedding = TongyiEmbeddings(api_key=embedding_key)
                model = ChatDeepSeek(model=os.getenv("PAPERAGENT_CHAT_MODEL", "deepseek-chat"), api_key=key,
                                     temperature=0, timeout=60, max_retries=2)
                # 学习时可从这里修改预算，也可修改 CompressionConfig 的默认值。
                # 参数单位为保守 Token 估算，实际用量记录在每轮 metrics 中。
                compression = CompressionConfig.from_environment()
                registry = self.tool_registry.copy()
                if self.ingestion is not None and any(item["name"] == "search_papers" for item in registry.list_tools()):
                    registry.register(create_download_tool(self.ingestion, self.store, session["collection_name"]), tags=("paper", "ingestion"))
                vector_store = self.documents.scoped_store(session["collection_name"], embedding, options["document_ids"])
                retriever = PaperRetriever(vector_store, config, cross_encoder=self.cross_encoder)
                rewriter = None
                if options["rewrite_query"]:
                    # 改写链只负责消解指代，其输出送入现有检索流程。
                    rewriter = rewrite_prompt | model.bind(max_tokens=256)
                # 摘要链与答案链共享客户端，提示词、输入预算和输出上限分别控制。
                # 超过近期历史窗口时才调用摘要模型，短对话没有摘要调用开销。
                summary_chain = summary_prompt | model.bind(max_tokens=512)
                tool_call_chain, web_answer_chain = None, None
                if config.mode == "hybrid_rerank":
                    # 只有低相关度路由会执行该调用链，正常本地回答不增加模型调用。
                    tool_call_chain, web_answer_chain = create_search_chains(
                        model, registry, max_output_tokens=compression.max_output_tokens)
                graph = build_graph(retriever=retriever,
                                    answer_chain=prompt | model.bind(max_tokens=compression.max_output_tokens),
                                    checkpointer=self.checkpointer, query_rewriter=rewriter,
                                    include_source_text=True, compression_config=compression,
                                    summary_chain=summary_chain, tool_registry=registry,
                                    tool_call_chain=tool_call_chain, web_answer_chain=web_answer_chain,
                                    paper_model=model, collection_name=session["collection_name"])
            config = {"configurable": {"thread_id": session["id"]}}
            if self.checkpointer.get_tuple(config) is None:
                # 兼容已有业务数据：若图检查点缺失，用成功问答重建完整消息。
                # 摘要字段可缺省，下次 prepare_history 节点按新预算自动生成。
                messages = []
                catalog = []
                for turn in self.store.turns(session["id"]):
                    if turn["status"] == "completed":
                        if turn.get("metrics", {}).get("evidence_origin") == "papers":
                            catalog = [{"arxiv_id": item["arxiv_id"], "title": item["source"],
                                        "summary": item.get("text", ""), "url": item["url"],
                                        "pdf_url": f"https://arxiv.org/pdf/{item['arxiv_id']}"}
                                       for item in turn["sources"] if item.get("arxiv_id")]
                        messages.extend([
                            HumanMessage(content=turn["question"], id=f"{turn['id']}:human"),
                            AIMessage(content=turn["answer"], id=f"{turn['id']}:ai")
                        ])
                if messages:
                    graph.update_state(config, {"messages": messages, "paper_catalog": catalog}, as_node="generate")
            self._graphs[session["id"]] = graph
            if len(self._graphs) > 16:
                self._graphs.popitem(last=False)
            return graph

    def _checkpoint_result(self, turn):
        """恢复一个确实完成的问答，摘要完成或检索完成都不代表答案完成。"""
        config = {"configurable": {"thread_id": turn["session_id"]}}
        checkpoint = self.checkpointer.get_tuple(config)
        if checkpoint is None:
            return None
        values = checkpoint.checkpoint.get("channel_values", {})
        message_ids = {message.id for message in values.get("messages", [])}
        expected = {f"{turn['id']}:human", f"{turn['id']}:ai"}
        # 两个稳定 ID 同时存在才认为本轮生成成功，重试不会重复追加消息。
        if not expected.issubset(message_ids):
            return None
        if values.get("turn_id") == turn["id"] and values.get("answer"):
            return values
        # 后续轮次已完成时，从该会话的历史检查点恢复原轮次证据快照。
        for previous in self.checkpointer.list(config):
            values = previous.checkpoint.get("channel_values", {})
            if values.get("turn_id") == turn["id"] and values.get("answer"):
                return values
        return None

    def _format_turn(self, turn):
        # 渲染 Markdown 时关闭 HTML，并为可访问的本地证据附加 PDF 页码链接。
        """渲染答案 Markdown，并为仍可访问的本地证据构造 PDF 页码链接。"""
        turn = dict(turn)
        turn["answer_html"] = MARKDOWN.render(turn.get("answer") or "")
        sources = []
        for source in turn["sources"]:
            source = dict(source)
            source["file_url"] = None
            document_id = source.get("document_id")
            if document_id:
                try:
                    self.documents.file_path(document_id)
                    page = source.get("page", 1)
                    source["file_url"] = f"/documents/{document_id}/file#page={page}"
                except ProjectError:
                    pass
            sources.append(source)
        turn["sources"] = sources
        return turn

    def history(self, session_id):
        # 前端读取完整业务历史，模型的短历史视图不会改变这里的返回内容。
        """返回会话信息与完整业务轮次，供前端恢复已有对话。"""
        session = self.store.session(session_id)
        session["turns"] = [self._format_turn(turn) for turn in self.store.turns(session_id)]
        return session

    def chat(self, session_id, request_id, question):
        """对外问答入口，处理幂等请求、串行执行、恢复与业务落盘。"""
        with self.session_lock(session_id):
            session = self.store.session(session_id)
            turn = self.store.reserve_turn(session_id, request_id, question)
            # 相同 request_id 已成功时直接返回原结果，避免重复花费模型调用。
            if turn["status"] == "completed":
                return self._format_turn(turn)
            try:
                result = self._checkpoint_result(turn)
                started = time.perf_counter()
                if result is None:
                    # 失败旧请求重试时按真实执行顺序排队，保持历史顺序一致。
                    self.store.order_turn_for_execution(turn["id"])
                    result = ask(self._graph(session), question, session_id, turn_id=turn["id"])
                metrics = {**result.get("metrics", {}), "request_seconds": time.perf_counter() - started}
                # 新增压缩指标沿用原有 metrics 字段存储，无需数据库迁移。
                self.store.finish_turn(turn["id"], result["answer"], result["sources"], metrics)
                if session["title"] == "新对话":
                    self.store.rename_session(session_id, question[:24])
            except Exception as exc:
                # 摘要、改写、检索和生成任一环节失败，都保留可重试的业务状态。
                self.store.fail_turn(turn["id"], str(exc))
                if isinstance(exc, ProjectError):
                    raise
                if isinstance(exc, QuestionBudgetError):
                    raise ProjectError(str(exc), 422) from exc
                cause = str(exc.__cause__) if exc.__cause__ else str(exc)
                if str(exc).startswith(LOCAL_KNOWLEDGE_NOTICE):
                    cause = str(exc)
                raise ProjectError(f"问答失败：{cause}", 502) from exc
            saved = next(item for item in self.store.turns(session_id) if item["id"] == turn["id"])
            return self._format_turn(saved)

    def delete_session(self, session_id):
        # 会话删除时检查点中的完整历史及摘要一起删除。
        """在会话锁内删除持久化检查点和业务会话，并移除图缓存。"""
        with self.session_lock(session_id):
            self.checkpointer.delete_thread(session_id)
            self.store.delete_session(session_id)
            with self._guard:
                self._graphs.pop(session_id, None)
        return {"status": "deleted"}
