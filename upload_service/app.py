"""FastAPI 文件上传、持久化任务及任务查询入口。"""

# HTTP 适配层。解析请求、选择响应状态并调用业务对象，业务对象在 lifespan 中集中初始化。

from contextlib import asynccontextmanager
from functools import partial
import logging
import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import psutil
from starlette.concurrency import run_in_threadpool

from pdfprocess.ingest_common import processing_lock
from .config import ServiceSettings
from .schemas import SessionRequest, ChatRequest, RenameRequest, ReindexRequest
from .ingestion import IngestionService
from .application_store import ApplicationStore
from .chat import ChatService, MARKDOWN
from .documents import DocumentManager
from .errors import ProjectError
from .storage import FileStorage, UploadRejected
from .worker import ParserPool, WorkerPool, run_pipeline


logger = logging.getLogger(__name__)


def process_usage():
    """采样当前服务及子进程的 RSS 和累计 CPU 时间，供健康检查与上传实验记录。"""
    root = psutil.Process(os.getpid())
    processes = [root] + root.children(recursive=True)
    rss = 0
    cpu_seconds = 0.0
    for process in processes:
        try:
            rss += process.memory_info().rss
            cpu = process.cpu_times()
            cpu_seconds += cpu.user + cpu.system
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {"pid": root.pid, "process_count": len(processes),
            "rss_bytes": rss, "cpu_seconds": cpu_seconds}


def create_app(settings=None, *, runner=None, graph_factory=None):
    """组装 FastAPI、生命周期和路由，支持注入入库 runner 与问答图工厂进行隔离测试。"""
    settings = settings or ServiceSettings.from_environment()
    root = Path(settings.service_root).resolve()
    store = ApplicationStore(root / "service.sqlite3", settings.sqlite_journal_mode)
    # 工厂阶段组装依赖，持久化结构和后台任务在线程进入 lifespan 时初始化。
    storage = FileStorage(root / "files", settings.max_file_bytes)

    @asynccontextmanager
    async def lifespan(application):
        """在单实例文件锁内恢复任务和删除状态，启动后台工作对象并在关闭时释放资源。"""
        with processing_lock(root / ".service.lock"):
            # 单实例锁在整个服务生命周期持有，防止多个服务同时领取同一份本机队列。
            store.initialize()
            cleaned = storage.cleanup_staging()
            recovered = store.recover_interrupted()
            documents = DocumentManager(store, settings)
            documents.recover()
            # 删除恢复完成后再创建问答服务，检索范围可按最终文档状态计算。
            parser_pool = ParserPool(settings.parse_processes)
            actual_runner = runner or partial(run_pipeline, parser_pool=parser_pool)
            workers = WorkerPool(store, actual_runner, settings.worker_count)
            ingestion = IngestionService(store, storage, settings, workers.notify)
            chat = ChatService(store, settings, documents, graph_factory=graph_factory, ingestion=ingestion)
            application.state.task_store = store
            application.state.workers = workers
            application.state.recovered_tasks = recovered
            application.state.staging_files_cleaned = cleaned
            application.state.documents = documents
            application.state.chat = chat
            application.state.ingestion = ingestion
            # application.state 是 HTTP 请求与本应用业务对象的连接点。
            try:
                workers.start()
                yield
            finally:
                # 先等工作线程退出，再关闭解析池和会话检查点，避免任务访问已关闭资源。
                await run_in_threadpool(workers.stop)
                await run_in_threadpool(parser_pool.stop)
                chat.close()

    application = FastAPI(
        title="PaperAgent",
        description="论文入库、文档管理、证据检索与持久化多轮问答。",
        version="2.0.0", lifespan=lifespan
    )


    @application.post("/documents", status_code=202, summary="上传一个或多个 PDF")
    async def upload_documents(
        files: Annotated[list[UploadFile], File(description="可选择多个 PDF 文件")],
        scheme: Annotated[int, Form(ge=1, le=3, description="1 本地文本，2 本地 OCR，3 本地 OCR + VLM 双路")] = 3
    ):
        """逐份接收 PDF，在线程池执行阻塞存储操作，汇总接收结果与 HTTP 状态。"""
        results = []
        for upload in files:
            # 每份文件单独处理，批次中其他有效 PDF 可以继续接收并得到独立 task_id。
            try:
                result = await run_in_threadpool(application.state.ingestion.submit, upload.filename, upload.file, scheme)
                # 文件读写和 PDF 结构校验放在线程池，异步请求处理器等待其结果。
            except UploadRejected as exc:
                result = {"filename": upload.filename, "accepted": False,
                          "error": {"code": exc.code, "message": str(exc), "http_status": exc.http_status}}
            except ProjectError as exc:
                result = {"filename": upload.filename, "accepted": False,
                          "error": {"code": "document_busy", "message": str(exc), "http_status": exc.status_code}}
            except Exception as exc:
                logger.exception("上传接收或任务入队失败")
                result = {"filename": upload.filename, "accepted": False,
                          "error": {"code": "submission_failed", "message": str(exc), "http_status": 503}}
            finally:
                # UploadFile 可能使用磁盘临时文件，逐份关闭释放底层资源。
                await upload.close()
            results.append(result)
        accepted = sum(item["accepted"] for item in results)
        status_code = 202
        if not accepted:
            # 整批失败时按大小、冲突或服务错误选择响应码，逐项错误仍保留在 items。
            status_code = 422
            if any(item["error"]["http_status"] == 503 for item in results):
                status_code = 503
            elif results and all(item["error"]["http_status"] == 413 for item in results):
                status_code = 413
            elif results and all(item["error"]["http_status"] == 409 for item in results):
                status_code = 409
        return JSONResponse(status_code=status_code, content={
            "accepted_count": accepted, "rejected_count": len(results) - accepted, "items": results
        })

    @application.get("/tasks/{task_id}", summary="查询任务状态、阶段、耗时和结果")
    def get_task(task_id: str):
        """返回持久化任务、阶段事件及耗时，未知任务返回 404。"""
        task = store.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="任务不存在")
        return task

    @application.get("/health", summary="查询服务、队列及进程资源采样")
    def health():
        """返回线程存活、队列计数、服务配置与进程采样值。"""
        return {
            "status": "ok", "workers_alive": application.state.workers.alive_count(),
            "worker_count": settings.worker_count, "parse_processes": settings.parse_processes,
            "runner_name": getattr(runner, "__name__", type(runner).__name__) if runner is not None else "run_pipeline",
            "max_file_bytes": settings.max_file_bytes, "sqlite_journal_mode": settings.sqlite_journal_mode,
            "recovered_tasks": application.state.recovered_tasks,
            "staging_files_cleaned": application.state.staging_files_cleaned,
            "tasks": store.counts(), "process": process_usage()
        }

    @application.exception_handler(ProjectError)
    async def domain_error(request, error):
        """把 ProjectError 中约定的状态码和说明转换为统一 JSON 响应。"""
        return JSONResponse(status_code=error.status_code, content={"detail": str(error)})

    @application.get("/documents")
    def list_documents():
        """列出当前业务库中的文档及各集合最新入库状态。"""
        return {"items": store.documents()}

    @application.get("/collections")
    def list_collections():
        """列出向量集合及当前业务就绪范围内可检索的 Chunk 数量。"""
        return {"items": application.state.documents.collections()}

    @application.get("/documents/{document_id}/file")
    def read_pdf(document_id: str):
        """通过文档服务校验可见性与文件路径后，以内联 PDF 响应返回原文件。"""
        path, document = application.state.documents.file_path(document_id)
        return FileResponse(path, media_type="application/pdf", filename=document["name"], content_disposition_type="inline")

    @application.delete("/documents/{document_id}")
    def delete_document(document_id: str):
        """调用文档服务执行可恢复的逻辑删除、向量清理及文件清理。"""
        return application.state.documents.delete(document_id)

    @application.post("/documents/{document_id}/reindex", status_code=202)
    def reindex_document(document_id: str, body: ReindexRequest):
        """检查文档处于活动状态，再按新方案和缓存策略提交重建任务。"""
        document = store.document(document_id)
        if document["lifecycle"] != "active":
            raise ProjectError("文档正在删除")
        return application.state.ingestion.enqueue(document, body.scheme, body.refresh)

    @application.post("/sessions", status_code=201)
    def create_session(body: SessionRequest):
        """清理会话标题，交由问答服务验证集合、文档范围与检索参数。"""
        if not body.title.strip():
            raise HTTPException(422, "会话名称不能为空")
        options = body.model_dump(exclude={"title", "collection_name"})
        try:
            return application.state.chat.create_session(body.title.strip(), body.collection_name, options)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @application.get("/sessions")
    def list_sessions():
        """按更新时间返回会话摘要列表。"""
        return {"items": store.sessions()}

    @application.get("/sessions/{session_id}")
    def session_history(session_id: str):
        """返回完整业务问答历史、渲染后的答案和来源链接。"""
        return application.state.chat.history(session_id)

    @application.patch("/sessions/{session_id}")
    def rename_session(session_id: str, body: RenameRequest):
        """校验非空标题并更新会话名称。"""
        if not body.title.strip():
            raise HTTPException(422, "会话名称不能为空")
        return store.rename_session(session_id, body.title.strip())

    @application.delete("/sessions/{session_id}")
    def delete_session(session_id: str):
        """在会话锁内删除图检查点及业务会话记录。"""
        return application.state.chat.delete_session(session_id)

    @application.post("/chat")
    def chat(body: ChatRequest):
        """校验并清理新问题，再调用带请求幂等和会话锁的 ChatService.chat。"""
        if not body.question.strip():
            raise HTTPException(422, "问题不能为空")
        return application.state.chat.chat(body.thread_id, body.request_id, body.question.strip())

    static = Path(__file__).resolve().parent.parent / "web"
    # 资源路径按项目文件位置计算，命令行工作目录可以由调用方选择。
    if static.is_dir():
        application.mount("/assets", StaticFiles(directory=static), name="assets")

    @application.get("/", include_in_schema=False)
    def homepage():
        """返回随项目提供的网页首页。"""
        return FileResponse(static / "index.html")

    @application.get("/guide", response_class=HTMLResponse, include_in_schema=False)
    def technical_guide():
        """把项目技术说明渲染为 HTML，供本地服务中的文档页面阅读。"""
        guide = Path(__file__).resolve().parent.parent / "PROJECT_TECHNICAL_GUIDE.md"
        if not guide.is_file():
            raise HTTPException(404, "技术文档尚未生成")
        content = MARKDOWN.render(guide.read_text(encoding="utf-8"))
        return HTMLResponse('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
                            '<title>PaperAgent 技术文档</title><link rel="stylesheet" href="/assets/style.css">'
                            '<body class="guide"><article>' + content + '</article></body></html>')

    return application


app = create_app()
