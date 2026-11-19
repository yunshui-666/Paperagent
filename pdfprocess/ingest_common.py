"""三个方案共用的配置、页级缓存、切块、入库和批量上传。"""

# 入库编排层。围绕文件身份、解析缓存、页级 Chunk、向量写入和完成标记组织可重试流程。

from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from uuid import uuid4

import chromadb
from dotenv import load_dotenv

from .embeddings import EMBEDDING_MODEL, TongyiEmbeddings
from .ingest_config import PipelineConfig
from .paths import DATA_DIR, PROJECT_DIR
from .paper_chunk import prepare_chunks
from .vector_store import save_chunks




class ProcessingBusyError(RuntimeError):
    """同一任务的处理锁正由另一个执行者持有。"""




def load_pdf_library():
    """延迟导入 PyMuPDF，缺少依赖时给出可定位到安装清单的错误。"""
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("请先安装 requirements-ingest.txt 中的 PDF 解析依赖") from exc
    return pymupdf


def write_json(path: Path, value):
    """先写同目录临时文件，再原子替换目标 JSON，结束时清理残留临时文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
    # 临时文件和目标文件在同目录，os.replace 用于发布一个完整的 JSON 快照。
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def processing_lock(path: Path):
    """通过操作系统文件锁保护同一任务，进程退出时自动释放。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                # Windows 锁住文件首字节；LK_NBLCK 在冲突时立即报错供任务退避。
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                # POSIX 使用文件级排他锁，进程退出时由操作系统回收锁。
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ProcessingBusyError("该文件正在处理，请在当前任务完成后重试") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def file_identity(pdf_path, scheme: int, config: PipelineConfig):
    """返回绝对 PDF 路径、内容哈希、处理配置指纹和缓存目录。
    同一 PDF 内容决定 paper_id，解析配置和依赖版本决定处理指纹。
    """
    path = Path(pdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PDF 文件不存在：{path}")
    with path.open("rb") as stream:
        # 文件内容决定 paper_id，与显示名和上传时采用的解析方案分别维护。
        paper_id = hashlib.file_digest(stream, "sha256").hexdigest()
    options = asdict(config)
    for key in ("data_root", "chroma_dir", "collection_name", "use_cache"):
        # 目录和缓存开关不改变解析产物，指纹集中描述处理参数及依赖版本。
        options.pop(key)
    options.update({"schema_version": 2, "scheme": scheme})
    if scheme in (2, 3):
        try:
            options["rapidocr_version"] = importlib.metadata.version("rapidocr")
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError("请先安装 requirements-ingest.txt 中的本地 OCR 依赖") from exc
    try:
        options["pymupdf_version"] = importlib.metadata.version("pymupdf")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("请先安装 requirements-ingest.txt 中的 PDF 解析依赖") from exc
    fingerprint = hashlib.sha256(
        json.dumps(options, sort_keys=True).encode("utf-8")
    ).hexdigest()
    directory = Path(config.data_root).resolve() / paper_id / f"scheme{scheme}" / fingerprint[:16]
    # 每个内容、方案和指纹拥有独立产物目录，页缓存可精确对应到处理配置。
    return path, paper_id, fingerprint, directory


def parse_document(pdf_path, scheme, page_parser, *, config=None, ocr_client=None, vlm_client=None):
    """逐页执行解析。成功页可缓存，失败页在下一次执行时重试。"""
    config = config or PipelineConfig()
    path, paper_id, fingerprint, directory = file_identity(pdf_path, scheme, config)
    pymupdf = load_pdf_library()
    pdf_bytes = path.read_bytes()
    if hashlib.sha256(pdf_bytes).hexdigest() != paper_id:
        # 从确定身份到读取实际解析字节之间再次核对，避免文件变化导致身份错配。
        raise RuntimeError("源文件发生变化，请重新提交")
    started = time.perf_counter()
    metrics = {"ocr_calls": 0, "vlm_calls": 0, "input_tokens": 0,
               "output_tokens": 0, "cached_pages": 0,
               "local_ocr_seconds": 0.0, "vlm_seconds": 0.0, "dual_page_seconds": 0.0,
               "ocr_line_count": 0, "unmatched_ocr_lines": 0}
    context = {"config": config, "directory": directory, "metrics": metrics,
               "ocr_client": ocr_client, "vlm_client": vlm_client}
    pages = []
    with ExitStack() as resources, processing_lock(directory / ".parse.lock"):
        # ExitStack 统一关闭本轮懒创建的 OCR/VLM 客户端，文件锁串行化同一解析产物。
        context["resources"] = resources
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            if not document.is_pdf or document.needs_pass or not len(document):
                raise ValueError("需要可读取的非加密 PDF，且至少包含一页")
            for index, page in enumerate(document, start=1):
                # PDF 页码从 1 开始，写入缓存、Chunk 元数据和前端跳页链接。
                cache_path = directory / "pages" / f"page-{index:04d}.json"
                cached = None
                if config.use_cache and cache_path.exists():
                    try:
                        cached = json.loads(cache_path.read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        pass
                if (isinstance(cached, dict) and cached.get("status") == "completed"
                        and cached.get("page") == index and cached.get("fingerprint") == fingerprint):
                    pages.append(cached)
                    metrics["cached_pages"] += 1
                    # 成功且身份匹配的页直接复用，失败页会落到下面重新解析。
                    continue
                page_started = time.perf_counter()
                try:
                    result = page_parser(document, page, index, context)
                    if not result["markdown"].strip() and not result.get("blank", False):
                        raise ValueError("该页解析结果为空，请检查 OCR 或图像内容")
                    result.update({"page": index, "status": "completed", "fingerprint": fingerprint})
                except Exception as exc:
                    # 页错误保存到产物中，继续记录其余页，整份结果随后标记 failed。
                    result = {"page": index, "status": "failed", "markdown": "",
                              "blocks": [], "error": str(exc), "fingerprint": fingerprint}
                result["seconds"] = time.perf_counter() - page_started
                write_json(cache_path, result)
                pages.append(result)

        markdown = "\n\n---\n\n".join(
            # 页边界写入可读产物，真正入库的 Chunk 在后续仍逐页构造。
            f"<!-- PDF page {item['page']} -->\n\n{item['markdown']}"
            for item in pages if item["status"] == "completed"
        )
        markdown_path = directory / "output.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        result = {
            "paper_id": paper_id, "scheme": scheme, "source": path.name,
            "fingerprint": fingerprint, "page_count": len(pages), "pages": pages,
            "status": "completed", "markdown_path": str(markdown_path),
            "content_hash": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            "artifacts_dir": str(directory), "metrics": metrics,
            "parse_seconds": time.perf_counter() - started
        }
        if any(item["status"] == "failed" for item in pages):
            # 完整文档的成功状态要求全部页面成功，后台会据此决定是否发布索引。
            result["status"] = "failed"
        write_json(directory / "parse_result.json", result)
    return result


def create_embedding_model(*, request_timeout=90):
    """在真实入库需要向量时读取环境配置并创建 TongyiEmbeddings。"""
    load_dotenv(PROJECT_DIR / ".env")
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise ValueError("未配置 DASHSCOPE_API_KEY，解析对比可使用 --parse-only")
    return TongyiEmbeddings(api_key=key, request_timeout=request_timeout)


def run_ingestion(pdf_path, scheme, parser, *, config=None, embedding_model=None,
                  collection=None, ocr_client=None, vlm_client=None, source_name=None,
                  progress_callback=None, parse_executor=None, expected_paper_id=None):
    """编排 PDF 解析、页级切块、向量化、Chroma 写入及完成标记。
    返回 Chunk ID、产物路径、模型用量和阶段耗时。失败向上传播给任务执行器。
    可注入解析执行器和模型，便于进程隔离与离线回归测试。
    """
    config = config or PipelineConfig()
    if embedding_model is None and config.embedding_model_id != EMBEDDING_MODEL:
        raise ValueError("自定义 embedding_model_id 时需要同时传入对应的 embedding_model")
    path, paper_id, fingerprint, directory = file_identity(pdf_path, scheme, config)
    if expected_paper_id is not None and paper_id != expected_paper_id:
        raise RuntimeError("保存的 PDF 内容与上传时的文件标识不一致")
    source = source_name or path.name
    started = time.perf_counter()
    collection_name = config.collection_name or f"paper_chunks{scheme}"
    if collection is not None:
        collection_name = collection.name
    lock_key = hashlib.sha256(f"{collection_name}:{paper_id}:{scheme}".encode()).hexdigest()
    # 同一文档、集合和方案的写入串行执行，重试时能够清理旧版本 Chunk。
    lock_path = Path(config.chroma_dir).resolve() / ".ingest-locks" / f"{lock_key}.lock"
    with processing_lock(lock_path):
        if progress_callback is not None:
            progress_callback("parsing")
        if parse_executor is None:
            # 脚本/测试可以直接执行；服务入口注入 ParserPool 将解析放到独立进程。
            parsed = parser(path, config=config, ocr_client=ocr_client, vlm_client=vlm_client)
        else:
            if ocr_client is not None or vlm_client is not None:
                raise ValueError("进程解析时由子进程创建模型客户端，请通过环境变量配置密钥")
            parsed = parse_executor.parse(parser, path, config)
        if parsed["paper_id"] != paper_id:
            raise RuntimeError("源文件在入库过程中发生变化，请重新提交")
        if parsed["status"] != "completed":
            failed_pages = [page["page"] for page in parsed["pages"] if page["status"] == "failed"]
            raise RuntimeError(f"PDF 解析失败，页码 {failed_pages}，详见 {directory / 'parse_result.json'}")

        if progress_callback is not None:
            progress_callback("checking_cache", {"page_count": parsed["page_count"]})

        if collection is None:
            client = chromadb.PersistentClient(path=str(Path(config.chroma_dir).resolve()))
            collection = client.get_or_create_collection(
                name=collection_name, metadata={"embedding_model": config.embedding_model_id}
            )
        actual_model = (collection.metadata or {}).get("embedding_model")
        # 集合中的所有向量使用一致模型语义空间，模型身份记录在集合及 Chunk 元数据。
        if actual_model and actual_model != config.embedding_model_id:
            raise ValueError("Collection 的 Embedding 模型与当前配置不一致，请使用独立 Collection")
        collection_name = collection.name
        storage_key = hashlib.sha256(
            f"{Path(config.chroma_dir).resolve()}:{collection_name}".encode()
        ).hexdigest()[:16]
        completion_path = directory / f"completed-{storage_key}.json"
        # 完成标记同时区分 Chroma 绝对目录和集合，避免把另一个库的缓存误当成已入库。
        if config.use_cache and completion_path.exists():
            try:
                completed = json.loads(completion_path.read_text(encoding="utf-8"))
                ids = completed["chunk_ids"]
                if (completed.get("content_hash") == parsed["content_hash"] and ids
                        and set(collection.get(ids=ids, include=[])["ids"]) == set(ids)):
                    completed.update({"skipped": True, "metrics": parsed["metrics"],
                                      # 缓存命中时只记录本次解析/校验开销。
                                      "timings": {"parse": parsed["parse_seconds"], "split": 0,
                                                  "embed": 0, "save": 0,
                                                  "total": time.perf_counter() - started}})
                    return completed
            except (OSError, ValueError, KeyError, TypeError):
                # 缓存损坏或字段缺失时继续执行完整入库，稳定 ID 可覆盖同身份数据。
                pass

        if progress_callback is not None:
            progress_callback("chunking", {"page_count": parsed["page_count"]})
        stage_started = time.perf_counter()
        chunks = []
        for page in parsed["pages"]:
            # 按页切块保证每段证据携带明确 PDF 页码，便于人工标注与引用验证。
            page_chunks = prepare_chunks(
                page["markdown"], paper_id, source,
                chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
            )
            for chunk in page_chunks:
                index = len(chunks)
                chunk.metadata.update({
                    # Chunk ID 同时包含内容身份、方案、配置指纹和顺序，便于评测数据版本核对。
                    "page": page["page"], "scheme": scheme, "pipeline_id": fingerprint,
                    "chunk_index": index, "chunk_id": f"{paper_id}_{scheme}_{fingerprint[:16]}_{index}",
                    "parse_method": page["method"],
                    "has_vlm_content": any(block.get("generated", False) for block in page["blocks"]),
                    "embedding_model": config.embedding_model_id
                })
                chunks.append(chunk)
        if not chunks:
            raise ValueError("PDF 没有可入库的文本内容")
        split_seconds = time.perf_counter() - stage_started

        if progress_callback is not None:
            progress_callback("embedding", {"chunk_count": len(chunks)})
        stage_started = time.perf_counter()
        if embedding_model is None:
            embedding_model = create_embedding_model(request_timeout=config.request_timeout)
        vectors = embedding_model.embed_documents([chunk.page_content for chunk in chunks])
        # 模型适配层负责每条正文与返回向量顺序对应，向量写入层再次校验数量与维度。
        embed_seconds = time.perf_counter() - stage_started

        if progress_callback is not None:
            progress_callback("saving", {"chunk_count": len(chunks)})
        stage_started = time.perf_counter()
        save_chunks(chunks, vectors, collection=collection)
        where = {"$and": [{"paper_id": paper_id}, {"scheme": scheme}]}
        stored_ids = set(collection.get(where=where, include=[])["ids"])
        expected_ids = {chunk.metadata["chunk_id"] for chunk in chunks}
        stale_ids = stored_ids - expected_ids
        # 新数据先 upsert，之后清理同文档同方案的旧身份；业务层在任务完成后发布可见性。
        if stale_ids:
            collection.delete(ids=sorted(stale_ids))
        if set(collection.get(ids=sorted(expected_ids), include=[])["ids"]) != expected_ids:
            # 完成标记只在所有预期 ID 可查询时写入，部分写入中断后可重新提交补齐。
            raise RuntimeError("向量写入数量校验失败，请重新执行该文件入库")
        result = {
            "paper_id": paper_id, "scheme": scheme, "source": source,
            "status": "completed", "skipped": False, "page_count": parsed["page_count"],
            "content_hash": parsed["content_hash"],
            "chunk_count": len(chunks), "chunk_ids": sorted(expected_ids),
            "collection_name": collection_name, "chroma_dir": str(Path(config.chroma_dir).resolve()),
            "markdown_path": parsed["markdown_path"], "artifacts_dir": str(directory),
            "metrics": parsed["metrics"],
            "timings": {"parse": parsed["parse_seconds"], "split": split_seconds,
                        "embed": embed_seconds, "save": time.perf_counter() - stage_started,
                        "total": time.perf_counter() - started}
        }
        write_json(completion_path, result)
        return result


def save_uploaded_pdf(filename: str, upload_bytes: bytes, *, config=None) -> Path:
    """校验上传字节及 PDF 页面，再按内容哈希保存供脚本调用的原文件。"""
    config = config or PipelineConfig()
    name = Path(filename.replace("\\", "/")).name
    if not name or Path(name).suffix.lower() != ".pdf":
        raise ValueError("上传文件名需要以 .pdf 结尾")
    if not isinstance(upload_bytes, bytes) or not upload_bytes:
        raise ValueError("上传内容需要为非空 bytes")
    pymupdf = load_pdf_library()
    with pymupdf.open(stream=upload_bytes, filetype="pdf") as document:
        if document.needs_pass or not len(document):
            raise ValueError("上传的 PDF 已加密或没有页面")
    paper_id = hashlib.sha256(upload_bytes).hexdigest()
    destination = Path(config.data_root).resolve() / "uploads" / paper_id / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(upload_bytes)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def ingest_many(pdf_paths, ingest_function, **kwargs):
    """串行批处理，每个文件返回自己的成功结果或错误。"""
    if isinstance(pdf_paths, (str, bytes, Path)):
        raise TypeError("批量入口需要文件路径列表，例如 [path1, path2]")
    results = []
    for path in pdf_paths:
        try:
            results.append(ingest_function(path, **kwargs))
        except Exception as exc:
            results.append({"source": str(path), "status": "failed", "error": str(exc)})
    return results


def upload_one(filename, upload_bytes, ingest_function, **kwargs):
    """将字节上传保存为本地 PDF，再转入指定方案的完整入库函数。"""
    path = save_uploaded_pdf(filename, upload_bytes, config=kwargs.get("config"))
    return ingest_function(path, **kwargs)


def upload_many(files, upload_function, **kwargs):
    """files 为 [(原文件名, PDF bytes), ...]，可直接对接上传控件。"""
    if isinstance(files, (str, bytes, Path)):
        raise TypeError("批量上传入口需要 [(文件名, PDF bytes), ...]")
    results = []
    for item in files:
        try:
            filename, content = item
            results.append(upload_function(filename, content, **kwargs))
        except Exception as exc:
            results.append({"source": str(item[0]) if isinstance(item, (list, tuple)) and item else "",
                            "status": "failed", "error": str(exc)})
    return results
