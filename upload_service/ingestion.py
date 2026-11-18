"""上传落盘与任务入队的业务服务，供上传和重建索引两个 HTTP 入口复用。

这里只接受文件并提交持久化任务，PDF 解析、Embedding 和 Chroma 写入由
worker.py 领取任务后执行。notify 在入队成功后唤醒工作线程。
"""

# 上传与入队业务层。生成持久化配置快照和稳定任务键，连接文件存储及后台队列。

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from pdfprocess.ingest_common import file_identity


class IngestionService:
    """组合文件存储、入库配置、任务数据库与工作线程通知接口。"""

    def __init__(self, store, storage, settings, notify):
        """保存应用生命周期提供的依赖，构造阶段仅组装对象。"""
        self.store = store
        self.storage = storage
        self.settings = settings
        self.notify = notify

    def enqueue(self, document, scheme, refresh=False, *, collection_name=None):
        """创建配置快照和稳定任务键，将同一内容及参数的活动任务合并。"""
        # refresh 控制本次任务缓存策略，配置快照随后由后台进程重新构造。
        config = replace(self.settings.pipeline, use_cache=False) if refresh else self.settings.pipeline
        if collection_name is not None:
            if not isinstance(collection_name, str) or not collection_name.strip():
                raise ValueError("目标知识库名称不能为空")
            config = replace(config, collection_name=collection_name)
        _, paper_id, fingerprint, _ = file_identity(document["path"], scheme, config)
        if paper_id != document["id"]:
            raise RuntimeError("已保存文件的内容哈希与上传文件不一致")
        collection_name = config.collection_name or f"paper_chunks{scheme}"
        snapshot = asdict(config)
        # Path 统一转为绝对字符串，使任务可以通过 JSON 保存并跨进程传递。
        snapshot["data_root"] = str(Path(config.data_root).resolve())
        snapshot["chroma_dir"] = str(Path(config.chroma_dir).resolve())
        job_key = hashlib.sha256(json.dumps([
            paper_id, scheme, fingerprint, snapshot, collection_name
        ], sort_keys=True).encode("utf-8")).hexdigest()
        task = self.store.enqueue(document, scheme, collection_name, snapshot, job_key)
        # 数据库成功提交后才通知工作线程，线程领取时可以读到完整任务。
        self.notify()
        return task

    def submit(self, filename, stream, scheme, *, collection_name=None):
        """保存并校验 PDF，返回上传结果和入队后的任务标识。"""
        document = self.storage.save(filename, stream)
        task = self.enqueue(document, scheme, collection_name=collection_name)
        return {
            "filename": document["name"], "document_id": document["id"],
            "size_bytes": document["size_bytes"], "page_count": document["page_count"],
            "reused_file": document["reused_file"],
            "copy_seconds": document["copy_seconds"], "validation_seconds": document["validation_seconds"],
            "accepted": True, **task
        }
