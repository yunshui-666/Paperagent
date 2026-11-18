"""文档业务及可见范围层。Dense 与 BM25 使用同一文档就绪状态和用户选择范围。"""

import json
from pathlib import Path
import re
import shutil
from threading import RLock

import chromadb
from langchain_chroma import Chroma

from .errors import ProjectError


class DocumentManager:
    """组合业务就绪状态、Chroma 客户端和文件系统，管理文档可见性与清理。"""
    def __init__(self, store, settings):
        """保存存储和配置，并创建按绝对目录复用的 Chroma 客户端缓存。"""
        self.store = store
        self.settings = settings
        self._clients = {}
        self._lock = RLock()

    def client(self, directory=None):
        """以绝对目录为键获取或创建 Chroma PersistentClient。"""
        directory = str(Path(directory or self.settings.pipeline.chroma_dir).resolve())
        with self._lock:
            if directory not in self._clients:
                self._clients[directory] = chromadb.PersistentClient(path=directory)
            return self._clients[directory]

    def collections(self):
        """列出集合总 Chunk 数及当前可见 Chunk 数，兼容已有未登记向量。"""
        result = []
        for collection in self.client().list_collections():
            value = self.client().get_collection(collection.name)
            known, ready = self.store.availability(collection.name)
            metadata = value.get(include=["metadatas"])["metadatas"]
            # 已有旧向量通过 known 判断兼容，新登记文档按 ready 状态发布给检索。
            visible = sum((item or {}).get("paper_id") not in known or
                          (item or {}).get("paper_id") in ready for item in metadata)
            result.append({"name": collection.name, "chunk_count": value.count(),
                           "ready_chunk_count": visible})
        return sorted(result, key=lambda item: item["name"])

    def file_path(self, document_id):
        """校验活动文档和受管理目录边界，返回现存原始 PDF 路径与业务记录。"""
        document = self.store.document(document_id)
        if document["lifecycle"] != "active":
            raise ProjectError("文档正在删除", 409)
        path = Path(document["path"]).resolve()
        root = Path(self.settings.service_root).resolve() / "files"
        if not path.is_relative_to(root) or not path.is_file():
            raise ProjectError("原 PDF 文件不可用", 404)
        return path, document

    def delete(self, document_id):
        """先标记 deleting，再清理相关集合、缓存和原文件，最后提交 deleted。
        中途失败保留可恢复状态，下一次调用或服务启动可继续清理。
        """
        if not re.fullmatch(r"[0-9a-f]{64}", document_id):
            raise ProjectError("文档 ID 格式不正确", 422)
        with self._lock:
            document = self.store.document(document_id, include_deleted=True)
            self.store.begin_delete(document_id)
            # 历史任务保存了实际存储目录，清理时把当前配置和历史配置一起考虑。
            targets = {(str(Path(self.settings.pipeline.chroma_dir).resolve()),
                        str(Path(self.settings.pipeline.data_root).resolve()))}
            for task in self.store.document_tasks(document_id):
                config = json.loads(task["config_json"])
                targets.add((str(Path(config["chroma_dir"]).resolve()), str(Path(config["data_root"]).resolve())))
            for chroma_dir, data_root in targets:
                for collection in self.client(chroma_dir).list_collections():
                    self.client(chroma_dir).get_collection(collection.name).delete(where={"paper_id": document_id})
                root = Path(data_root).resolve()
                cache = (root / document_id).resolve()
                # 删除前解析并检查绝对路径边界，清理范围限于该内容身份的缓存目录。
                if cache.is_relative_to(root) and cache.is_dir():
                    shutil.rmtree(cache)
            path = Path(document["path"]).resolve()
            root = Path(self.settings.service_root).resolve() / "files"
            if path.is_relative_to(root):
                path.unlink(missing_ok=True)
            self.store.finish_delete(document_id)
            # 只有全部清理成功才标记 deleted，失败时保留 deleting 供重试恢复。
        return {"document_id": document_id, "status": "deleted"}

    def recover(self):
        """继续处理业务库中尚处于 deleting 的文档清理任务。"""
        for document_id in self.store.unfinished_deletions():
            self.delete(document_id)

    def scoped_store(self, collection_name, embedding, document_ids):
        """验证集合存在后构造带业务可见范围的 Chroma 适配对象。"""
        available = {item["name"] for item in self.collections()}
        if collection_name not in available:
            raise ProjectError("该知识库尚未完成入库，请先上传并等待任务完成", 404)
        store = Chroma(client=self.client(), collection_name=collection_name, embedding_function=embedding,
                       create_collection_if_not_exists=False)
        return ScopedStore(store, embedding, self.store, collection_name, document_ids)


class ScopedStore:
    """向量与 BM25 共用同一文档可见范围，检索时仅向量化问题一次。"""
    def __init__(self, store, embedding, database, collection_name, document_ids):
        """保存原生向量库、问题向量模型、业务状态库和用户选定论文集合。"""
        self.store = store
        self.embedding = embedding
        self.database = database
        self.collection_name = collection_name
        self.document_ids = set(document_ids)

    def get(self, **kwargs):
        """读取语料后按入库就绪状态和用户选择范围过滤，供 BM25 和向量候选共同使用。"""
        raw = self.store.get(include=["documents", "metadatas"])
        known, ready = self.database.availability(self.collection_name)
        ids, texts, metadata = [], [], []
        for chunk_id, text, meta in zip(raw["ids"], raw["documents"], raw["metadatas"]):
            meta = meta or {}
            paper_id = meta.get("paper_id")
            visible = paper_id not in known or paper_id in ready
            # 业务就绪状态与用户论文选择范围同时满足时，才将证据交给上层。
            if visible and (not self.document_ids or paper_id in self.document_ids):
                ids.append(chunk_id)
                texts.append(text)
                metadata.append(meta)
        return {"ids": ids, "documents": texts, "metadatas": metadata}

    def similarity_search_with_score(self, *, query, k):
        """问题向量化一次，逐步扩大候选池并过滤可见 ID，尽量返回范围内前 K 条证据。"""
        allowed = set(self.get()["ids"])
        if not allowed:
            return []
        total = len(self.store.get(include=[])["ids"])
        vector = self.embedding.embed_query(query)
        # 过滤会减少候选，先取较宽候选池；后续扩池复用同一个问题向量。
        size = min(total, max(k * 2, 20))
        while True:
            hits = self.store.similarity_search_by_vector_with_relevance_scores(vector, k=size)
            # LangChain 此入口返回的距离继续沿用现有检索适配接口传给 RetrievalHit。
            filtered = [(document, distance) for document, distance in hits if document.id in allowed]
            if len(filtered) >= k or size >= total:
                return filtered[:k]
            size = min(total, size * 2)
            # 每次最多翻倍，达到全集仍不足 K 时返回当前全部可见结果。
