"""Chroma 写入适配层。校验向量维度和 Chunk ID 后，分批按稳定 ID 执行 upsert。"""


import math
from pathlib import Path

import chromadb


CHROMA_DIR = Path(__file__).resolve().parent / "data" / "chroma_db"


def save_chunks(chunks, embeddings, *, collection=None,
                persist_directory=None, collection_name="paper_chunks", batch_size=128):
    """校验后按固定 ID 分批 upsert，兼容原有双参数调用。"""
    if not chunks:
        raise ValueError("没有可保存的 Chunk")
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk 数量与向量数量不一致")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    dimension = len(embeddings[0])
    # 在任何 upsert 前完成输入校验，避免输入本身不一致造成部分批次写入。
    if not dimension:
        raise ValueError("Embedding 向量不能为空")
    for vector in embeddings:
        if len(vector) != dimension or any(not math.isfinite(value) for value in vector):
            raise ValueError("Embedding 维度不一致或包含非有限数值")

    ids = [chunk.metadata["chunk_id"] for chunk in chunks]
    # 唯一稳定 ID 同时用于入库重试、引用回查和检索评测标注。
    if len(ids) != len(set(ids)):
        raise ValueError("Chunk ID 必须唯一")
    if collection is None:
        directory = CHROMA_DIR
        if persist_directory is not None:
            directory = Path(persist_directory).resolve()
        client = chromadb.PersistentClient(path=str(directory))
        collection = client.get_or_create_collection(name=collection_name)
        batch_size = min(batch_size, client.get_max_batch_size())
        # 与 Chroma 当前服务端批量上限对齐，避免单次请求超过集合接口限制。
    existing = collection.get(limit=1, include=["embeddings"])["embeddings"]
    # 已有集合的向量维度必须和本次写入一致，维度错误在写入前报告。
    if existing is not None and len(existing) and len(existing[0]) != dimension:
        raise ValueError("Embedding 维度与已有 Collection 不一致")

    for start in range(0, len(chunks), batch_size):
        # 同一切片范围同时选取正文、元数据、ID 和向量，维持逐条对应关系。
        batch = chunks[start:start + batch_size]
        collection.upsert(
            ids=ids[start:start + batch_size],
            documents=[chunk.page_content for chunk in batch],
            metadatas=[chunk.metadata for chunk in batch],
            embeddings=embeddings[start:start + batch_size]
        )
    print(f"数据保存成功，当前 Chunk 数量：{collection.count()}")
    return collection
