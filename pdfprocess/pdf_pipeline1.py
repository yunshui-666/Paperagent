"""方案 1：本地 Markdown 解析。所有入口使用相同参数约定。"""

# 方案一入口。页处理器选用本地 Markdown 提取，其余上传、切块、Embedding 和持久化复用公共编排。

from .ingest_common import (
    ingest_many, parse_document, run_ingestion, upload_many, upload_one
)
from .pdf_vision import local_page


def pdf_to_markdown1(pdf_path, *, config=None, ocr_client=None, vlm_client=None):
    """解析单个 PDF，返回页级结果、Markdown 路径及调用指标。"""
    return parse_document(
        pdf_path, 1, local_page, config=config,
        ocr_client=ocr_client, vlm_client=vlm_client
    )


def ingest_pdf1(pdf_path, **kwargs):
    """单文件解析、切块、Embedding 及 Chroma 入库。"""
    return run_ingestion(pdf_path, 1, pdf_to_markdown1, **kwargs)


def ingest_pdfs1(pdf_paths, **kwargs):
    """批量处理本地路径，每个文件返回独立结果。"""
    return ingest_many(pdf_paths, ingest_pdf1, **kwargs)


def upload_pdf1(filename, upload_bytes, **kwargs):
    """接收文件名及 bytes，保存后执行完整入库。"""
    return upload_one(filename, upload_bytes, ingest_pdf1, **kwargs)


def batch_upload_pdfs1(files, **kwargs):
    """批量上传，files 为 [(filename, PDF bytes), ...]。"""
    return upload_many(files, upload_pdf1, **kwargs)
