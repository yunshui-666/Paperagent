"""解析方案选择入口，网页任务与 CLI 共用同一份方案约定。"""

from importlib import import_module

SCHEMES = (1, 2, 3)


def get_pipeline(scheme):
    """按方案号读取已有的 pdf_pipeline1/2/3 模块，保留三套学习接口。"""
    if isinstance(scheme, bool) or not isinstance(scheme, int) or scheme not in SCHEMES:
        raise ValueError("PDF 处理方案需要为 1、2 或 3")
    return import_module(f"pdfprocess.pdf_pipeline{scheme}")
