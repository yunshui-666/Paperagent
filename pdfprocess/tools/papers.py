"""论文工具接口，知网和万方适配由 campus_papers.py 负责。"""

import hashlib
import time
from typing import Annotated, Literal

from langchain_core.tools import tool
from pydantic import Field

from .campus_papers import CampusPaperClient, PAPER_URL_PATTERN, canonical_id


CAMPUS_CLIENT = CampusPaperClient()


@tool
def search_papers(query: Annotated[str, Field(min_length=1,max_length=2000)],
                  max_results: Annotated[int, Field(ge=1,le=10,strict=True)] = 5,
                  source: Literal["all","cnki","wanfang"] = "all") -> dict:
    """检索知网和万方论文，支持中文关键词。返回官方详情链接 paper_id、标题、摘要和来源错误。"""
    return CAMPUS_CLIENT.search(query,max_results,source)


def create_download_tool(ingestion, store, collection_name, *, client=CAMPUS_CLIENT, wait_seconds=60):
    """绑定当前会话目标库，官网 PDF 下载成功后提交已有方案三入库。"""
    @tool
    def download_paper(paper_id: str) -> dict:
        """下载知网或万方论文的官方详情链接 paper_id，自动解析、切块、Embedding 入库并返回真实状态。"""
        paper_id = canonical_id(paper_id)
        directory = ingestion.storage.root / "downloads"
        filename = "paper_" + hashlib.sha256(paper_id.encode("utf-8")).hexdigest()[:16] + ".pdf"
        with client.open_pdf(paper_id,max_bytes=ingestion.settings.max_file_bytes,directory=directory) as stream:
            accepted = ingestion.submit(filename,stream,3,collection_name=collection_name)
        deadline = time.monotonic()+wait_seconds
        task = store.get(accepted["task_id"])
        while task and task["status"] in {"queued","running"} and time.monotonic()<deadline:
            time.sleep(0.2)
            task = store.get(accepted["task_id"])
        if task is None:
            raise RuntimeError("论文已下载，但未能查询入库任务")
        return {"paper_id":paper_id,"url":paper_id,"document_id":accepted["document_id"],
                "task_id":task["id"],"collection_name":collection_name,"status":task["status"],
                "ingested":task["status"]=="completed","chunk_count":(task.get("result") or {}).get("chunk_count",0),
                "error":task.get("error")}
    return download_paper
