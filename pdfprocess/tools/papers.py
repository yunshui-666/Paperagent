"""arXiv 开放论文搜索与下载，下载后复用应用的持久化入库队列。"""

from contextlib import contextmanager
import re
from pathlib import Path
import tempfile
from threading import Lock
import time
from typing import Annotated
from urllib.parse import urljoin, urlsplit
from xml.etree import ElementTree

from langchain_core.tools import tool
from pydantic import Field
import requests


ARXIV_ID_PATTERN = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v[1-9]\d*)?"
ATOM = {"a": "http://www.w3.org/2005/Atom"}


def canonical_id(paper_id):
    """验证并返回 arXiv 标识，兼容现代编号和早期学科编号。"""
    if not isinstance(paper_id, str) or not re.fullmatch(ARXIV_ID_PATTERN, paper_id.strip()):
        raise ValueError("请提供有效 arXiv ID，例如 1706.03762 或 hep-th/9901001")
    return paper_id.strip()


class ArxivClient:
    """固定官方入口的有界 HTTP 客户端，查询启动间隔至少三秒。"""

    def __init__(self, *, timeout=30):
        """保存超时与查询节流状态，构造时仅初始化本地对象。"""
        self.timeout = timeout
        self._lock = Lock()
        self._last_query = 0.0

    def search(self, query, max_results=5):
        """返回论文标题、作者、摘要、日期和规范化的公开 PDF 地址。"""
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("论文搜索词需要为 1 到 2000 字符")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 10:
            raise ValueError("论文结果数量需要在 1 到 10 之间")
        with self._lock:
            time.sleep(max(0.0, 3 - (time.monotonic() - self._last_query)))
            self._last_query = time.monotonic()
        payload, size = [], 0
        with requests.get("https://export.arxiv.org/api/query", params={"search_query": query.strip(),
                "start": 0, "max_results": max_results, "sortBy": "relevance"},
                headers={"User-Agent": "PaperAgent/2.0", "Accept": "application/atom+xml"},
                timeout=self.timeout, stream=True) as response:
            response.raise_for_status()
            for block in response.iter_content(chunk_size=65536):
                size += len(block)
                if size > 2 * 1024 * 1024:
                    raise RuntimeError("论文搜索响应超过大小限制")
                payload.append(block)
        root = ElementTree.fromstring(b"".join(payload))
        if root.tag != "{http://www.w3.org/2005/Atom}feed":
            raise RuntimeError("论文搜索返回了无效的 Atom 数据")
        result, seen = [], set()
        for entry in root.findall("a:entry", ATOM):
            url = entry.findtext("a:id", default="", namespaces=ATOM)
            parsed = urlsplit(url)
            if parsed.hostname not in {"arxiv.org", "export.arxiv.org"} or not parsed.path.startswith("/abs/"):
                raise RuntimeError("论文搜索返回了异常论文标识")
            paper_id = canonical_id(parsed.path.removeprefix("/abs/"))
            title = " ".join(entry.findtext("a:title", default="", namespaces=ATOM).split())[:500]
            if not title or paper_id in seen:
                continue
            seen.add(paper_id)
            result.append({"arxiv_id": paper_id, "title": title,
                "authors": [author.findtext("a:name", default="", namespaces=ATOM)[:150]
                            for author in entry.findall("a:author", ATOM)[:100]],
                "summary": " ".join(entry.findtext("a:summary", default="", namespaces=ATOM).split())[:3000],
                "published": entry.findtext("a:published", default="", namespaces=ATOM)[:40],
                "url": f"https://arxiv.org/abs/{paper_id}", "pdf_url": f"https://arxiv.org/pdf/{paper_id}"})
        return result[:max_results]

    @contextmanager
    def open_pdf(self, paper_id, *, max_bytes, directory):
        """流式下载至受管理临时文件，逐跳检查公开来源并限制实际响应大小。"""
        paper_id = canonical_id(paper_id)
        url = f"https://arxiv.org/pdf/{paper_id}"
        Path(directory).mkdir(parents=True, exist_ok=True)
        with tempfile.SpooledTemporaryFile(max_size=4 * 1024 * 1024, mode="w+b", dir=directory) as output:
            for _ in range(4):
                parsed = urlsplit(url)
                if (parsed.scheme != "https" or parsed.hostname not in {"arxiv.org", "export.arxiv.org"}
                        or parsed.username or parsed.password or parsed.port not in (None, 443)
                        or not parsed.path.startswith("/pdf/")):
                    raise ValueError("论文下载地址需要为官方 arXiv HTTPS PDF 链接")
                with requests.get(url, headers={"User-Agent": "PaperAgent/2.0", "Accept": "application/pdf"},
                                  timeout=self.timeout, stream=True, allow_redirects=False) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        url = urljoin(url, response.headers.get("Location", ""))
                        continue
                    response.raise_for_status()
                    size = 0
                    for block in response.iter_content(chunk_size=65536):
                        size += len(block)
                        if size > max_bytes:
                            raise ValueError("论文 PDF 超过当前上传大小上限")
                        output.write(block)
                    if not size:
                        raise ValueError("论文 PDF 下载为空")
                    output.seek(0)
                    # 原有 FileStorage 随后验证真实 PDF 格式、页面数与加密状态。
                    yield output
                    return
            raise RuntimeError("论文 PDF 重定向次数过多")


ARXIV_CLIENT = ArxivClient()


@tool
def search_papers(query: Annotated[str, Field(min_length=1, max_length=2000)],
                  max_results: Annotated[int, Field(ge=1, le=10, strict=True)] = 5) -> list[dict]:
    """搜索 arXiv 开放访问论文，使用简洁英文关键词或 arXiv 查询语法，返回 ID、作者、摘要和 PDF 链接。"""
    return ARXIV_CLIENT.search(query, max_results)


def create_download_tool(ingestion, store, collection_name, *, client=ARXIV_CLIENT, wait_seconds=60):
    """绑定当前会话知识库，下载后提交方案三入库并观察任务状态。"""
    @tool
    def download_paper(paper_id: str) -> dict:
        """下载指定 arXiv ID 的公开 PDF，并自动解析、切块、Embedding 入库。返回任务 ID 和实际状态。"""
        paper_id = canonical_id(paper_id)
        directory = ingestion.storage.root / "downloads"
        with client.open_pdf(paper_id, max_bytes=ingestion.settings.max_file_bytes, directory=directory) as stream:
            accepted = ingestion.submit(paper_id.replace("/", "_") + ".pdf", stream, 3,
                                        collection_name=collection_name)
        deadline = time.monotonic() + wait_seconds
        task = store.get(accepted["task_id"])
        while task and task["status"] in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.2)
            task = store.get(accepted["task_id"])
        if task is None:
            raise RuntimeError("论文已下载，但未能查询入库任务")
        return {"arxiv_id": paper_id, "document_id": accepted["document_id"], "task_id": task["id"],
                "collection_name": collection_name, "status": task["status"],
                "ingested": task["status"] == "completed", "chunk_count": (task.get("result") or {}).get("chunk_count", 0),
                "error": task.get("error"), "pdf_url": f"https://arxiv.org/pdf/{paper_id}"}
    return download_paper
