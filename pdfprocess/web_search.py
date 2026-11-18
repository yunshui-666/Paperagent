"""联网搜索工具，使用 Bing RSS 返回网页标题、链接和摘要。"""

# 公开网页搜索适配层。固定 RSS 入口返回标题、URL 和文本摘要，下游继续校验及分配证据预算。

from html.parser import HTMLParser
import time
from urllib.parse import urlsplit
from xml.etree import ElementTree

from langchain_core.tools import tool
import requests


class WebSearchError(RuntimeError):
    """搜索服务不可用或返回格式异常。"""


class _PlainText(HTMLParser):
    """收集 HTML 的文字节点，丢掉标签，便于作为普通文本放入提示词。"""
    def __init__(self):
        """启用 HTML 字符实体转换，并为文字节点准备累积列表。"""
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        """由 HTMLParser 回调收集文字节点，供最终摘要拼接。"""
        self.parts.append(data)


def _plain_text(value):
    # convert_charrefs=True 会将 &amp; 等实体还原，再归一化多余空白。
    """移除 HTML 标签并归一化空白，返回普通文本。"""
    parser = _PlainText()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())


def clean_results(results, *, limit=5):
    """清理工具返回值，只接收具有 HTTP 链接和正文摘要的结果。"""
    if not isinstance(results, list):
        raise WebSearchError("搜索结果需要为列表")
    cleaned = []
    seen = set()
    for result in results:
        # 工具或测试替身都可能返回异常字段，逐条检查后再构造统一结果。
        if not isinstance(result, dict):
            continue
        url = result.get("url")
        content = result.get("content")
        title = result.get("title") or url
        if not all(isinstance(value, str) for value in (url, title, content)):
            continue
        url = url.strip()
        try:
            parsed = urlsplit(url)
            # 来源链接仅接收 HTTP(S)，拒绝含用户名/密码的地址。
            valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                     and not parsed.username and not parsed.password)
        except ValueError:
            continue
        content = _plain_text(content)[:3000]
        # 同一网页的不同锚点只保留一次，避免占用多个搜索结果名额。
        identity = url.split("#", 1)[0]
        if not valid or len(url) > 4096 or not content or identity in seen:
            continue
        seen.add(identity)
        cleaned.append({"title": _plain_text(title)[:300] or url, "url": url, "content": content})
        if len(cleaned) >= limit:
            break
    return cleaned


class BingWebSearch:
    """固定 RSS 入口搜索，返回摘要。下游压缩模块继续分配证据预算。"""
    def __init__(self, *, timeout=15, max_results=5):
        """保存请求超时和结果条数上限，实际请求发生在 search 中。"""
        self.timeout = timeout
        self.max_results = max_results

    def search(self, query):
        """查询固定 Bing RSS 地址，限制响应大小，解析后返回规范化网页列表。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("联网搜索问题需要为非空字符串")
        # RSS 入口无需密钥。限制响应大小，超时和临时 HTTP 错误最多重试一次。
        for attempt in range(2):
            try:
                with requests.get(
                    "https://www.bing.com/search",
                    params={"q": query.strip(), "format": "rss", "count": self.max_results},
                    headers={"User-Agent": "PaperAgent/2.0", "Accept": "application/rss+xml, application/xml, text/xml"},
                    timeout=self.timeout, stream=True
                ) as response:
                    response.raise_for_status()
                    # 流式读取使大小检查在下载过程中生效。
                    chunks = []
                    size = 0
                    for chunk in response.iter_content(chunk_size=65536):
                        # 每次累加真实响应字节数，避免只依赖 Content-Length。
                        size += len(chunk)
                        if size > 1024 * 1024:
                            raise WebSearchError("联网搜索响应超过大小限制")
                        chunks.append(chunk)
                    payload = b"".join(chunks)
                break
            except requests.RequestException as exc:
                # 超时、连接失败、429 和 5xx 可短暂重试，其他 4xx 直接返回错误。
                status = exc.response.status_code if exc.response is not None else None
                if attempt == 1 or (status is not None and status != 429 and status < 500):
                    raise WebSearchError("联网搜索服务请求失败，请检查网络后重试") from exc
                time.sleep(0.3)
        try:
            # RSS 是 XML，解析失败与“搜索结果为空”分别处理。
            root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as exc:
            raise WebSearchError("联网搜索服务返回了无效的 RSS") from exc
        if root.tag != "rss" or root.find("channel") is None:
            raise WebSearchError("联网搜索服务返回格式异常")
        results = [{"title": item.findtext("title") or "", "url": item.findtext("link") or "",
                    "content": item.findtext("description") or ""}
                   for item in root.findall("./channel/item")]
        return clean_results(results, limit=self.max_results)


@tool
def web_search(query: str) -> list[dict]:
    """搜索互联网公开网页，返回与独立检索问题相关的标题、HTTP 链接和正文摘要。"""
    # @tool 让工作流通过 .invoke({"query": ...}) 调用，测试可替换这个边界。
    return BingWebSearch().search(query)
