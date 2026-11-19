"""知网与万方论文来源适配，使用部署主机的网络和站点授予的会话权限。"""

from contextlib import contextmanager
from html.parser import HTMLParser
from itertools import zip_longest
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit

import requests

from ..web_search import BingWebSearch, _plain_text
from .wanfang_protocol import decode_search, encode_search


PROVIDERS = {"cnki": ("cnki.net", "cnki.com.cn"), "wanfang": ("wanfangdata.com.cn",)}
PAPER_URL_PATTERN = r'https?://[^\s<>"\u201c\u201d]+'


class PaperAccessError(RuntimeError):
    """站点要求验证、权限不足、返回非 PDF 或上游响应异常。"""


def provider_of(url):
    """只接受知网和万方官方域名，检查凭据、端口和控制字符。"""
    if not isinstance(url,str) or len(url)>4096 or any(ord(char)<32 for char in url):
        raise ValueError("论文链接格式不正确")
    parsed = urlsplit(url.strip())
    if (parsed.scheme not in {"http","https"} or parsed.username or parsed.password
            or parsed.port not in (None,80,443)):
        raise ValueError("请使用知网或万方的官方论文链接")
    hostname = (parsed.hostname or "").lower()
    for name,domains in PROVIDERS.items():
        if any(hostname==domain or hostname.endswith("."+domain) for domain in domains):
            return name
    raise ValueError("论文来源需要为知网或万方")


def canonical_id(paper_id):
    """用官方详情链接标识论文，移除网页锚点并保留必要查询参数。"""
    provider = provider_of(paper_id)
    parsed = urlsplit(paper_id.strip())
    path = unquote(parsed.path).lower()
    prefixes = ("/kcms", "/article", "/doi", "/periodical/", "/thesis/", "/conference/")
    if not path.startswith(prefixes):
        raise ValueError("请提供论文详情链接，可直接复制 search_papers 的 paper_id")
    if provider == "wanfang" and not path.startswith(("/periodical/","/thesis/","/conference/","/doi")):
        raise ValueError("万方论文链接需要指向期刊、学位或会议论文")
    return urlunsplit((parsed.scheme.lower(),parsed.netloc.lower(),parsed.path,parsed.query,""))


class _DownloadLinks(HTMLParser):
    """读取页面直接提供的 PDF 地址，跳过需要执行的 JavaScript。"""
    def __init__(self, base):
        """保存详情页地址，解析相对链接时复用。"""
        super().__init__(convert_charrefs=True)
        self.base = base
        self.links = []

    def handle_starttag(self, tag, attrs):
        """收集 citation_pdf_url、PDF 下载锚点与内嵌 PDF。"""
        item = dict(attrs)
        if tag=="meta" and item.get("name","").lower()=="citation_pdf_url":
            self.links.append(urljoin(self.base,item.get("content","")))
        for key in ("href","src","data-url"):
            value = item.get(key,"")
            if value and not value.lower().startswith("javascript:") and re.search(r"\.pdf(?:[?#]|$)|pdfdown|dflag=pdf|download",value,re.I):
                self.links.append(urljoin(self.base,value))


def _read_limited(response, limit):
    """按真实响应字节限制元数据和 HTML，避免无界读取。"""
    chunks = []
    size = 0
    for chunk in response.iter_content(chunk_size=65536):
        size += len(chunk)
        if size > limit:
            raise PaperAccessError("论文站点响应超过大小限制")
        chunks.append(chunk)
    return b"".join(chunks)


def _save_response(response, output, max_bytes):
    """把响应流写入可回卷文件，检查真实大小并判断 PDF 文件头。"""
    output.seek(0)
    output.truncate()
    size = 0
    for chunk in response.iter_content(chunk_size=65536):
        size += len(chunk)
        if size>max_bytes:
            raise PaperAccessError("论文 PDF 超过上传大小上限")
        output.write(chunk)
    output.seek(0)
    is_pdf = output.read(1024).lstrip().startswith(b"%PDF-")
    output.seek(0)
    return is_pdf


class CampusPaperClient:
    """万方原生检索与知网官网页面定位，全文下载复用官方详情和全文入口。"""
    def __init__(self, *, timeout=30):
        """只保存配置；每次操作使用自己的 HTTP 会话，避免跨线程共享 Cookie。"""
        self.timeout = timeout

    def _session(self):
        """创建当前操作的会话，是否使用环境代理由校园网部署配置决定。"""
        session = requests.Session()
        session.trust_env = os.getenv("PAPERAGENT_PAPERS_USE_PROXY","true").lower() in {"true","1","yes"}
        session.headers.update({"User-Agent":"Mozilla/5.0 PaperAgent/3.0"})
        return session

    def _get(self, session, url, provider):
        """逐跳验证官方来源，保持当前操作的 Cookie 与 Referer。"""
        for _ in range(5):
            try:
                actual_provider = provider_of(url)
            except ValueError as exc:
                raise PaperAccessError("论文下载跳转到了未支持的来源") from exc
            if actual_provider != provider:
                raise PaperAccessError("论文下载跳转到了其他来源")
            # 官网全文服务使用 Cookies 头转交同一次会话获得的站点 Cookie。
            cookies = "; ".join(f"{item.name}={item.value}" for item in session.cookies)
            response = session.get(url,timeout=self.timeout,stream=True,allow_redirects=False,
                                   headers={"Cookies":cookies})
            if response.status_code in {301,302,303,307,308}:
                url = urljoin(url,response.headers.get("Location",""))
                response.close()
                continue
            if response.status_code in {401,403}:
                response.close()
                raise PaperAccessError("站点未授予当前服务会话全文权限，请核对校园网识别与机构授权")
            try:
                response.raise_for_status()
            except Exception:
                response.close()
                raise
            if "/verify/" in urlsplit(url).path:
                response.close()
                raise PaperAccessError("论文站点要求安全验证，请在官网完成验证后重试")
            return response
        raise PaperAccessError("论文下载跳转次数过多")

    def _search_wanfang(self, query, max_results):
        """调用已核对的万方官网检索协议，读取期刊、学位和会议论文。"""
        with self._session() as session:
            with session.post("https://s.wanfangdata.com.cn/SearchService.SearchService/search",
                    data=encode_search(query,max_results),timeout=self.timeout,stream=True,allow_redirects=False,
                    headers={"Content-Type":"application/grpc-web+proto","X-Grpc-Web":"1",
                             "X-User-Agent":"grpc-web-javascript/0.1","Referer":"https://s.wanfangdata.com.cn/paper"}) as response:
                response.raise_for_status()
                result = decode_search(_read_limited(response,2*1024*1024))
        papers = []
        for resource in result.resources:
            kind = resource.type.lower()
            if kind not in {"periodical","thesis","conference"}:
                continue
            article = getattr(resource,kind)
            if not article.id or not article.titles:
                continue
            url = f"https://d.wanfangdata.com.cn/{kind}/{quote(article.id,safe='')}"
            title = re.sub(r"</?span\b[^>]*>", "", article.titles[0], flags=re.I)
            papers.append({"paper_id":canonical_id(url),"title":_plain_text(title)[:500],
                "source":"wanfang","summary":_plain_text(" ".join(article.abstracts))[:3000],
                "published":str(article.year) if article.year else "","doi":article.doi,
                "url":url,"search_via":"wanfang_grpc"})
        return papers[:max_results]

    def _search_cnki(self, query, max_results):
        """用网页检索定位知网官方详情页，结果域名与论文路径再次校验。"""
        results = BingWebSearch(timeout=self.timeout,max_results=max_results).search(
            f"(site:cnki.net OR site:cnki.com.cn) {query}")
        papers = []
        for item in results:
            try:
                paper_id = canonical_id(item["url"])
                if provider_of(paper_id) != "cnki":
                    continue
            except ValueError:
                continue
            papers.append({"paper_id":paper_id,"title":item["title"],"source":"cnki",
                "summary":item["content"],"published":"","url":paper_id,"search_via":"web_index"})
        return papers[:max_results]

    def search(self, query, max_results=5, source="all"):
        """返回论文目录和各来源的错误，部分来源失败时保留其他来源的结果。"""
        if not isinstance(query,str) or not query.strip() or len(query)>2000:
            raise ValueError("论文搜索词需要为 1 到 2000 字符")
        if isinstance(max_results,bool) or not isinstance(max_results,int) or not 1<=max_results<=10:
            raise ValueError("论文结果数量需要在 1 到 10 之间")
        if source not in {"all","cnki","wanfang"}:
            raise ValueError("论文来源需要为 cnki、wanfang 或 all")
        selected = ["wanfang","cnki"] if source=="all" else [source]
        papers, errors, seen = [], [], set()
        for name in selected:
            try:
                function = self._search_wanfang if name=="wanfang" else self._search_cnki
                for item in function(query.strip(),max_results):
                    if item["paper_id"] not in seen:
                        seen.add(item["paper_id"])
                        papers.append(item)
            except Exception as exc:
                errors.append({"source":name,"message":str(exc)[:300]})
        # 两个来源交错展示，避免总量限制让排在后面的来源始终不可见。
        groups = [[item for item in papers if item["source"]==name] for name in selected]
        interleaved = []
        for row in zip_longest(*groups):
            interleaved.extend(item for item in row if item is not None)
        return {"papers":interleaved[:max_results],"errors":errors}

    @contextmanager
    def open_pdf(self, paper_id, *, max_bytes, directory):
        """取得官网 PDF 后交给原入库服务，校验失败时不会提交入库任务。"""
        paper_id = canonical_id(paper_id)
        provider = provider_of(paper_id)
        Path(directory).mkdir(parents=True,exist_ok=True)
        with self._session() as session, tempfile.SpooledTemporaryFile(max_size=4*1024*1024,mode="w+b",dir=directory) as output:
            session.headers["Referer"] = paper_id
            with self._get(session,paper_id,provider) as page:
                if _save_response(page,output,max_bytes):
                    yield output
                    return
                html = output.read(2*1024*1024+1)
                if len(html)>2*1024*1024:
                    raise PaperAccessError("论文详情页面超过大小限制")
                text = html.decode(page.encoding or "utf-8",errors="replace")
                if "安全验证" in text[:20000] or "captchaType" in text[:20000]:
                    raise PaperAccessError("论文站点要求安全验证，请在官网完成验证后重试")
            parser = _DownloadLinks(paper_id)
            parser.feed(text)
            candidates = parser.links
            if provider=="wanfang":
                parts = urlsplit(paper_id).path.strip("/").split("/")
                if len(parts)==2 and parts[0] in {"periodical","thesis","conference"}:
                    candidates.append(f"https://s.wanfangdata.com.cn/fulltext/api/get/pc/{parts[0]}/{parts[1]}")
            errors = []
            for url in dict.fromkeys(candidates):
                try:
                    with self._get(session,url,provider) as response:
                        if not _save_response(response,output,max_bytes):
                            raise PaperAccessError("官网全文入口返回了验证页或非 PDF 内容")
                except Exception as exc:
                    errors.append(str(exc))
                    continue
                # 离开 HTTP 响应后再交给入库；下游失败由原任务机制处理。
                yield output
                return
            reason = errors[-1] if errors else "详情页未提供可用的 PDF 下载入口，可能仅提供 CAJ 或需要站点授权"
            raise PaperAccessError(reason)
