"""将上传文件分块落盘、校验 PDF，并按内容哈希保存。"""

# 原始 PDF 存储层。流式保存、哈希标识、PDF 校验和临时文件清理由该模块处理。

import hashlib
import os
from pathlib import Path
import time
from uuid import uuid4

from pypdf import PdfReader


class UploadRejected(ValueError):
    """上传校验失败的结构化业务异常，携带错误代码及建议 HTTP 状态。"""
    def __init__(self, code, message, http_status=400):
        """保存可读说明、机器可读错误码与上传响应状态。"""
        super().__init__(message)
        self.code = code
        self.http_status = http_status


class FileStorage:
    """原始上传文件的流式接收、PDF 校验与内容寻址存储。"""
    def __init__(self, root, max_file_bytes):
        """保存受管理文件根目录及单文件大小上限。"""
        self.root = Path(root)
        self.max_file_bytes = max_file_bytes

    def save(self, filename, stream):
        """每次读一块上传流并累计哈希和大小，校验 PDF 后原子发布到内容哈希文件名。
        返回文件身份、页数、耗时与是否复用文件；失败也清理本次临时文件。
        """
        filename = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        # 原始文件名仅用于显示，实际落盘使用内容哈希，避免用户路径进入存储路径。
        if not filename or Path(filename).suffix.lower() != ".pdf":
            raise UploadRejected("invalid_extension", "请上传扩展名为 .pdf 的文件")
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / "staging"
        staging.mkdir(exist_ok=True)
        temporary = staging / f"{uuid4().hex}.part"
        started = time.perf_counter()
        digest = hashlib.sha256()
        size = 0
        try:
            stream.seek(0)
            with temporary.open("wb") as output:
                while chunk := stream.read(1024 * 1024):
                    # 一块最多 1 MiB，同时累计大小和哈希，上传接收的内存开销受块大小约束。
                    size += len(chunk)
                    if size > self.max_file_bytes:
                        raise UploadRejected("file_too_large", f"单文件大小上限为 {self.max_file_bytes} 字节", 413)
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                # 先把缓冲写入文件，再请求系统同步，然后再校验并发布文件。
                os.fsync(output.fileno())
            if size == 0:
                raise UploadRejected("empty_file", "PDF 内容为空")
            copy_seconds = time.perf_counter() - started
            validation_started = time.perf_counter()
            # 扩展名检查之后继续检查 PDF 结构、加密状态和页面数。
            try:
                with temporary.open("rb") as pdf_stream:
                    document = PdfReader(pdf_stream)
                    if document.is_encrypted or not len(document.pages):
                        raise ValueError("需要至少包含一页的非加密 PDF")
                    page_count = len(document.pages)
            except Exception as exc:
                raise UploadRejected("invalid_pdf", f"PDF 校验失败：{exc}") from exc
            file_id = digest.hexdigest()
            destination = self.root / f"{file_id}.pdf"
            # 相同字节具有相同身份，不同原文件名上传同一内容时可以复用落盘文件。
            reused_file = destination.exists()
            if not reused_file:
                try:
                    os.replace(temporary, destination)
                    # 同一文件系统内发布完整文件，后台解析可以读取完整内容。
                except PermissionError:
                    # Windows 下另一上传可能已发布同一文件，并被工作线程打开。
                    if not destination.exists():
                        raise
                    reused_file = True
            return {
                "id": file_id, "name": filename, "path": str(destination.resolve()),
                "size_bytes": size, "page_count": page_count,
                "copy_seconds": copy_seconds,
                "validation_seconds": time.perf_counter() - validation_started,
                "reused_file": reused_file
            }
        finally:
            # 正常发布会移走临时文件，异常和重复上传也统一执行清理。
            temporary.unlink(missing_ok=True)

    def cleanup_staging(self):
        """在服务单实例锁内清理上次中断遗留的本服务临时文件。"""
        removed = 0
        for path in (self.root / "staging").glob("*.part"):
            # 只匹配本服务 UUID 命名并位于受管理目录内的暂存文件。
            if (len(path.stem) == 32 and all(char in "0123456789abcdef" for char in path.stem)
                    and path.is_file() and path.resolve().is_relative_to(self.root.resolve())):
                path.unlink()
                removed += 1
        return removed
