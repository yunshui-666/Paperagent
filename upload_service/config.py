"""服务配置层。环境变量在启动时转成明确类型，PipelineConfig 保存 PDF 处理相关参数。"""

from dataclasses import dataclass, field
import os
from pathlib import Path

from pdfprocess.ingest_common import DATA_DIR, PipelineConfig


@dataclass(frozen=True)
class ServiceSettings:
    """不可变的服务配置，保存业务目录、入库配置、线程进程数量及文件限制。"""
    service_root: Path = DATA_DIR / "upload_service"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    # worker_count 控制任务消费线程，parse_processes 控制独立 PDF 解析进程数量。
    worker_count: int = 1
    parse_processes: int = 1
    max_file_bytes: int = 50 * 1024 * 1024
    sqlite_journal_mode: str = "WAL"

    def __post_init__(self):
        """校验服务并发数量、文件字节上限和 SQLite 日志模式。"""
        if isinstance(self.worker_count, bool) or not isinstance(self.worker_count, int) or self.worker_count < 1:
            raise ValueError("worker_count 必须为正整数")
        if isinstance(self.parse_processes, bool) or not isinstance(self.parse_processes, int) or self.parse_processes < 1:
            raise ValueError("parse_processes 必须为正整数")
        if isinstance(self.max_file_bytes, bool) or not isinstance(self.max_file_bytes, int) or self.max_file_bytes < 1:
            raise ValueError("max_file_bytes 必须为正整数")
        if self.sqlite_journal_mode not in {"WAL", "DELETE"}:
            raise ValueError("sqlite_journal_mode 需要为 WAL 或 DELETE")

    @classmethod
    def from_environment(cls):
        """从 PAPERAGENT 环境变量读取配置，并在边界处完成字符串到类型的转换。"""
        # PAPERAGENT_SERVICE_ROOT 保存业务数据库与原文件，INGEST_ROOT 保存解析缓存。
        # CHROMA_DIR 保存向量库，三个目录在用途和生命周期上分别管理。
        return cls(
            service_root=Path(os.getenv("PAPERAGENT_SERVICE_ROOT", str(DATA_DIR / "upload_service"))),
            pipeline=PipelineConfig(
                data_root=Path(os.getenv("PAPERAGENT_INGEST_ROOT", str(DATA_DIR / "ingestion"))),
                chroma_dir=Path(os.getenv("PAPERAGENT_CHROMA_DIR", str(DATA_DIR / "chroma_db"))),
                ocr_threads=int(os.getenv("PAPERAGENT_OCR_THREADS", "2")),
                ocr_min_confidence=float(os.getenv("PAPERAGENT_OCR_MIN_CONFIDENCE", "0.5")),
                ocr_match_threshold=float(os.getenv("PAPERAGENT_OCR_MATCH_THRESHOLD", "0.5")),
                vlm_model=os.getenv("PAPERAGENT_VLM_MODEL", "qwen3-vl-flash")
            ),
            worker_count=int(os.getenv("PAPERAGENT_UPLOAD_WORKERS", "1")),
            parse_processes=int(os.getenv("PAPERAGENT_PARSE_PROCESSES", "1")),
            max_file_bytes=int(os.getenv("PAPERAGENT_UPLOAD_MAX_MB", "50")) * 1024 * 1024,
            sqlite_journal_mode=os.getenv("PAPERAGENT_SQLITE_JOURNAL_MODE", "WAL").upper()
        )
