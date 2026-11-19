"""三套 PDF 处理方案共享的不可变配置与环境读取入口。"""

from dataclasses import dataclass
import os
from pathlib import Path

from .embeddings import EMBEDDING_MODEL
from .local_ocr import LOCAL_OCR_MODEL_ID
from .paths import CHROMA_DIR, DATA_DIR

@dataclass(frozen=True)
class PipelineConfig:
    """不可变的解析与入库配置。切块、模型、缓存和目录参数参与任务与缓存身份计算。"""
    data_root: Path = DATA_DIR / "ingestion"
    chroma_dir: Path = DATA_DIR / "chroma_db"
    collection_name: str | None = None
    # chunk_size/overlap 的单位为字符，最终模型请求另有 Token 估算预算。
    chunk_size: int = 800
    chunk_overlap: int = 120
    dpi: int = 144
    min_text_chars: int = 80
    max_bad_char_ratio: float = 0.05
    ocr_model: str = LOCAL_OCR_MODEL_ID
    vlm_model: str = "qwen3-vl-flash"
    ocr_coordinate_space: str = "normalized_1000"
    # 本地 OCR 输出像素坐标，此字段约定 VLM 布局输出的坐标空间。
    ocr_threads: int = 2
    ocr_min_confidence: float = 0.5
    ocr_match_threshold: float = 0.5
    request_timeout: float = 90
    max_tokens: int = 8192
    use_cache: bool = True
    embedding_model_id: str = EMBEDDING_MODEL

    def __post_init__(self):
        """校验切块重叠、图片 DPI、文本质量阈值、OCR 坐标空间和请求参数。"""
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("需要 chunk_size > chunk_overlap >= 0")
        if self.dpi <= 0 or self.min_text_chars < 0:
            raise ValueError("dpi 必须大于 0，min_text_chars 必须非负")
        if not 0 <= self.max_bad_char_ratio <= 1:
            raise ValueError("max_bad_char_ratio 必须在 0 到 1 之间")
        if self.ocr_coordinate_space not in {"normalized_1000", "pixels"}:
            raise ValueError("ocr_coordinate_space 需要为 normalized_1000 或 pixels")
        if self.request_timeout <= 0 or self.max_tokens <= 0:
            raise ValueError("request_timeout 和 max_tokens 必须大于 0")
        if isinstance(self.ocr_threads, bool) or not isinstance(self.ocr_threads, int) or self.ocr_threads < 1:
            raise ValueError("ocr_threads 必须为正整数")
        for name in ("ocr_min_confidence", "ocr_match_threshold"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} 必须在 0 到 1 之间")


    @classmethod
    def from_environment(cls, **overrides):
        """统一读取网页和 PDF CLI 的环境配置，显式参数覆盖环境默认值。"""
        options = {
            "data_root": Path(os.getenv("PAPERAGENT_INGEST_ROOT", str(DATA_DIR / "ingestion"))),
            "chroma_dir": Path(os.getenv("PAPERAGENT_CHROMA_DIR", str(CHROMA_DIR))),
            "ocr_threads": int(os.getenv("PAPERAGENT_OCR_THREADS", "2")),
            "ocr_min_confidence": float(os.getenv("PAPERAGENT_OCR_MIN_CONFIDENCE", "0.5")),
            "ocr_match_threshold": float(os.getenv("PAPERAGENT_OCR_MATCH_THRESHOLD", "0.5")),
            "vlm_model": os.getenv("PAPERAGENT_VLM_MODEL", "qwen3-vl-flash"),
        }
        options.update(overrides)
        return cls(**options)
