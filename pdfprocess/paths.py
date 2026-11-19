"""统一项目资源路径，默认数据目录继续位于 pdfprocess/data。"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "pdfprocess" / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
MODEL_DIR = DATA_DIR / "models" / "mmarco-minilm"
DOCS_DIR = PROJECT_DIR / "docs"


def chroma_directory():
    """运行时读取向量库路径，让网页、CLI 与容器使用相同环境配置。"""
    return Path(os.getenv("PAPERAGENT_CHROMA_DIR", str(CHROMA_DIR)))
