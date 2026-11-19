"""历史拼写的兼容入口，检索调试实现统一位于 retrieval_cli.py。"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pdfprocess"

from .retrieval_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
