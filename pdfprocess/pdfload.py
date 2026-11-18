"""本地 PDF 命令行入口，默认使用本地 OCR 与通义 VLM 双路方案。

复用批量处理和产物报告，例如
python -m pdfprocess.pdfload --pdfs paper.pdf --parse-only。
"""

from pathlib import Path
import sys

if __package__:
    from .compare_ingest import main as process_main
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pdfprocess.compare_ingest import main as process_main


def main(argv=None):
    """读取本地 PDF 路径，缺省选择方案三，返回批量处理退出码。"""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(argument.split("=", 1)[0] == "--schemes" for argument in arguments):
        arguments = ["--schemes", "3", *arguments]
    return process_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
