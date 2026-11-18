"""同一批 PDF 的解析与入库对比入口。"""

# 解析方案实验入口。使用同一批 PDF、配置和参考词统计解析结果及调用成本。

import argparse
from dataclasses import asdict
import importlib
from pathlib import Path
from uuid import uuid4

from .ingest_common import PipelineConfig, write_json


def main(argv=None):
    """读取 CLI 参数，逐份运行选定解析方案并写出包含结果和耗时的 JSON 报告。"""
    parser = argparse.ArgumentParser(description="比较三套 PDF 解析入库方案")
    parser.add_argument("--pdfs", nargs="+", required=True, help="同一批 PDF 文件路径")
    parser.add_argument("--schemes", nargs="+", type=int, choices=(1, 2, 3), default=[1])
    parser.add_argument("--parse-only", action="store_true", help="仅解析，不调用 Embedding 和 Chroma")
    parser.add_argument("--refresh", action="store_true", help="重新解析，测量实际模型调用及耗时")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--chroma-dir", type=Path)
    parser.add_argument("--ocr-model", help="本地 OCR 模型标识，默认 rapidocr-ppocrv6-small")
    parser.add_argument("--vlm-model", help="图表理解服务实际支持的模型名")
    parser.add_argument("--output", type=Path, help="对比结果 JSON 保存位置")
    args = parser.parse_args(argv)
    options = {"use_cache": not args.refresh}
    # 只覆盖 CLI 明确提供的参数，其余参数由同一个 PipelineConfig 默认值决定。
    if args.data_root is not None:
        options["data_root"] = args.data_root
    if args.chroma_dir is not None:
        options["chroma_dir"] = args.chroma_dir
    if args.ocr_model is not None:
        options["ocr_model"] = args.ocr_model
    if args.vlm_model is not None:
        options["vlm_model"] = args.vlm_model
    config = PipelineConfig(**options)
    results = []
    for scheme in args.schemes:
        # 动态选择同名不同后缀的入口，三套方案接收相同文件和配置结构。
        module = importlib.import_module(f".pdf_pipeline{scheme}", package=__package__)
        function = getattr(module, f"ingest_pdf{scheme}")
        if args.parse_only:
            # 解析对比可以单独观察页面产物和 OCR/VLM 用量。
            function = getattr(module, f"pdf_to_markdown{scheme}")
        for path in args.pdfs:
            try:
                result = function(path, config=config)
            except Exception as exc:
                # 每个文件保留自己的错误，后续文件仍继续形成完整实验记录。
                result = {"source": str(path), "scheme": scheme,
                          "status": "failed", "error": str(exc)}
            results.append(result)
            metrics = result.get("metrics", {})
            print(f"方案 {scheme} | {Path(path).name} | {result['status']} | "
                  f"OCR={metrics.get('ocr_calls', 0)} VLM={metrics.get('vlm_calls', 0)} "
                  f"缓存页={metrics.get('cached_pages', 0)}")
            if "error" in result:
                print(result["error"])
            for page in result.get("pages", []):
                if page["status"] == "failed":
                    print(f"第 {page['page']} 页：{page['error']}")
    output = args.output or Path(config.data_root) / "reports" / f"comparison-{uuid4().hex[:12]}.json"
    configuration = {key: str(value) if isinstance(value, Path) else value
                     # Path 转为 JSON 字符串，保存本轮参数快照便于复现实验。
                     for key, value in asdict(config).items()}
    write_json(output, {"config": configuration, "parse_only": args.parse_only, "results": results})
    print(f"对比结果：{output.resolve()}")
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
