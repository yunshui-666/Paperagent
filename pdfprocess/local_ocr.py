"""本机 OCR 适配器，使用 RapidOCR 自带的 PP-OCRv6 ONNX 权重。

模型文件随固定版本 wheel 安装，识别过程在 CPU 上执行。每个解析进程复用
已加载的模型，并以锁保护同一个推理实例。输出为可以直接写入 JSON 的文字框。
"""

from functools import lru_cache
import math
from pathlib import Path
from threading import Lock


LOCAL_OCR_MODEL_ID = "rapidocr-ppocrv6-small"


@lru_cache(maxsize=2)
def _load_engine(threads):
    """按 CPU 线程设置复用引擎及推理锁，首次初始化读取已安装的模型文件。"""
    try:
        import rapidocr
        from rapidocr import RapidOCR
    except ImportError as exc:
        raise RuntimeError("请先安装 requirements-ingest.txt 中的本地 OCR 依赖") from exc
    directory = Path(rapidocr.__file__).resolve().parent / "models"
    paths = {"Det": directory / "PP-OCRv6_det_small.onnx",
             "Cls": directory / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
             "Rec": directory / "PP-OCRv6_rec_small.onnx"}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"本地 OCR 权重缺失，请按 requirements-ingest.txt 安装 RapidOCR：{missing}")
    params = {f"{kind}.model_path": str(path) for kind, path in paths.items()}
    params.update({"Global.log_level": "warning",
                   "EngineConfig.onnxruntime.intra_op_num_threads": threads,
                   "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                   "EngineConfig.onnxruntime.use_cuda": False,
                   "EngineConfig.onnxruntime.use_dml": False})
    return RapidOCR(params=params), Lock()


class LocalOCR:
    """将一页渲染后的图片识别为文字、像素坐标和置信度列表。"""

    def __init__(self, *, model_id=LOCAL_OCR_MODEL_ID, threads=2, min_confidence=0.5):
        """保存推理配置，可通过 prepare() 在发出 VLM 请求前检查本地引擎。"""
        if model_id != LOCAL_OCR_MODEL_ID:
            raise ValueError("当前 OCR 使用 rapidocr-ppocrv6-small，请使用新配置重建旧任务")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("OCR threads 必须为正整数")
        if (isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float))
                or not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1):
            raise ValueError("OCR min_confidence 必须在 0 到 1 之间")
        self.model_id = model_id
        self.threads = threads
        self.min_confidence = min_confidence

    def prepare(self):
        """预加载 CPU 模型，依赖或权重错误在双路任务启动前报告。"""
        _load_engine(self.threads)

    def recognize(self, image_bytes):
        """只在本机执行推理，返回普通 Python 值组成的文字框列表。"""
        engine, lock = _load_engine(self.threads)
        with lock:
            output = engine(image_bytes, text_score=self.min_confidence)
        if output.txts is None:
            return []
        if output.boxes is None or output.scores is None or not (len(output.boxes) == len(output.txts) == len(output.scores)):
            raise RuntimeError("本地 OCR 的文字、坐标与置信度数量不一致")
        lines = []
        for polygon, text, score in zip(output.boxes, output.txts, output.scores):
            points = [[float(x), float(y)] for x, y in polygon]
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            lines.append({"text": text, "confidence": float(score), "polygon": points,
                          "bbox": [min(xs), min(ys), max(xs), max(ys)]})
        return lines


def validate_lines(lines, width, height):
    """验证本地 OCR 统一返回值，并把矩形裁到本页图片范围。"""
    if not isinstance(lines, list):
        raise ValueError("本地 OCR 需要返回文字框列表")
    cleaned = []
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("text"), str):
            raise ValueError("本地 OCR 文字框缺少 text")
        box, score = line.get("bbox"), line.get("confidence")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in box)
                or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("本地 OCR 坐标或置信度无效")
        box = [max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("本地 OCR 文字框超出页面范围")
        if line["text"].strip():
            cleaned.append({**line, "text": line["text"].strip(), "bbox": box})
    return cleaned
