"""融合本地 OCR 文字与 VLM 版面，保留每个区域的来源和原始结果。"""

import json
import math


def parse_layout(text, width, height, coordinate_space="normalized_1000"):
    """读取 VLM 的 JSON 版面，按模型给出的阅读顺序转换为像素矩形。"""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    value = json.loads(stripped)
    if not isinstance(value, dict) or not isinstance(value.get("blocks"), list) or len(value["blocks"]) > 512:
        raise ValueError("VLM 版面需要包含 blocks 列表，区域数量上限为 512")
    blocks, seen = [], set()
    for item in value["blocks"]:
        if not isinstance(item, dict):
            raise ValueError("VLM 版面区域需要为对象")
        kind = item.get("type")
        if kind not in {"text", "title", "table", "formula", "image"}:
            raise ValueError("VLM 版面包含未知区域类型")
        box, content = item.get("bbox"), item.get("content", "")
        if (not isinstance(box, list) or len(box) != 4 or not isinstance(content, str)
                or any(isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) for number in box)):
            raise ValueError("VLM 区域需要四个有限坐标和文本 content")
        if coordinate_space == "normalized_1000":
            box = [box[0] * width / 1000, box[1] * height / 1000,
                   box[2] * width / 1000, box[3] * height / 1000]
        elif coordinate_space != "pixels":
            raise ValueError("未知 VLM 布局坐标空间")
        box = [max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3])]
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("VLM 返回无效版面区域")
        # 同类型同位置同内容的完全重复区域只保留一次。
        identity = (kind, tuple(box), content.strip())
        if identity in seen:
            continue
        seen.add(identity)
        blocks.append({"type": kind, "bbox": box, "content": content.strip(), "coordinate_space": "pixels"})
    return blocks


def _area(box):
    """计算已校验矩形的面积，供覆盖比例和并列选择使用。"""
    return (box[2] - box[0]) * (box[3] - box[1])


def _coverage(line_box, region_box):
    """计算 OCR 文字框被版面区域覆盖的面积比例。"""
    width = max(0, min(line_box[2], region_box[2]) - max(line_box[0], region_box[0]))
    height = max(0, min(line_box[3], region_box[3]) - max(line_box[1], region_box[1]))
    return width * height / _area(line_box)


def fuse_page(lines, layout, *, match_threshold=0.5):
    """文字按区域归属融合，每个 OCR 框最多分配一次，未匹配文字追加到页末。

    正文和标题优先保留本地 OCR 原文。图表及公式使用 VLM 结构化转写，
    同时附上该区域的 OCR 可见文字，便于核对数值、单位和生成内容。
    """
    groups = [[] for _ in layout]
    unmatched = []
    for index, line in enumerate(lines):
        matches = [(position, _coverage(line["bbox"], block["bbox"])) for position, block in enumerate(layout)]
        matches = [(position, ratio) for position, ratio in matches if ratio >= match_threshold]
        if not matches:
            unmatched.append(index)
            continue
        # 优先覆盖比例最高的区域，并列时选较小区域，最后按既有阅读顺序确定。
        position, _ = max(matches, key=lambda item: (item[1], -_area(layout[item[0]]["bbox"]), -item[0]))
        groups[position].append(index)
    blocks, parts = [], []
    for region, ids in zip(layout, groups):
        ocr_text = "\n".join(lines[index]["text"] for index in ids)
        visual_text = region["content"]
        generated = region["type"] in {"image", "table", "formula"} or not ocr_text
        content = visual_text if generated else ocr_text
        if not content:
            content, generated = ocr_text, False
        if not content:
            continue
        block = {**region, "content": content, "ocr_text": ocr_text, "vlm_content": visual_text,
                 "generated": generated, "ocr_line_ids": ids,
                 "provenance": "vlm" if generated else "local_ocr+vlm_layout"}
        blocks.append(block)
        if region["type"] == "title":
            rendered = "## " + content.replace("\n", " ")
        elif generated:
            rendered = "### VLM 区域转写或描述\n\n" + content
            if ocr_text:
                rendered += "\n\n**本地 OCR 可见文字**\n\n" + ocr_text
        else:
            rendered = content
        parts.append(rendered)
    if unmatched:
        content = "\n".join(lines[index]["text"] for index in unmatched)
        boxes = [lines[index]["bbox"] for index in unmatched]
        blocks.append({"type": "text", "content": content, "ocr_text": content, "vlm_content": "",
                       "generated": False, "provenance": "local_ocr", "ocr_line_ids": unmatched,
                       "bbox": [min(box[0] for box in boxes), min(box[1] for box in boxes),
                                max(box[2] for box in boxes), max(box[3] for box in boxes)],
                       "coordinate_space": "pixels"})
        parts.append("### 本地 OCR 补充文字\n\n" + content)
    return {"markdown": "\n\n".join(parts), "blocks": blocks, "unmatched_ocr_lines": len(unmatched)}
