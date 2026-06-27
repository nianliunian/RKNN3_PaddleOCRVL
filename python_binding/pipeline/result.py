"""Result objects aligned with PaddleOCRVLResult."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_IMAGE_LABELS = frozenset({
    "image", "figure_title", "header_image",
    "footer_image", "seal", "vision_footnote",
})


def _is_formula_label(label: str) -> bool:
    return "formula" in label and label != "formula_number"


@dataclass
class PaddleOCRVLBlock:
    """A single block recognized by the pipeline."""
    label: str
    bbox: list[float]
    content: str
    polygon_points: list | None = None

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "bbox": self.bbox,
            "content": self.content,
            "polygon_points": self.polygon_points,
        }


def _default_model_settings() -> dict:
    return {
        "pipeline_version": "v1.6-rknn-so",
        "use_layout_detection": True,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_chart_recognition": True,
        "use_seal_recognition": False,
        "use_ocr_for_image_block": False,
    }


@dataclass
class PipelineResult:
    """Pipeline output, structurally compatible with PaddleOCRVLResult."""
    input_path: str
    page_index: int = 0
    page_count: int = 1
    width: int = 0
    height: int = 0
    model_settings: dict = field(default_factory=_default_model_settings)
    layout_det_res: dict = field(default_factory=lambda: {"boxes": []})
    parsing_res_list: list[PaddleOCRVLBlock] = field(default_factory=list)
    metrics: dict | None = None
    _img: Any = None

    def to_dict(self) -> dict:
        return {
            "input_path": self.input_path,
            "page_index": self.page_index,
            "page_count": self.page_count,
            "width": self.width,
            "height": self.height,
            "model_settings": self.model_settings,
            "layout_det_res": self.layout_det_res,
            "parsing_res_list": [b.to_dict() for b in self.parsing_res_list],
            "metrics": self.metrics,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        parts: list[str] = []
        for block in self.parsing_res_list:
            content = block.content or ""
            if block.label == "table":
                parts.append(f"\n{content}\n")
            elif _is_formula_label(block.label):
                stripped = content.strip()
                if stripped.startswith("$$") and stripped.endswith("$$"):
                    parts.append(f"\n{stripped}\n")
                elif stripped.startswith("$") and stripped.endswith("$"):
                    parts.append(f"\n{stripped}\n")
                elif block.label == "display_formula":
                    parts.append(f"\n$${stripped}$$\n")
                else:
                    parts.append(f" ${stripped}$ ")
            elif block.label in _IMAGE_LABELS:
                parts.append("")
            else:
                parts.append(f"{content}\n\n")
        return "".join(parts)

    def to_html(self) -> str:
        parts = ["<div class=\"doc-result\">"]
        for block in self.parsing_res_list:
            content = block.content or ""
            if block.label == "table" and content:
                parts.append(f"<div class=\"block-table\">{content}</div>")
            elif _is_formula_label(block.label):
                parts.append(f"<div class=\"block-formula\">{content}</div>")
            elif block.label in _IMAGE_LABELS:
                parts.append("<div class=\"block-image\"></div>")
            else:
                escaped = (
                    content.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                )
                parts.append(f"<p>{escaped}</p>")
        parts.append("</div>")
        return "\n".join(parts)

    def save_all(self, save_path: str) -> None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.with_suffix(".json").write_text(self.to_json(), encoding="utf-8")
        p.with_suffix(".md").write_text(self.to_markdown(), encoding="utf-8")
        p.with_suffix(".html").write_text(self.to_html(), encoding="utf-8")

    @property
    def img(self):
        return self._img

    @img.setter
    def img(self, value):
        self._img = value
