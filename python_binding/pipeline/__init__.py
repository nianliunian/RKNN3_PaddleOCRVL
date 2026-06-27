"""Default paths and DocLayoutV2 label list for the layout+VLM pipeline."""
from pathlib import Path

DEFAULT_PATHS = {
    "onnx_path": "/root/demo_paddleocr_vl/PP_onnx/PP-DocLayoutV2.onnx",
    "model_dir": "/root/demo_paddleocr_vl/model",
}

# DocLayoutV2 label list (25 classes), from config.json
DOCLAYOUT_LABELS = [
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer",
    "footer_image", "footnote", "formula_number", "header",
    "header_image", "image", "inline_formula", "number",
    "paragraph_title", "reference", "reference_content", "seal",
    "table", "text", "vertical_text", "vision_footnote",
]
