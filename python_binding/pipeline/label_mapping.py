"""Map DocLayoutV2 25-class labels to VLM prompt labels.

VLM (infer_simular.py) supports: ocr, table, formula, chart.
Image-type labels return None (skip VLM, no text content expected).
"""
from pipeline import DOCLAYOUT_LABELS

SUPPORTED_PROMPTS = {"ocr", "table", "formula", "chart"}

# label -> prompt label (None = skip VLM)
_LABEL_TO_PROMPT = {
    # Text-like -> ocr
    "text": "ocr",
    "paragraph_title": "ocr",
    "doc_title": "ocr",
    "reference": "ocr",
    "reference_content": "ocr",
    "content": "ocr",
    "abstract": "ocr",
    "header": "ocr",
    "footer": "ocr",
    "footnote": "ocr",
    "aside_text": "ocr",
    "vertical_text": "ocr",
    "number": "ocr",
    "formula_number": "ocr",
    "algorithm": "ocr",
    # Table
    "table": "table",
    # Formula
    "display_formula": "formula",
    "inline_formula": "formula",
    # Chart
    "chart": "chart",
    # Image-like -> skip
    "image": None,
    "figure_title": None,
    "header_image": None,
    "footer_image": None,
    "seal": None,
    "vision_footnote": None,
}


def label_to_prompt(label: str) -> str | None:
    """Return VLM prompt label for a DocLayoutV2 label, or None to skip.

    Raises ValueError if label is not in the 25-class label list.
    """
    if label not in _LABEL_TO_PROMPT:
        raise ValueError(
            f"Unknown label: {label!r}. Expected one of {DOCLAYOUT_LABELS}"
        )
    return _LABEL_TO_PROMPT[label]
