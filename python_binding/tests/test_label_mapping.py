"""Tests for label_to_prompt mapping."""
import pytest
from pipeline.label_mapping import label_to_prompt, SUPPORTED_PROMPTS


class TestLabelToPrompt:
    def test_text_labels_map_to_ocr(self):
        for label in [
            "text", "paragraph_title", "doc_title", "reference",
            "reference_content", "content", "abstract", "header",
            "footer", "footnote", "aside_text", "vertical_text",
            "number", "formula_number", "algorithm",
        ]:
            assert label_to_prompt(label) == "ocr", f"{label} should map to ocr"

    def test_table_maps_to_table(self):
        assert label_to_prompt("table") == "table"

    def test_formula_labels_map_to_formula(self):
        assert label_to_prompt("display_formula") == "formula"
        assert label_to_prompt("inline_formula") == "formula"

    def test_formula_number_maps_to_ocr_not_formula(self):
        """formula_number is a text label, not a formula."""
        assert label_to_prompt("formula_number") == "ocr"

    def test_chart_maps_to_chart(self):
        assert label_to_prompt("chart") == "chart"

    def test_image_labels_return_none(self):
        for label in [
            "image", "figure_title", "header_image", "footer_image",
            "seal", "vision_footnote",
        ]:
            assert label_to_prompt(label) is None, f"{label} should map to None"

    def test_all_25_labels_covered(self):
        """Every label in DOCLAYOUT_LABELS must have a defined mapping."""
        from pipeline import DOCLAYOUT_LABELS
        for label in DOCLAYOUT_LABELS:
            result = label_to_prompt(label)
            assert result is None or result in SUPPORTED_PROMPTS

    def test_unknown_label_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown label"):
            label_to_prompt("nonexistent_label")
