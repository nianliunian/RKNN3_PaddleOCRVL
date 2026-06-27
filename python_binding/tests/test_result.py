"""Tests for PaddleOCRVLBlock and PipelineResult."""
import json
from pathlib import Path

from pipeline.result import PaddleOCRVLBlock, PipelineResult


class TestPaddleOCRVLBlock:
    def test_construct_basic_block(self):
        block = PaddleOCRVLBlock(
            label="text",
            bbox=[10.0, 20.0, 100.0, 50.0],
            content="hello world",
        )
        assert block.label == "text"
        assert block.bbox == [10.0, 20.0, 100.0, 50.0]
        assert block.content == "hello world"
        assert block.polygon_points is None

    def test_to_dict(self):
        block = PaddleOCRVLBlock(
            label="table", bbox=[0, 0, 10, 10], content="<table/>"
        )
        d = block.to_dict()
        assert d == {
            "label": "table",
            "bbox": [0, 0, 10, 10],
            "content": "<table/>",
            "polygon_points": None,
        }


class TestPipelineResultMetrics:
    def test_default_metrics_is_none(self):
        r = PipelineResult(input_path="x.png")
        assert r.metrics is None

    def test_metrics_field_round_trips_in_to_dict(self):
        m = {"layout_ms": 12.3, "vlm_total_ms": 45.6}
        r = PipelineResult(input_path="x.png", metrics=m)
        d = r.to_dict()
        assert d["metrics"] == m

    def test_to_json_includes_metrics(self):
        r = PipelineResult(
            input_path="x.png",
            metrics={"layout_ms": 1.0, "vlm_total_ms": 2.0},
        )
        j = json.loads(r.to_json())
        assert j["metrics"] == {"layout_ms": 1.0, "vlm_total_ms": 2.0}


class TestPipelineResultMarkdown:
    def test_text_block_appends_paragraph(self):
        block = PaddleOCRVLBlock(label="text", bbox=[0, 0, 1, 1], content="hi")
        r = PipelineResult(input_path="x.png", parsing_res_list=[block])
        assert r.to_markdown() == "hi\n\n"

    def test_table_block_kept_as_is(self):
        block = PaddleOCRVLBlock(label="table", bbox=[0, 0, 1, 1], content="<tbl>")
        r = PipelineResult(input_path="x.png", parsing_res_list=[block])
        assert r.to_markdown() == "\n<tbl>\n"

    def test_display_formula_wrapped_in_double_dollar(self):
        block = PaddleOCRVLBlock(
            label="display_formula", bbox=[0, 0, 1, 1], content="x^2"
        )
        r = PipelineResult(input_path="x.png", parsing_res_list=[block])
        assert r.to_markdown() == "\n$$x^2$$\n"

    def test_save_all_writes_three_files(self, tmp_path):
        r = PipelineResult(input_path="x.png", parsing_res_list=[
            PaddleOCRVLBlock(label="text", bbox=[0, 0, 1, 1], content="hi")
        ])
        save_path = tmp_path / "result"
        r.save_all(str(save_path))
        assert save_path.with_suffix(".json").exists()
        assert save_path.with_suffix(".md").exists()
        assert save_path.with_suffix(".html").exists()
