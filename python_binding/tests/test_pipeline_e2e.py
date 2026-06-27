"""End-to-end Pipeline tests with mocked VLMRunnerSo.

Uses real DocLayoutRunner only if ONNX is available; otherwise skips layout
tests and tests pipeline orchestration with a mocked layout_runner too.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pipeline.pipeline import Pipeline
from pipeline.result import PaddleOCRVLBlock, PipelineResult


def _make_test_image(path: Path):
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    img.save(str(path))


class TestPipelineOrchestration:
    """Tests that mock both DocLayoutRunner and VLMRunnerSo — no ONNX/.so needed."""

    def _make_pipeline_with_mocks(self, boxes, vlm_responses):
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.layout_threshold = 0.5
        pipeline.layout_runner = MagicMock()
        pipeline.layout_runner.detect.return_value = boxes
        pipeline.vlm_runner = MagicMock()
        pipeline.vlm_runner.run.side_effect = vlm_responses
        return pipeline

    def test_process_image_runs_layout_then_vlm_per_text_block(self, tmp_path):
        img_path = tmp_path / "input.png"
        _make_test_image(img_path)
        boxes = [
            {"label": "text", "score": 0.9, "bbox": [0, 0, 100, 50]},
            {"label": "table", "score": 0.8, "bbox": [0, 60, 100, 100]},
        ]
        vlm_responses = [
            ("hello text", {"vision_latency": 5, "prefill_latency": 10,
                            "prefill_tokens": 3, "decode_latency": 50,
                            "decode_tokens": 5}),
            ("<table/>", {"vision_latency": 8, "prefill_latency": 15,
                          "prefill_tokens": 4, "decode_latency": 80,
                          "decode_tokens": 8}),
        ]
        pipeline = self._make_pipeline_with_mocks(boxes, vlm_responses)
        result = pipeline.process_image(str(img_path))
        assert isinstance(result, PipelineResult)
        assert len(result.parsing_res_list) == 2
        assert result.parsing_res_list[0].label == "text"
        assert result.parsing_res_list[0].content == "hello text"
        assert result.parsing_res_list[1].label == "table"
        assert result.parsing_res_list[1].content == "<table/>"
        # Both blocks ran VLM
        assert pipeline.vlm_runner.run.call_count == 2
        # Metrics populated
        assert result.metrics is not None
        assert result.metrics["vlm_blocks_run"] == 2
        assert result.metrics["vlm_blocks_skipped"] == 0
        assert result.metrics["layout_boxes"] == 2

    def test_image_label_skipped_no_vlm_call(self, tmp_path):
        img_path = tmp_path / "input.png"
        _make_test_image(img_path)
        boxes = [
            {"label": "image", "score": 0.9, "bbox": [0, 0, 100, 100]},
            {"label": "text", "score": 0.8, "bbox": [0, 0, 50, 50]},
        ]
        vlm_responses = [("text content", {"vision_latency": 1, "prefill_latency": 1,
                                           "prefill_tokens": 1, "decode_latency": 1,
                                           "decode_tokens": 1})]
        pipeline = self._make_pipeline_with_mocks(boxes, vlm_responses)
        result = pipeline.process_image(str(img_path))
        # Only one VLM call (for text block)
        assert pipeline.vlm_runner.run.call_count == 1
        # Image block recorded as empty
        assert result.parsing_res_list[0].label == "image"
        assert result.parsing_res_list[0].content == ""
        assert result.metrics["vlm_blocks_run"] == 1
        assert result.metrics["vlm_blocks_skipped"] == 1

    def test_vlm_exception_does_not_break_pipeline(self, tmp_path):
        img_path = tmp_path / "input.png"
        _make_test_image(img_path)
        boxes = [
            {"label": "text", "score": 0.9, "bbox": [0, 0, 50, 50]},
            {"label": "text", "score": 0.8, "bbox": [50, 50, 100, 100]},
        ]
        side_effects = [
            RuntimeError("VLM failed"),
            ("recovered", {"vision_latency": 1, "prefill_latency": 1,
                           "prefill_tokens": 1, "decode_latency": 1,
                           "decode_tokens": 1}),
        ]
        pipeline = self._make_pipeline_with_mocks(boxes, side_effects)
        result = pipeline.process_image(str(img_path))
        # First block failed → content empty; second block succeeded
        assert result.parsing_res_list[0].content == ""
        assert result.parsing_res_list[1].content == "recovered"
        assert pipeline.vlm_runner.run.call_count == 2

    def test_visualize_output_writes_png(self, tmp_path):
        img_path = tmp_path / "input.png"
        _make_test_image(img_path)
        vis_path = tmp_path / "vis.png"
        boxes = [{"label": "text", "score": 0.9, "bbox": [0, 0, 50, 50]}]
        vlm_responses = [("hi", {"vision_latency": 1, "prefill_latency": 1,
                                  "prefill_tokens": 1, "decode_latency": 1,
                                  "decode_tokens": 1})]
        pipeline = self._make_pipeline_with_mocks(boxes, vlm_responses)
        result = pipeline.process_image(str(img_path), visualize_output=str(vis_path))
        assert vis_path.exists()
        assert result.img is not None

    def test_layout_det_res_captures_all_boxes(self, tmp_path):
        img_path = tmp_path / "input.png"
        _make_test_image(img_path)
        boxes = [
            {"label": "text", "score": 0.9, "bbox": [0, 0, 100, 50]},
            {"label": "image", "score": 0.8, "bbox": [0, 60, 100, 100]},
        ]
        vlm_responses = []
        pipeline = self._make_pipeline_with_mocks(boxes, vlm_responses)
        result = pipeline.process_image(str(img_path))
        assert len(result.layout_det_res["boxes"]) == 2
        assert result.layout_det_res["boxes"][0]["label"] == "text"
        assert result.layout_det_res["boxes"][0]["score"] == 0.9
