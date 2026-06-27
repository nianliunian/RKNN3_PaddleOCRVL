"""Tests for VLMRunnerSo — uses fake paddleocr_vl module, no real .so needed."""
import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def fake_paddleocr_vl(monkeypatch):
    """Inject a fake `paddleocr_vl` module into sys.modules."""
    fake_module = MagicMock()
    fake_model = MagicMock()
    fake_model.run.return_value = {
        "text": "hello world",
        "metrics": {"vision_latency": 10.0, "prefill_latency": 20.0,
                    "prefill_tokens": 5, "decode_latency": 100.0,
                    "decode_tokens": 10},
    }
    fake_module.PaddleOCRVL.return_value = fake_model
    monkeypatch.setitem(sys.modules, "paddleocr_vl", fake_module)
    return fake_module, fake_model


def _make_pil_image():
    return Image.new("RGB", (50, 50), color=(255, 0, 0))


class TestVLMRunnerSo:
    def test_init_calls_paddleocr_vl_init(self, fake_paddleocr_vl):
        fake_module, fake_model = fake_paddleocr_vl
        from pipeline.vlm_runner_so import VLMRunnerSo
        runner = VLMRunnerSo(
            model_dir="/fake/model",
            vision_core_mask=0x1,
            mlpar_core_mask=0x2,
            llm_core_mask=0x4,
        )
        fake_module.PaddleOCRVL.assert_called_once()
        fake_model.init.assert_called_once_with(
            "/fake/model",
            vision_core_mask=0x1,
            mlpar_core_mask=0x2,
            llm_core_mask=0x4,
        )

    def test_run_with_pil_image_writes_tmp_and_returns_text_metrics(self, fake_paddleocr_vl):
        _, fake_model = fake_paddleocr_vl
        from pipeline.vlm_runner_so import VLMRunnerSo
        runner = VLMRunnerSo(model_dir="/fake/model")
        text, metrics = runner.run(_make_pil_image(), "ocr")
        assert text == "hello world"
        assert metrics["vision_latency"] == 10.0
        # model.run was called with a path that points to an existing tmp file
        call_args = fake_model.run.call_args
        tmp_path = call_args.kwargs.get("image_path") or call_args.args[0]
        # After run returns, tmp file must be cleaned up
        assert not os.path.exists(tmp_path), f"tmp file {tmp_path} should be deleted"
        # prompt_type passed correctly
        assert call_args.kwargs.get("prompt_type") == "ocr"

    def test_run_with_path_passes_path_directly(self, fake_paddleocr_vl, tmp_path):
        """When given a str path, no tmp file is created; path passed as-is."""
        _, fake_model = fake_paddleocr_vl
        img_path = tmp_path / "input.png"
        _make_pil_image().save(str(img_path))
        from pipeline.vlm_runner_so import VLMRunnerSo
        runner = VLMRunnerSo(model_dir="/fake/model")
        text, _ = runner.run(str(img_path), "table")
        assert text == "hello world"
        call_args = fake_model.run.call_args
        # Path passed through unchanged
        passed_path = call_args.args[0] if call_args.args else call_args.kwargs["image_path"]
        assert os.path.abspath(passed_path) == os.path.abspath(str(img_path))
        # Input file untouched
        assert img_path.exists()

    def test_run_with_ndarray_writes_tmp(self, fake_paddleocr_vl):
        _, fake_model = fake_paddleocr_vl
        from pipeline.vlm_runner_so import VLMRunnerSo
        runner = VLMRunnerSo(model_dir="/fake/model")
        arr = np.zeros((30, 40, 3), dtype=np.uint8)
        text, _ = runner.run(arr, "formula")
        assert text == "hello world"
        tmp_path = fake_model.run.call_args.args[0]
        assert not os.path.exists(tmp_path)

    def test_run_cleans_up_tmp_on_exception(self, fake_paddleocr_vl):
        """If model.run raises, tmp file must still be deleted."""
        _, fake_model = fake_paddleocr_vl
        fake_model.run.side_effect = RuntimeError("NPU error")
        from pipeline.vlm_runner_so import VLMRunnerSo
        runner = VLMRunnerSo(model_dir="/fake/model")
        with pytest.raises(RuntimeError, match="NPU error"):
            runner.run(_make_pil_image(), "ocr")
        # The tmp path was the first positional arg to model.run
        tmp_path = fake_model.run.call_args.args[0]
        assert not os.path.exists(tmp_path), "tmp file must be cleaned up on exception"

    def test_run_invalid_prompt_label_raises(self, fake_paddleocr_vl):
        from pipeline.vlm_runner_so import VLMRunnerSo
        runner = VLMRunnerSo(model_dir="/fake/model")
        with pytest.raises(ValueError, match="prompt_label"):
            runner.run(_make_pil_image(), "unknown")

    def test_close_releases_model(self, fake_paddleocr_vl):
        _, fake_model = fake_paddleocr_vl
        from pipeline.vlm_runner_so import VLMRunnerSo
        runner = VLMRunnerSo(model_dir="/fake/model")
        runner.close()
        fake_model.release.assert_called_once()
