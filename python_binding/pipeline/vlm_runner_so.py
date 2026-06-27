"""Thin wrapper over paddleocr_vl.PaddleOCRVL .so.

Replaces the old vlm_runner.py (which used torch/transformers/RKNN Python API).
Accepts PIL.Image / ndarray / path; writes a tmp PNG for non-path inputs.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_PROMPTS = {"ocr", "table", "formula", "chart"}

DEFAULT_PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


class VLMRunnerSo:
    """Wraps paddleocr_vl.PaddleOCRVL for repeatable VLM inference.

    The paddleocr_vl .so does NOT support multiple PaddleOCRVL instances in
    the same process — a second init() corrupts internal global state and
    both instances return empty text. So if a PaddleOCRVL instance already
    exists (e.g. the server's `/ocr` route singleton), pass it in via
    `model=` and VLMRunnerSo will reuse it without calling init()/release().
    """

    def __init__(
        self,
        model_dir: str | None = None,
        vision_core_mask: int = 0xff,
        mlpar_core_mask: int = 0xff,
        llm_core_mask: int = 0xff,
        model=None,
    ):
        if model is not None:
            self._model = model
            self._owns_model = False
            logger.info("VLMRunnerSo initialized with external model instance")
            return

        if model_dir is None:
            raise ValueError(
                "model_dir is required when model= is not provided"
            )
        import paddleocr_vl
        self._model = paddleocr_vl.PaddleOCRVL()
        self._model.init(
            model_dir,
            vision_core_mask=vision_core_mask,
            mlpar_core_mask=mlpar_core_mask,
            llm_core_mask=llm_core_mask,
        )
        self._owns_model = True
        logger.info("VLMRunnerSo initialized (model_dir=%s)", model_dir)

    @staticmethod
    def _validate_prompt_label(prompt_label: str) -> None:
        if prompt_label not in SUPPORTED_PROMPTS:
            raise ValueError(
                f"Unsupported prompt_label: {prompt_label!r}. "
                f"Must be one of {sorted(SUPPORTED_PROMPTS)}"
            )

    @staticmethod
    def _to_pil_image(image):
        if isinstance(image, Image.Image):
            return image
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image not found: {image}")
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 ndarray, got shape {image.shape}")
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            "Expected str (path), np.ndarray, or PIL.Image.Image"
        )

    def run(
        self,
        image,
        prompt_label: str,
    ) -> tuple[str, dict]:
        """Run VLM inference on an image with a given prompt label.

        Args:
            image: path / ndarray / PIL.Image
            prompt_label: one of {ocr, table, formula, chart}

        Returns:
            (text, metrics) — text is VLM output, metrics is the per-block
            metrics dict from model.run().
        """
        self._validate_prompt_label(prompt_label)

        # If image is already a path, pass it through (no tmp file).
        if isinstance(image, str):
            result = self._model.run(
                image,
                prompt=DEFAULT_PROMPTS[prompt_label],
                prompt_type=prompt_label,
            )
            return result["text"], result["metrics"]

        # PIL / ndarray → write tmp PNG, clean up in finally.
        pil_image = self._to_pil_image(image)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".png", prefix="vlm_block_"
            ) as tmp:
                pil_image.save(tmp, format="PNG")
                tmp_path = tmp.name
            result = self._model.run(
                tmp_path,
                prompt=DEFAULT_PROMPTS[prompt_label],
                prompt_type=prompt_label,
            )
            return result["text"], result["metrics"]
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError as e:
                    logger.warning("Failed to delete tmp %s: %s", tmp_path, e)

    def close(self) -> None:
        if self._model is not None and self._owns_model:
            try:
                self._model.release()
            except Exception as e:
                logger.warning("Failed to release PaddleOCRVL: %s", e)
        self._model = None
