"""Pipeline orchestration: layout detection -> crop -> VLM -> assemble."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pipeline.doc_layout_runner import DocLayoutRunner
from pipeline.label_mapping import label_to_prompt
from pipeline.metrics_aggregator import MetricsAggregator
from pipeline.result import PaddleOCRVLBlock, PipelineResult
from pipeline.vlm_runner_so import VLMRunnerSo

logger = logging.getLogger(__name__)

MIN_BLOCK_SIZE = 200


def _pad_to_min(img: Image.Image, min_size: int = MIN_BLOCK_SIZE) -> Image.Image:
    """Pad an image with white background to at least min_size on each side.

    The .so internal preprocessing resizes any input to a fixed 504x504 grid
    (or similar). Small crops (e.g. 87x25 text lines) get upscaled ~10x with
    extreme aspect-ratio distortion, causing the VLM to emit blank output.
    Padding small sides up to min_size preserves aspect ratio closer to the
    model's training distribution.
    """
    w, h = img.size
    if w >= min_size and h >= min_size:
        return img
    new_w = max(w, min_size)
    new_h = max(h, min_size)
    canvas = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    canvas.paste(img, ((new_w - w) // 2, (new_h - h) // 2))
    return canvas


class Pipeline:
    """Orchestrates DocLayout + VLM (via paddleocr_vl .so) recognition.

    Use as a context manager to ensure resources are released:
        with Pipeline(...) as pipeline:
            result = pipeline.process_image("page.png")
    """

    def __init__(
        self,
        onnx_path: str,
        model_dir: str | None = None,
        layout_threshold: float = 0.5,
        vision_core_mask: int = 0xff,
        mlpar_core_mask: int = 0xff,
        llm_core_mask: int = 0xff,
        model=None,
    ):
        self.layout_threshold = layout_threshold
        self.layout_runner = DocLayoutRunner(
            onnx_path=onnx_path, threshold=layout_threshold
        )
        self.vlm_runner = VLMRunnerSo(
            model_dir=model_dir,
            vision_core_mask=vision_core_mask,
            mlpar_core_mask=mlpar_core_mask,
            llm_core_mask=llm_core_mask,
            model=model,
        )

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return None

    def close(self) -> None:
        self.vlm_runner.close()

    def _crop_block(self, source_img: Image.Image, block: dict) -> Image.Image:
        xmin, ymin, xmax, ymax = block["bbox"]
        w, h = source_img.size
        xmin_i = max(0, int(xmin))
        ymin_i = max(0, int(ymin))
        xmax_i = min(w, int(xmax))
        ymax_i = min(h, int(ymax))
        crop = source_img.crop((xmin_i, ymin_i, xmax_i, ymax_i))
        return _pad_to_min(crop, MIN_BLOCK_SIZE)

    def _postprocess_content(self, label: str, raw_content: str) -> str:
        if not raw_content:
            return ""
        if label == "table":
            return raw_content
        if "formula" in label and label != "formula_number":
            stripped = raw_content.strip()
            if stripped.startswith("$$") and stripped.endswith("$$"):
                return stripped
            if stripped.startswith("$") and stripped.endswith("$"):
                return stripped
            if label == "display_formula":
                return f"$${stripped}$$"
            return f"${stripped}$"
        return raw_content

    def _visualize(
        self,
        image_path: str,
        boxes: list[dict],
        output_path: str | None = None,
    ) -> np.ndarray | None:
        if output_path is None:
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        for box in boxes:
            xmin, ymin, xmax, ymax = [int(v) for v in box["bbox"]]
            label = box.get("label", "?")
            score = box.get("score", 0.0)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            text = f"{label} {score:.2f}"
            cv2.putText(
                img, text, (xmin, max(0, ymin - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )
        cv2.imwrite(output_path, img)
        return img

    def process_image(
        self,
        image_path: str,
        visualize_output: str | None = None,
    ) -> PipelineResult:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        t_start = time.time()
        t_layout_start = time.time()
        boxes = self.layout_runner.detect(image_path)
        layout_ms = (time.time() - t_layout_start) * 1000.0
        logger.info(
            "Layout detection: %d boxes (%.1f ms)",
            len(boxes), layout_ms,
        )

        source_img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = source_img.size

        parsing_res_list: list[PaddleOCRVLBlock] = []
        layout_det_boxes = []
        accumulator = MetricsAggregator()
        blocks_skipped = 0

        for idx, box in enumerate(boxes):
            label = box["label"]
            score = box["score"]
            bbox = list(box["bbox"])
            layout_det_boxes.append({
                "label": label,
                "coordinate": bbox,
                "score": score,
            })

            prompt_label = label_to_prompt(label)
            if prompt_label is None:
                logger.debug("Skip block %d (%s): image-type", idx, label)
                parsing_res_list.append(PaddleOCRVLBlock(
                    label=label, bbox=bbox, content=""
                ))
                blocks_skipped += 1
                continue

            t_block = time.time()
            try:
                block_img = self._crop_block(source_img, box)
                raw_text, block_metrics = self.vlm_runner.run(block_img, prompt_label)
                content = self._postprocess_content(label, raw_text)
                accumulator.add_block(block_metrics)
                logger.debug(
                    "Block %d (%s/%s): %d chars (%.1f ms)",
                    idx, label, prompt_label,
                    len(content), (time.time() - t_block) * 1000.0,
                )
            except Exception as e:
                logger.warning(
                    "Block %d (%s) failed: %s; content set to empty",
                    idx, label, e,
                )
                content = ""

            parsing_res_list.append(PaddleOCRVLBlock(
                label=label, bbox=bbox, content=content
            ))

        metrics = accumulator.build(
            layout_ms=layout_ms,
            layout_boxes=len(boxes),
            blocks_skipped=blocks_skipped,
        )
        logger.info(
            "Pipeline total: %d blocks, %.1fs",
            len(parsing_res_list), time.time() - t_start,
        )

        result = PipelineResult(
            input_path=image_path,
            page_index=0,
            page_count=1,
            width=orig_w,
            height=orig_h,
            layout_det_res={"boxes": layout_det_boxes},
            parsing_res_list=parsing_res_list,
            metrics=metrics,
        )
        if visualize_output:
            vis_img = self._visualize(image_path, boxes, visualize_output)
            if vis_img is not None:
                result.img = vis_img

        source_img.close()
        return result

    def process_pdf(
        self,
        pdf_path: str,
        output_dir: str,
        dpi: int = 200,
    ) -> list[PipelineResult]:
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        import fitz

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results: list[PipelineResult] = []
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        logger.info("PDF: %d pages, dpi=%d", total_pages, dpi)
        try:
            for page_idx in range(total_pages):
                page = doc[page_idx]
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                page_png = out_dir / f"page_{page_idx:02d}.png"
                pix.save(str(page_png))

                t_page = time.time()
                logger.info("=== Page %d/%d ===", page_idx + 1, total_pages)
                vis_path = str(
                    out_dir / f"result_page{page_idx:02d}_vis.png"
                )
                result = self.process_image(
                    str(page_png), visualize_output=vis_path
                )
                result.page_index = page_idx
                result.page_count = total_pages

                save_path = out_dir / f"result_page{page_idx:02d}"
                result.save_all(save_path=str(save_path))
                logger.info(
                    "Page %d/%d done (%.1fs)",
                    page_idx + 1, total_pages, time.time() - t_page,
                )
                results.append(result)
        finally:
            doc.close()
        return results
