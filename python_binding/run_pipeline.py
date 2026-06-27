#!/usr/bin/env python3
"""CLI entry: DocLayout + PaddleOCR-VL .so pipeline (no HTTP).

Runs on the board (requires paddleocr_vl .so in sys.path).
Mirrors the HTTP /ocr_layout_image and /ocr_layout_pdf endpoints but
prints results to stdout and writes .md/.json/.html to output_dir.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from pipeline import DEFAULT_PATHS
from pipeline.pipeline import Pipeline


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_PDF_EXTS = {".pdf"}


def _validate_path(path: str, allowed_exts: set[str], kind_desc: str) -> str:
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"{path}: not found")
    if not p.is_file():
        raise argparse.ArgumentTypeError(f"{path}: not a regular file")
    if p.suffix.lower() not in allowed_exts:
        raise argparse.ArgumentTypeError(
            f"{path}: not a {kind_desc} (expected one of {sorted(allowed_exts)})"
        )
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DocLayout + PaddleOCR-VL .so pipeline (no HTTP)"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", type=lambda v: _validate_path(v, _IMAGE_EXTS, "image file"),
        help="input image path",
    )
    group.add_argument(
        "--pdf", type=lambda v: _validate_path(v, _PDF_EXTS, "PDF file"),
        help="input PDF path",
    )

    p.add_argument("--output_dir", type=str, default="./output",
                   help="output directory (default: ./output)")
    p.add_argument("--layout_threshold", type=float, default=0.5,
                   help="DocLayoutV2 score threshold (default: 0.5)")
    p.add_argument("--onnx_path", type=str,
                   default=str(DEFAULT_PATHS["onnx_path"]),
                   help="PP-DocLayoutV2 ONNX model path")
    p.add_argument("--model_dir", type=str,
                   default=str(DEFAULT_PATHS["model_dir"]),
                   help="PaddleOCR-VL model directory")
    p.add_argument("--dpi", type=int, default=200,
                   help="PDF rendering DPI (default: 200)")
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_kwargs = dict(
        onnx_path=args.onnx_path,
        model_dir=args.model_dir,
        layout_threshold=args.layout_threshold,
    )

    if args.image:
        with Pipeline(**pipeline_kwargs) as pipeline:
            stem = Path(args.image).stem
            vis_path = str(out_dir / f"result_{stem}_vis.png")
            result = pipeline.process_image(
                args.image, visualize_output=vis_path
            )
            save_path = out_dir / f"result_{stem}"
            result.save_all(save_path=str(save_path))
            print(f"Saved: {save_path}.json / .md / .html")
            print(f"Visualization: {vis_path}")
            print(f"Metrics: {result.metrics}")
    else:
        with Pipeline(**pipeline_kwargs) as pipeline:
            t0 = time.time()
            results = pipeline.process_pdf(args.pdf, str(out_dir), dpi=args.dpi)
            print(
                f"Processed {len(results)} pages -> {out_dir} "
                f"(total {time.time() - t0:.1f}s)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
