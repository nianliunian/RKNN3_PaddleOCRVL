"""Debug script: crop blocks from page0, save orig+pad versions, run .so on each.

Single NPU process: load .so once, run all blocks, release. Saves artifacts
to ./debug_blocks/ (persistent across reboots).
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "build" / "build_local_rk3588_Release"))

import paddleocr_vl
from PIL import Image

from pipeline.doc_layout_runner import DocLayoutRunner
from pipeline.pipeline import _pad_to_min, MIN_BLOCK_SIZE

BOARD_ROOT = Path(__file__).parent
MODEL_DIR = "/root/demo_paddleocr_vl/model"
ONNX_PATH = "/root/demo_paddleocr_vl/PP_onnx/PP-DocLayoutV2.onnx"
IMG_PATH = "/root/demo_paddleocr_vl/pdf_process/output_pdf/page0_scale2.0_pymupdf.png"
OUT_DIR = BOARD_ROOT / "debug_blocks"


def main() -> int:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"=== debug_blocks output to {OUT_DIR} ===", flush=True)

    # Step 1: DocLayout detect
    print(f"[1/3] DocLayout detect: {IMG_PATH}", flush=True)
    layout = DocLayoutRunner(onnx_path=ONNX_PATH, threshold=0.5)
    boxes = layout.detect(IMG_PATH)
    print(f"  -> {len(boxes)} boxes", flush=True)

    # Step 2: Save orig + pad crops for ALL blocks
    N_SAVE = len(boxes)
    print(f"[2/3] Saving orig+pad crops for all {N_SAVE} blocks", flush=True)
    source = Image.open(IMG_PATH).convert("RGB")
    w, h = source.size

    samples = []
    for idx, box in enumerate(boxes[:N_SAVE]):
        label = box["label"]
        xmin, ymin, xmax, ymax = box["bbox"]
        xmin_i = max(0, int(xmin)); ymin_i = max(0, int(ymin))
        xmax_i = min(w, int(xmax)); ymax_i = min(h, int(ymax))
        orig = source.crop((xmin_i, ymin_i, xmax_i, ymax_i))
        pad = _pad_to_min(orig, MIN_BLOCK_SIZE)
        orig_path = OUT_DIR / f"block{idx:02d}_{label}_orig.png"
        pad_path = OUT_DIR / f"block{idx:02d}_{label}_pad.png"
        orig.save(str(orig_path))
        pad.save(str(pad_path))
        samples.append({
            "idx": idx, "label": label,
            "bbox": [xmin_i, ymin_i, xmax_i, ymax_i],
            "orig_size": orig.size, "pad_size": pad.size,
            "orig_path": str(orig_path), "pad_path": str(pad_path),
        })
        print(f"  [{idx}] {label} bbox={[xmin_i,ymin_i,xmax_i,ymax_i]} orig={orig.size} pad={pad.size}", flush=True)

    # Step 3: Load .so, run on orig + pad + full image
    print(f"[3/3] Loading .so and running VLM on samples", flush=True)
    model = paddleocr_vl.PaddleOCRVL()
    model.init(MODEL_DIR, vision_core_mask=0xff, mlpar_core_mask=0xff, llm_core_mask=0xff)

    results = []
    for s in samples:
        for variant in ("orig", "pad"):
            path = s[f"{variant}_path"]
            r = model.run(path, prompt="OCR:", prompt_type="ocr")
            text = r.get("text", "")
            m = r.get("metrics", {})
            tps = m.get("decode_tps", 0)
            print(f"  [{s['idx']} {s['label']} {variant}] len={len(text)} decode_tps={tps:.1f} head={text[:120]!r}", flush=True)
            results.append({
                "idx": s["idx"], "label": s["label"], "variant": variant,
                "text_len": len(text), "text_head": text[:200],
                "decode_tokens": m.get("decode_tokens"), "decode_tps": tps,
            })

    # Reference: full image
    print(f"  [FULL page0] running...", flush=True)
    r = model.run(IMG_PATH, prompt="OCR:", prompt_type="ocr")
    text = r.get("text", "")
    m = r.get("metrics", {})
    print(f"  [FULL page0] len={len(text)} decode_tps={m.get('decode_tps',0):.1f} head={text[:120]!r}", flush=True)
    results.append({"idx": "full", "label": "full_page", "variant": "orig",
                    "text_len": len(text), "text_head": text[:200],
                    "decode_tokens": m.get("decode_tokens"), "decode_tps": m.get("decode_tps",0)})

    model.release()

    out_json = OUT_DIR / "results.json"
    out_json.write_text(json.dumps({"samples": samples, "results": results}, indent=2, ensure_ascii=False))
    print(f"\n=== Done. Results: {out_json} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
