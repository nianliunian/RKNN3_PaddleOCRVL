#!/usr/bin/env python3
"""
PaddleOCR-VL Client Demo

Sends an image or PDF to the PaddleOCR-VL server and prints the OCR result.

Usage:
  python3 paddleocr_vl_client.py --image test.png
  python3 paddleocr_vl_client.py --image test.png --server http://192.168.46.102:8080 --prompt_type table
  python3 paddleocr_vl_client.py --pdf document.pdf
  python3 paddleocr_vl_client.py --pdf document.pdf --pdf-dpi 300 --server http://192.168.46.102:8080
"""

import argparse
import sys
import os
import json

try:
    import requests
except ImportError:
    print("requests library not found. Install with: pip3 install requests")
    sys.exit(1)


def _print_metrics(m):
    if not m:
        return
    print("-" * 90)
    hdr = "  {:<12} {:>15} {:>8} {:>18} {:>18}".format(
        "Stage", "Total Time (ms)", "Tokens", "Time/Token (ms)", "Tokens/sec")
    print(hdr)
    print("-" * 90)

    prefill_ms = m["prefill_latency"]
    prefill_tokens = int(m["prefill_tokens"])
    prefill_tpt = m["prefill_tpt"]
    prefill_tps = m["prefill_tps"]
    print("  {:<12} {:>15.2f} {:>8} {:>18.2f} {:>18.2f}".format(
        "Prefill", prefill_ms, prefill_tokens, prefill_tpt, prefill_tps))

    decode_ms = m["decode_latency"]
    decode_tokens = int(m["decode_tokens"])
    decode_tpt = m["decode_tpt"]
    decode_tps = m["decode_tps"]
    print("  {:<12} {:>15.2f} {:>8} {:>18.2f} {:>18.2f}".format(
        "Generate", decode_ms, decode_tokens, decode_tpt, decode_tps))

    print("-" * 90)
    print("  TTFT: {:.2f} ms".format(m["ttft"]))
    print("  Vision latency: {:.2f} ms".format(m["vision_latency"]))
    print("  Total tokens: {}".format(prefill_tokens + decode_tokens))
    print("  Overall TPS: {:.2f} tokens/s".format(m["tps"]))
    print("=" * 90)


def _handle_image(args):
    url = args.server.rstrip("/") + "/ocr"

    print(f"[client] Sending {args.image} to {url} ...")

    with open(args.image, "rb") as f:
        files = {"image": (os.path.basename(args.image), f)}
        data = {"prompt": args.prompt, "prompt_type": args.prompt_type}
        resp = requests.post(url, files=files, data=data, timeout=120)

    if resp.status_code != 200:
        print(f"[client] Error: HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    result = resp.json()

    print()
    print("=" * 90)
    print("OCR Result:")
    print(result["text"])
    print()

    print("-" * 90)
    print("Markdown:")
    print(result.get("markdown", result["text"]))
    print()

    _print_metrics(result.get("metrics"))


def _handle_pdf(args):
    url = args.server.rstrip("/") + "/ocr_pdf"

    print(f"[client] Sending PDF {args.pdf} to {url} ...")

    with open(args.pdf, "rb") as f:
        files = {"file": (os.path.basename(args.pdf), f)}
        data = {"prompt": args.prompt, "prompt_type": args.prompt_type, "dpi": str(args.pdf_dpi)}
        resp = requests.post(url, files=files, data=data, timeout=600)

    if resp.status_code != 200:
        print(f"[client] Error: HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    result = resp.json()

    print()
    print("=" * 90)
    print(f"PDF OCR Result ({result['total_pages']} pages)")
    if result.get("truncated"):
        print("(Results limited to first 10 pages)")
    print("=" * 90)

    for page in result["pages"]:
        print()
        print(f"--- Page {page['page']} ---")
        print(page["text"])
        if "error" in page:
            print(f"[ERROR] {page['error']}")

    print()
    print("-" * 90)
    print("Full Markdown:")
    print(result["markdown"])
    print()

    if "metrics" in result:
        _print_metrics(result["metrics"])


def _print_layout_metrics(m):
    if not m:
        return
    print("-" * 60)
    print("Page-level metrics:")
    print(f"  layout_ms              : {m.get('layout_ms', 0):.2f}")
    print(f"  layout_boxes           : {m.get('layout_boxes', 0)}")
    print(f"  vlm_blocks_run         : {m.get('vlm_blocks_run', 0)}")
    print(f"  vlm_blocks_skipped     : {m.get('vlm_blocks_skipped', 0)}")
    print("-" * 60)
    print("VLM breakdown (totals across all blocks):")
    print(f"  vision_latency_total_ms: {m.get('vision_latency_total_ms', 0):.2f}")
    print(f"  prefill_latency_total  : {m.get('prefill_latency_total_ms', 0):.2f} ms / {int(m.get('prefill_tokens_total', 0))} tokens")
    print(f"  decode_latency_total   : {m.get('decode_latency_total_ms', 0):.2f} ms / {int(m.get('decode_tokens_total', 0))} tokens")
    print(f"  ttft_avg               : {m.get('ttft_avg_ms', 0):.2f} ms (avg per block)")
    if m.get('decode_tokens_total'):
        decode_tps = m['decode_tokens_total'] / (m.get('decode_latency_total_ms', 1) / 1000.0)
        print(f"  decode_tps (derived)   : {decode_tps:.2f} tok/s")
    if m.get('prefill_tokens_total'):
        prefill_tps = m['prefill_tokens_total'] / (m.get('prefill_latency_total_ms', 1) / 1000.0)
        print(f"  prefill_tps (derived)  : {prefill_tps:.2f} tok/s")
    print("-" * 60)
    print(f"  vlm_total_ms           : {m.get('vlm_total_ms', 0):.2f}")
    print(f"  page_total_ms          : {m.get('page_total_ms', 0):.2f}")
    print(f"  vlm_tps_avg            : {m.get('vlm_tps_avg', 0):.2f}")
    print("-" * 60)


def _save_layout_image_outputs(args, result):
    """Save md/json/html/vis to args.output_dir. Returns list of saved paths."""
    if not args.output_dir:
        return []
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.layout_image))[0]
    saved = []
    with open(os.path.join(args.output_dir, base + ".md"), "w", encoding="utf-8") as f:
        f.write(result["markdown"])
    saved.append(base + ".md")
    with open(os.path.join(args.output_dir, base + ".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    saved.append(base + ".json")
    if result.get("html"):
        with open(os.path.join(args.output_dir, base + ".html"), "w", encoding="utf-8") as f:
            f.write(result["html"])
        saved.append(base + ".html")
    if result.get("visualization_b64"):
        import base64
        vis_path = os.path.join(args.output_dir, base + "_vis.png")
        with open(vis_path, "wb") as f:
            f.write(base64.b64decode(result["visualization_b64"]))
        saved.append(base + "_vis.png")
    return saved


def _handle_layout_image(args):
    url = args.server.rstrip("/") + "/ocr_layout_image"

    print(f"[client] Sending {args.layout_image} to {url} ...")

    with open(args.layout_image, "rb") as f:
        files = {"image": (os.path.basename(args.layout_image), f)}
        data = {
            "layout_threshold": str(args.layout_threshold),
            "visualize": "true" if args.visualize else "false",
        }
        resp = requests.post(url, files=files, data=data, timeout=600)

    if resp.status_code != 200:
        print(f"[client] Error: HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    result = resp.json()

    print()
    print("=" * 90)
    print("Layout OCR Result:")
    print(result["text"])
    print()

    _print_layout_metrics(result.get("metrics"))

    saved = _save_layout_image_outputs(args, result)
    for name in saved:
        print(f"[client] Saved: {os.path.join(args.output_dir, name)}")


def _save_layout_pdf_outputs(args, result):
    """Save md/json/html/vis to args.output_dir. Returns list of saved paths."""
    if not args.output_dir:
        return []
    import base64
    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.layout_pdf))[0]
    saved = []
    with open(os.path.join(args.output_dir, base + ".md"), "w", encoding="utf-8") as f:
        f.write(result["markdown"])
    saved.append(base + ".md")
    with open(os.path.join(args.output_dir, base + ".json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    saved.append(base + ".json")
    # Combined html: concatenate per-page html (separated by hr)
    html_parts = [p.get("html", "") for p in result.get("pages", []) if p.get("html")]
    if html_parts:
        with open(os.path.join(args.output_dir, base + ".html"), "w", encoding="utf-8") as f:
            f.write("\n<hr>\n".join(html_parts))
        saved.append(base + ".html")
    # Per-page visualization PNGs
    for page in result.get("pages", []):
        vis_b64 = page.get("visualization_b64")
        if not vis_b64:
            continue
        vis_path = os.path.join(
            args.output_dir, f"{base}_page{page['page']}_vis.png"
        )
        with open(vis_path, "wb") as f:
            f.write(base64.b64decode(vis_b64))
        saved.append(os.path.basename(vis_path))
    return saved


def _handle_layout_pdf(args):
    url = args.server.rstrip("/") + "/ocr_layout_pdf"

    print(f"[client] Sending PDF {args.layout_pdf} to {url} ...")

    with open(args.layout_pdf, "rb") as f:
        files = {"file": (os.path.basename(args.layout_pdf), f)}
        data = {
            "layout_threshold": str(args.layout_threshold),
            "dpi": str(args.pdf_dpi),
            "visualize": "true" if args.visualize else "false",
        }
        if args.pages:
            data["pages"] = args.pages
        resp = requests.post(url, files=files, data=data, timeout=1800)

    if resp.status_code != 200:
        print(f"[client] Error: HTTP {resp.status_code}")
        print(resp.text)
        sys.exit(1)

    result = resp.json()

    print()
    print("=" * 90)
    print(f"Layout OCR PDF Result ({result['total_pages_requested']}/{result['total_pages_in_doc']} pages)")
    if result.get("truncated"):
        print(f"(Truncated: requested {result['total_pages_requested']} of {result['total_pages_in_doc']} pages)")
    print("=" * 90)

    for page in result["pages"]:
        print()
        print(f"--- Page {page['page']} ---")
        print(page["text"])
        if "error" in page:
            print(f"[ERROR] {page['error']}")
        _print_layout_metrics(page.get("metrics"))

    print()
    print("-" * 90)
    print("Full Markdown:")
    print(result["markdown"])

    saved = _save_layout_pdf_outputs(args, result)
    for name in saved:
        print(f"[client] Saved: {os.path.join(args.output_dir, name)}")


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Client")
    parser.add_argument("--server", type=str, default="http://192.168.46.104:8080",
                        help="Server URL (default: http://192.168.46.104:8080)")
    # Original /ocr routes
    parser.add_argument("--image", type=str, default=None, help="Path to image file (uses /ocr)")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF file (uses /ocr_pdf)")
    parser.add_argument("--pdf-dpi", type=int, default=200, help="PDF rendering DPI (default: 200)")
    parser.add_argument("--prompt", type=str, default="OCR:", help="Prompt text (default: OCR:)")
    parser.add_argument("--prompt_type", type=str, default="ocr",
                        choices=["ocr", "table", "chart", "formula", "image"],
                        help="Prompt type (default: ocr)")
    # New /ocr_layout_image and /ocr_layout_pdf routes
    parser.add_argument("--layout_image", type=str, default=None,
                        help="Path to image file (uses /ocr_layout_image: DocLayout + VLM)")
    parser.add_argument("--layout_pdf", type=str, default=None,
                        help="Path to PDF file (uses /ocr_layout_pdf: per-page DocLayout + VLM)")
    parser.add_argument("--layout_threshold", type=float, default=0.5,
                        help="DocLayout score threshold (default: 0.5)")
    parser.add_argument("--pages", type=str, default=None,
                        help='PDF page selection, e.g. "1-5,7,10-12" (1-based inclusive)')
    parser.add_argument("--visualize", action="store_true",
                        help="Request visualization PNG (base64-encoded in response)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save .md/.json output (optional)")
    args = parser.parse_args()

    modes = [args.image, args.pdf, args.layout_image, args.layout_pdf]
    selected = [m for m in modes if m]
    if len(selected) == 0:
        print("[client] Error: Provide one of --image, --pdf, --layout_image, --layout_pdf")
        sys.exit(1)
    if len(selected) > 1:
        print("[client] Error: Only one of --image, --pdf, --layout_image, --layout_pdf may be used")
        sys.exit(1)

    if args.image:
        _handle_image(args)
    elif args.pdf:
        _handle_pdf(args)
    elif args.layout_image:
        _handle_layout_image(args)
    else:
        _handle_layout_pdf(args)


if __name__ == "__main__":
    main()