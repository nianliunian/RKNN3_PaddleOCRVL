#!/usr/bin/env python3
"""
PaddleOCR-VL Client Demo

Sends an image to the PaddleOCR-VL server and prints the OCR result.

Usage:
  python3 paddleocr_vl_client.py --image test.png
  python3 paddleocr_vl_client.py --image test.png --server http://192.168.46.102:8080 --prompt_type table
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


def main():
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Client")
    parser.add_argument("--server", type=str, default="http://192.168.46.102:8080",
                        help="Server URL (default: http://192.168.46.102:8080)")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--prompt", type=str, default="OCR:", help="Prompt text (default: OCR:)")
    parser.add_argument("--prompt_type", type=str, default="ocr",
                        choices=["ocr", "table", "chart", "formula", "image"],
                        help="Prompt type (default: ocr)")
    args = parser.parse_args()

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

    m = result["metrics"]
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


if __name__ == "__main__":
    main()
