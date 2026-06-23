#!/usr/bin/env python3
"""
PaddleOCR-VL Server Demo (Client-Server Model)

Server runs on the RK3588 development board, exposing an HTTP API for OCR inference.
Clients send images via HTTP POST and receive OCR text + metrics in JSON.

Usage:
  Start server:
    python3 paddleocr_vl_server.py --port 8080

  Test with curl:
    curl -X POST http://192.168.46.102:8080/ocr \
      -F "image=@/path/to/image.png" \
      -F "prompt=OCR:" \
      -F "prompt_type=ocr"

  Or use the client script:
    python3 paddleocr_vl_client.py --server http://192.168.46.102:8080 --image test.png
"""

import sys
import os
import argparse
import tempfile
import traceback

lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build/build_local_rk3588_Release")
sys.path.insert(0, lib_dir)

from flask import Flask, request, jsonify
import paddleocr_vl

app = Flask(__name__)

MODEL_DIR = r"/root/demo_paddleocr_vl/model"
model = None


def init_model():
    global model
    model = paddleocr_vl.PaddleOCRVL()
    model.init(MODEL_DIR, vision_core_mask=0xFF, mlpar_core_mask=0xFF, llm_core_mask=0xFF)
    print("[server] Model initialized successfully")


@app.route("/ocr", methods=["POST"])
def ocr():
    if model is None:
        return jsonify({"error": "Model not initialized"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    prompt = request.form.get("prompt", "OCR:")
    prompt_type = request.form.get("prompt_type", "ocr")

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        result = model.run(tmp_path, prompt=prompt, prompt_type=prompt_type)

        return jsonify({
            "text": result["text"],
            "metrics": result["metrics"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/ocr_base64", methods=["POST"])
def ocr_base64():
    """Accept base64-encoded image in JSON body."""
    import base64

    if model is None:
        return jsonify({"error": "Model not initialized"}), 503

    data = request.get_json(force=True)
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "No 'image' field (base64) in JSON body"}), 400

    prompt = data.get("prompt", "OCR:")
    prompt_type = data.get("prompt_type", "ocr")

    tmp_path = None
    try:
        img_bytes = base64.b64decode(image_b64)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        result = model.run(tmp_path, prompt=prompt, prompt_type=prompt_type)

        return jsonify({
            "text": result["text"],
            "metrics": result["metrics"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model is not None else "not_ready",
        "model_dir": MODEL_DIR,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PaddleOCR-VL Inference Server")
    parser.add_argument("--port", type=int, default=8080, help="Listen port (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Listen host (default: 0.0.0.0)")
    args = parser.parse_args()

    print(f"[server] Initializing model from {MODEL_DIR} ...")
    init_model()
    print(f"[server] Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)
