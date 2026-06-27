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
import fitz  # PyMuPDF
import base64

from pipeline import DEFAULT_PATHS
from pipeline.pipeline import Pipeline
from pipeline.pages_parser import parse_pages_param, PageSpecError

app = Flask(__name__)

MODEL_DIR = r"/root/demo_paddleocr_vl/model"
LAYOUT_ONNX_PATH = DEFAULT_PATHS["onnx_path"]
LAYOUT_DEFAULT_THRESHOLD = 0.5

model = None        # PaddleOCRVL singleton (used by /ocr, /ocr_base64, /ocr_pdf)
pipeline = None     # Pipeline singleton (used by /ocr_layout_image, /ocr_layout_pdf)

# PDF OCR constants
MAX_PDF_PAGES = 10
PDF_DEFAULT_DPI = 200


def init_model():
    global model
    model = paddleocr_vl.PaddleOCRVL()
    model.init(MODEL_DIR, vision_core_mask=0xFF, mlpar_core_mask=0xFF, llm_core_mask=0xFF)
    print("[server] Model initialized successfully")


def init_pipeline():
    """Initialize the DocLayout+VLM Pipeline singleton.

    Shares the same PaddleOCRVL instance as `model` above — the paddleocr_vl
    .so does NOT support two PaddleOCRVL instances in one process (both
    return empty text), so the pipeline must reuse the `/ocr` route's
    singleton instead of creating its own.
    """
    global pipeline
    if model is None:
        raise RuntimeError(
            "init_model() must be called before init_pipeline() — "
            "pipeline shares the model's PaddleOCRVL instance"
        )
    pipeline = Pipeline(
        onnx_path=LAYOUT_ONNX_PATH,
        model_dir=MODEL_DIR,
        layout_threshold=LAYOUT_DEFAULT_THRESHOLD,
        vision_core_mask=0xFF,
        mlpar_core_mask=0xFF,
        llm_core_mask=0xFF,
        model=model,
    )
    print(f"[server] Pipeline initialized (onnx={LAYOUT_ONNX_PATH}, shared model)")


def _build_markdown(text, page_num=None):
    """Build markdown text: single text as-is, or with ## Page N header."""
    if page_num is not None:
        return f"## Page {page_num}\n\n{text}"
    return text


def _ocr_single(tmp_path, prompt, prompt_type):
    """Run a single image through the model, return text + metrics."""
    result = model.run(tmp_path, prompt=prompt, prompt_type=prompt_type)
    return {
        "text": result["text"],
        "metrics": result["metrics"],
    }


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
            "markdown": _build_markdown(result["text"]),
            "metrics": result["metrics"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/ocr_layout_image", methods=["POST"])
def ocr_layout_image():
    """Layout-aware OCR: DocLayout → crop per block → VLM → assemble.

    Multipart form:
      image: file (required)
      layout_threshold: float (optional, default 0.5)
      visualize: "true"/"false" (optional, default false)
    """
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use form field 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        layout_threshold = float(
            request.form.get("layout_threshold", LAYOUT_DEFAULT_THRESHOLD)
        )
    except ValueError:
        layout_threshold = LAYOUT_DEFAULT_THRESHOLD
    visualize = request.form.get("visualize", "false").lower() == "true"

    # Update threshold at runtime (cheap; just a field on DocLayoutRunner)
    pipeline.layout_threshold = layout_threshold
    pipeline.layout_runner.threshold = layout_threshold

    tmp_path = None
    vis_path = None
    try:
        suffix = os.path.splitext(file.filename)[1] or ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp)
            tmp_path = tmp.name

        if visualize:
            vis_path = tmp_path + "_vis.png"

        result = pipeline.process_image(
            tmp_path, visualize_output=vis_path
        )

        response = {
            "text": result.to_markdown(),
            "markdown": result.to_markdown(),
            "html": result.to_html(),
            "blocks": [b.to_dict() for b in result.parsing_res_list],
            "layout_det_res": result.layout_det_res,
            "metrics": result.metrics,
        }
        if vis_path and os.path.exists(vis_path):
            with open(vis_path, "rb") as f:
                response["visualization_b64"] = base64.b64encode(f.read()).decode("ascii")
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if vis_path and os.path.exists(vis_path):
            os.unlink(vis_path)


@app.route("/ocr_layout_pdf", methods=["POST"])
def ocr_layout_pdf():
    """Layout-aware OCR over a PDF: render selected pages → per-page Pipeline.

    Multipart form:
      file: PDF file (required)
      layout_threshold: float (optional, default 0.5)
      dpi: int (optional, default 200)
      pages: str (optional, e.g. "1-5,7,10-12"; 1-based inclusive)
      visualize: "true"/"false" (optional, default false)
    """
    if pipeline is None:
        return jsonify({"error": "Pipeline not initialized"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No PDF file provided. Use form field 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF (.pdf)"}), 400

    try:
        layout_threshold = float(
            request.form.get("layout_threshold", LAYOUT_DEFAULT_THRESHOLD)
        )
    except ValueError:
        layout_threshold = LAYOUT_DEFAULT_THRESHOLD
    try:
        dpi = int(request.form.get("dpi", PDF_DEFAULT_DPI))
    except ValueError:
        dpi = PDF_DEFAULT_DPI
    pages_str = request.form.get("pages", None) or None
    visualize = request.form.get("visualize", "false").lower() == "true"

    pipeline.layout_threshold = layout_threshold
    pipeline.layout_runner.threshold = layout_threshold

    pdf_tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp)
            pdf_tmp_path = tmp.name

        try:
            doc = fitz.open(pdf_tmp_path)
        except fitz.FileDataError:
            return jsonify({"error": "Invalid or corrupted PDF file"}), 400

        total_in_doc = len(doc)

        try:
            page_indices = parse_pages_param(pages_str, total_in_doc, MAX_PDF_PAGES)
        except PageSpecError as e:
            doc.close()
            return jsonify({"error": f"Invalid 'pages' parameter: {e}"}), 400

        pages_result = []
        combined_markdown_parts = []

        for idx_in_selection, page_idx in enumerate(page_indices):
            page = doc[page_idx]
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)

            page_tmp_path = None
            vis_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    pix.save(tmp)
                    page_tmp_path = tmp.name

                if visualize:
                    vis_path = page_tmp_path + "_vis.png"

                result = pipeline.process_image(
                    page_tmp_path, visualize_output=vis_path
                )

                page_entry = {
                    "page": page_idx + 1,  # 1-based, matches input spec
                    "text": result.to_markdown(),
                    "markdown": result.to_markdown(),
                    "html": result.to_html(),
                    "blocks": [b.to_dict() for b in result.parsing_res_list],
                    "layout_det_res": result.layout_det_res,
                    "metrics": result.metrics,
                }
                if vis_path and os.path.exists(vis_path):
                    with open(vis_path, "rb") as f:
                        page_entry["visualization_b64"] = (
                            base64.b64encode(f.read()).decode("ascii")
                        )
                pages_result.append(page_entry)
                combined_markdown_parts.append(page_entry["markdown"])

            except Exception as e:
                traceback.print_exc()
                pages_result.append({
                    "page": page_idx + 1,
                    "text": "",
                    "markdown": f"[Error on page {page_idx + 1}: {e}]",
                    "html": "",
                    "blocks": [],
                    "layout_det_res": {"boxes": []},
                    "metrics": None,
                    "error": str(e),
                })
                combined_markdown_parts.append(
                    f"[Error on page {page_idx + 1}: {e}]"
                )
            finally:
                if page_tmp_path and os.path.exists(page_tmp_path):
                    os.unlink(page_tmp_path)
                if vis_path and os.path.exists(vis_path):
                    os.unlink(vis_path)

        truncated = len(page_indices) < total_in_doc
        doc.close()

        return jsonify({
            "pages": pages_result,
            "markdown": "\n\n---\n\n".join(combined_markdown_parts),
            "total_pages_requested": len(page_indices),
            "total_pages_in_doc": total_in_doc,
            "truncated": truncated,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if pdf_tmp_path and os.path.exists(pdf_tmp_path):
            os.unlink(pdf_tmp_path)


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
            "markdown": _build_markdown(result["text"]),
            "metrics": result["metrics"],
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/ocr_pdf", methods=["POST"])
def ocr_pdf():
    if model is None:
        return jsonify({"error": "Model not initialized"}), 503

    if "file" not in request.files:
        return jsonify({"error": "No PDF file provided. Use form field 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "File must be a PDF (.pdf)"}), 400

    prompt = request.form.get("prompt", "OCR:")
    prompt_type = request.form.get("prompt_type", "ocr")
    try:
        dpi = int(request.form.get("dpi", PDF_DEFAULT_DPI))
    except ValueError:
        dpi = PDF_DEFAULT_DPI

    pdf_tmp_path = None
    try:
        # Save PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            file.save(tmp)
            pdf_tmp_path = tmp.name

        # Open with PyMuPDF
        doc = fitz.open(pdf_tmp_path)
        total = len(doc)
        if total > MAX_PDF_PAGES:
            total = MAX_PDF_PAGES

        pages_result = []
        combined_markdown_parts = []
        first_metrics = None

        for i in range(total):
            page = doc[i]
            pix = page.get_pixmap(dpi=dpi)
            png_bytes = pix.tobytes("png")

            page_tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(png_bytes)
                    page_tmp_path = tmp.name

                ocr_result = model.run(page_tmp_path, prompt=prompt, prompt_type=prompt_type)
                page_text = ocr_result["text"]

                page_md = _build_markdown(page_text, page_num=i + 1)
                page_entry = {
                    "page": i + 1,
                    "text": page_text,
                    "markdown": page_md,
                }
                pages_result.append(page_entry)
                combined_markdown_parts.append(page_md)

                if i == 0:
                    first_metrics = ocr_result["metrics"]

            except Exception as e:
                pages_result.append({
                    "page": i + 1,
                    "text": "",
                    "markdown": _build_markdown(f"[Error] {str(e)}", page_num=i + 1),
                    "error": str(e),
                })
                combined_markdown_parts.append(_build_markdown(f"[Error] {str(e)}", page_num=i + 1))
            finally:
                if page_tmp_path and os.path.exists(page_tmp_path):
                    os.unlink(page_tmp_path)

        truncated = len(doc) > MAX_PDF_PAGES
        doc.close()

        response = {
            "pages": pages_result,
            "markdown": "\n\n---\n\n".join(combined_markdown_parts),
            "total_pages": total,
        }
        if first_metrics is not None:
            response["metrics"] = first_metrics
        if truncated:
            response["truncated"] = True

        return jsonify(response)

    except fitz.FileDataError:
        return jsonify({"error": "Invalid or corrupted PDF file"}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if pdf_tmp_path and os.path.exists(pdf_tmp_path):
            os.unlink(pdf_tmp_path)


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
    print(f"[server] Initializing pipeline (onnx={LAYOUT_ONNX_PATH}) ...")
    init_pipeline()
    print(f"[server] Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False)
