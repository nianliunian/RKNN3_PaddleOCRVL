#!/usr/bin/env python3
"""
PaddleOCR-VL Python Binding Demo
使用 pybind11 封装的 C++ RKNN3 推理代码进行 OCR 识别。
"""
import sys
import os

# 确保能找到 paddleocr_vl 模块
# 部署后通常不需要这行，直接 import paddleocr_vl 即可
# sys.path.insert(0, "/path/to/paddleocr_vl.so/directory")

import paddleocr_vl


def demo_basic():
    """基础用法：单张图片 OCR"""
    print("=" * 60)
    print("Demo 1: 基础 OCR")
    print("=" * 60)

    model = paddleocr_vl.PaddleOCRVL()

    # 默认模型目录在当前脚本同级的 model/
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

    # 初始化（加载三个 RKNN 子模型）
    print(f"[init] Loading models from: {model_dir}")
    model.init(
        model_dir=model_dir,
        vision_core_mask=0x1,
        mlpar_core_mask=0x2,
        llm_core_mask=0x4,
    )
    print("[init] Done!")

    # 推理
    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vision", "test.png")
    print(f"\n[inference] Running on: {image_path}")
    print("-" * 40)

    result = model.run(image_path, prompt="OCR:", prompt_type="ocr")

    print(f"\n[output] Response:")
    print(result["text"])
    print("\n" + "-" * 40)

    # 性能数据
    print("\n[performance]:")
    m = result["metrics"]
    print(f"  vision_latency   : {m['vision_latency']:.2f} ms")
    print(f"  prefill_latency  : {m['prefill_latency']:.2f} ms  ({m['prefill_tokens']} tokens)")
    print(f"  decode_latency   : {m['decode_latency']:.2f} ms  ({m['decode_tokens']} tokens)")
    print(f"  ttft             : {m['ttft']:.2f} ms")
    print(f"  tps              : {m['tps']:.1f} tokens/sec")
    print(f"  tpt              : {m['tpt']:.3f} ms/token")

    # 释放
    model.release()
    print("[release] Done!")


def demo_table_recognition():
    """表格识别"""
    print("\n" + "=" * 60)
    print("Demo 2: 表格识别")
    print("=" * 60)

    model = paddleocr_vl.PaddleOCRVL()
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

    model.init(model_dir=model_dir)

    image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vision", "test.png")
    result = model.run(image_path, prompt_type="table")

    print(f"\n[output] Table recognition result:")
    print(result["text"])

    model.release()


def demo_multiple_images():
    """多次推理（一次 init，多次 run）"""
    print("\n" + "=" * 60)
    print("Demo 3: 多次推理")
    print("=" * 60)

    model = paddleocr_vl.PaddleOCRVL()
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    model.init(model_dir=model_dir)

    test_images = [
        ("data/vision/test.png", "OCR:"),
        ("data/vision/test.png", "Chart Recognition:"),
        ("data/vision/test.png", "Formula Recognition:"),
    ]

    for img_path, prompt in test_images:
        full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), img_path)
        print(f"\n[{os.path.basename(img_path)}]:")
        result = model.run(full_path, prompt=prompt)
        print(f"  Text: {result['text'][:100]}...")
        m = result["metrics"]
        print(f"  TPS: {m['tps']:.1f} tok/s")

    model.release()


if __name__ == "__main__":
    try:
        # 选择运行哪个 demo
        if len(sys.argv) > 1:
            demo_name = sys.argv[1]
            if demo_name == "basic":
                demo_basic()
            elif demo_name == "table":
                demo_table_recognition()
            elif demo_name == "multi":
                demo_multiple_images()
            else:
                print(f"Unknown demo: {demo_name}")
                print("Usage: demo.py [basic|table|multi]")
        else:
            demo_basic()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
