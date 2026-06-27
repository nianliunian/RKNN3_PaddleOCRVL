---
base_model: paddlepaddle/PP-DocLayoutV2
tags:
- ocr
- onnx
- layout-detection
- paddle
license: apache-2.0
---

This model is an ONNX version of [`paddlepaddle/PP-DocLayoutV2`](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2), created with [Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX).
---

Example Python code to run this model:


```
# Install dependencies:
# pip install numpy opencv-python onnxruntime

import numpy as np
import cv2
import onnxruntime as ort

def preprocess_image_doclayout(image, target_input_size=(800, 800)):
    """
    Preprocessing for DocLayoutV2 with 800x800 input
    """
    orig_h, orig_w = image.shape[:2]
    # Resize, do not preserve aspect ratio
    target_h, target_w = target_input_size
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w

    new_h, new_w = int(orig_h * scale_h), int(orig_w * scale_w)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Convert to RGB and normalize
    padded = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_blob = padded.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_blob = (input_blob - mean) / std

    # Transpose to CHW format and add batch dimension
    input_blob = input_blob.transpose(2, 0, 1)[np.newaxis, ...]

    return input_blob, scale_h, scale_w


def run_doclayout_onnx():
    # Specify onnx model path here
    model = ort.InferenceSession('path/to/PP-DocLayoutV2.onnx')  # Update your path to ONNX model here
    input_names = [i.name for i in model.get_inputs()]
    output_names = [o.name for o in model.get_outputs()]

    image_path = 'path/to/input_image.png'  # Update path to your input image here
    image = cv2.imread(image_path)
    input_blob, scale_h, scale_w = preprocess_image_doclayout(image)
    preprocess_shape = [np.array([800, 800], dtype=np.float32)]
    input_feed = {input_names[0]: preprocess_shape,
                  input_names[1]: input_blob,
                  input_names[2]: [[scale_h, scale_w]]}

    # shape=(300, 8), First 6 values are [label_index, score, xmin, ymin, xmax, ymax]
    output = model.run(output_names, input_feed)[0]

    # Filter out low-confidence boxes
    boxes = output[output[:, 1] > 0.5]
    print('--- DocLayoutV2 ONNX Output: ---')
    print_doclayout_res(boxes[np.argsort(boxes[:, 2])])


def print_doclayout_res(boxes):
    print('cls_id\tscore\txmin\tymin\txmax\tymax')
    if isinstance(boxes, np.ndarray):
        for box in boxes:
            print(f"{box[0]:.0f}\t{box[1]:.3f}\t{box[2]:.2f}\t{box[3]:.2f}\t{box[4]:.2f}\t{box[5]:.2f}")
    else:
        for box in sorted(boxes, key=lambda x: x['coordinate'][0]):
            xmin, ymin, xmax, ymax = box['coordinate']
            print(f"{box['cls_id']:.0f}\t{box['score']:.3f}"
                  f"\t{xmin:.2f}\t{ymin:.2f}\t{xmax:.2f}\t{ymax:.2f}")

if __name__ == '__main__':
    run_doclayout_onnx()
```
