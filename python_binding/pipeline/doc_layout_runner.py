"""PP-DocLayoutV2 ONNX inference wrapper."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from pipeline import DOCLAYOUT_LABELS

# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_TARGET_SIZE = (800, 800)


class DocLayoutRunner:
    """Wraps PP-DocLayoutV2 ONNX model for layout detection."""

    def __init__(self, onnx_path: str, threshold: float = 0.5):
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        self.onnx_path = onnx_path
        self.threshold = threshold
        self.session = ort.InferenceSession(onnx_path)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess_image(
        self, image: np.ndarray, target_input_size: tuple = _TARGET_SIZE
    ) -> tuple[np.ndarray, float, float]:
        """Preprocess image for DocLayoutV2: resize, normalize, transpose.

        Returns (input_blob, scale_h, scale_w).
        """
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = target_input_size
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = (blob - _MEAN) / _STD
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, scale_h, scale_w

    def filter_boxes(
        self, output: np.ndarray, threshold: float
    ) -> list[dict]:
        """Filter ONNX output by score threshold.

        ONNX output shape: (N, 8), each row [cls_id, score, xmin, ymin, xmax, ymax, ...].
        """
        kept = output[output[:, 1] > threshold]
        boxes = []
        for row in kept:
            boxes.append({
                "label_index": int(row[0]),
                "score": float(row[1]),
                "bbox": [float(row[2]), float(row[3]),
                         float(row[4]), float(row[5])],
            })
        return boxes

    def rescale_boxes(
        self,
        boxes: list[dict],
        scale_h: float,
        scale_w: float,
        orig_h: int,
        orig_w: int,
    ) -> list[dict]:
        """Rescale boxes from 800x800 space back to original image space."""
        for box in boxes:
            xmin, ymin, xmax, ymax = box["bbox"]
            box["bbox"] = [
                max(0.0, xmin / scale_w),
                max(0.0, ymin / scale_h),
                min(float(orig_w), xmax / scale_w),
                min(float(orig_h), ymax / scale_h),
            ]
            label_idx = box["label_index"]
            if 0 <= label_idx < len(DOCLAYOUT_LABELS):
                box["label"] = DOCLAYOUT_LABELS[label_idx]
            else:
                box["label"] = f"unknown_{label_idx}"
        return boxes

    def detect(self, image_input: str | np.ndarray) -> list[dict]:
        """Run layout detection on an image path or ndarray.

        Returns list of {label, label_index, score, bbox} with bbox in original
        image coordinates.
        """
        if isinstance(image_input, str):
            if not Path(image_input).exists():
                raise FileNotFoundError(f"Image not found: {image_input}")
            image = cv2.imread(image_input)
            if image is None:
                raise RuntimeError(f"Failed to read image: {image_input}")
        else:
            image = image_input

        orig_h, orig_w = image.shape[:2]
        blob, scale_h, scale_w = self.preprocess_image(image)
        preprocess_shape = [np.array(list(_TARGET_SIZE), dtype=np.float32)]
        feed = {
            self.input_names[0]: preprocess_shape,
            self.input_names[1]: blob,
            self.input_names[2]: [[scale_h, scale_w]],
        }
        output = self.session.run(self.output_names, feed)[0]
        boxes = self.filter_boxes(output, self.threshold)
        # The ONNX model uses the `scale_factor` input to internally map
        # coordinates back to original-image space, so output boxes are already
        # in original coordinates. Pass scale=1.0 to `rescale_boxes` so it only
        # attaches labels and clips to image bounds (no double-division).
        boxes = self.rescale_boxes(boxes, 1.0, 1.0, orig_h, orig_w)
        return boxes
