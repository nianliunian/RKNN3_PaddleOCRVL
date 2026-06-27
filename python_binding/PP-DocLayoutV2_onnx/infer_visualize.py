"""Visualize PP-DocLayoutV2 ONNX inference results with bounding boxes and labels."""
import numpy as np
import cv2
import onnxruntime as ort
import json
import os
import sys
print(1)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), 'PP-DocLayoutV2', 'config.json')
print(2)
# Color palette for labels
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (192, 192, 192), (128, 128, 128),
    (255, 165, 0), (255, 105, 180), (100, 149, 237), (255, 69, 0),
    (85, 107, 47), (47, 79, 79), (205, 133, 63), (72, 209, 204),
    (255, 139, 0), (138, 43, 226),
]

def load_labels():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    print(config)
    return config['label_list']

def preprocess_image(image, target_input_size=(800, 800)):
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_input_size
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w

    new_h, new_w = int(orig_h * scale_h), int(orig_w * scale_w)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    padded = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_blob = padded.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_blob = (input_blob - mean) / std

    input_blob = input_blob.transpose(2, 0, 1)[np.newaxis, ...]
    return input_blob, scale_h, scale_w

def draw_bounding_boxes(image, boxes, labels, output_path):
    img = image.copy()
    for i, box in enumerate(boxes):
        cls_id = int(box[0])
        score = box[1]
        xmin = int(box[2])
        ymin = int(box[3])
        xmax = int(box[4])
        ymax = int(box[5])

        color = COLORS[cls_id % len(COLORS)]
        label = labels[cls_id]

        # Draw bounding box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)

        # Draw label with background
        text = f"{label} ({score:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 0.6, 1)[0]
        cv2.rectangle(img, (xmin, ymin - text_size[1] - 5), (xmin + text_size[0], ymin), color, -1)
        cv2.putText(img, text, (xmin, ymin - 5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(output_path, img)
    print(f"Result saved to: {output_path}")
if __name__ == '__main__':
    print("begin")
    labels = load_labels()
    print(f"Loaded {len(labels)} labels: {labels}")

    model = ort.InferenceSession(os.path.join(SCRIPT_DIR, 'PP-DocLayoutV2.onnx'))
    input_names = [i.name for i in model.get_inputs()]
    output_names = [o.name for o in model.get_outputs()]
    print(f"Model inputs: {input_names}")
    print(f"Model outputs: {output_names}")

    image_path = '/home/nianliu/tmp/pp/pdf_render_demo/pymupdf_page10.png'
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image {image_path}")
        sys.exit(1)

    input_blob, scale_h, scale_w = preprocess_image(image)
    preprocess_shape = [np.array([800, 800], dtype=np.float32)]
    input_feed = {
        input_names[0]: preprocess_shape,
        input_names[1]: input_blob,
        input_names[2]: [[scale_h, scale_w]]
    }

    output = model.run(output_names, input_feed)[0]

    # Filter boxes with score > 0.5
    boxes = output[output[:, 1] > 0.5]
    print('\n--- PP-DocLayoutV2 Inference Results ---')
    print('cls_id\tlabel\tscore\txmin\tymin\txmax\tymax')
    sorted_boxes = boxes[np.argsort(boxes[:, 2])]
    for box in sorted_boxes:
        cls_id = int(box[0])
        label = labels[cls_id]
        print(f"{cls_id}\t{label}\t{box[1]:.3f}\t{box[2]:.2f}\t{box[3]:.2f}\t{box[4]:.2f}\t{box[5]:.2f}")

    print(f"\nTotal detections: {len(boxes)}")

    # Draw results
    output_path = os.path.join(SCRIPT_DIR, 'result_visualization.png')
    draw_bounding_boxes(image, boxes, labels, output_path)
