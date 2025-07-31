import cv2
import torch
from ultralytics import YOLO
import torchvision.ops as ops

model = YOLO('models/best.pt')  # Load once globally for efficiency

def detect(frame):
    # Run inference
    results = model(frame)

    # Grab raw tensors
    boxes   = results[0].boxes.xyxy.cpu()
    scores  = results[0].boxes.conf.cpu()
    cls_ids = results[0].boxes.cls.cpu()

    # NMS
    keep = ops.nms(boxes, scores, iou_threshold=0.2)

    # Annotate image
    annotated = frame.copy()
    counter = 0

    for i in keep:
        x1, y1, x2, y2 = boxes[i].int().tolist()
        conf = scores[i].item()
        cls_name = model.names[int(cls_ids[i])]
        label = f"{cls_name} {conf:.2f}"

        if cls_name == "uv-line":
            counter += 1

        # Draw box and label
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save annotated image
    cv2.imwrite("result.jpg", annotated)

    return counter
