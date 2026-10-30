"""
detector.py — YOLOv8 Wrapper using Ultralytics
Detects 80 COCO object classes with confidence scoring
"""

from ultralytics import YOLO
import numpy as np

# Color palette per class (clean, high-contrast, no purple)
CLASS_COLORS = {
    'person':'#2563eb','car':'#0284c7','bicycle':'#0d9488',
    'motorcycle':'#ea580c','bus':'#10b981','truck':'#d97706',
    'cat':'#e11d48','dog':'#0284c7','bird':'#059669',
    'horse':'#16a34a','sheep':'#ca8a04','cow':'#dc2626',
    'elephant':'#0891b2','bear':'#b45309','zebra':'#475569',
    'giraffe':'#d97706','backpack':'#2563eb','umbrella':'#ea580c',
    'handbag':'#0d9488','tie':'#0284c7','suitcase':'#65a30d',
    'frisbee':'#e11d48','skis':'#0284c7','snowboard':'#2563eb',
    'sports ball':'#ca8a04','kite':'#10b981','baseball bat':'#dc2626',
    'baseball glove':'#2563eb','skateboard':'#0d9488','surfboard':'#0891b2',
    'tennis racket':'#ea580c','bottle':'#0284c7','wine glass':'#0d9488',
    'cup':'#2563eb','fork':'#ca8a04','knife':'#dc2626','spoon':'#65a30d',
    'bowl':'#ea580c','banana':'#eab308','apple':'#16a34a','sandwich':'#ca8a04',
    'orange':'#ea580c','broccoli':'#15803d','carrot':'#ea580c',
    'hot dog':'#dc2626','pizza':'#d97706','donut':'#ca8a04','cake':'#e11d48',
    'chair':'#2563eb','couch':'#0d9488','potted plant':'#10b981',
    'bed':'#0284c7','dining table':'#b45309','toilet':'#0891b2',
    'tv':'#2563eb','laptop':'#0284c7','mouse':'#0d9488','remote':'#dc2626',
    'keyboard':'#16a34a','cell phone':'#ca8a04','microwave':'#475569',
    'oven':'#dc2626','toaster':'#ea580c','sink':'#0891b2',
    'refrigerator':'#10b981','book':'#ca8a04','clock':'#2563eb',
    'vase':'#0d9488','scissors':'#0891b2','teddy bear':'#b45309',
    'hair drier':'#0284c7','toothbrush':'#0d9488',
}

def get_class_color(label: str) -> str:
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    # Deterministic hash-based color for unknown classes
    h = abs(hash(label)) % 0xFFFFFF
    return f'#{h:06x}'


class Detector:
    def __init__(self, model_size: str = 'yolov8n'):
        """
        model_size options:
          yolov8n  – nano  (~6 MB)  fastest
          yolov8s  – small (~22 MB) balanced
          yolov8m  – medium         accurate
        """
        print(f'[Detector] Loading {model_size}.pt ...')
        self.model = YOLO(f'{model_size}.pt')
        print(f'[Detector] {model_size} ready')

    def detect(self, frame, confidence: float = 0.5) -> list:
        """
        Returns list of dicts:
          { label, score, bbox:[x,y,w,h], color }
        """
        results = self.model(
            frame,
            conf=confidence,
            verbose=False,
            stream=False,
        )

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label  = self.model.names[cls_id]
                score  = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    'label': label,
                    'score': round(score, 3),
                    'bbox':  [x1, y1, x2 - x1, y2 - y1],  # x,y,w,h
                    'color': get_class_color(label),
                })

        return detections
