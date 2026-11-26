import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ray import serve
from starlette.requests import Request
from starlette.responses import JSONResponse
from ultralytics import YOLO

# Reuse the same model and inference settings as profiling/profile_yolo.py
root = Path(__file__).resolve().parent.parent
model_file = root / 'yolo11n.pt'


@serve.deployment(name='YOLOWorker', num_replicas=1, ray_actor_options={'num_cpus': 0.5}, max_ongoing_requests=1, max_queued_requests=50)
class YOLOWorker:
    def __init__(self):
        if not model_file.exists():
            raise FileNotFoundError('Missing YOLO model: ' + str(model_file))

        # Give each CPU replica one PyTorch thread
        torch.set_num_threads(1)
        cv2.setNumThreads(0)
        self.model = YOLO(str(model_file), task='detect')
        # Warm up before the replica starts receiving requests
        sample = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(sample, device='cpu', imgsz=640, batch=1, rect=False, conf=0.25, iou=0.7, max_det=300, save=False, verbose=False)

    async def __call__(self, request: Request):
        # Clients send raw JPEG bytes in the POST request body
        image_bytes = await request.body()
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse({'error': 'Send a valid JPEG frame'}, status_code=400)
        start = time.perf_counter()
        result = self.model.predict(image, device='cpu', imgsz=640, batch=1, rect=False, conf=0.25, iou=0.7, max_det=300, save=False, verbose=False)[0]
        model_ms = (time.perf_counter() - start) * 1000
        detections = []
        for box in result.boxes:
            detections.append({'class_id': int(box.cls[0]),'confidence': round(float(box.conf[0]), 3),'bbox_xyxy': [round(float(v), 1) for v in box.xyxy[0]]})
        return {'detections': detections,'count': len(detections),'model_ms': round(model_ms, 2),'replica_pid': os.getpid()}

app = YOLOWorker.bind()
