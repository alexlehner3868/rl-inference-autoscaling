import asyncio
import logging
import os
import threading
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
logger = logging.getLogger(__name__)


@serve.deployment(name='YOLOWorker', num_replicas=1, ray_actor_options={'num_cpus': 0.5}, max_ongoing_requests=1, max_queued_requests=50)
class YOLOWorker:
    def __init__(self):
        if not model_file.exists():
            raise FileNotFoundError('Missing YOLO model: ' + str(model_file))

        # Give each CPU replica one PyTorch thread
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cv2.setNumThreads(0)
        self.model = YOLO(str(model_file), task='detect')
        # Keep the limit if YOLO changes it during predictor setup.
        def limit_predict_threads(_predictor):
            if torch.get_num_threads() != 1:
                torch.set_num_threads(1)

        self.model.add_callback('on_predict_start', limit_predict_threads)
        # A disconnected client can cancel the coroutine while its thread is
        # still running. Keep subsequent calls from using the model concurrently.
        self.model_lock = threading.Lock()
        # Warm up before the replica starts receiving requests
        sample = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(sample, device='cpu', imgsz=640, batch=1, rect=False, conf=0.25, iou=0.7, max_det=300, save=False, verbose=False)

    async def __call__(self, request: Request):
        # Clients send raw JPEG bytes in the POST request body
        image_bytes = await request.body()
        cancelled = threading.Event()
        try:
            return await asyncio.to_thread(self._predict_bytes, image_bytes, cancelled)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    def _predict_bytes(self, image_bytes, cancelled):
        if cancelled.is_set():
            return None
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse({'error': 'Send a valid JPEG frame'}, status_code=400)
        with self.model_lock:
            if cancelled.is_set():
                return None
            # Prediction runs in asyncio.to_thread, not the constructor thread.
            if torch.get_num_threads() != 1:
                torch.set_num_threads(1)
            start = time.perf_counter()
            result = self.model.predict(image, device='cpu', imgsz=640, batch=1, rect=False, conf=0.25, iou=0.7, max_det=300, save=False, verbose=False)[0]
            model_ms = (time.perf_counter() - start) * 1000
        if model_ms > 500:
            logger.warning('YOLO inference took %.1f ms in replica %s', model_ms, os.getpid())
        detections = []
        for box in result.boxes:
            detections.append({'class_id': int(box.cls[0]),'confidence': round(float(box.conf[0]), 3),'bbox_xyxy': [round(float(v), 1) for v in box.xyxy[0]]})
        return {'detections': detections,'count': len(detections),'model_ms': round(model_ms, 2),'replica_pid': os.getpid()}

app = YOLOWorker.bind()
