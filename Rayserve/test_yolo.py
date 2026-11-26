import time
from pathlib import Path

import cv2
import requests

root = Path(__file__).resolve().parent.parent
videos = sorted((root / 'TrafficCamData/Dataset_A').rglob('*.mp4'))
if not videos:
    raise SystemExit('No MP4 videos found in TrafficCamData/Dataset_A')
# Extract a real frame from the first recorded traffic video
capture = cv2.VideoCapture(str(videos[0]))
ok, frame = capture.read()
capture.release()
if not ok:
    raise SystemExit('Cannot read the first frame from ' + str(videos[0]))
ok, encoded = cv2.imencode('.jpg', frame)
if not ok:
    raise SystemExit('Could not encode the frame as JPEG')
# Send the frame to Ray Serve, measure HTTP round-trip latency
start = time.perf_counter()
response = requests.post('http://127.0.0.1:8000/', data=encoded.tobytes(), headers={'Content-Type': 'image/jpeg'}, timeout=60)
elapsed = time.perf_counter() - start
response.raise_for_status()
result = response.json()

print('Video:', videos[0].name)
print('HTTP status:', response.status_code)
print('Detections:', result['count'])
print('Model time (ms):', result['model_ms'])
print('End-to-end HTTP latency (ms):', round(elapsed * 1000, 2))
print('Completed within 1 second:', elapsed <= 1.0)
print('Replica process:', result['replica_pid'])
print('First three detections:', result['detections'][:3])
