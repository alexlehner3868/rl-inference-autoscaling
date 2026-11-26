import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument('workers', type=int, choices=[1, 2, 3, 4])
args = parser.parse_args()

# Ray Serve's external-scaling API changes the target replica count.
url = 'http://127.0.0.1:8265/api/v1/applications/camera-app/deployments/YOLOWorker/scale'
response = requests.post(url, json={'target_num_replicas': args.workers}, timeout=15)
response.raise_for_status()

print('Requested replicas:', args.workers)
print('HTTP status:', response.status_code)
print('This is the target count. Run "serve status" to check readiness.')
