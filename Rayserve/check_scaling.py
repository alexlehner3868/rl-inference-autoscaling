"""Check that Ray Serve can scale from 1 worker to N workers and back."""

import argparse
import time
from Rayserve.live_autoscale import set_workers, wait_for_workers

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--timeout', type=float, default=120)
args = parser.parse_args()

if args.workers < 2:
    parser.error('The worker count must be at least 2')

try:
    set_workers(1)
    wait_for_workers(1, timeout=args.timeout)
    print('Ready workers: 1')

    start = time.monotonic()
    set_workers(args.workers)
    wait_for_workers(args.workers, timeout=args.timeout)
    print('Ready workers:', args.workers)
    print('Scale-up time:', round(time.monotonic() - start, 2), 'seconds')

    start = time.monotonic()
    set_workers(1)
    wait_for_workers(1, timeout=args.timeout)
    print('Ready workers: 1')
    print('Scale-down time:', round(time.monotonic() - start, 2), 'seconds')
finally:
    set_workers(1)
