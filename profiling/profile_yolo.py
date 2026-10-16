"""Benchmark every video in TrafficCamData and save two combined result files."""

import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import tempfile
import time

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "TrafficCamData"
DEVICE = "cpu"

# Update these for the calibration settings
SETTINGS = {
    "weights": str(Path(__file__).resolve().parent.parent / "yolo11n.pt"),
    "imgsz": 640, # nxn size that the images get resized to
    "frames": 300, # look at the first n number of frames
    "warmup": 20, # run this many pred before starting benchmark testing
    "repeats": 3, # num of times to repeat the complete sweep (with new workers)
    "startup_repeats": 5, # start up this many workers 
    "threads": 1, # use this many pytorch threads 
}

# Returned summarized stats
def statistics(values):
    import numpy as np
    values = np.asarray(values, dtype=float)
    return {
            "count": len(values), 
            "mean": float(values.mean()), 
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0, 
            "min": float(values.min()), 
            "p50": float(np.percentile(values, 50)), 
            "p95": float(np.percentile(values, 95)), 
            "p99": float(np.percentile(values, 99)), 
            "max": float(values.max())
        } 

# Waits for word to finish to correctly measure time of operaiton
def synchronize(torch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


# Gets times of the differnet stages in YOLO
def time_stages(model, torch):
    timings = {}
    for name in ("preprocess", "inference", "postprocess"):
        original = getattr(model.predictor, name)

        def timed(*args, _original=original, _name=name, **kwargs):
            synchronize(torch, model.predictor.device) # Wait for any other work to finish
            start = time.perf_counter() # Start timer 
            result = _original(*args, **kwargs) # Run the YOLO stage 
            synchronize(torch, model.predictor.device) # Wait for work to finish
            timings[_name + "_ms"] = (time.perf_counter() - start) * 1000 # Calculate time [ms]
            return result

        setattr(model.predictor, name, timed)
    return timings


# Run each worker in a new process
def worker(connection, videos, frame_file, device, settings, measure_frames, launched):
    try:
        # Imports
        import cv2
        import numpy as np
        import psutil
        import torch
        from ultralytics import YOLO

        # Set up device 
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS inst avail on device")

        # Keep settings fixed across trials 
        cv2.setNumThreads(0)
        torch.set_num_threads(settings["threads"])
        torch.set_num_interop_threads(1)
        model = YOLO(settings["weights"], task="detect")

        # Reset to our settings after YOLO pred starts so YOLO cant change number of threads
        def fix_threads(predictor):
            if torch.get_num_threads() != settings["threads"]:
                torch.set_num_threads(settings["threads"])
        model.add_callback("on_predict_start", fix_threads) # Run before prediction

        # Define how to process frames 
        def predict(frame):
            results = model.predict(frame, device=device, imgsz=settings["imgsz"],
                                    batch=1, rect=False, quantize=32, conf=0.25,
                                    iou=0.7, max_det=300, save=False, verbose=False)
            for result in results:
                result.boxes.data.detach().cpu().numpy()
            synchronize(torch, model.predictor.device)

        # Measure start up delay
        first_frame = np.load(frame_file, allow_pickle=False)
        predict(first_frame)
        startup_seconds = time.perf_counter() - launched

        rows = []
        process = psutil.Process()
        peak_rss = process.memory_info().rss

        if measure_frames:
            # First pred only counts for warmup, finish the other predictions and add timers
            for _ in range(settings["warmup"] - 1):
                predict(first_frame)
            timings = time_stages(model, torch)

            # Open each video
            for number, video in enumerate(videos, start=1):
                print(f"  Video {number}/{len(videos)}: {Path(video).name}", flush=True)
                capture = cv2.VideoCapture(video)
                count = 0
                try:
                    # Iterate over the specified number of frames
                    for _ in range(settings["frames"]):
                        # Get time to read and decode frame
                        start = time.perf_counter()
                        ok, frame = capture.read()
                        decode_ms = (time.perf_counter() - start) * 1000

                        if not ok:
                            break

                        # Get full prediction time (incl prepping input, inference, postprocessing, and putting detections on CPU)
                        start = time.perf_counter()
                        predict(frame)
                        service_ms = (time.perf_counter() - start) * 1000
                        
                        # Save results 
                        rows.append({"video": video, "decode_ms": decode_ms, **timings, "service_ms": service_ms, "decode_plus_service_ms": decode_ms + service_ms})
                        
                        count += 1 # Incr count of processed framed
                        peak_rss = max(peak_rss, process.memory_info().rss) # Record highest mem usage 
                finally:
                    # Close the video
                    capture.release()
                if not count:
                    raise ValueError(f"No frames could be decoded from {video}")

        # Get OS memory usage peak
        if sys.platform != "win32":
            import resource
            os_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_rss = max(peak_rss, os_peak if sys.platform == "darwin" else os_peak * 1024)
        
        # Get GPU memory usage (varies by device)
        actual_device = model.predictor.device
        gpu_memory = {}
        if actual_device.type == "cuda":
            gpu_memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(actual_device),
                          "peak_reserved_bytes": torch.cuda.max_memory_reserved(actual_device)}
        elif actual_device.type == "mps":
            gpu_memory = {"current_allocated_bytes": torch.mps.current_allocated_memory(),
                          "driver_allocated_bytes": torch.mps.driver_allocated_memory()}

        # Send results back
        connection.send({"startup_seconds": startup_seconds, "rows": rows,
                         "peak_rss_bytes": peak_rss, "gpu_memory": gpu_memory,
                         "environment": {"platform": platform.platform(),
                                         "processor": platform.processor(),
                                         "python": platform.python_version(),
                                         "device": str(actual_device),
                                         "accelerator": torch.cuda.get_device_name(actual_device)
                                         if actual_device.type == "cuda" else platform.processor(),
                                         "torch_threads": torch.get_num_threads(),
                                         "model_input_hw": list(model.predictor.imgsz),
                                         "versions": {name: importlib.metadata.version(name) for name in
                                                      ("ultralytics", "torch", "opencv-python", "numpy", "psutil")}}})
    except Exception as error:
        connection.send({"error": str(error)})
    finally:
        connection.close()

# Starts  one worker process and retrieves its results
def launch_worker(videos, frame_file, device, settings, measure_frames):
    # Create a fresh env and communication channel
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)

    # Prepare worker process
    process = context.Process(target=worker, args=(child, videos, frame_file, device, settings, measure_frames, time.perf_counter()))

    
    try:
        process.start() # Start the process
        child.close() # close main program's version of process - only worker has its own copy now

        # Wait for a response 
        if not parent.poll(1800):
            raise RuntimeError("Worker took too long (over 30min)")

        # Get the result and worker exits 
        result = parent.recv()
        process.join()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result
    # Clean up the process
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
        child.close()

# Faciliate the benchmark of the videos 
def benchmark(data_dir, device, output, settings=SETTINGS):
    import cv2
    import numpy as np
    data_dir = Path(data_dir).resolve()
    output = Path(output)

    # Validate inputs
    for name in ("imgsz", "frames", "warmup", "repeats", "startup_repeats", "threads"):
        if not isinstance(settings[name], int) or settings[name] <= 0:
            raise ValueError(f"SETTINGS['{name}'] must be a positive integer")
    if settings["imgsz"] % 32:
        raise ValueError("Image size must be divisible by 32")
    if not data_dir.is_dir():
        raise ValueError(f"Video folder does not exist: {data_dir}")
    
    videos = sorted(str(path) for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
    
    if not videos:
        raise ValueError(f"No videos found in {data_dir}")
    if not Path(settings["weights"]).is_file():
        raise ValueError(f"Model weights do not exist: {settings['weights']}")

    # Prep the starting frame
    capture = cv2.VideoCapture(videos[0])
    try:
        ok, first_frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError(f"Cannot decode video: {videos[0]}")

    # Create the output folder
    output.mkdir(parents=True, exist_ok=False)
    print(f"Found {len(videos)} videos in {data_dir}", flush=True)
    startups, runs = [], []

    # Temporarilty save the starting frame
    with tempfile.TemporaryDirectory(prefix="yolo-profile-") as temporary:
        frame_file = str(Path(temporary) / "frame.npy")
        np.save(frame_file, first_frame, allow_pickle=False)

        # Run the start up tests and benchmark testing
        for repeat in range(settings["startup_repeats"]):
            print(f"Startup trial {repeat + 1}/{settings['startup_repeats']}", flush=True)
            startups.append(launch_worker(videos, frame_file, device, settings, False))
        for repeat in range(settings["repeats"]):
            print(f"Benchmark run {repeat + 1}/{settings['repeats']}", flush=True)
            runs.append(launch_worker(videos, frame_file, device, settings, True))

    # Save results
    rows = [row for run in runs for row in run["rows"]]
    timings = {name: statistics([row[name] for row in rows]) for name in rows[0] if name != "video"}
    startup = statistics([run["startup_seconds"] for run in startups])
    peak_memory = max(run["peak_rss_bytes"] for run in startups + runs)
    capacity = 1000 / timings["service_ms"]["mean"]
    per_video = {}

    # Get results per video and save
    for video in videos:
        video_rows = [row for row in rows if row["video"] == video]
        per_video[video] = {
            "frame_count": len(video_rows),
            "frames_per_run": [sum(row["video"] == video for row in run["rows"]) for run in runs],
            "timings_ms": {name: statistics([row[name] for row in video_rows]) for name in timings},
        }
    summary = {"schema_version": 4, "data_dir": str(data_dir), "video_count": len(videos),
               "configuration": {**settings, "batch_size": 1, "precision": "float32",
                                 "conf": 0.25, "iou": 0.7, "max_det": 300, "rect": False},
               "environment": runs[0]["environment"], "frame_count": len(rows),
               "frames_per_run": [len(run["rows"]) for run in runs],
               "per_video": per_video,
               "timings_ms": timings, "startup_seconds": startup,
               "service_capacity_fps": capacity, "worker_rss_peak_bytes": peak_memory,
               "gpu_memory_observations": [run["gpu_memory"] for run in startups + runs],
               "notes": ["Service time excludes video decoding and includes host detection output.",
                         "Startup includes process launch, imports, model setup and first prediction; files are local.",
                         "Repeated runs reuse the same frames; short videos produce fewer samples.",
                         "Each repeat sweeps every video with one loaded worker; overall timings pool all measured frames.",
                         "Startup trials use the first video in sorted order.",
                         "GPU synchronization adds timing overhead. MPS memory observations are not peaks.",
                         "Single-worker capacity does not establish capacity when workers share hardware."]}
    parameters = {"schema_version": 4, "source_summary": "summary.json", "batch_size": 1,
                  "service_time_mean_seconds": timings["service_ms"]["mean"] / 1000,
                  "service_time_p95_seconds": timings["service_ms"]["p95"] / 1000,
                  "startup_mean_seconds": startup["mean"], "startup_p95_seconds": startup["p95"],
                  "worker_capacity_fps": capacity, "worker_memory_peak_bytes": peak_memory,
                  "includes_video_decode": False,
                  "memory_caveat": "Process RSS alone does not describe GPU memory limits."}
    for filename, data in [("summary.json", summary), ("simulator_parameters.json", parameters)]:
        (output / filename).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"Saved results to {output.resolve()}")
    print(f"Mean service: {timings['service_ms']['mean']:.2f} ms; "
          f"capacity: {capacity:.2f} FPS; mean startup: {startup['mean']:.2f} s")


def main():
    output = PROJECT_DIR / "results" / time.strftime("benchmark_%Y%m%d_%H%M%S")
    try:
        benchmark(DATA_DIR, DEVICE, output)
    except (ValueError, RuntimeError, OSError, EOFError) as error:
        sys.exit(f"Profiling failed: {error}")


if __name__ == "__main__":
    main()
