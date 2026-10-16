### Helper code to generate workloads 

import json
import math
from pathlib import Path
import random


CAMERA_FPS = 10.0 # FPS per active camera 
DECISION_SECONDS = 1.0
DURATION_SECONDS = 300 # Length of a simulated workload
MIN_CAMERAS = 0
STANDARD_MAX_CAMERAS = 5  # Baseline max cameras
MAX_CAMERAS = 12         # Max num of cameras for stress testing 
WORKLOAD_COUNTS = {"train": 30, "validation": 20, "test": 10} # Amount of workloads to make for each dataset
STRESS_COUNTS = {"train": 10, "validation": 5, "test": 5} # Number of stress test workloads to make for each dataset 
SPLIT_SEEDS = {"train": 1, "validation": 2, "test": 3} # Seeds to use for each dataset (for reporducabiliy)
OUTPUT_DIR = Path(__file__).resolve().parent


# Changes the number of active cameras in a series for a period of time
def demand_change(series, start, end, count):
    for index in range(start, min(end, len(series))):
        series[index] = count


# Craete a workload that has lots of short bursts of demand
def multi_burst(rng, steps, limit):
    series = [rng.randint(MIN_CAMERAS, MIN_CAMERAS + 1)] * steps # Baseline schedule 
    start = rng.randint(5, 12)
    first = True

    # Randomize when and how high to add bursts for a given time 
    while start < steps - 10:
        width = rng.randint(1, 8)
        height = limit if first else rng.randint(max(MIN_CAMERAS + 3, limit // 2), limit)
        demand_change(series, start, start + width, height)
        first = False
        start += width + rng.randint(3, 15)
    
    return series


# Create workloads that have increased demand with spikes 
def flash_crowds(rng, steps, limit):
    series = [rng.randint(MIN_CAMERAS, MIN_CAMERAS + 1)] * steps
    start = rng.randint(8, 15)

    while start < steps - 20:
        end = min(start + rng.randint(15, 35), steps - 8)
        demand_change(series, start, end, rng.randint(max(3, limit - 2), limit - 1))
        
        spike = rng.randint(start + 2, end - 2)
        demand_change(series, spike, min(spike + rng.randint(1, 5), end), limit)
        start = end + rng.randint(8, 20)
    
    return series

# Create a workload that has gradual changes to demand with spikes
def ramp_with_spikes(rng, steps, limit):
    series = []
    count, direction = MIN_CAMERAS, 1
    
    while len(series) < steps:
        series.extend([count] * rng.randint(5, 12))
        if count == limit:
            direction = -1
        elif count == MIN_CAMERAS:
            direction = 1
        count += direction
    series = series[:steps]
    
    for _ in range(rng.randint(6, 12)):
        start = rng.randint(5, steps - 8)
        demand_change(series, start, start + rng.randint(1, 4), limit)
    return series

# Workloads with demand decreases and some recovery
def mass_dropoff(rng, steps, limit):
    series = []
    
    while len(series) < steps:
        series.extend([limit] * rng.randint(12, 25))
        low = rng.randint(MIN_CAMERAS, MIN_CAMERAS + 1)
        series.extend([low] * rng.randint(8, 18))
        
        for count in range(low + 1, limit + 1):
            series.extend([count] * rng.randint(3, 7))
    
    return series[:steps]

# Irregular demand changes 
def mixed_churn(rng, steps, limit):
    series = []
    count = rng.randint(MIN_CAMERAS, MIN_CAMERAS + 2)
    
    while len(series) < steps:
        event = rng.random()
        if event < 0.2:
            count = limit
        elif event < 0.4:
            count = MIN_CAMERAS
        else:
            count = max(MIN_CAMERAS, min(limit, count + rng.randint(-3, 3)))
        series.extend([count] * rng.randint(2, 10))
    
    series = series[:steps]
    
    # Need at least one full size influx
    start = rng.randint(steps // 3, 2 * steps // 3)
    demand_change(series, start, start + rng.randint(2, 6), limit)

    return series


PATTERNS = {"multi_burst": multi_burst, "flash_crowds": flash_crowds,
            "ramp_with_spikes": ramp_with_spikes, "mass_dropoff": mass_dropoff,
            "mixed_churn": mixed_churn}


# Checks that the workload is valid
def validate_schedule(schedule):
    # Gets timing settings 
    decision, duration = schedule["decision_seconds"], schedule["duration_seconds"]

    # Confirms that timeing and frame rate values are positive valid
    if any(not math.isfinite(value) or value <= 0
           for value in (decision, duration, schedule["camera_fps"])):
        raise ValueError("Frame rate and times must be > 0 and finite")

    # Get camaera count bounds 
    bounds = schedule["generation_settings"]
    previous_end = 0

    # Iterate over schedule intervals 
    for interval in schedule["intervals"]:

        # CHeck that the intervals connect properly (no gaps or overlaps)
        start, end = interval["start_seconds"], interval["end_seconds"]
        if start != previous_end or not start < end <= duration:
            raise ValueError("Bad Intervals (overlapping or gaps)")
        
        # Confirm changes happen on boundaries (changes need to occur on simulator steps)
        for boundary in (start, end):
            # Divide boundary by decision should give an int 
            if not math.isclose(boundary / decision, round(boundary / decision), rel_tol=0, abs_tol=1e-9): 
                raise ValueError("Inverval changes need to happen at boundaries")
            
        previous_end = end
    if previous_end != duration:
        raise ValueError("Intervals must end at the workload duration")


# Create a description of the workload 
def demand_signature(schedule):
    points = []
    for interval in schedule["intervals"]:
        count = len(interval["camera_ids"])
        if not points or points[-1][1] != count:
            points.append((interval["start_seconds"], count))
    return json.dumps([schedule["duration_seconds"], schedule["camera_fps"], points])

# Create one workload
def make_schedule(pattern, seed, load_level="standard"):
    rng = random.Random(seed) # Set up random seed for repoducability

    # Define camera limits based on type of workload 
    limit = STANDARD_MAX_CAMERAS if load_level == "standard" else rng.randint(STANDARD_MAX_CAMERAS + 2, MAX_CAMERAS)

    # Set up camera counts over time 
    steps = round(DURATION_SECONDS / DECISION_SECONDS)
    counts = PATTERNS[pattern](rng, steps, limit)

    intervals = []
    active = []
    next_id = 0

    # Iterate over counts
    for step, count in enumerate(counts):
        previous_count = len(active)

        # remove cameras when demand falls
        if count < len(active):
            active = sorted(rng.sample(active, count))

        # Add cameras when demand incerases
        added = count - len(active)
        active.extend(range(next_id, next_id + added))
        next_id += added

        # Create more complex workloads with mixed_churn types 
        if pattern == "mixed_churn" and active and count == previous_count and rng.random() < 0.12:
            replacements = rng.randint(1, max(1, len(active) // 2))
            active = rng.sample(active, len(active) - replacements)
            active.extend(range(next_id, next_id + replacements))
            next_id += replacements
        
        active.sort() # Keep Camera IDs ordered

        # Create new intervals
        if not intervals or active != intervals[-1]["camera_ids"]:
            boundary = round(step * DECISION_SECONDS, 9)
            
            if intervals:
                intervals[-1]["end_seconds"] = boundary
            intervals.append({"start_seconds": boundary, "end_seconds": DURATION_SECONDS, "camera_ids": active.copy()})

    # Consturct schedule 
    schedule = {
        "generator_version": 2, "pattern": pattern, "seed": seed, "load_level": load_level,
        "camera_fps": CAMERA_FPS, "duration_seconds": DURATION_SECONDS,
        "decision_seconds": DECISION_SECONDS,
        "generation_settings": {"min_cameras": MIN_CAMERAS, "max_cameras": limit},
        "peak_demand_fps": max(counts) * CAMERA_FPS,
        "demand_changes": sum(a != b for a, b in zip(counts, counts[1:])),
        "camera_session_changes": len(intervals) - 1,
        "total_camera_sessions": next_id,
        "intervals": intervals,
    }
    # Check it 
    validate_schedule(schedule)

    return schedule


# Creates the collection of unqiue schedules 
def make_schedules():
    schedules, seen = {}, set()

    # loop through each split
    for split, total in WORKLOAD_COUNTS.items():
        stress = STRESS_COUNTS[split]

        # Check that workload types can be divided by total number of generated schedules 
        if (type(total) is not int or total <= 0 or total % len(PATTERNS) or type(stress) is not int or not 0 <= stress <= total or stress % len(PATTERNS)):
            raise ValueError("Workload counts must be multiples of the pattern count")

        # Get num of variants per pattern
        variants = total // len(PATTERNS)
        stress_variants = stress // len(PATTERNS)

        # Look through all patterns
        for pattern_index, pattern in enumerate(PATTERNS):
            # Iterate for the number of variants you need
            for variation in range(variants):
                level = "stress" if variation >= variants - stress_variants else "standard"
                name = f"{pattern}_{variation + 1:02d}"

                # Try to create a workload
                for attempt in range(1000):
                    seed = SPLIT_SEEDS[split] + pattern_index * 10_000 + variation * 1_000 + attempt
                    schedule = make_schedule(pattern, seed, level)
                    signature = demand_signature(schedule)
                    # Check that it is unique, else try again 
                    if signature not in seen and schedule["demand_changes"] >= 8:
                        break
                else:
                    raise ValueError(f"Couldn't create a viabke workload for {split}/{name}")
                
                seen.add(signature)
                schedule.update(name=name, split=split)
                schedules[f"{split}/{name}.json"] = schedule

    return schedules


# Create the dataset of workloads 
def generate(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)

    schedules = make_schedules()

    files = {output_dir / name: json.dumps(data, indent=2, allow_nan=False) + "\n" for name, data in schedules.items()}
    
    for split in WORKLOAD_COUNTS:
        for path in (output_dir / split).glob("*.json"):
            if path not in files:
                raise ValueError(f"Old workload {path}; move it before trying again")
    
    for path, content in files.items():
        if path.exists() and path.read_text() != content:
            raise ValueError(f"{path} alraedy has workloads - use a new output directory or remove files")
    
    for path, content in files.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
    
    for split, count in WORKLOAD_COUNTS.items():
        print(f"{split}: {count} workloads")
    
    return schedules


if __name__ == "__main__":
    generate()
