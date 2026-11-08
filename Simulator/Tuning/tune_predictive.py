from datetime import datetime
from itertools import product
import json

import matplotlib

from Simulator.predictive_baseline import (
    ROOT, SCALE_DOWN_STEPS, WINDOW, run_workload, write_csv,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paramters to sweep for tuning 
WINDOWS = (3, 5, 10)
SCALE_DOWN_DELAYS = (1, 3, 5)
REFERENCE = (WINDOW, SCALE_DOWN_STEPS) # Get the current hyperparams ones specified in predictive_baseline.py 

# Calculate the on time percentage across mutliple workloads 
def on_time_rate(rows):
    arrived = sum(row["arrived"] for row in rows)
    return 100 * sum(row["on_time"] for row in rows) / arrived if arrived else None


# Convert the 20 workloads into one config result 
def summarize(settings, rows):
    # Seperate the workloads by type 
    standard = [row for row in rows if row["load_level"] == "standard"]
    stress = [row for row in rows if row["load_level"] == "stress"]
    bursts = [row for row in rows if row["pattern"] == "multi_burst"]
    # Create and return dict  
    return {
        **settings,
        "workloads": len(rows),
        "on_time_percentage": on_time_rate(rows),
        "standard_on_time_percentage": on_time_rate(standard),
        "stress_on_time_percentage": on_time_rate(stress),
        "multi_burst_on_time_percentage": on_time_rate(bursts),
        "worker_seconds": round(sum(row["worker_seconds"] for row in rows), 6),
        "dropped_frames": sum(row["dropped_deadline"] + row["dropped_full"] for row in rows),
        "scaling_changes": sum(row["scaling_changes"] for row in rows),
    }


def plot_tradeoff(output, summary):
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")

    # Plot each hyperparamter on the scatter plot
    for row in summary:
        pair = row["window"], row["scale_down_steps"]
        cost, rate = row["worker_seconds"], row["on_time_percentage"]
        reference = pair == REFERENCE
        ax.scatter(cost, rate, color="#bc5425" if reference else "#226889", s=55)
        label_below = pair[0] == max(WINDOWS)
        ax.annotate(f"{pair[0]} / {pair[1]}", (cost, rate), xytext=(6, -5 if label_below else 5), textcoords="offset points", a="top" if label_below else "bottom", fontsize=9)
    
    ax.set(title=f"Predictive tuning on {summary[0]['workloads']} validation workloads", xlabel="Allocated worker-seconds (lower is better)", ylabel="Frames completed on time (%)")
    ax.text(0.02, 0.98, f"Labels: window / scale-down steps\nOrange: {REFERENCE[0]} / {REFERENCE[1]} reference", transform=ax.transAxes, va="top", fontsize=9)
    ax.margins(x=0.12, y=0.15)
    ax.grid(alpha=0.25)
    fig.savefig(output / "tradeoff.png", dpi=160)
    
    plt.close(fig)



def plot_heatmap(output, summary):
    # Create lookup table with key winow, scale down steps 
    by_setting = {(row["window"], row["scale_down_steps"]): row for row in summary}
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")

    # Choose what is shown on the the heatmap 
    metrics = (("on_time_percentage", "Frames completed on time (%)", ".2f", "Blues"), ("worker_seconds", "Allocated worker-seconds", ",.0f", "Oranges_r"),)

    # Loop through 2 heat maps 
    for ax, (key, title, number_format, colors) in zip(axes, metrics):
        values = [[by_setting[window, delay][key] for delay in SCALE_DOWN_DELAYS] for window in WINDOWS]
        grid = ax.imshow(values, cmap=colors)

        # Write the numbers into the grid 
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                red, green, blue, _ = grid.cmap(grid.norm(value))
                brightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "white" if brightness < 0.48 else "#15232e"
                ax.text(j, i, format(value, number_format), ha="center", va="center", color=color)
        
        ax.set_xticks(range(len(SCALE_DOWN_DELAYS)), SCALE_DOWN_DELAYS)
        ax.set_yticks(range(len(WINDOWS)), WINDOWS)
        ax.set(title=title, xlabel="Scale-down steps", ylabel="Demand window (steps)")
    
    fig.savefig(output / "parameter_grid.png", dpi=160)
    plt.close(fig)


def main():
    files = sorted((ROOT / "workloads" / "validation").glob("*.json"))
    calibration = json.loads((ROOT / "results/calibration/simulator_parameters.json").read_text())

    summary, per_workload = [], []

    # Do grid search of paramters 
    for window, delay in product(WINDOWS, SCALE_DOWN_DELAYS):
        settings = {"window": window, "scale_down_steps": delay}
        rows = []
        
        for file in files:
            result, _ = run_workload(file, None, calibration, **settings)
            rows.append({**settings, **result})
        
        per_workload.extend(rows)
        totals = summarize(settings, rows)
        summary.append(totals)
        
        print(f"{window} / {delay}: {totals['on_time_percentage']:.2f}% on time, {totals['worker_seconds']:,.1f} worker-seconds", flush=True)

    output = ROOT / "results" / "predictive_baseline" / "tuning" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)

    # Save results
    write_csv(output / "summary.csv", summary)
    write_csv(output / "per_workload.csv", per_workload)

    # Save plots 
    plot_tradeoff(output, summary)
    plot_heatmap(output, summary)
    
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
