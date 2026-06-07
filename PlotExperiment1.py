import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any

# ----------------------------------------------------------------------
# Experiment configuration – update these paths as needed
# ----------------------------------------------------------------------
BASE_RESULTS_DIR = "./Results"          # where Config.save_dir points to
OUTPUT_DIR = os.path.join("Results", "Experiment_1_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = {
    "smoothing": {
        "param_name": "Smoothing sigma",
        "param_values": [None, 0.5, 1, 2, 3],
        "is_categorical": False,          # treat as continuous, map None->0
    },
    "angle_ordering": {
        "param_name": "Angle ordering",
        "param_values": ["sequential", "random", "interleaved"],
        "is_categorical": True,
    },
    "InitialARM": {
        "param_name": "Initial ARM iterations",
        "param_values": [5, 10, 50, 200],
        "is_categorical": False,
    },
    "InternalARM": {
        "param_name": "Internal ARM iterations",
        "param_values": [1, 3, 5, 10, 20],
        "is_categorical": False,
    },
    "sart_relaxation": {
        "param_name": "SART relaxation λ",
        "param_values": [0.1, 0.5, 1.0, 1.5, 2.0],
        "is_categorical": False,
    },
}

PHANTOMS = ["phantom_1", "phantom_2", "phantom_3", "phantom_4"]
PROJECTIONS = [10, 25]
N_ITERS = 10   # must match N_ITERS used in the experiments

# ----------------------------------------------------------------------
# Helper: load results for a given experiment, hyperparameter value,
# number of projections, and phantom name.
# Returns a list of k_error values (length = N_ITERS) and a list of times.
# ----------------------------------------------------------------------
def load_results(exp_type: str, param_val: Any, n_proj: int, phantom: str) -> Tuple[List[float], List[float]]:
    # Convert None to string 'None' for folder name
    param_str = "None" if param_val is None else str(param_val)
    folder = os.path.join(BASE_RESULTS_DIR, exp_type, param_str, f"projections_{n_proj}")
    file_path = os.path.join(folder, f"{phantom}.pkl")
    if not os.path.exists(file_path):
        print(f"Warning: missing {file_path}")
        return [], []
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    k_errors = []
    times = []
    for idx in range(N_ITERS):
        if idx not in data:
            continue
        k_errors.append(data[idx]["K_error"])
        times.append(data[idx]["Time"])
    return k_errors, times

# ----------------------------------------------------------------------
# Plot all experiments in one large figure.
# Each experiment becomes a row with two subplots (10 and 25 projections).
# Saves to OUTPUT_DIR.
# ----------------------------------------------------------------------
def plot_all_experiments_combined():
    n_experiments = len(EXPERIMENTS)
    fig, axes = plt.subplots(n_experiments, 2, figsize=(14, 4 * n_experiments))
    fig.suptitle("Hyperparameter Ablation Studies: k_error", fontsize=16, y=1.02)

    if n_experiments == 1:
        axes = axes.reshape(1, -1)

    for row, (exp_type, exp_cfg) in enumerate(EXPERIMENTS.items()):
        param_name = exp_cfg["param_name"]
        param_vals = exp_cfg["param_values"]
        is_cat = exp_cfg["is_categorical"]

        # For numeric experiments, map None -> 0 for plotting
        if not is_cat:
            numeric_vals = []
            for v in param_vals:
                if v is None:
                    numeric_vals.append(0.0)
                else:
                    numeric_vals.append(float(v))
        else:
            numeric_vals = param_vals  # keep original for categorical

        for col, n_proj in enumerate(PROJECTIONS):
            ax = axes[row, col]
            ax.set_title(f"{param_name} – {n_proj} projections")
            ax.set_xlabel(param_name)
            if col == 0:
                ax.set_ylabel("k_error")
            ax.set_yscale('log') 

            for phantom in PHANTOMS:
                means = []
                stds = []
                for pv in param_vals:
                    k_errs, _ = load_results(exp_type, pv, n_proj, phantom)
                    if len(k_errs) == 0:
                        means.append(np.nan)
                        stds.append(np.nan)
                    else:
                        means.append(np.mean(k_errs))
                        stds.append(np.std(k_errs))

                if is_cat:
                    x = np.arange(len(param_vals))
                    width = 0.2
                    offset = (PHANTOMS.index(phantom) - len(PHANTOMS)/2 + 0.5) * width
                    ax.bar(x + offset, means, width=width,
                           yerr=stds, capsize=2, label=phantom)
                else:
                    # Use numeric_vals (0, 0.5, 1, 2, 3) as x-coordinates
                    x = numeric_vals
                    ax.errorbar(x, means, yerr=stds, marker='o', label=phantom, capsize=3)

            if is_cat:
                ax.set_xticks(np.arange(len(param_vals)))
                # For categorical, show original param values (None becomes "None")
                display_vals = ["None" if v is None else v for v in param_vals]
                ax.set_xticklabels(display_vals)
            else:
                # For numeric, ensure ticks show 0 instead of None
                ax.set_xticks(numeric_vals)
                # Format tick labels: 0 is shown as 0, not None
                ax.set_xticklabels([f"{v:.1f}" if isinstance(v, float) else str(v) for v in numeric_vals])

            ax.legend(loc='best', fontsize='small')
            ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "all_experiments_k_error.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {save_path}")
    plt.show()

# ----------------------------------------------------------------------
# Print a table of average reconstruction times (seconds) for each
# experiment, hyperparameter, projection count, and phantom.
# Also saves the table as CSV.
# ----------------------------------------------------------------------
def print_time_table():
    rows = []
    for exp_type, exp_cfg in EXPERIMENTS.items():
        param_vals = exp_cfg["param_values"]
        for pv in param_vals:
            for n_proj in PROJECTIONS:
                for phantom in PHANTOMS:
                    _, times = load_results(exp_type, pv, n_proj, phantom)
                    if times:
                        avg_time = np.mean(times)
                        # Show parameter as 0 for None in the table for clarity
                        param_display = 0 if pv is None else pv
                        rows.append({
                            "Experiment": exp_type,
                            "Parameter": param_display,
                            "Projections": n_proj,
                            "Phantom": phantom,
                            "Avg time (s)": avg_time
                        })
    if not rows:
        print("No time data found.")
        return

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index=["Experiment", "Parameter", "Projections"],
                           columns="Phantom", values="Avg time (s)")
    print("\n=== Average reconstruction times (seconds) ===\n")
    print(pivot.round(3))

    csv_path = os.path.join(OUTPUT_DIR, "reconstruction_times.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nTime table saved to {csv_path}")

# ----------------------------------------------------------------------
# Main function
# ----------------------------------------------------------------------
def main():
    plot_all_experiments_combined()
    print_time_table()

if __name__ == "__main__":
    main()