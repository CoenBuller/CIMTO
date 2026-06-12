import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# ============================================================================
# GLOBAL PLOTTING PARAMETERS - Adjust these as needed
# ============================================================================
# Font sizes
TITLE_SIZE = 30          # Size for subplot titles
AXIS_LABEL_SIZE = 30     # Size for x and y axis labels
TICK_LABEL_SIZE = 30     # Size for tick labels
LEGEND_FONT_SIZE = 18    # Size for legend text
LEGEND_TITLE_SIZE = 18   # Size for legend title (if any)

# Figure dimensions
FIGURE_WIDTH = 20        # Width of the entire figure in inches
FIGURE_HEIGHT_PER_ROW = 7# Height per experiment row in inches

# Marker and line settings
MARKER_SIZE = 12          # Size of markers in line plots
LINE_WIDTH = 5           # Width of lines in line plots
BAR_WIDTH = 0.2          # Width of bars in bar plots
ERRORBAR_CAPSIZE = 5     # Size of error bar caps

# Grid and transparency
GRID_ALPHA = 0.8         # Transparency of grid lines
ERRORBAR_ALPHA = 0.9     # Transparency of error bars
BAR_ALPHA = 1            # Transparency of bars

# Image quality
DPI = 300                # Resolution for saved figures

# ============================================================================
# Experiment configuration
# ============================================================================
BASE_RESULTS_DIR = "./Results"          # where Config.save_dir points to
OUTPUT_DIR = os.path.join("Results", "Experiment_1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPERIMENTS = {
    "smoothing": {
        "param_name": r"Smoothing $\sigma$",
        "param_values": [None, 0.5, 1, 2, 3],
        "is_categorical": False,
    },
    "angle_ordering": {
        "param_name": "Angle ordering ",
        "param_values": ["sequential", "random", "maximally seperated"],
        "is_categorical": True,
    },
    "InitialARM": {
        "param_name": r"Initial ARM iterations $t_{0}$",
        "param_values": [5, 10, 50, 200],
        "is_categorical": False,
    },
    "InternalARM": {
        "param_name": r"Internal ARM iterations $t$",
        "param_values": [1, 3, 5, 10, 20],
        "is_categorical": False,
    },
    "sart_relaxation": {
        "param_name": r"SART relaxation $\lambda$",
        "param_values": [0.1, 0.5, 1.0, 1.5, 2.0],
        "is_categorical": False,
    },
}

PHANTOMS = ["phantom_1", "phantom_2", "phantom_3", "phantom_4"]
PHANTOM_NAMES ={"phantom_1": 'Phantom (1)', 
                "phantom_2": 'Phantom (2)', 
                "phantom_3": 'Phantom (3)', 
                "phantom_4": 'Phantom (4)'}
PROJECTIONS = [10, 25]
N_ITERS = 10

# ============================================================================
# Helper: load results and return the LAST k_error value from each run
# ============================================================================
def load_results(exp_type: str, param_val: Any, n_proj: int, phantom: str) -> List[float]:
    param_str = "None" if param_val is None else str(param_val)
    folder = os.path.join(BASE_RESULTS_DIR, exp_type, param_str, f"projections_{n_proj}")
    file_path = os.path.join(folder, f"{phantom}.pkl")
    
    if not os.path.exists(file_path):
        print(f"Warning: missing {file_path}")
        return []
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    final_k_errors = []
    for idx in range(N_ITERS):
        if idx not in data:
            continue
        k_error_list = data[idx]["K_error"]
        if len(k_error_list) > 0:
            final_k_errors.append(k_error_list[-1])
    
    return final_k_errors

# ============================================================================
# Plot all experiments in one large figure
# ============================================================================
def plot_all_experiments_combined():
    n_experiments = len(EXPERIMENTS)
    fig, axes = plt.subplots(n_experiments, 2, 
                            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT_PER_ROW * n_experiments))

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
            numeric_vals = param_vals

        for col, n_proj in enumerate(PROJECTIONS):
            ax = axes[row, col]
            ax.set_title(f"{param_name} – {n_proj} projections", fontsize=TITLE_SIZE)
            if not is_cat:
                ax.set_xlabel(param_name.split(" ")[-1], fontsize=AXIS_LABEL_SIZE)
            if col == 0:
                ax.set_ylabel(r"Final $k$-error", fontsize=AXIS_LABEL_SIZE)
            
            # Use log scale for y-axis
            ax.set_yscale('log')
            
            # Set tick label sizes

            ax.tick_params(axis='both', labelsize=TICK_LABEL_SIZE)
            for phantom in PHANTOMS:
                means = []
                stds = []
                for pv in param_vals:
                    final_k_errors = load_results(exp_type, pv, n_proj, phantom)
                    if len(final_k_errors) == 0:
                        means.append(np.nan)
                        stds.append(np.nan)
                    else:
                        means.append(np.mean(final_k_errors))
                        stds.append(np.std(final_k_errors))

                if is_cat:
                    x = np.arange(len(param_vals))
                    offset = (PHANTOMS.index(phantom) - len(PHANTOMS)/2 + 0.5) * BAR_WIDTH
                    ax.bar(x + offset, means, width=BAR_WIDTH,
                           yerr=stds, capsize=ERRORBAR_CAPSIZE, 
                           label=PHANTOM_NAMES[phantom], alpha=BAR_ALPHA)
                else:
                    x = numeric_vals
                    # Filter out nan values for plotting
                    valid_mask = ~np.isnan(means)
                    if np.any(valid_mask):
                        x_valid = np.array(x)[valid_mask]
                        means_valid = np.array(means)[valid_mask]
                        stds_valid = np.array(stds)[valid_mask]
                        
                        # Ensure all values are positive for log scale
                        eps = 1e-10
                        means_valid_plot = np.maximum(means_valid, eps)
                        
                        ax.errorbar(x_valid, means_valid_plot, yerr=stds_valid, 
                                  marker='o', label=PHANTOM_NAMES[phantom], capsize=ERRORBAR_CAPSIZE, 
                                  markersize=MARKER_SIZE, linewidth=LINE_WIDTH, 
                                  alpha=ERRORBAR_ALPHA)

            if is_cat:
                ax.set_xticks(np.arange(len(param_vals)))
                display_vals = ["None" if v is None else v for v in param_vals]
                ax.set_xticklabels(display_vals, fontsize=24, 
                                  rotation=45 if len(display_vals) > 3 else 0)
            else:
                ax.set_xticks(numeric_vals)
                ax.set_xticklabels([f"{v:.1f}" if isinstance(v, float) else str(v) for v in numeric_vals], 
                                  fontsize=TICK_LABEL_SIZE)
                
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=GRID_ALPHA, which='both')
            ax.set_axisbelow(True)
            
            # Add legend
            if len(PHANTOMS) > 1:
                ax.legend(loc='best', fontsize=LEGEND_FONT_SIZE, framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "all_experiments_k_error.png")
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    print(f"Figure saved to {save_path}")

# ============================================================================
# Main function
# ============================================================================
def main():
    plot_all_experiments_combined()

if __name__ == "__main__":
    main()