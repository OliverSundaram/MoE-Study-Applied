# Imports
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import numpy as np
from matplotlib.pyplot import Axes



# Hyperparameters
ROOT = Path.cwd()

MODELS = ["base", "sft"]
SPEED_DIR = ROOT / "results"
OUT_DIR = ROOT / "assets" / "speed.png"
RESULTS = {
    "base": {
        "no_cached": "base_model_no_cached.pt",
        "cached": "base_model_cached.pt"
    },

    "sft": {
        "no_cached": "sft_model_no_cached.pt",
        "cached": "sft_model_cached.pt"
    }
}



# Functions
def load_results(results: dict[str, dict[str, str]], speed_dir):
    base_no_cached_path = results["base"]["no_cached"]
    base_cached_path = results["base"]["cached"]
    sft_no_cached_path = results["sft"]["no_cached"]
    sft_cached_path = results["sft"]["cached"]

    return (
        torch.load(speed_dir / base_no_cached_path),
        torch.load(speed_dir / base_cached_path),
        torch.load(speed_dir / sft_no_cached_path),
        torch.load(speed_dir / sft_cached_path),
    )


def plot(ax: Axes, base_no_cached, base_cached, sft_no_cached, sft_cached, value: str):
    x = np.array([0, 1])
    width = 0.35
    nudge = 0.02

    # Graph bars. Graph blue/cached first, then orange/not cached second
    bars1 = ax.bar(x=x - width / 2 - nudge,
                   height=[base_cached[value], sft_cached[value]],
                   width=width,
                   color="#4C72B0",
                   label="KV cache",
                   edgecolor="white",
                   linewidth="1")

    bars2 = ax.bar(x=x + width / 2 + nudge,
                   height=[base_no_cached[value], sft_no_cached[value]],
                   width=width,
                   color="#DD8452",
                   label="no cache",
                   edgecolor="white",
                   linewidth="1")

    # Add values on top of bars
    ax.bar_label(bars1, fmt="%.2f", padding=3, color="grey")
    ax.bar_label(bars2, fmt="%.2f", padding=3, color="grey")

    # Set x labels to [base, sft]
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS)

    # Set graph specific values
    if value == "tok_per_sec":
        ax.set_title("base vs. sft tokens / sec (higher is better)", fontweight="bold", fontsize=11)
        ax.set_xlabel("Model")
        ax.set_ylabel("tokens / sec")

        # Set y-axis to be larger to shrink bars since they are too high
        ax.set_yticks(list(range(0, 31, 5)))

    elif value == "time_to_first_tok":
        ax.set_title("base vs. sft time to first token (lower is better)", fontweight="bold", fontsize=11)
        ax.set_xlabel("Model")
        ax.set_ylabel("seconds")

        # Set y-axis to be larger to shrink bars since they are too high
        ax.set_yticks(np.array(list(range(0, 36, 5))) / 100)

    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.5, color="grey", linestyle="-")

    return ax


def main():
    base_no_cached, base_cached, sft_no_cached, sft_cached = load_results(RESULTS, SPEED_DIR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    ax1 = plot(ax1, base_no_cached, base_cached, sft_no_cached, sft_cached, value="tok_per_sec")
    ax2 = plot(ax2, base_no_cached, base_cached, sft_no_cached, sft_cached, value="time_to_first_tok")


    plt.suptitle("Inference speed — KV cache vs no cache", fontweight="bold", fontsize=16)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    # Adjust layout to make room for the bottom legend and super title
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    plt.savefig(OUT_DIR)


if __name__ == "__main__":
    main()