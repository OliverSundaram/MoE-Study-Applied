# Imports
import torch

import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

from pathlib import Path



# Hyperparameters
ROOT = Path.cwd()

MODELS = ["base", "sft"]
MODEL_NAME = "base"
assert MODEL_NAME in MODELS, f"{MODEL_NAME} unknown. Please select from {MODELS}"
STATE_DIR = ROOT.parent / "model" / MODEL_NAME / "trainer_state.pt"
OUT_DIR = ROOT / "assets" / f"{MODEL_NAME}_losses.png"

VALUES = ["losses", "aux_losses"]



# Functions
def load_state(state_dir) -> dict:
    print(f"Loading state from -> {state_dir}")
    return torch.load(state_dir)


def plot(ax: Axes, state: dict, value):
    assert value in VALUES, f"{value} is not valid. Please select from {VALUES}"

    # Determine what value to graph
    if value == "losses":
        train_losses = state["train_losses"]
        val_losses = state["val_losses"]
        test_loss = state["test_loss"]
    else:
        train_losses = state["train_aux_losses"]
        val_losses = state["val_aux_losses"]
        test_loss = state["test_aux_loss"]
    steps = state["steps"]
    val_steps = state["val_steps"]

    # Plot train loss line
    ax.plot(steps, train_losses, color="#4C72B0", linewidth=1, alpha=0.9, label="train")
    # Plot val loss line and dots
    ax.plot(val_steps, val_losses, color="#DD8452", linewidth=2, label="validation", marker="o", markersize=4)
    # Plot test loss line over x-axis
    ax.axhline(test_loss, color="grey", linewidth=2, label=f"test ({test_loss:.3f})", linestyle="--")

    ax.grid(axis="y", alpha=0.5, color="grey")
    ax.grid(axis="x", alpha=0.5, color="grey")

    if value == "losses":
        ax.set_title("Loss", fontweight="bold", fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-entropy loss")
        ax.legend(loc="upper right")
    else:
        ax.set_title("Auxiliary (load-balancing) loss", fontweight="bold", fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Aux loss")
        ax.legend(loc="upper right")

    return ax


def main():
    state = load_state(STATE_DIR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1 = plot(ax1, state, "losses")
    ax2 = plot(ax2, state, "aux_losses")

    plt.suptitle(f"{MODEL_NAME} model — training losses", fontweight="bold", fontsize=16)
    fig.supxlabel(f"{len(state["steps"]):,} steps — {state["training_time"] / 3600:.1f} h of training — no train curve smoothening", y=0.03, color="grey", fontsize=11)

    plt.tight_layout(rect=[0.0, 0.01, 1, 1])

    plt.savefig(OUT_DIR)


if __name__ == "__main__":
    main()