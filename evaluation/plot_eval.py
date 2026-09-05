# Imports
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np

import json

from sympy.printing.pretty.pretty_symbology import line_width

# Hyperparameters
ROOT = Path.cwd()

RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"

MODELS = ["base", "sft"]

MAIN_METRICS = {
    "arc_easy": "acc_norm,none",
    "piqa": "acc_norm,none",
    "wikitext": "word_perplexity,none",
    "lambada_openai": "acc,none",
    "winogrande": "acc,none",
    "hellaswag": "acc_norm,none",
    "arc_challenge": "acc_norm,none"
}

METRIC_CHANCE = {
    "arc_easy": 0.25,
    "piqa": 0.50,
    "wikitext": None,
    "lambada_openai": 0.00,
    "winogrande": 0.50,
    "hellaswag": 0.25,
    "arc_challenge": 0.25
}

WIDTH = 0.7
NUM_TICKS = 7



# Functions
def load_eval(results_dir) -> tuple[dict, dict]:
    """
    Opens the base and sft eval.json files from *results_dir*, loads them with `json`, and returns both.
    :param results_dir: The directory where the base and sft model results are.
    :return:
    """
    with (
        open(f"{str(results_dir / "base")}_eval.json", "r") as f1,
        open(f"{str(results_dir / "sft")}_eval.json", "r") as f2,
    ):
        base_eval = json.load(f1)
        sft_eval = json.load(f2)

    return base_eval, sft_eval


def get_info(eval_file) -> list[tuple[str | int | float]]:
    """
    Goes through each benchmark and extracts the necessary information
    (e.g., main metric and main metric results), and returns a list with
    a tuple for each benchmark.
     Expects *eval_file["results"]* to return a dict.
    :param eval_file: The read eval results using json.
    :return:
    """
    extracted_info = []

    benchmark_count = -1
    for benchmark, result in eval_file["results"].items():
        benchmark_count += 1

        fewshot = eval_file["n_shot"][benchmark]

        main_metric = MAIN_METRICS[benchmark]
        main_metric_result = result[main_metric]
        sample_len = result["sample_len"]

        extracted_info.append((benchmark, fewshot, main_metric, main_metric_result, sample_len))

    return extracted_info


def get_avg(x, y):
    return round((x + y) / 2, 2)


def plot(base_info, sft_info):
    for i in range(len(base_info)):


        # Unpack results for each model
        benchmark, few_shot, main_metric, main_metric_result_base, sample_len = base_info[i]
        benchmark, few_shot, main_metric, main_metric_result_sft, sample_len = sft_info[i]


        SAVE_PATH = ASSETS_DIR / f"{benchmark}_eval.png"


        fig, ax = plt.subplots(figsize=(8, 8))

        # Create bars
        x = (0, 1)
        bar1 = ax.bar(
            x=x[0],
            height=main_metric_result_base,
            width=WIDTH,
            color="#4C72B0",
            edgecolor="white",
            linewidth=2,
            label="base"
        )
        bar2 = ax.bar(
            x[1],
            height=main_metric_result_sft,
            width=WIDTH,
            color="#DD8452",
            edgecolor="white",
            linewidth=2,
            label="sft"
        )


        # Add chance line
        chance = METRIC_CHANCE[benchmark]
        if chance is not None:
            ax.axhline(chance, linewidth=2, color="grey", linestyle="--", label=f"chance ({chance})")


        # Add values op top of bars
        ax.bar_label(bar1, fmt="%.4f", padding=3, color="grey")
        ax.bar_label(bar2, fmt="%.4f", padding=3, color="grey")


        # Add text

        # Reformat y ticks to make bars shorter to fit legend at top right
        # Make the top value

        # Multiply by two to make the top value double the bars, then multiply by 0.8 to make bars slightly bigger
        top_value = get_avg(main_metric_result_base, main_metric_result_sft) * 2 * 0.8
        # Multiply by 100 to get ints, since range doesn't use floats. Divide by 100 once nums are generated
        top_value *= 100
        # Convert to int. Top value is int but with trailing .0
        top_value = int(top_value)

        # Make the step value
        # Divide top value by num ticks to get step
        step = int(top_value / NUM_TICKS)

        ax.set_yticks(
            np.array(range(
                0,
                top_value,
                step
            )) / 100
        )

        if main_metric == "acc,none" or main_metric == "acc_norm,none":
            name = "accuracy" if main_metric == "acc,none" else "normalized accuracy"
            ax.set_title(
                label=f"{name} (higher is better)",
                fontsize=12,
                fontweight="bold"
            )

        else:
            name = "word perplexity"
            ax.set_title(
                label=f"{name} (lower is better)",
                fontsize=12,
                fontweight="bold"
            )


        ax.set_xlabel("Model")
        ax.set_ylabel(name)
        fig.suptitle(t=f"{benchmark} — base vs. sft", fontsize=18, fontweight="bold")
        fig.supxlabel(f"{few_shot}-shot — {sample_len} samples", fontsize=12, color="grey")


        # Set x labels
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS)


        # Make grid for decoration
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.5, color="grey")


        plt.legend(loc="upper right")

        plt.savefig(SAVE_PATH)


def main():
    print(f"Loading results from -> {RESULTS_DIR}")
    base_eval, sft_eval = load_eval(RESULTS_DIR)

    base_info = get_info(base_eval)
    sft_info = get_info(sft_eval)

    plot(base_info, sft_info)


if __name__ == "__main__":
    main()