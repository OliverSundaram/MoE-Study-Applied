# Imports
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch

import lm_eval
from lm_eval.models.huggingface import HFLM

import json



# Hyperparameters
ROOT = Path.cwd()

MODELS = ["base", "sft"]
MODEL_NAME = "sft"
assert MODEL_NAME in MODELS, f"{MODEL_NAME} is unknown. Please select from {MODELS}"
MODEL_DIR = ROOT.parent / "model" / MODEL_NAME
TOK_DIR = MODEL_DIR
OUTPUT_DIR = ROOT / "results"

BATCH_SIZE = 8
LIMIT = None
device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

GROUPS = {
    0 : ["arc_easy", "piqa", "wikitext", "lambada_openai"],
    5 : ["winogrande", "hellaswag"                       ],
    15: ["arc_challenge"                                 ]
}



# Functions
def main():
    # Loading model and tokenizer
    print(f"Loading model from -> {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)

    torch.manual_seed(SEED)
    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device).eval()

    # Set up lm_eval model
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        device=device,
        max_length=model.config.context_length
    )

    # Running benchmarks
    results = {}
    n_shot = {}

    for fewshot, tasks in GROUPS.items():

        out = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=fewshot,
            limit=LIMIT,
            log_samples=False
        )

        results.update(out["results"])
        n_shot.update({task: fewshot for task in tasks})

        data = {
            "model": MODEL_NAME,
            "model_dir": str(MODEL_DIR),
            "device": device,
            "lm_eval_version": lm_eval.__version__,
            "limit": LIMIT,
            "n_shot": n_shot,
            "results": results
        }

        with open(str(OUTPUT_DIR / f"{MODEL_NAME}_eval.json"), "w") as f:
            json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()