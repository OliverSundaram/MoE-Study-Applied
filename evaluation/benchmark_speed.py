# Imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from datasets import load_dataset
from random import randint



# Hyperparameters
ROOT = Path.cwd()

MODELS = ["base", "sft"]
MODEL = "sft"
assert MODEL in MODELS, f"{MODEL} not known. Please choose from {MODELS}."

MODEL_DIR = ROOT.parent / "model" / MODEL
TOK_DIR = MODEL_DIR
DATA_DIR = "EdinburghNLP/xsum"
OUT_DIR = ROOT / "results"

MAX_NEW_TOKENS = 500
CONTEXT_LENGTH = 1024
TOP_K = 50
TEMP = 0.8
USE_CACHED = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = randint(0, 1000)
SPLIT = "test"
assert SPLIT in ["train", "validation", "test"], f"{SPLIT} split unknown."
PRINT_TEXT = True



# Functions
def get_random_prompt(DATA_DIR, SEED, SPLIT, tokenizer, context_length, MAX_NEW_TOKENS):
    ds = iter(load_dataset(DATA_DIR, split=SPLIT).shuffle(SEED))

    while True:
        prompt = next(ds)["document"]
        formatted_prompt = "<|user|>" + prompt + "<|end|>" + "<|assistant|>"
        ids = tokenizer(formatted_prompt).input_ids
        if len(ids) <= context_length - MAX_NEW_TOKENS:
            break

    return prompt


def main():

    print(f"Getting model from: {MODEL_DIR}")
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)

    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    prompt = get_random_prompt(DATA_DIR, SEED, SPLIT, tokenizer, CONTEXT_LENGTH, MAX_NEW_TOKENS)
    if PRINT_TEXT:
        print("Prompt:")
        print("-------------")
        print(prompt)
        print()
        print("Summary:")
        print("-------------")

    result, tok_per_sec, time_to_first_tok = model.generate(prompt, tokenizer, device, max_new_tokens=MAX_NEW_TOKENS, top_k=TOP_K, temp=TEMP, use_cached=USE_CACHED)
    if PRINT_TEXT:
        print()
        print("-------------")
        print(tok_per_sec)
        print(time_to_first_tok)

    torch.save(
        {
            "prompt": prompt,
            "result": result,
            "tok_per_sec": tok_per_sec,
            "time_to_first_tok": time_to_first_tok
        },
        OUT_DIR / f"{MODEL}_model_{"" if USE_CACHED else "no"}_cached.pt"
    )




if __name__ == "__main__":
    main()