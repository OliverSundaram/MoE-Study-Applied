import torch
from torch import nn
from torch.utils.data import DataLoader

from training.trainer import Trainer

from transformers import AutoTokenizer, AutoModelForCausalLM

from preparation.dataset import get_dataset

from pathlib import Path

ROOT = Path.cwd().parent



def main():
    # --------------- #
    # Hyperparameters #
    # --------------- #
    AUX_SCALE = 0.01
    CHECKPOINT_PATH = ROOT / "runs" / "sft"
    BATCH_SIZE = 2
    LR_RATE = 5e-5
    WEIGHT_DECAY = 0.1
    MAX_NORM = 1.0
    GRAD_ACCUM_STEPS = 16
    VAL_EVERY_STEPS = 700
    NUM_WORKERS = 2
    SEED = 42
    VAL_MAX_BATCHES = 250
    MAX_OPTIM_STEPS = None
    TOK_DIR = ROOT / "model" / "base"
    MODEL_DIR = ROOT / "model" / "base"

    hyperparams = {
        "AUX_SCALE": AUX_SCALE,
        "BATCH_SIZE": BATCH_SIZE,
        "LR_RATE": LR_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "MAX_NORM": MAX_NORM,
        "GRAD_ACCUM_STEPS": GRAD_ACCUM_STEPS,
    }

    # ----------------------- #
    # Device Selection & Seed #
    # ----------------------- #
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU required for this training run, but none was detected.")
    print(f"Using device: {device}")

    # Optimize matrix multiplication on Tensor Cores is available
    torch.set_float32_matmul_precision("high")

    # ----------- #
    # DataLoaders #
    # ----------- #
    print("Loading Datasets...")

    train_ds = get_dataset("train")
    val_ds = get_dataset("val")
    test_ds = get_dataset("test")

    train_loader = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=False)
    test_loader = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=False)

    # -------------------- #
    # Model Initialization #
    # -------------------- #
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(TOK_DIR)

    model.resize_token_embeddings(len(tokenizer))
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model = model.to(device)

    # ---------------------------- #
    # Loss, Optimizer, & Scheduler #
    # ---------------------------- #
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=LR_RATE, eps=1e-8, fused=True)

    loss_fn = nn.CrossEntropyLoss()

    total_optim_steps = len(train_loader) // GRAD_ACCUM_STEPS
    if MAX_OPTIM_STEPS is not None:
        total_optim_steps = min(total_optim_steps, MAX_OPTIM_STEPS)
    warmup_steps = max(2, int(0.03 * total_optim_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR_RATE,
        total_steps=total_optim_steps,
        pct_start=warmup_steps / total_optim_steps,
        anneal_strategy="cos",
        cycle_momentum=False
    )
    # ----------------- #
    # Run Training Loop #
    # ----------------- #
    print("Starting training...")

    trainer = Trainer(
        model=model,
        hyperparams=hyperparams,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        max_optim_steps=total_optim_steps,
        checkpoint_path=CHECKPOINT_PATH,
        val_every_steps=VAL_EVERY_STEPS,
        val_max_batches=VAL_MAX_BATCHES,
    )

    trainer.train()

if __name__ == "__main__":
    main()