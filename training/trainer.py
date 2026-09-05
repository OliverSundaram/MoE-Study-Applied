from pathlib import Path

import torch

import time

from tqdm import tqdm



class Trainer:
    """
    The base class for training a llm.
    """


    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.hyperparams = kwargs["hyperparams"]
        self.tokenizer = kwargs["tokenizer"]

        self.loss_fn = kwargs["loss_fn"]
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]

        self.train_loader = kwargs["train_loader"]
        self.val_loader = kwargs["val_loader"]
        self.test_loader = kwargs["test_loader"]

        self.device = kwargs["device"]
        self.max_optim_steps = kwargs["max_optim_steps"]
        self.checkpoint_path = kwargs["checkpoint_path"]
        self.val_every_steps = kwargs["val_every_steps"]
        self.val_max_batches = kwargs["val_max_batches"]





    def train(self):

        def save_model(model,  tokenizer, path: str | Path, **state):
            model.save_pretrained(path)
            tokenizer.save_pretrained(path)

            torch.save(
                state,
                path / "trainer_state.pt"
            )

        # -------------------- #
        # Initialize Variables #
        # -------------------- #
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)

        train_losses, val_losses, train_aux_losses, val_aux_losses = [], [], [], []
        optim_steps, val_steps  = [], []

        moe_layers = [m for m in self.model.modules() if isinstance(m, type(self.model.trans_blocks[0].ff))]

        micro_step = 0
        optim_step = 0
        accum_ce_loss = 0.0
        accum_aux_loss = 0.0

        AUX_SCALE = self.hyperparams["AUX_SCALE"]
        MAX_NORM = self.hyperparams["MAX_NORM"]
        GRAD_ACCUM_STEPS = self.hyperparams["GRAD_ACCUM_STEPS"]

        # ------------ #
        # Run Training #
        # ------------ #
        self.model.train()
        self.optimizer.zero_grad()
        start = time.time()
        for input_ids, targets in tqdm(self.train_loader, desc="Training"):
            micro_step += 1

            # ------------------ #
            # Perform Micro-Step #
            # ------------------ #
            input_ids = input_ids.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(input_ids).logits
                ce_loss = self.loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))

            aux_loss = sum(m.aux_loss for m in moe_layers) / len(moe_layers)
            total_loss = (ce_loss + aux_loss * AUX_SCALE) / GRAD_ACCUM_STEPS

            accum_ce_loss += ce_loss.item() / GRAD_ACCUM_STEPS
            accum_aux_loss += aux_loss.item() / GRAD_ACCUM_STEPS

            total_loss.backward()

            # ---------------------- #
            # Perform Optimizer Step #
            # ---------------------- #
            if micro_step % GRAD_ACCUM_STEPS == 0:

                optim_step += 1
                optim_steps.append(optim_step)
                train_losses.append(accum_ce_loss)
                train_aux_losses.append(accum_aux_loss)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), MAX_NORM)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()

                accum_ce_loss, accum_aux_loss = 0.0, 0.0

                # ------------------------- #
                # Validation and Checkpoint #
                # ------------------------- #
                if optim_step % self.val_every_steps == 0:

                    # ---------- #
                    # Validation #
                    # ---------- #
                    total_val_loss = 0.0
                    total_val_aux_loss = 0.0
                    val_batches = 0
                    self.model.eval()
                    with torch.inference_mode():
                        for input_ids, targets in tqdm(self.val_loader, desc="Validation"):
                            input_ids = input_ids.to(self.device, non_blocking=True)
                            targets = targets.to(self.device, non_blocking=True)

                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                                logits = self.model(input_ids).logits
                                ce_loss = self.loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))

                            aux_loss = sum(m.aux_loss for m in moe_layers) / len(moe_layers)

                            total_val_aux_loss += aux_loss.item()
                            total_val_loss += ce_loss.item()

                            val_batches += 1
                            if self.val_max_batches is not None and val_batches >= self.val_max_batches:
                                break

                        val_avg_aux_loss = total_val_aux_loss / val_batches
                        val_avg_loss =  total_val_loss / val_batches

                        val_steps.append(optim_step)
                        val_losses.append(val_avg_loss)
                        val_aux_losses.append(val_avg_aux_loss)

                    # ---------- #
                    # Checkpoint #
                    # ---------- #
                    step_dir = self.checkpoint_path / f"step_{optim_step}"
                    save_model(
                        self.model,
                        self.tokenizer,
                        step_dir,

                        optimizer_state_dict=self.optimizer.state_dict(),
                        scheduler_state_dict=self.scheduler.state_dict() if self.scheduler is not None else None,

                        train_losses=train_losses,
                        val_losses=val_losses,
                        train_aux_losses=train_aux_losses,
                        val_aux_losses=val_aux_losses,

                        step=optim_step,
                        steps=optim_steps,
                        val_steps=val_steps,
                    )

                    print(f"\n[Step {optim_step}] Train Loss: {train_losses[-1]:.4f} (Aux Loss: {train_aux_losses[-1]:.4f}) | Val Loss: {val_avg_loss:.4f} (Aux Loss: {val_avg_aux_loss:.4f})")
                    self.model.train()

                if self.max_optim_steps is not None and optim_step >= self.max_optim_steps:
                    break

        # ------- #
        # Testing #
        # ------- #
        print("\nTraining complete. Running full test evaluation...\n")
        total_test_loss = 0.0
        total_test_aux_loss = 0.0

        self.model.eval()
        with torch.inference_mode():
            for input_ids, targets in tqdm(self.test_loader, desc="Testing"):
                input_ids = input_ids.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(input_ids).logits
                    ce_loss = self.loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))

                    aux_loss = sum(m.aux_loss for m in moe_layers) / len(moe_layers)

                total_test_aux_loss += aux_loss.item()
                total_test_loss += ce_loss.item()

            test_avg_aux_loss = total_test_aux_loss / len(self.test_loader)
            test_avg_loss =  total_test_loss / len(self.test_loader)

        training_time = time.time() - start
        print(f"Test Loss: {test_avg_loss:.4f} (Aux Loss: {test_avg_aux_loss:.4f})| Total time: {training_time:.4f}s")

        # ---------------- #
        # Save final model #
        # ---------------- #
        final_dir = self.checkpoint_path / "final"
        final_dir.mkdir(parents=True, exist_ok=True)

        save_model(
            self.model,
            self.tokenizer,
            final_dir,

            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict() if self.scheduler is not None else None,

            train_losses=train_losses,
            val_losses=val_losses,
            train_aux_losses=train_aux_losses,
            val_aux_losses=val_aux_losses,

            test_loss=test_avg_loss,
            test_aux_loss=test_avg_aux_loss,

            step=optim_step,
            steps=optim_steps,
            val_steps=val_steps,
            training_time=training_time
        )

        return None