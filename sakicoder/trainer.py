import math
from typing import Optional
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from pathlib import Path
from sakicoder.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(self, model, config, device, mixed_precision=False):
        self.model = model
        self.config = config
        self.device = device
        self.mixed_precision = mixed_precision

    def train(self, train_data: torch.Tensor, val_data: Optional[torch.Tensor], tokenizer, out_dir: str, max_steps: int = 1000, grad_accum_steps: int = 1, resume_checkpoint: Optional[str] = None):
        if train_data is None or len(train_data) == 0:
            raise ValueError("Training data is empty")

        model = self.model.to(self.device)
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate)

        dataset = TensorDataset(torch.from_numpy(train_data).long())
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

        global_step = 0
        scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        log_token_count = 0
        log_start = time.time()

        if resume_checkpoint:
            global_step = load_checkpoint(model, optimizer, resume_checkpoint, device=str(self.device))
            print(f"Resumed from checkpoint {resume_checkpoint} at step {global_step}")

        for epoch in range(999999):
            for batch in loader:
                x = batch[0].to(self.device)
                inputs = x[:, :-1]
                targets = x[:, 1:]
                log_token_count += int(inputs.numel())
                model.train()
                optimizer.zero_grad()
                logits, loss = model(inputs, targets)
                loss = loss / grad_accum_steps
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                global_step += 1
                if global_step % 10 == 0:
                    elapsed = max(time.time() - log_start, 1e-6)
                    tps = log_token_count / elapsed
                    print(f"step {global_step} loss {loss.item():.4f} tok/s {tps:.1f}")
                    log_token_count = 0
                    log_start = time.time()
                if global_step % 50 == 0:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    save_checkpoint(model, optimizer, global_step, f"{out_dir}/step-{global_step}.pt")
                    # Also save a copy as latest.pt for easy loading by scripts
                    save_checkpoint(model, optimizer, global_step, f"{out_dir}/latest.pt")
                if global_step >= max_steps:
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    save_checkpoint(model, optimizer, global_step, f"{out_dir}/latest.pt")
                    return
