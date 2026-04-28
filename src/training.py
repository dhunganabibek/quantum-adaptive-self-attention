"""Training utilities and metrics."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute regression metrics."""
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(1 - ((target - pred) ** 2).sum() / (ss_tot + 1e-8))
    return {"mse": mse, "mae": mae, "rmse": math.sqrt(mse), "r2": r2}


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        """Returns True if new best."""
        if metric < self.best:
            self.best, self.counter = metric, 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float | None = None,
    log_every: int | None = None,
) -> tuple[float, dict[str, float]]:
    """Run one epoch of training or evaluation."""
    is_train = optimizer is not None
    model.train(is_train)

    losses, preds, targets = [], [], []

    for step, (x, y) in enumerate(loader, 1):
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = F.mse_loss(pred, y)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        losses.append(loss.item())
        preds.append(pred.detach().cpu())
        targets.append(y.detach().cpu())

        if is_train and log_every and step % log_every == 0:
            import logging

            logging.info(f"  step={step}  loss={float(np.mean(losses)):.4f}")

    return float(np.mean(losses)), compute_metrics(torch.cat(preds), torch.cat(targets))
