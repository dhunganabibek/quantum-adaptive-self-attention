"""
QASA — Quantum Adaptive Self-Attention
Time-series forecasting with hybrid quantum-classical models.
"""

import argparse
import logging
import os
from dataclasses import asdict
from pathlib import Path

import torch

from config import DataConfig, ModelConfig, TrainConfig
from data import build_dataloaders
from models import build_model
from training import EarlyStopping, run_epoch
from utils import is_local, save_json, set_seed, setup_logging


def train(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig):
    """Run training experiment."""
    set_seed(train_cfg.seed)
    output_dir = Path(train_cfg.output_dir)
    setup_logging(output_dir)

    backend = "local" if is_local() else f"IBM ({os.environ.get('IBM_BACKEND', 'ibm_brisbane')})"
    logging.info("=" * 60)
    logging.info(f"QASA | Model: {model_cfg.model_name} | Backend: {backend}")
    logging.info("=" * 60)

    # Build dataset and model
    train_loader, val_loader, test_loader = build_dataloaders(data_cfg, train_cfg)
    device = torch.device(train_cfg.device)
    model = build_model(data_cfg.window_size, model_cfg).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parameters: {n_params} | Device: {device}")

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg.scheduler_tmax)
    stopper = EarlyStopping(train_cfg.early_stopping_patience)

    save_json(output_dir / "config.json", {
        "data": asdict(data_cfg),
        "model": asdict(model_cfg),
        "train": asdict(train_cfg)
    })

    best_path = output_dir / "best_model.pt"
    history = []

    # Training loop
    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, optimizer, device,
                                  train_cfg.grad_clip, train_cfg.log_every)
        val_loss, val_m = run_epoch(model, val_loader, None, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_r2": val_m["r2"]
        })
        logging.info(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} r2={val_m['r2']:.3f}")

        if stopper.step(val_loss):
            torch.save({"model_state": model.state_dict()}, best_path)
        if stopper.should_stop:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    save_json(output_dir / "history.json", {"history": history})

    # Test evaluation
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    test_loss, test_m = run_epoch(model, test_loader, None, device)
    logging.info(f"TEST: loss={test_loss:.4f} r2={test_m['r2']:.3f}")
    save_json(output_dir / "test_metrics.json", {"test_loss": test_loss, **test_m})

    return test_m


def main():
    """Main entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Quick demo (5 epochs, small data)")
    
    # Data
    p.add_argument("--dataset-type", default="mixed", choices=["sine", "damped", "chirp", "mixed"])
    p.add_argument("--num-series", type=int, default=2400)
    p.add_argument("--window-size", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=32)
    
    # Model
    p.add_argument("--model", default="qasa_transformer",
                   choices=["classical_baseline", "single_qubit", "classical_transformer", "qasa_transformer"])
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--q-layers", type=int, default=3)
    
    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    
    args = p.parse_args()

    if args.demo:
        # Quick demo settings
        data_cfg = DataConfig(num_series=100, window_size=12, series_length=60,
                             batch_size=16, dataset_type="sine")
        model_cfg = ModelConfig(model_name=args.model, n_qubits=2, q_layers=1,
                               d_model=32, num_heads=2, num_classical_layers=1)
        train_cfg = TrainConfig(epochs=5, output_dir="./outputs/demo", log_every=5)
    else:
        # Full training settings
        output_dir = args.output_dir or f"./outputs/{'local' if is_local() else 'ibm'}"
        data_cfg = DataConfig(num_series=args.num_series, window_size=args.window_size,
                             batch_size=args.batch_size, dataset_type=args.dataset_type)
        model_cfg = ModelConfig(model_name=args.model, n_qubits=args.n_qubits, q_layers=args.q_layers)
        train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr, seed=args.seed,
                               output_dir=output_dir, scheduler_tmax=args.epochs)

    train(data_cfg, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
