"""Time-series forecasting with MLP, Classical Transformer, and QASA."""

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from config import DataConfig, ModelConfig, TrainConfig
from data import build_dataloaders
from models import build_model
from training import EarlyStopping, run_epoch
from utils import is_local, save_json, set_seed, setup_logging


def _apply_backend_flag(backend: str | None) -> None:
    if backend == "ibm":
        os.environ["RUN_LOCAL"] = "false"
    elif backend == "local":
        os.environ["RUN_LOCAL"] = "true"


def train(data_cfg: DataConfig, model_cfg: ModelConfig, train_cfg: TrainConfig) -> dict:
    """Train a model from scratch and evaluate on the test set."""
    set_seed(train_cfg.seed)
    output_dir = Path(train_cfg.output_dir)
    setup_logging(output_dir)

    backend = "local" if is_local() else f"IBM ({os.environ.get('IBM_BACKEND', 'ibm_sherbrooke')})"
    logging.info(f"Model: {model_cfg.model_name} | Backend: {backend}")

    train_loader, val_loader, test_loader = build_dataloaders(data_cfg, train_cfg)
    device = torch.device(train_cfg.device)
    model = build_model(data_cfg.window_size, model_cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Parameters: {n_params} | Device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.scheduler_tmax
    )
    stopper = EarlyStopping(train_cfg.early_stopping_patience)

    save_json(
        output_dir / "config.json",
        {"data": asdict(data_cfg), "model": asdict(model_cfg), "train": asdict(train_cfg)},
    )

    best_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, _ = run_epoch(
            model, train_loader, optimizer, device, train_cfg.grad_clip, train_cfg.log_every
        )
        val_loss, val_m = run_epoch(model, val_loader, None, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_r2": val_m["r2"],
        })
        logging.info(
            f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} r2={val_m['r2']:.3f}"
        )

        if stopper.step(val_loss):
            torch.save({"model_state": model.state_dict()}, best_path)
        if stopper.should_stop:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    save_json(output_dir / "history.json", {"history": history})

    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    test_loss, test_m = run_epoch(model, test_loader, None, device)
    logging.info(f"TEST: loss={test_loss:.4f} r2={test_m['r2']:.3f}")
    save_json(output_dir / "test_metrics.json", {"test_loss": test_loss, **test_m})

    return test_m


def eval_on_ibm(local_output_dir: str, ibm_output_dir: str, max_samples: int = 64) -> dict:
    """
    Load weights trained locally, rebuild the model using the IBM backend,
    and evaluate only the test set on IBM hardware.
    """
    local_dir = Path(local_output_dir)
    ibm_dir   = Path(ibm_output_dir)
    ibm_dir.mkdir(parents=True, exist_ok=True)

    # Load the config that was used during local training
    cfg_path = local_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"No config.json in {local_dir} — run 'just run-all' first to train locally."
        )
    with open(cfg_path) as f:
        cfg_dict = json.load(f)

    data_cfg  = DataConfig(**cfg_dict["data"])
    model_cfg = ModelConfig(**cfg_dict["model"])
    train_cfg = TrainConfig(**cfg_dict["train"])

    setup_logging(ibm_dir)
    set_seed(train_cfg.seed)

    ibm_backend = os.environ.get("IBM_BACKEND", "ibm_sherbrooke")
    logging.info(f"Eval-only | Model: {model_cfg.model_name} | Backend: IBM ({ibm_backend})")
    logging.info("Building model with IBM quantum device — connecting to IBM Quantum Platform...")

    # Build model: quantum circuit now uses IBM hardware device
    device = torch.device(train_cfg.device)
    model = build_model(data_cfg.window_size, model_cfg).to(device)

    # Load the locally-trained weights
    best_path = local_dir / "best_model.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No best_model.pt in {local_dir} — did training complete?")
    model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
    logging.info("Loaded locally-trained weights.")

    # Only need the test loader, capped to max_samples to limit IBM job count
    _, _, test_loader = build_dataloaders(data_cfg, train_cfg)
    test_dataset = test_loader.dataset
    n_test = len(test_loader.dataset)  # type: ignore[arg-type]
    if max_samples < n_test:
        test_dataset = Subset(test_dataset, list(range(max_samples)))
        logging.info(f"Capping IBM eval to {max_samples} samples (full test set: {n_test})")
    test_loader = DataLoader(test_dataset, batch_size=max_samples, shuffle=False)

    logging.info("Running test set forward pass on IBM hardware (this may take a while)...")
    test_loss, test_m = run_epoch(model, test_loader, None, device)

    logging.info(f"IBM TEST: loss={test_loss:.4f} r2={test_m['r2']:.3f}")
    save_json(ibm_dir / "test_metrics.json", {"test_loss": test_loss, **test_m})
    logging.info(f"IBM results saved to {ibm_dir}/")

    return test_m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Tiny dataset, all models in ~1 min")
    p.add_argument("--fast", action="store_true", help="Small dataset, fast settings (~3-5 min)")
    p.add_argument("--full", action="store_true", help="Full dataset, better settings (~10-20 min)")
    p.add_argument("--backend", choices=["local", "ibm"], default=None)
    p.add_argument("--dataset-type", default="mixed", choices=["sine", "damped", "chirp", "mixed"])
    p.add_argument("--model", default="qasa_transformer",
                   choices=["mlp", "single_qubit", "classical_transformer", "qasa_transformer"])
    p.add_argument("--n-qubits", type=int, default=4)
    p.add_argument("--q-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)

    # Eval-only mode: skip training, load local weights, run test set on IBM
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training — load weights from --local-dir and eval on IBM hardware")
    p.add_argument("--local-dir", default=None,
                   help="Path to local training output (used with --eval-only)")
    p.add_argument("--ibm-samples", type=int, default=64,
                   help="Max samples to send to IBM hardware (one batch = one job)")

    args = p.parse_args()
    _apply_backend_flag(args.backend)

    if args.eval_only:
        if not args.local_dir:
            p.error("--eval-only requires --local-dir pointing to a completed local training run")
        if not args.output_dir:
            p.error("--eval-only requires --output-dir for IBM results")
        eval_on_ibm(args.local_dir, args.output_dir, max_samples=args.ibm_samples)
        return

    if args.demo:
        output_dir = args.output_dir or "./outputs/comparison/demo/local"
        data_cfg = DataConfig(
            num_series=40, window_size=8, series_length=32,
            batch_size=32, dataset_type=args.dataset_type,
        )
        model_cfg = ModelConfig(
            model_name=args.model, n_qubits=2, q_layers=1,
            d_model=16, num_heads=2, num_layers=2,
        )
        train_cfg = TrainConfig(
            epochs=5, lr=0.05 if args.model == "single_qubit" else args.lr,
            seed=args.seed, output_dir=output_dir, scheduler_tmax=5,
        )

    elif args.fast:
        output_dir = args.output_dir or "./outputs/comparison/fast/local"
        data_cfg = DataConfig(
            num_series=200, window_size=12, series_length=60,
            batch_size=32, dataset_type=args.dataset_type,
        )
        model_cfg = ModelConfig(
            model_name=args.model, n_qubits=2, q_layers=1,
            d_model=32, num_heads=2, num_layers=2,
        )
        # Single-qubit needs higher lr and more epochs due to barren plateau
        if args.model == "single_qubit":
            train_cfg = TrainConfig(
                epochs=30, lr=0.05, seed=args.seed,
                output_dir=output_dir, scheduler_tmax=30,
            )
        else:
            train_cfg = TrainConfig(
                epochs=10, lr=args.lr, seed=args.seed,
                output_dir=output_dir, scheduler_tmax=10,
            )

    elif args.full:
        output_dir = args.output_dir or "./outputs/comparison/full/local"
        data_cfg = DataConfig(
            num_series=2400, window_size=24, series_length=160,
            batch_size=32, dataset_type=args.dataset_type,
        )
        model_cfg = ModelConfig(
            model_name=args.model, n_qubits=4, q_layers=2,
            d_model=64, num_heads=4, num_layers=3,
        )
        if args.model == "single_qubit":
            train_cfg = TrainConfig(
                epochs=60, lr=0.05, seed=args.seed,
                output_dir=output_dir, scheduler_tmax=60,
            )
        elif args.model in ("classical_transformer", "qasa_transformer"):
            train_cfg = TrainConfig(
                epochs=30, lr=args.lr, seed=args.seed,
                output_dir=output_dir, scheduler_tmax=30,
            )
        else:
            train_cfg = TrainConfig(
                epochs=20, lr=args.lr, seed=args.seed,
                output_dir=output_dir, scheduler_tmax=20,
            )

    else:
        output_dir = args.output_dir or f"./outputs/local/{args.model}"
        data_cfg = DataConfig(dataset_type=args.dataset_type)
        model_cfg = ModelConfig(
            model_name=args.model, n_qubits=args.n_qubits, q_layers=args.q_layers
        )
        train_cfg = TrainConfig(
            epochs=args.epochs, lr=args.lr, seed=args.seed,
            output_dir=output_dir, scheduler_tmax=args.epochs,
        )

    train(data_cfg, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
