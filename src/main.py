"""
QASA — Quantum Adaptive Self-Attention
Time-series forecasting with a hybrid quantum-classical Transformer.

Models
------
single_qubit       1-qubit data-reuploading regressor.
qasa_transformer   Classical Transformer + quantum-enhanced final encoder block.


Quick demo
---------------------------------------
    python src/main.py --demo
    python src/main.py --demo --model qasa_transformer

Full training
-------------
    python src/main.py --model single_qubit --epochs 20
    python src/main.py --model qasa_transformer --epochs 30 --n-qubits 4
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

try:
    import pennylane as qml
except ImportError as exc:
    raise ImportError(
        "PennyLane is required. Install with:\n  pip install pennylane pennylane-lightning"
    ) from exc


def _load_dotenv(path: Path) -> None:
    """Load a .env file into os.environ without overwriting existing variables."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _is_local() -> bool:
    return os.environ.get("RUN_LOCAL", "true").strip().lower() not in ("false", "0", "no")


def make_device(n_wires: int) -> tuple:
    """
    Return (qml.device, label_str).

    RUN_LOCAL=true  → default.qubit  (no credentials needed)
    RUN_LOCAL=false → qiskit.ibmq    (needs IBM_QUANTUM_TOKEN in .env)
    """
    if _is_local():
        dev = qml.device("default.qubit", wires=n_wires)
        return dev, "local (default.qubit)"

    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    backend = os.environ.get("IBM_BACKEND", "ibm_brisbane")
    if not token:
        raise OSError(
            "IBM_QUANTUM_TOKEN must be set in .env when RUN_LOCAL=false.\n"
            "Get a free token at https://quantum.ibm.com"
        )

    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Install the IBM Quantum plugin:\n"
            "  pip install pennylane-qiskit qiskit-ibm-runtime"
        ) from exc

    dev = qml.device("qiskit.ibmq", wires=n_wires, backend=backend, ibmqx_token=token)
    return dev, f"IBM Quantum ({backend})"


# Config
@dataclass
class DataConfig:
    num_series: int = 2400
    window_size: int = 24
    horizon: int = 1
    series_length: int = 160
    noise_std: float = 0.03
    dataset_type: str = "mixed"   # mixed | sine | damped | chirp
    batch_size: int = 32
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    normalize_per_series: bool = True


@dataclass
class ModelConfig:
    model_name: str = "qasa_transformer"   # single_qubit | qasa_transformer
    d_model: int = 64
    num_heads: int = 4
    num_classical_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.10
    n_qubits: int = 4
    q_layers: int = 3
    use_timestep_conditioning: bool = True


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    num_workers: int = 0
    output_dir: str = "./outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_every: int = 20
    scheduler_tmax: int = 20


# Demo preset — small data
DEMO_DATA = DataConfig(num_series=100, window_size=12, series_length=60, batch_size=16, dataset_type="sine")
DEMO_MODEL = ModelConfig(model_name="single_qubit", n_qubits=2, q_layers=1, d_model=32, num_heads=2, num_classical_layers=1)
DEMO_TRAIN = TrainConfig(epochs=5, early_stopping_patience=5, log_every=5, output_dir="./outputs/demo")


# Utilities
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# Dataset
class WindowedForecastDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / (std if std > 1e-8 else 1.0)


def _sine(length: int, noise: float) -> np.ndarray:
    t = np.linspace(0, 1, length)
    freq = np.random.uniform(1, 6)
    y = np.random.uniform(0.6, 1.4) * np.sin(2 * math.pi * freq * t + np.random.uniform(0, 2 * math.pi))
    return y + np.random.normal(scale=noise, size=length)


def _damped(length: int, noise: float) -> np.ndarray:
    t = np.linspace(0, 4, length)
    y = np.random.uniform(0.8, 1.5) * np.exp(-np.random.uniform(0.1, 0.7) * t) * np.cos(
        np.random.uniform(4, 10) * t + np.random.uniform(0, 2 * math.pi)
    )
    return y + np.random.normal(scale=noise, size=length)


def _chirp(length: int, noise: float) -> np.ndarray:
    t = np.linspace(0, 1, length)
    f0, f1 = np.random.uniform(1, 3), np.random.uniform(8, 14)
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) * t * t)
    return np.sin(phase) + np.random.normal(scale=noise, size=length)


def _mixed(length: int, noise: float) -> np.ndarray:
    gens = [_sine, _damped, _chirp]
    a, b = np.random.choice(gens, 2, replace=False)
    series = 0.6 * a(length, noise) + 0.4 * b(length, noise)
    jump = np.random.randint(length // 5, 4 * length // 5)
    series[jump:] += np.random.uniform(-0.4, 0.4)
    return series


_GENERATORS: dict[str, Callable] = {"sine": _sine, "damped": _damped, "chirp": _chirp, "mixed": _mixed}


def build_dataset(cfg: DataConfig) -> WindowedForecastDataset:
    gen = _GENERATORS[cfg.dataset_type]
    xs, ys = [], []
    for _ in range(cfg.num_series):
        s = gen(cfg.series_length, cfg.noise_std)
        if cfg.normalize_per_series:
            s = _zscore(s)
        max_start = cfg.series_length - cfg.window_size - cfg.horizon + 1
        for start in range(max_start):
            end = start + cfg.window_size
            xs.append(s[start:end].astype(np.float32))
            ys.append(float(s[end + cfg.horizon - 1]))
    return WindowedForecastDataset(np.stack(xs), np.array(ys, dtype=np.float32))


def build_dataloaders(data_cfg: DataConfig, train_cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = build_dataset(data_cfg)
    n = len(dataset)
    n_train = int(n * data_cfg.train_ratio)
    n_val = int(n * data_cfg.val_ratio)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )
    kw = dict(batch_size=data_cfg.batch_size, num_workers=train_cfg.num_workers,
              pin_memory=train_cfg.device.startswith("cuda"))
    return (
        DataLoader(train_set, shuffle=True, **kw),
        DataLoader(val_set, shuffle=False, **kw),
        DataLoader(test_set, shuffle=False, **kw),
    )



# Classical model components
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassicalEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_mult, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        return self.norm2(x + self.ffn(x))



# Quantum layers
class SingleQubitReuploadCell(nn.Module):
    """
    1-qubit data re-uploading regressor.

    For each timestep t:  RX(x_t) → RY(θ) → RZ(φ)
    Final measurement: <Z> passed through a linear readout.

    PennyLane broadcasts over the batch dimension natively, so no Python loop
    over batch items.
    """

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.theta = nn.Parameter(0.01 * torch.randn(window_size, 2))
        self.readout = nn.Linear(1, 1)
        nn.init.zeros_(self.readout.bias)

        dev, label = make_device(1)
        logging.info("SingleQubitReuploadCell: backend = %s", label)

        @qml.qnode(dev, interface="torch", diff_method="best")
        def _circuit(inputs: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
            for t in range(window_size):
                qml.RX(inputs[..., t], wires=0)
                qml.RY(theta[t, 0], wires=0)
                qml.RZ(theta[t, 1], wires=0)
            return qml.expval(qml.PauliZ(0))

        self._circuit = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = self._circuit(x, self.theta).to(x.dtype).unsqueeze(-1)  # [B, 1]
        return self.readout(q_out).squeeze(-1)                           # [B]


class QuantumTokenProjection(nn.Module):
    """
    Per-token quantum projection inside the hybrid encoder block.

    Pipeline per token:
      x ∈ R^d  →  Linear(d → n_qubits)  →  PQC  →  Linear(n_qubits → d)

    The PQC uses data re-uploading (RX+RZ per qubit per layer), learnable
    RY+RZ rotations, and circular CNOT entanglement.
    """

    def __init__(self, d_model: int, n_qubits: int, q_layers: int, dropout: float,
                 use_timestep_conditioning: bool):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layers = q_layers
        self.use_timestep_conditioning = use_timestep_conditioning

        self.in_proj = nn.Linear(d_model, n_qubits)
        self.in_norm = nn.LayerNorm(n_qubits)
        self.out_proj = nn.Linear(n_qubits, d_model)
        self.drop = nn.Dropout(dropout)
        self.q_weights = nn.Parameter(0.05 * torch.randn(q_layers, n_qubits, 2))

        dev, label = make_device(n_qubits)
        logging.info("QuantumTokenProjection: backend = %s", label)

        @qml.qnode(dev, interface="torch", diff_method="best")
        def _circuit(features: torch.Tensor, weights: torch.Tensor) -> list[torch.Tensor]:
            for layer in range(q_layers):
                qml.AngleEmbedding(features, wires=range(n_qubits), rotation="X")
                qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Z")
                for q in range(n_qubits):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)
                if n_qubits > 1:
                    for q in range(n_qubits):
                        qml.CNOT(wires=[q, (q + 1) % n_qubits])
            return [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)]

        self._circuit = _circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        hq = torch.tanh(self.in_proj(x))
        hq = self.in_norm(hq)

        if self.use_timestep_conditioning:
            t = torch.linspace(0, 1, seq_len, device=x.device, dtype=x.dtype)
            hq = hq + t.unsqueeze(0).unsqueeze(-1)

        flat = hq.reshape(batch_size * seq_len, self.n_qubits)
        q_out = self._circuit(flat, self.q_weights)          # list of [B*L]
        q_out = torch.stack(q_out, dim=-1)                   # [B*L, n_qubits]
        q_out = q_out.view(batch_size, seq_len, self.n_qubits).to(x.dtype)
        return x + self.drop(self.out_proj(q_out))


class QuantumEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_mult: int, dropout: float,
                 n_qubits: int, q_layers: int, use_timestep_conditioning: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.quantum = QuantumTokenProjection(d_model, n_qubits, q_layers, dropout, use_timestep_conditioning)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_mult, dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(self.quantum(x))
        return self.norm3(x + self.ffn(x))



# Top-level models
class QASATransformerRegressor(nn.Module):
    def __init__(self, window_size: int, cfg: ModelConfig):
        super().__init__()
        self.window_size = window_size
        self.embed = nn.Sequential(
            nn.Linear(1, cfg.d_model), nn.LayerNorm(cfg.d_model), nn.Dropout(cfg.dropout),
        )
        self.pos_enc = PositionalEncoding(cfg.d_model, max_len=max(4096, window_size + 8))
        self.classical_layers = nn.ModuleList([
            ClassicalEncoderBlock(cfg.d_model, cfg.num_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.num_classical_layers)
        ])
        self.quantum_layer = QuantumEncoderBlock(
            d_model=cfg.d_model, num_heads=cfg.num_heads, ff_mult=cfg.ff_mult,
            dropout=cfg.dropout, n_qubits=cfg.n_qubits, q_layers=cfg.q_layers,
            use_timestep_conditioning=cfg.use_timestep_conditioning,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.embed(x.unsqueeze(-1)))
        for layer in self.classical_layers:
            h = layer(h)
        h = self.quantum_layer(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def build_model(data_cfg: DataConfig, model_cfg: ModelConfig) -> nn.Module:
    if model_cfg.model_name == "single_qubit":
        return SingleQubitReuploadCell(data_cfg.window_size)
    if model_cfg.model_name == "qasa_transformer":
        return QASATransformerRegressor(data_cfg.window_size, model_cfg)
    raise ValueError(f"Unknown model: {model_cfg.model_name!r}")



# Training
class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        if metric < self.best:
            self.best, self.counter = metric, 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = float(1 - ((target - pred) ** 2).sum() / (ss_tot + 1e-8))
    return {"mse": mse, "mae": mae, "rmse": math.sqrt(mse), "r2": r2}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float | None = None,
    log_every: int | None = None,
) -> tuple[float, dict[str, float]]:
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
            logging.info("  step=%d  loss=%.4f", step, float(np.mean(losses)))

    return float(np.mean(losses)), _metrics(torch.cat(preds), torch.cat(targets))



# Main experiment
def train_experiment(
    data_cfg: DataConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    is_demo: bool = False,
) -> dict[str, float]:
    set_seed(train_cfg.seed)
    output_dir = Path(train_cfg.output_dir)
    setup_logging(output_dir)

    backend_label = ("local (default.qubit)" if _is_local()
                     else f"IBM Quantum ({os.environ.get('IBM_BACKEND', 'ibm_brisbane')})")

    logging.info("=" * 55)
    logging.info("  QASA — Quantum Adaptive Self-Attention%s", "  [DEMO]" if is_demo else "")
    logging.info("  Model: %s  |  backend: %s", model_cfg.model_name, backend_label)
    logging.info("=" * 55)

    logging.info("Building dataset...")
    train_loader, val_loader, test_loader = build_dataloaders(data_cfg, train_cfg)

    device = torch.device(train_cfg.device)
    model = build_model(data_cfg, model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Model: %s  |  params: %d  |  device: %s",
                 model_cfg.model_name, n_params, train_cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr,
                                  weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, train_cfg.scheduler_tmax))
    stopper = EarlyStopping(patience=train_cfg.early_stopping_patience)

    save_json(output_dir / "config.json",
              {"data": asdict(data_cfg), "model": asdict(model_cfg), "train": asdict(train_cfg)})

    best_ckpt = output_dir / "best_model.pt"
    history: list[dict] = []

    for epoch in range(1, train_cfg.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, optimizer, device,
                                  grad_clip=train_cfg.grad_clip, log_every=train_cfg.log_every)
        val_loss, val_m = run_epoch(model, val_loader, None, device)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_mae": val_m["mae"], "val_r2": val_m["r2"]}
        history.append(row)
        logging.info("epoch=%d  train=%.4f  val=%.4f  val_r2=%.3f",
                     epoch, train_loss, val_loss, val_m["r2"])

        if stopper.step(val_loss):
            torch.save({"model_state": model.state_dict()}, best_ckpt)
        if stopper.should_stop:
            logging.info("Early stopping at epoch %d", epoch)
            break

    save_json(output_dir / "history.json", {"history": history})

    model.load_state_dict(torch.load(best_ckpt, map_location=train_cfg.device)["model_state"])
    test_loss, test_m = run_epoch(model, test_loader, None, device)
    result = {"test_loss": test_loss, **test_m}
    logging.info("TEST  loss=%.4f  mse=%.4f  mae=%.4f  r2=%.3f",
                 test_loss, test_m["mse"], test_m["mae"], test_m["r2"])
    save_json(output_dir / "test_metrics.json", result)

    return result


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train QASA models")

    p.add_argument("--demo", action="store_true",
                   help="Quick demo mode: tiny data, 5 epochs, runs in ~30 s")

    # Data
    p.add_argument("--dataset-type",  default="mixed", choices=list(_GENERATORS))
    p.add_argument("--num-series",    type=int, default=2400)
    p.add_argument("--series-length", type=int, default=160)
    p.add_argument("--window-size",   type=int, default=24)
    p.add_argument("--batch-size",    type=int, default=32)

    # Model
    p.add_argument("--model", default="qasa_transformer",
                   choices=["single_qubit", "qasa_transformer"])
    p.add_argument("--d-model",        type=int,   default=64)
    p.add_argument("--num-heads",      type=int,   default=4)
    p.add_argument("--num-classical-layers", type=int, default=2)
    p.add_argument("--n-qubits",       type=int,   default=4)
    p.add_argument("--q-layers",       type=int,   default=3)
    p.add_argument("--dropout",        type=float, default=0.10)
    p.add_argument("--disable-timestep-conditioning", action="store_true")

    # Train
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--patience",      type=int,   default=8)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir",    default=None,
                   help="Where to save results. Defaults to outputs/local or outputs/ibm based on RUN_LOCAL.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.demo:
        demo_model = DEMO_MODEL
        if args.model != "qasa_transformer":
            demo_model.model_name = args.model
        train_experiment(DEMO_DATA, demo_model, DEMO_TRAIN, is_demo=True)
    else:
        if args.output_dir is None:
            backend_dir = "local" if _is_local() else "ibm"
            output_dir = f"./outputs/{backend_dir}"
        else:
            output_dir = args.output_dir

        data_cfg = DataConfig(
            num_series=args.num_series, window_size=args.window_size,
            series_length=args.series_length, batch_size=args.batch_size,
            dataset_type=args.dataset_type,
        )
        model_cfg = ModelConfig(
            model_name=args.model, d_model=args.d_model, num_heads=args.num_heads,
            num_classical_layers=args.num_classical_layers, n_qubits=args.n_qubits,
            q_layers=args.q_layers, dropout=args.dropout,
            use_timestep_conditioning=not args.disable_timestep_conditioning,
        )
        train_cfg = TrainConfig(
            epochs=args.epochs, lr=args.lr, seed=args.seed,
            early_stopping_patience=args.patience, device=args.device,
            output_dir=output_dir, scheduler_tmax=args.epochs,
        )
        train_experiment(data_cfg, model_cfg, train_cfg)


if __name__ == "__main__":
    main()
