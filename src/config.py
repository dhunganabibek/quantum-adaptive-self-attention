"""Configuration for QASA models."""

from dataclasses import dataclass


@dataclass
class DataConfig:
    """Dataset configuration."""
    num_series: int = 2400
    window_size: int = 24
    horizon: int = 1
    series_length: int = 160
    noise_std: float = 0.03
    dataset_type: str = "mixed"
    batch_size: int = 32
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    normalize_per_series: bool = True


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "qasa_transformer"
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
    """Training configuration."""
    seed: int = 42
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    num_workers: int = 0
    output_dir: str = "./outputs"
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    log_every: int = 50
    scheduler_tmax: int = 50
