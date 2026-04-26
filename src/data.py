"""Dataset generation and loading for time-series forecasting."""

import math
import random
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from config import DataConfig, TrainConfig


class WindowedForecastDataset(Dataset):
    """Dataset of windowed time-series for next-step forecasting."""
    
    def __init__(self, inputs: np.ndarray, targets: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def _zscore(x: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    std = x.std()
    return (x - x.mean()) / (std if std > 1e-8 else 1.0)


def _sine(length: int, noise: float) -> np.ndarray:
    """Generate sine wave with random frequency and phase."""
    t = np.linspace(0, 1, length)
    freq = np.random.uniform(1, 6)
    y = np.random.uniform(0.6, 1.4) * np.sin(
        2 * math.pi * freq * t + np.random.uniform(0, 2 * math.pi)
    )
    return y + np.random.normal(scale=noise, size=length)


def _damped(length: int, noise: float) -> np.ndarray:
    """Generate damped oscillation."""
    t = np.linspace(0, 4, length)
    y = np.random.uniform(0.8, 1.5) * np.exp(-np.random.uniform(0.1, 0.7) * t) * np.cos(
        np.random.uniform(4, 10) * t + np.random.uniform(0, 2 * math.pi)
    )
    return y + np.random.normal(scale=noise, size=length)


def _chirp(length: int, noise: float) -> np.ndarray:
    """Generate chirp signal (frequency sweep)."""
    t = np.linspace(0, 1, length)
    f0, f1 = np.random.uniform(1, 3), np.random.uniform(8, 14)
    phase = 2 * math.pi * (f0 * t + 0.5 * (f1 - f0) * t * t)
    return np.sin(phase) + np.random.normal(scale=noise, size=length)


def _mixed(length: int, noise: float) -> np.ndarray:
    """Generate mixed signal with regime shift."""
    gens = [_sine, _damped, _chirp]
    a, b = random.sample(gens, 2)
    series = 0.6 * a(length, noise) + 0.4 * b(length, noise)
    jump = np.random.randint(length // 5, 4 * length // 5)
    series[jump:] += np.random.uniform(-0.4, 0.4)
    return series


GENERATORS: dict[str, Callable[[int, float], np.ndarray]] = {
    "sine": _sine,
    "damped": _damped,
    "chirp": _chirp,
    "mixed": _mixed,
}


def build_dataset(cfg: DataConfig) -> WindowedForecastDataset:
    """
    Build a windowed forecasting dataset from synthetic time series.
    """
    gen = GENERATORS[cfg.dataset_type]
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


def build_dataloaders(
    data_cfg: DataConfig,
    train_cfg: TrainConfig
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, validation, and test dataloaders.
    
    Args:
        data_cfg: Data configuration
        train_cfg: Training configuration
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset = build_dataset(data_cfg)
    n = len(dataset)
    n_train = int(n * data_cfg.train_ratio)
    n_val = int(n * data_cfg.val_ratio)
    n_test = n - n_train - n_val
    
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(train_cfg.seed),
    )
    
    batch_size = data_cfg.batch_size
    num_workers = train_cfg.num_workers
    pin_memory = train_cfg.device.startswith("cuda")
    
    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,
                  num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(val_set, batch_size=batch_size, shuffle=False,
                  num_workers=num_workers, pin_memory=pin_memory),
        DataLoader(test_set, batch_size=batch_size, shuffle=False,
                  num_workers=num_workers, pin_memory=pin_memory),
    )
