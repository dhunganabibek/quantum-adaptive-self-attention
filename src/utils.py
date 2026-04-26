"""Utility functions and device management."""

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=False)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: Path) -> None:
    """Setup logging to file and console."""
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
    """Save dictionary to JSON file."""
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


# Quantum device management
def is_local() -> bool:
    """Check if running on local simulator or IBM Quantum hardware."""
    return os.environ.get("RUN_LOCAL", "true").strip().lower() not in ("false", "0", "no")


def make_device(n_wires: int) -> tuple[qml.Device, str]:
    """Create a quantum device (local or IBM)."""
    if is_local():
        dev = qml.device("default.qubit", wires=n_wires)
        return dev, "local (default.qubit)"

    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    backend = os.environ.get("IBM_BACKEND", "ibm_brisbane")
    
    if not token:
        raise OSError("IBM_QUANTUM_TOKEN must be set in .env file")

    try:
        import qiskit_ibm_runtime  # noqa: F401
    except ImportError as exc:
        raise ImportError("qiskit_ibm_runtime required for IBM backend") from exc

    dev = qml.device("qiskit.ibmq", wires=n_wires, backend=backend, ibmqx_token=token)
    logging.info(f"Connected to IBM Quantum: {backend}")
    return dev, f"IBM Quantum ({backend})"
