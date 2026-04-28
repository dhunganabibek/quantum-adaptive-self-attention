"""Utility functions and device management."""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

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
    return os.environ.get("RUN_LOCAL", "true").strip().lower() not in ("false", "0", "no")


def is_aer() -> bool:
    return os.environ.get("USE_AER", "false").strip().lower() in ("true", "1", "yes")


def make_device(n_wires: int) -> tuple[Any, str]:
    """
    Three backends:
      local  (default)  — default.qubit, pure Python, fast
      aer    (USE_AER)  — qiskit.aer, noiseless or noisy (AER_NOISE=true)
      ibm    (RUN_LOCAL=false) — real IBM hardware via qiskit.remote, slow queue
    """
    if is_local() and not is_aer():
        dev = qml.device("default.qubit", wires=n_wires)
        return dev, "local (default.qubit)"

    if is_aer():
        from qiskit_aer import AerSimulator
        use_noise = os.environ.get("AER_NOISE", "false").strip().lower() in ("true", "1", "yes")
        if use_noise:
            from qiskit_aer.noise import NoiseModel
            from qiskit_ibm_runtime import QiskitRuntimeService
            token = os.environ.get("IBM_QUANTUM_TOKEN", "")
            backend_name = os.environ.get("IBM_BACKEND", "ibm_sherbrooke")
            if not token:
                raise OSError("IBM_QUANTUM_TOKEN must be set to fetch noise model")
            instance = os.environ.get("IBM_INSTANCE", None)
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
            real_backend = service.backend(backend_name)
            noise_model = NoiseModel.from_backend(real_backend)
            aer_backend = AerSimulator(noise_model=noise_model)
            dev = qml.device("qiskit.aer", wires=n_wires, backend=aer_backend)
            logging.info(f"Aer simulator with noise model from {backend_name}")
            return dev, f"Aer + noise ({backend_name})"
        aer_backend = AerSimulator()
        dev = qml.device("qiskit.aer", wires=n_wires, backend=aer_backend)
        return dev, "Aer simulator (noiseless)"

    # Real IBM hardware
    token = os.environ.get("IBM_QUANTUM_TOKEN", "")
    backend_name = os.environ.get("IBM_BACKEND", "ibm_sherbrooke")
    if not token:
        raise OSError("IBM_QUANTUM_TOKEN must be set in .env")

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        raise ImportError("qiskit_ibm_runtime is required for IBM backend") from exc

    instance = os.environ.get("IBM_INSTANCE", None)
    shots = int(os.environ.get("IBM_SHOTS", "256"))
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    backend = service.backend(backend_name)
    dev = qml.device("qiskit.remote", wires=n_wires, backend=backend, shots=shots)
    logging.info(f"Connected to IBM Quantum: {backend_name} | shots={shots}")
    return dev, f"IBM Quantum ({backend_name})"
