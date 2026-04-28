"""Model implementations: MLP, Classical Transformer, and QASA."""

import logging
import math
import os
from typing import Any, cast

import pennylane as qml
import torch
import torch.nn as nn

from config import ModelConfig
from utils import is_local, make_device


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
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_mult, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        return self.norm2(x + self.ffn(x))


# MLP baseline

class MLPRegressor(nn.Module):
    """Simple MLP: flattens the window and predicts the next value."""

    def __init__(self, window_size: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# Single-qubit quantum baseline

class SingleQubitRegressor(nn.Module):
    """
    Data re-uploading on a single qubit.
    For each of the 24 input time steps: encode value as RX, then apply
    trainable RY + RZ rotations. Read out PauliZ expectation.
    Classical counterpart to MLPRegressor — no attention, no embeddings.
    """

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.theta = nn.Parameter(0.01 * torch.randn(window_size, 2))
        self.readout = nn.Linear(1, 1)
        nn.init.zeros_(self.readout.bias)

        dev, label = make_device(1)
        logging.info(f"SingleQubitRegressor: {label}")

        shots = int(os.environ.get("IBM_SHOTS", "256")) if not is_local() else None

        @qml.qnode(dev, interface="torch", diff_method="best", shots=shots)
        def circuit(inputs: torch.Tensor, theta: torch.Tensor) -> Any:
            for t in range(window_size):
                qml.RX(cast(Any, inputs[..., t]), wires=0)
                qml.RY(cast(Any, theta[t, 0]), wires=0)
                qml.RZ(cast(Any, theta[t, 1]), wires=0)
            return cast(Any, qml.expval(qml.PauliZ(0)))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_out = cast(torch.Tensor, self.circuit(x, self.theta))
        return self.readout(q_out.to(x.dtype).unsqueeze(-1)).squeeze(-1)


# Classical Transformer

class ClassicalTransformerRegressor(nn.Module):
    """Standard Transformer encoder for time-series forecasting."""

    def __init__(self, window_size: int, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.pos_enc = PositionalEncoding(cfg.d_model, max(4096, window_size + 8))
        self.blocks = nn.ModuleList([
            EncoderBlock(cfg.d_model, cfg.num_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.embed(x.unsqueeze(-1)))
        for block in self.blocks:
            h = block(h)
        return self.head(h[:, -1, :]).squeeze(-1)


# QASA: Quantum replaces FFN in the last encoder block

class QuantumLayer(nn.Module):
    """
    Replaces the FFN in the final encoder block.
    Projects d_model → n_qubits, runs a variational quantum circuit,
    projects back to d_model.
    """

    def __init__(self, d_model: int, n_qubits: int, q_layers: int, dropout: float):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layers = q_layers

        self.in_proj = nn.Linear(d_model, n_qubits)
        self.in_norm = nn.LayerNorm(n_qubits)
        self.out_proj = nn.Linear(n_qubits, d_model)
        self.dropout = nn.Dropout(dropout)
        self.weights = nn.Parameter(0.05 * torch.randn(q_layers, n_qubits, 2))

        dev, label = make_device(n_qubits)
        logging.info(f"QuantumLayer: {label}")

        shots = int(os.environ.get("IBM_SHOTS", "256")) if not is_local() else None

        @qml.qnode(dev, interface="torch", diff_method="best", shots=shots)
        def circuit(features: torch.Tensor, weights: torch.Tensor) -> Any:
            for layer in range(q_layers):
                qml.AngleEmbedding(features, wires=range(n_qubits), rotation="X")
                qml.AngleEmbedding(features, wires=range(n_qubits), rotation="Z")
                for q in range(n_qubits):
                    qml.RY(cast(Any, weights[layer, q, 0]), wires=q)
                    qml.RZ(cast(Any, weights[layer, q, 1]), wires=q)
                if n_qubits > 1:
                    for q in range(n_qubits):
                        qml.CNOT(wires=[q, (q + 1) % n_qubits])
            return cast(Any, [qml.expval(qml.PauliZ(q)) for q in range(n_qubits)])

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        h = torch.tanh(self.in_proj(x))
        h = self.in_norm(h)
        flat = h.view(B * L, self.n_qubits)
        q_out = torch.stack(cast(list, self.circuit(flat, self.weights)), dim=-1)
        q_out = q_out.view(B, L, self.n_qubits).to(x.dtype)
        return x + self.dropout(self.out_proj(q_out))


class QuantumEncoderBlock(nn.Module):
    """
    Attention + Quantum (no FFN).
    The quantum circuit IS the non-linear transformation — it replaces the FFN.
    """

    def __init__(self, d_model: int, num_heads: int, n_qubits: int, q_layers: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.quantum = QuantumLayer(d_model, n_qubits, q_layers, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        # Only run the quantum circuit on the last token (the one the head reads).
        # This reduces IBM submissions from B*L to B.
        last = x[:, -1:, :]                        # [B, 1, d_model]
        q_last = self.quantum(last)                 # [B, 1, d_model]
        out = torch.cat([x[:, :-1, :], q_last], dim=1)  # [B, L, d_model]
        return self.norm2(out)


class QASATransformerRegressor(nn.Module):
    """
    QASA: classical encoder blocks followed by one quantum encoder block.
    The final block uses a quantum circuit instead of an FFN.
    """

    def __init__(self, window_size: int, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(1, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.Dropout(cfg.dropout),
        )
        self.pos_enc = PositionalEncoding(cfg.d_model, max(4096, window_size + 8))
        self.classical_blocks = nn.ModuleList([
            EncoderBlock(cfg.d_model, cfg.num_heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.num_layers - 1)
        ])
        self.quantum_block = QuantumEncoderBlock(
            cfg.d_model, cfg.num_heads, cfg.n_qubits, cfg.q_layers, cfg.dropout
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pos_enc(self.embed(x.unsqueeze(-1)))
        for block in self.classical_blocks:
            h = block(h)
        h = self.quantum_block(h)
        return self.head(h[:, -1, :]).squeeze(-1)


def build_model(window_size: int, cfg: ModelConfig) -> nn.Module:
    models = {
        "mlp":                    lambda: MLPRegressor(window_size),
        "single_qubit":           lambda: SingleQubitRegressor(window_size),
        "classical_transformer":  lambda: ClassicalTransformerRegressor(window_size, cfg),
        "qasa_transformer":       lambda: QASATransformerRegressor(window_size, cfg),
    }
    if cfg.model_name not in models:
        raise ValueError(f"Unknown model: {cfg.model_name}. Choose from: {list(models)}")
    return models[cfg.model_name]()
