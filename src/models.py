"""All model implementations for QASA."""

import logging
import math
from typing import Any, cast

import pennylane as qml
import torch
import torch.nn as nn

from config import ModelConfig
from utils import make_device


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
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
    """Position-wise feed-forward network."""
    
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
    """Standard Transformer encoder block."""
    
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




class ClassicalBaselineRegressor(nn.Module):
    """Parameter-matched classical MLP baseline (~53 params)."""
    
    def __init__(self, window_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window_size, 2),
            nn.Tanh(),
            nn.Linear(2, 1),
        )
        # Small initialization like quantum model
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ClassicalTransformerRegressor(nn.Module):
    """Fully classical Transformer baseline."""
    
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
            for _ in range(cfg.num_classical_layers + 1)
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




class SingleQubitReuploadCell(nn.Module):
    """1-qubit data re-uploading regressor (~50 params)."""

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.theta = nn.Parameter(0.01 * torch.randn(window_size, 2))
        self.readout = nn.Linear(1, 1)
        nn.init.zeros_(self.readout.bias)

        dev, label = make_device(1)
        logging.info(f"SingleQubitReuploadCell: {label}")

        @qml.qnode(dev, interface="torch", diff_method="best")
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


class QuantumTokenProjection(nn.Module):
    """Quantum projection layer for Transformer."""

    def __init__(self, d_model: int, n_qubits: int, q_layers: int, dropout: float, use_conditioning: bool):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layers = q_layers
        self.use_conditioning = use_conditioning

        self.in_proj = nn.Linear(d_model, n_qubits)
        self.in_norm = nn.LayerNorm(n_qubits)
        self.out_proj = nn.Linear(n_qubits, d_model)
        self.dropout = nn.Dropout(dropout)
        self.weights = nn.Parameter(0.05 * torch.randn(q_layers, n_qubits, 2))

        dev, label = make_device(n_qubits)
        logging.info(f"QuantumTokenProjection: {label}")

        @qml.qnode(dev, interface="torch", diff_method="best")
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
        hq = torch.tanh(self.in_proj(x))
        hq = self.in_norm(hq)

        if self.use_conditioning:
            t = torch.linspace(0, 1, L, device=x.device, dtype=x.dtype)
            hq = hq + t.view(1, -1, 1)

        flat = hq.view(B * L, self.n_qubits)
        q_out = torch.stack(cast(list, self.circuit(flat, self.weights)), dim=-1)
        q_out = q_out.view(B, L, self.n_qubits).to(x.dtype)
        return x + self.dropout(self.out_proj(q_out))


class QuantumEncoderBlock(nn.Module):
    """Quantum-enhanced encoder block."""
    
    def __init__(self, d_model: int, num_heads: int, ff_mult: int, dropout: float,
                 n_qubits: int, q_layers: int, use_conditioning: bool):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.quantum = QuantumTokenProjection(d_model, n_qubits, q_layers, dropout, use_conditioning)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_mult, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(self.quantum(x))
        return self.norm3(x + self.ffn(x))


class QASATransformerRegressor(nn.Module):
    """Quantum Adaptive Self-Attention Transformer."""
    
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
            for _ in range(cfg.num_classical_layers)
        ])
        self.quantum_block = QuantumEncoderBlock(
            cfg.d_model, cfg.num_heads, cfg.ff_mult, cfg.dropout,
            cfg.n_qubits, cfg.q_layers, cfg.use_timestep_conditioning
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
    """Build a model based on configuration."""
    models = {
        "classical_baseline": lambda: ClassicalBaselineRegressor(window_size),
        "single_qubit": lambda: SingleQubitReuploadCell(window_size),
        "classical_transformer": lambda: ClassicalTransformerRegressor(window_size, cfg),
        "qasa_transformer": lambda: QASATransformerRegressor(window_size, cfg),
    }
    
    if cfg.model_name not in models:
        raise ValueError(f"Unknown model: {cfg.model_name}")
    
    return models[cfg.model_name]()
