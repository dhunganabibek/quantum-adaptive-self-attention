# Quantum Adaptive Self-Attention (QASA)

A hybrid quantum-classical Transformer for time-series forecasting. The final classical encoder block is replaced with a **Parameterized Quantum Circuit (PQC)** using data re-uploading and circular CNOT entanglement, trained end-to-end via backpropagation with PyTorch + PennyLane.

---

## Models

### `single_qubit` — `SingleQubitReuploadCell`

Minimal 1-qubit data re-uploading regressor. The quantum circuit *is* the entire model — no Transformer, no attention.

**Per timestep:** `RX(xₜ) → RY(θₜ) → RZ(φₜ)` — encode data, then apply two learned rotations.
**Readout:** `⟨Z⟩` expectation value → linear layer → scalar forecast.

48 trainable parameters (θ and φ for each of 24 timesteps). Achieves R² = 0.91 on synthetic mixed series after 20 epochs. Good for demos and ablation studies.

### `qasa_transformer` — `QASATransformerRegressor`

Full hybrid architecture:

1. Linear embedding + sinusoidal positional encoding
2. N classical Transformer encoder blocks (`ClassicalEncoderBlock`)
3. One **quantum-enhanced encoder block** (`QuantumEncoderBlock`):
   - Multi-head self-attention → residual + LayerNorm
   - `QuantumTokenProjection`: Linear(d → n_qubits) → PQC → Linear(n_qubits → d)
   - Feed-forward network → residual + LayerNorm
4. Regression head: last token → scalar forecast

**PQC per token** (repeated `q_layers` times):

- Data re-uploading: `AngleEmbedding` with RX then RZ
- Learnable rotations: `RY + RZ` per qubit
- Circular CNOT entanglement: `CNOT(0→1→2→…→0)`
- Readout: `⟨Zᵢ⟩` for each qubit

---

## Quick start

```bash
# Install dependencies
uv sync

# Quick demo — ~30 seconds, saves to outputs/demo/
uv run python src/main.py --demo

# Single-qubit model, full training
uv run python src/main.py --model single_qubit --epochs 20

# QASA Transformer, full training
uv run python src/main.py --model qasa_transformer --epochs 30 --n-qubits 4 --q-layers 3

# Regenerate plots from training outputs
uv run python src/plot_results.py
```

---

## CLI reference

| Argument | Default | Description |
| --- | --- | --- |
| `--demo` | — | Quick demo: tiny dataset, 5 epochs, ~30 s |
| `--model` | `qasa_transformer` | `single_qubit` or `qasa_transformer` |
| `--n-qubits` | `4` | Number of qubits in the PQC |
| `--q-layers` | `3` | PQC re-uploading layers |
| `--epochs` | `20` | Training epochs |
| `--num-series` | `2400` | Synthetic series to generate |
| `--series-length` | `160` | Length of each generated series |
| `--window-size` | `24` | Lookback window (input size) |
| `--dataset-type` | `mixed` | `sine`, `damped`, `chirp`, or `mixed` |
| `--batch-size` | `32` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--patience` | `8` | Early stopping patience (epochs) |
| `--output-dir` | auto | Defaults to `outputs/local` or `outputs/ibm` |
| `--device` | auto | `cpu` or `cuda` |

---

## Backend switching

Edit `.env` (copy from `.env.example`):

```bash
# Local simulator (default — no account needed, exact gradients)
RUN_LOCAL=true

# IBM Quantum hardware
RUN_LOCAL=false
IBM_QUANTUM_TOKEN=your_token   # free account at quantum.ibm.com
IBM_BACKEND=ibm_brisbane
```

No code changes needed. `make_device()` reads `RUN_LOCAL` at startup and wires the correct `qml.device` into every quantum layer automatically. Output is saved to `outputs/local/` or `outputs/ibm/` accordingly.

---

## Outputs

```text
outputs/
  local/            ← full local training run
    config.json     ← exact config used
    history.json    ← loss + R² per epoch
    test_metrics.json
    best_model.pt
    train.log
  ibm/              ← IBM Quantum run (same structure)
  demo/             ← --demo run (safe to overwrite)
  plots/            ← PNG figures from src/plot_results.py
```

---

## Synthetic datasets

| Type | Description |
| --- | --- |
| `sine` | Sine waves with random frequency, phase, and amplitude |
| `damped` | Damped cosine oscillators |
| `chirp` | Frequency-swept (chirp) signals |
| `mixed` | Weighted mix of two types with a random level shift |

---

## Stability features

- `LayerNorm` around classical and quantum submodules
- `Dropout` in embedding, attention, and FFN
- Gradient clipping, cosine LR schedule, early stopping
- Type-safe dtype casting between PyTorch and PennyLane outputs
- PennyLane batch broadcasting: the entire `[B×L, n_qubits]` tensor is evaluated in a single vectorized circuit call — no Python loop over batch items

---

## What we claim and what we don't

**We claim:**

- A PQC can be inserted into a Transformer encoder block and trained end-to-end with standard PyTorch backprop
- The hybrid model converges — R² = 0.91 on synthetic data in 20 epochs
- A 1-qubit circuit with 48 parameters captures nonlinear temporal patterns

**We do not claim:**

- Quantum speedup — local simulation is slower than a classical layer
- That the quantum block outperforms a size-matched classical MLP — no such baseline was run
- Validity on real-world data — only synthetic sine/damped/chirp series were tested
- Verified IBM Quantum results — the hardware code path is wired but not run end-to-end

---

## Development

```bash
# Type-check
uv run pyright

# Lint + format
uv run ruff check src/
uv run ruff format src/
```

---

## Limitations

- State-vector simulation scales as 2ⁿ in memory; practical limit is ~25–30 qubits locally.
- On IBM hardware, gradients use the parameter-shift rule (2 circuit evals per parameter per step) rather than backprop — significantly slower.
- Shot noise on real QPUs degrades accuracy proportional to 1/√shots.
