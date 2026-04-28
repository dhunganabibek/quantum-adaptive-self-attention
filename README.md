# QASA - Quantum Adaptive Self-Attention

Hybrid quantum-classical Transformer for time-series forecasting.  
Compares MLP, Single-Qubit, Classical Transformer, and QASA across local simulator and IBM hardware.

## Quick Start

```bash
just show-data              # plot training data

just run-all-fast           # fast: train all 4 models locally (~3-5 min)
just run-all-full           # full: train all 4 models locally (~10-20 min)

just run-ibm-fast           # fast local + eval quantum models on IBM hardware
just run-ibm-full           # full local + eval quantum models on IBM hardware

just plot                   # regenerate plots from fast results
just plot-full              # regenerate plots from full results
```

## Models

- MLP
- `single_qubit` 1-qubit data re-uploading circuit
- `classical_transformer` Standard Transformer encoder
- `qasa_transformer` Transformer where the final encoder block replaces its FFN with a quantum circuit 

## Architecture: What QASA Changes

Standard Transformer encoder block:

```text
Attention → Norm → FFN → Norm
```

QASA final encoder block (quantum replaces FFN):

```text
Attention → Norm → Quantum Circuit → Norm
```

The quantum circuit: projects `d_model → n_qubits`, runs AngleEmbedding + RY/RZ rotations + CNOT entanglement, projects back. Classical attention is unchanged.

## Commands

```bash
just show-data              # plot training data

just run-all-fast           # fast: train all 4 models locally (~3-5 min)
just run-all-full           # full: train all 4 models locally (~10-20 min)

just run-ibm-fast           # fast local + eval quantum models on IBM hardware
just run-ibm-full           # full local + eval quantum models on IBM hardware

just plot                   # regenerate plots from fast results
just plot-full              # regenerate plots from full results

```

## IBM Quantum Setup

Add to `.env` at project root:

```text
IBM_QUANTUM_TOKEN=your_token_here
IBM_INSTANCE=crn:v1:bluemix:public:quantum-computing:us-east:a/...::
IBM_BACKEND=ibm_sherbrooke
```

Each model output contains:

- `test_metrics.json` — R², RMSE, MAE
- `history.json` — per-epoch train/val loss
- `best_model.pt` — saved weights
