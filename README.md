# QASA - Quantum Adaptive Self-Attention

Hybrid quantum-classical Transformer for time-series forecasting.

## Quick Start

```bash
# Install
uv sync --all-extras

# Run demo
just demo

# Train models
just train-quantum
just train-qasa

# Compare models
just compare-all
```

## Models

| Model | Params | Description |
|-------|--------|-------------|
| `classical_baseline` | ~53 | Parameter-matched MLP baseline |
| `single_qubit` | ~50 | 1-qubit quantum circuit |
| `classical_transformer` | ~80K | Fully classical Transformer |
| `qasa_transformer` | ~75K | Quantum-enhanced Transformer |

## Commands

See all available commands:
```bash
just --list
```

Common commands:
```bash
just demo                    # Quick demo (~30s)
just train-quantum           # Train quantum model
just train-qasa              # Train QASA transformer
just compare-all             # Run all comparisons
just plot                    # Generate plots
just pdf                     # Compile presentation
```

## Structure

```
src/
├── config.py       # Configuration
├── data.py         # Dataset generation
├── models.py       # All 4 models
├── training.py     # Training loop
├── utils.py        # Utilities
└── main.py         # Entry point
```

## IBM Quantum

Edit `.env`:
```bash
RUN_LOCAL=false
IBM_QUANTUM_TOKEN=your_token
IBM_BACKEND=ibm_brisbane
```

Or use:
```bash
just use-ibm
```

## Results

After training, check `outputs/MODEL_NAME/`:
- `test_metrics.json` - Performance (R², RMSE, MAE)
- `history.json` - Training curves
- `best_model.pt` - Saved weights

## Documentation

- `justfile` - All available commands
