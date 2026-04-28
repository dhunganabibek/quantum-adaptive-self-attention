# QASA - Quantum Adaptive Self-Attention

default:
    @just --list

# Visualize training data 
show-data:
    uv run python src/demo_data_viz.py


run-demo:
    @echo "\nMLP (demo)"
    USE_AER=false RUN_LOCAL=true uv run python src/main.py --demo --model mlp \
        --output-dir outputs/comparison/demo/local/mlp
    @echo "\nSingle-Qubit (demo)"
    USE_AER=false RUN_LOCAL=true uv run python src/main.py --demo --model single_qubit \
        --output-dir outputs/comparison/demo/local/single_qubit
    @echo "\nClassical Transformer (demo)"
    USE_AER=false RUN_LOCAL=true uv run python src/main.py --demo --model classical_transformer \
        --output-dir outputs/comparison/demo/local/classical_transformer
    @echo "\nQASA Transformer (demo)"
    USE_AER=false RUN_LOCAL=true uv run python src/main.py --demo --model qasa_transformer \
        --output-dir outputs/comparison/demo/local/qasa_transformer
    @echo "\nGenerating plots"
    uv run python src/plot_results.py --base-dir outputs/comparison/demo
    @echo "\nDone! Open outputs/plots/"

run-all-fast:
    @echo "\nMLP (local)"
    USE_AER=false RUN_LOCAL=true uv run python src/main.py --fast --model mlp \
        --output-dir outputs/comparison/fast/local/mlp
    @echo "\nSingle-Qubit quantum (local)"
    USE_AER=false RUN_LOCAL=true uv run python src/main.py --fast --model single_qubit \
        --output-dir outputs/comparison/fast/local/single_qubit
    @echo "\nClassical Transformer (local)"
    USE_AER=false RUN_LOCAL=true uv run python src/main.py --fast --model classical_transformer \
        --output-dir outputs/comparison/fast/local/classical_transformer
    @echo "\nQASA Transformer (local)"
    USE_AER=false RUN_LOCAL=true uv run python src/main.py --fast --model qasa_transformer \
        --output-dir outputs/comparison/fast/local/qasa_transformer
    @echo "\nGenerating plots"
    uv run python src/plot_results.py --base-dir outputs/comparison/fast
    @echo "\nDone! Open outputs/plots/"

# Eval on Aer with IBM noise model
run-ibm-fast-aer:
    @echo "\nStep 1: train locally (skips if already done)"
    @if [ ! -f outputs/comparison/fast/local/qasa_transformer/best_model.pt ]; then just run-all; else echo "Local weights found, skipping training."; fi
    @echo "\nStep 2: eval Single-Qubit on Aer + noise"
    USE_AER=true AER_NOISE=true uv run python src/main.py --eval-only \
        --local-dir outputs/comparison/fast/local/single_qubit \
        --output-dir outputs/comparison/fast/ibm/single_qubit
    @echo "\nStep 3: eval QASA on Aer + noise"
    USE_AER=true AER_NOISE=true uv run python src/main.py --eval-only \
        --local-dir outputs/comparison/fast/local/qasa_transformer \
        --output-dir outputs/comparison/fast/ibm/qasa_transformer
    @echo "\nStep 4: regenerate plots"
    uv run python src/plot_results.py --base-dir outputs/comparison/fast
    @echo "\nDone! Open outputs/plots/"

# Eval on real IBM hardware
run-ibm-fast:
    @echo "\nStep 1: train locally (skips if already done)"
    @if [ ! -f outputs/comparison/fast/local/qasa_transformer/best_model.pt ]; then just run-all; else echo "Local weights found, skipping training."; fi
    @echo "\nStep 2: eval Single-Qubit on IBM hardware"
    USE_AER=false uv run python src/main.py --backend ibm --eval-only \
        --local-dir outputs/comparison/fast/local/single_qubit \
        --output-dir outputs/comparison/fast/ibm/single_qubit \
        --ibm-samples 16
    @echo "\nStep 3: eval QASA on IBM hardware"
    USE_AER=false uv run python src/main.py --backend ibm --eval-only \
        --local-dir outputs/comparison/fast/local/qasa_transformer \
        --output-dir outputs/comparison/fast/ibm/qasa_transformer \
        --ibm-samples 16
    @echo "\nStep 4: regenerate plots with IBM bars"
    uv run python src/plot_results.py --base-dir outputs/comparison/fast
    @echo "\nDone! Open outputs/plots/"


# Full dataset (2400 series, window 24), more epochs, 4 qubits.
run-all-full:
    @echo "\nMLP (full)"
    uv run python src/main.py --full --model mlp \
        --output-dir outputs/comparison/full/local/mlp
    @echo "\nSingle-Qubit quantum (full)"
    uv run python src/main.py --full --model single_qubit \
        --output-dir outputs/comparison/full/local/single_qubit
    @echo "\nClassical Transformer (full)"
    uv run python src/main.py --full --model classical_transformer \
        --output-dir outputs/comparison/full/local/classical_transformer
    @echo "\nQASA Transformer (full)"
    uv run python src/main.py --full --model qasa_transformer \
        --output-dir outputs/comparison/full/local/qasa_transformer
    @echo "\nGenerating plots"
    uv run python src/plot_results.py --base-dir outputs/comparison/full
    @echo "\nDone! Open outputs/plots/"


# Add IBM too.
run-ibm-full:
    @echo "\nStep 1: full training locally"
    just run-full
    @echo "\nStep 2: eval Single-Qubit on IBM hardware"
    uv run python src/main.py --backend ibm --eval-only \
        --local-dir outputs/comparison/full/local/single_qubit \
        --output-dir outputs/comparison/full/ibm/single_qubit \
        --ibm-samples 32
    @echo "\nStep 3: eval QASA on IBM hardware"
    uv run python src/main.py --backend ibm --eval-only \
        --local-dir outputs/comparison/full/local/qasa_transformer \
        --output-dir outputs/comparison/full/ibm/qasa_transformer \
        --ibm-samples 32
    @echo "\nStep 4: regenerate plots with IBM bars"
    uv run python src/plot_results.py --base-dir outputs/comparison/full
    @echo "\nDone! Open outputs/plots/"


# Regenerate plots from existing results
plot:
    uv run python src/plot_results.py --base-dir outputs/comparison/fast

plot-full:
    uv run python src/plot_results.py --base-dir outputs/comparison/full

# Install dependencies
install:
    uv sync --all-extras
