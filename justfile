# QASA - Quantum Adaptive Self-Attention
# Common commands for development and experimentation

# Show available commands
default:
    @just --list

# ============================================================================
# Quick Start
# ============================================================================

# Run quick demo (~30 seconds)
demo:
    uv run python src/main.py --demo

# Run demo with specific model
demo-model model="single_qubit":
    uv run python src/main.py --demo --model {{model}}

# ============================================================================
# Training
# ============================================================================

# Train classical baseline
train-classical epochs="20":
    uv run python src/main.py --model classical_baseline --epochs {{epochs}}

# Train single-qubit quantum model
train-quantum epochs="20":
    uv run python src/main.py --model single_qubit --epochs {{epochs}}

# Train classical transformer
train-transformer epochs="30":
    uv run python src/main.py --model classical_transformer --epochs {{epochs}}

# Train QASA (quantum-enhanced) transformer
train-qasa epochs="30" qubits="4" layers="3":
    uv run python src/main.py --model qasa_transformer --epochs {{epochs}} --n-qubits {{qubits}} --q-layers {{layers}}

# Train all models (for comparison)
train-all:
    @echo "Training all models..."
    @just train-classical 20
    @just train-quantum 20
    @just train-transformer 30
    @just train-qasa 30

# ============================================================================
# Comparisons
# ============================================================================

# Run parameter-matched comparison (classical vs quantum)
compare-simple:
    uv run python src/run_comparison.py --comparison simple

# Run architecture-matched comparison (transformers)
compare-transformer:
    uv run python src/run_comparison.py --comparison transformer

# Run all comparisons
compare-all:
    uv run python src/run_comparison.py --comparison all

# ============================================================================
# Visualization
# ============================================================================

# Generate plots from training results
plot:
    uv run python src/plot_results.py

# Generate plots for specific output directory
plot-dir dir="outputs/local":
    uv run python src/plot_results.py --output-dir {{dir}}

# ============================================================================
# LaTeX / Presentation
# ============================================================================

# Compile LaTeX presentation to PDF
pdf:
    cd assets/presentation && lualatex report.tex

# Compile and open PDF
pdf-open:
    cd assets/presentation && lualatex report.tex && open report.pdf

# Clean LaTeX auxiliary files
pdf-clean:
    cd assets/presentation && rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

# ============================================================================
# Development
# ============================================================================

# Install dependencies
install:
    uv sync --all-extras

# Run code formatter
format:
    uv run ruff format src/

# Run linter
lint:
    uv run ruff check src/

# Run linter and fix issues
lint-fix:
    uv run ruff check --fix src/

# Run type checker
typecheck:
    uv run pyright src/

# Run all checks (format, lint, typecheck)
check:
    @just format
    @just lint
    @just typecheck

# ============================================================================
# Testing
# ============================================================================

# Test imports
test-imports:
    cd src && uv run python -c "from models import build_model; from config import ModelConfig; print('✓ Imports OK')"

# Test model building
test-models:
    cd src && uv run python -c "from models import build_model; from config import ModelConfig; \
        for m in ['classical_baseline', 'single_qubit', 'classical_transformer', 'qasa_transformer']: \
            model = build_model(24, ModelConfig(model_name=m)); \
            print(f'✓ {m}: {sum(p.numel() for p in model.parameters())} params')"

# Run all tests
test:
    @just test-imports
    @just test-models
    @just demo

# ============================================================================
# Cleanup
# ============================================================================

# Clean Python cache files
clean-cache:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Clean output files
clean-outputs:
    rm -rf outputs/demo outputs/local outputs/ibm outputs/comparison

# Clean everything (cache + outputs)
clean-all:
    @just clean-cache
    @just clean-outputs

# ============================================================================
# IBM Quantum
# ============================================================================

# Switch to local simulator
use-local:
    @echo "RUN_LOCAL=true" > .env
    @echo "✓ Switched to local simulator"

# Switch to IBM Quantum (requires token in .env)
use-ibm:
    @echo "RUN_LOCAL=false" >> .env
    @echo "Make sure IBM_QUANTUM_TOKEN and IBM_BACKEND are set in .env"

# Show current backend
show-backend:
    @grep "RUN_LOCAL" .env || echo "RUN_LOCAL not set (defaults to true)"

# ============================================================================
# Utilities
# ============================================================================

# Count lines of code
loc:
    @echo "Lines of code:"
    @wc -l src/*.py | tail -1

# Show project structure
tree:
    @echo "Project structure:"
    @tree -L 2 -I '__pycache__|*.pyc|.venv|.git|.mypy_cache|.ruff_cache|outputs' .

# Show model parameters
params model="single_qubit":
    cd src && uv run python -c "from models import build_model; from config import ModelConfig; \
        m = build_model(24, ModelConfig(model_name='{{model}}')); \
        print(f'Model: {{model}}'); \
        print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}'); \
        print(f'Trainable: {sum(p.numel() for p in m.parameters() if p.requires_grad):,}')"

# ============================================================================
# Help
# ============================================================================

# Show detailed help
help:
    @echo "QASA - Quantum Adaptive Self-Attention"
    @echo ""
    @echo "Quick Start:"
    @echo "  just demo              # Run quick demo"
    @echo "  just train-quantum     # Train quantum model"
    @echo "  just compare-all       # Run all comparisons"
    @echo ""
    @echo "For all commands, run: just --list"
