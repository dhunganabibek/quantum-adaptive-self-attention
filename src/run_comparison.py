#!/usr/bin/env python3
"""
Run systematic comparisons between classical and quantum models.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> int:
    """Run a command and return exit code."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    return result.returncode


def load_metrics(output_dir: str) -> dict:
    """Load test metrics from output directory."""
    metrics_path = Path(output_dir) / "test_metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path) as f:
        return json.load(f)


def print_comparison(results: dict[str, dict]) -> None:
    """Print comparison table of results."""
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<30} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'Params':>10}")
    print(f"{'-'*80}")
    
    for model_name, metrics in results.items():
        if metrics:
            r2 = metrics.get('r2', 0.0)
            rmse = metrics.get('rmse', 0.0)
            mae = metrics.get('mae', 0.0)
            # Note: params would need to be extracted from config or logs
            print(f"{model_name:<30} {r2:>10.4f} {rmse:>10.4f} {mae:>10.4f} {'N/A':>10}")
        else:
            print(f"{model_name:<30} {'FAILED':>10} {'FAILED':>10} {'FAILED':>10} {'N/A':>10}")
    
    print(f"{'-'*80}\n")


def run_simple_comparison(epochs: int = 20) -> dict[str, dict]:
    """Run parameter-matched comparison: classical_baseline vs single_qubit."""
    results = {}
    
    # Classical baseline
    cmd = [
        "uv", "run", "python", "src/main.py",
        "--model", "classical_baseline",
        "--epochs", str(epochs),
        "--output-dir", "outputs/comparison/classical_baseline"
    ]
    if run_command(cmd) == 0:
        results["classical_baseline"] = load_metrics("outputs/comparison/classical_baseline")
    
    # Single qubit quantum
    cmd = [
        "uv", "run", "python", "src/main.py",
        "--model", "single_qubit",
        "--epochs", str(epochs),
        "--output-dir", "outputs/comparison/single_qubit"
    ]
    if run_command(cmd) == 0:
        results["single_qubit"] = load_metrics("outputs/comparison/single_qubit")
    
    return results


def run_transformer_comparison(epochs: int = 30) -> dict[str, dict]:
    """Run architecture-matched comparison: classical_transformer vs qasa_transformer."""
    results = {}
    
    # Classical Transformer
    cmd = [
        "uv", "run", "python", "src/main.py",
        "--model", "classical_transformer",
        "--epochs", str(epochs),
        "--output-dir", "outputs/comparison/classical_transformer"
    ]
    if run_command(cmd) == 0:
        results["classical_transformer"] = load_metrics("outputs/comparison/classical_transformer")
    
    # QASA Transformer
    cmd = [
        "uv", "run", "python", "src/main.py",
        "--model", "qasa_transformer",
        "--epochs", str(epochs),
        "--n-qubits", "4",
        "--q-layers", "3",
        "--output-dir", "outputs/comparison/qasa_transformer"
    ]
    if run_command(cmd) == 0:
        results["qasa_transformer"] = load_metrics("outputs/comparison/qasa_transformer")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run model comparisons")
    parser.add_argument(
        "--comparison",
        choices=["simple", "transformer", "all"],
        default="simple",
        help="Which comparison to run"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (default: 20 for simple, 30 for transformer)"
    )
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.comparison in ["simple", "all"]:
        epochs = args.epochs or 20
        print("\n" + "="*80)
        print("COMPARISON 1: Parameter-Matched (Simple Models)")
        print("="*80)
        results = run_simple_comparison(epochs)
        all_results.update(results)
        print_comparison(results)
    
    if args.comparison in ["transformer", "all"]:
        epochs = args.epochs or 30
        print("\n" + "="*80)
        print("COMPARISON 2: Architecture-Matched (Transformer Models)")
        print("="*80)
        results = run_transformer_comparison(epochs)
        all_results.update(results)
        print_comparison(results)
    
    if args.comparison == "all":
        print("\n" + "="*80)
        print("ALL RESULTS")
        print("="*80)
        print_comparison(all_results)
    
    # Save combined results
    output_path = Path("outputs/comparison/summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
