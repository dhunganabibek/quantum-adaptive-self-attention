"""Generate comparison plots from training results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY   = "#1B2A4A"
BLUE   = "#2D7DD2"
ORANGE = "#E76F51"
GREEN  = "#2DC653"
PURPLE = "#9B5DE5"
GRAY   = "#8899AA"
BG     = "#F7FAFD"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": NAVY,  "axes.labelcolor": NAVY,
    "xtick.color": NAVY,     "ytick.color": NAVY,
    "text.color": NAVY,      "font.family": "sans-serif",
    "font.size": 11,         "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#D0DCEA", "grid.linestyle": "--",
    "grid.linewidth": 0.6,   "savefig.dpi": 150,
    "savefig.bbox": "tight", "savefig.facecolor": BG,
})


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _bar(ax: plt.Axes, labels: list[str], vals: list[float],
         colors: list[str], title: str, ylabel: str) -> None:
    bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.45)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y")


def _loss_curves(ax: plt.Axes, runs: dict[str, list[dict]],
                 colors: list[str], title: str) -> None:
    for (label, hist), color in zip(runs.items(), colors):
        epochs = [r["epoch"] for r in hist]
        ax.plot(epochs, [r["val_loss"] for r in hist],
                lw=2, color=color, label=label)
        ax.plot(epochs, [r["train_loss"] for r in hist],
                lw=1.2, color=color, linestyle="--", alpha=0.45)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True)
    ax.text(0.98, 0.97, "— val   -- train",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color=GRAY, style="italic")


def plot_all(base: Path, out_dir: Path) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    local = base / "local"
    ibm   = base / "ibm"

    # ── Plot 1: MLP (local) vs Single-Qubit (local) vs Single-Qubit (IBM) ───
    ibm_sq = _load(ibm / "single_qubit" / "test_metrics.json")

    m_mlp = _load(local / "mlp" / "test_metrics.json")
    m_sq  = _load(local / "single_qubit" / "test_metrics.json")
    h_mlp = _load(local / "mlp" / "history.json")
    h_sq  = _load(local / "single_qubit" / "history.json")

    if m_mlp or m_sq:
        # bars: always show local results; add IBM bar when available
        entries = {"MLP\n(local)": m_mlp, "Single-Qubit\n(local)": m_sq}
        if ibm_sq:
            entries["Single-Qubit\n(IBM hardware)"] = ibm_sq
        entries = {k: v for k, v in entries.items() if v}

        labels = list(entries.keys())
        colors = [BLUE, ORANGE, GREEN][: len(labels)]

        n_panels = 3 if (h_mlp or h_sq) and not ibm_sq else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
        if n_panels == 2:
            axes = list(axes)

        title = (
            "Comparison 1 — MLP vs Single-Qubit: local vs IBM hardware"
            if ibm_sq
            else "Comparison 1 — MLP vs Single-Qubit (local simulator)"
        )
        fig.suptitle(title, fontsize=14, fontweight="bold")

        _bar(axes[0], labels, [entries[l]["r2"]   for l in labels],
             colors, "R²  (higher = better)", "R²")
        _bar(axes[1], labels, [entries[l]["rmse"] for l in labels],
             colors, "RMSE  (lower = better)", "RMSE")

        if n_panels == 3:
            curve_runs = {}
            if h_mlp: curve_runs["MLP"] = h_mlp["history"]
            if h_sq:  curve_runs["Single-Qubit (local)"] = h_sq["history"]
            _loss_curves(axes[2], curve_runs, [BLUE, ORANGE], "Training Curves")

        if ibm_sq:
            fig.text(0.5, -0.04,
                     "IBM result: locally-trained weights evaluated on real quantum hardware",
                     ha="center", fontsize=9, color=GRAY, style="italic")

        plt.tight_layout()
        p = out_dir / "plot1_mlp_vs_single_qubit.png"
        fig.savefig(p); plt.close(fig)
        print(f"  saved: {p}")

    # Plot 2: Classical Transformer vs QASA (local)
    runs_2 = {
        "Classical Transformer": local / "classical_transformer",
        "QASA Transformer":      local / "qasa_transformer",
    }
    metrics_2   = {k: _load(d / "test_metrics.json") for k, d in runs_2.items()}
    histories_2 = {k: _load(d / "history.json") for k, d in runs_2.items()}

    if any(metrics_2.values()):
        labels = [k for k, v in metrics_2.items() if v]
        colors = [BLUE, ORANGE]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle("Comparison 2 — Classical Transformer vs QASA (local simulator)",
                     fontsize=14, fontweight="bold")
        _bar(axes[0], labels, [metrics_2[l]["r2"] for l in labels],
             colors, "R²  (higher = better)", "R²")
        _bar(axes[1], labels, [metrics_2[l]["rmse"] for l in labels],
             colors, "RMSE  (lower = better)", "RMSE")
        curve_runs = {k: histories_2[k]["history"] for k in labels if histories_2.get(k)}
        if curve_runs:
            _loss_curves(axes[2], curve_runs, colors, "Training Curves")
        else:
            axes[2].set_visible(False)
        fig.text(0.5, -0.04,
                 "QASA: final encoder block uses a quantum circuit instead of FFN",
                 ha="center", fontsize=9, color=GRAY, style="italic")
        plt.tight_layout()
        p = out_dir / "plot2_transformer_vs_qasa.png"
        fig.savefig(p); plt.close(fig)
        print(f"  saved: {p}")

    # Plot 3: Classical (local) vs QASA (local) vs QASA (IBM) 
    # Exactly 3 bars — only generated when IBM results exist
    ibm_qasa = _load(ibm / "qasa_transformer" / "test_metrics.json")

    if ibm_qasa:
        entries = {
            "Classical\nTransformer\n(local)": metrics_2.get("Classical Transformer"),
            "QASA\n(local simulator)":          metrics_2.get("QASA Transformer"),
            "QASA\n(IBM hardware)":             ibm_qasa,
        }
        # drop any entry where we don't have results
        entries = {k: v for k, v in entries.items() if v}

        labels = list(entries.keys())
        colors = [BLUE, ORANGE, GREEN][:len(labels)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            "Comparison 3 — Classical (local) vs QASA (local) vs QASA (IBM hardware)",
            fontsize=14, fontweight="bold",
        )
        _bar(axes[0], labels, [entries[l]["r2"]   for l in labels],
             colors, "R²  (higher = better)", "R²")
        _bar(axes[1], labels, [entries[l]["rmse"] for l in labels],
             colors, "RMSE  (lower = better)", "RMSE")
        fig.text(
            0.5, -0.04,
            "IBM result: locally-trained weights evaluated on real quantum hardware",
            ha="center", fontsize=9, color=GRAY, style="italic",
        )
        plt.tight_layout()
        p = out_dir / "plot3_local_vs_ibm.png"
        fig.savefig(p); plt.close(fig)
        print(f"  saved: {p}")
    else:
        print("  Plot 3 skipped — no IBM results found (run: just run-ibm)")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="./outputs/comparison/fast")
    ap.add_argument("--out-dir",  default="./outputs/plots")
    args = ap.parse_args()

    print("Generating plots...")
    plot_all(Path(args.base_dir), Path(args.out_dir))
    print(f"\nDone. Plots saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
