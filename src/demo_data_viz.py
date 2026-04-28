"""
Visualize the training data for the class demo.
"""

from __future__ import annotations

import math
import random
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

OUT = Path("outputs/demo")
OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
random.seed(42)


# Signal generators
def _sine(length: int) -> np.ndarray:
    t = np.linspace(0, 1, length)
    return np.sin(2 * math.pi * np.random.uniform(1, 6) * t + np.random.uniform(0, 2 * math.pi))

def _damped(length: int) -> np.ndarray:
    t = np.linspace(0, 4, length)
    return (np.exp(-np.random.uniform(0.1, 0.7) * t)
            * np.cos(np.random.uniform(4, 10) * t + np.random.uniform(0, 2 * math.pi)))

def _chirp(length: int) -> np.ndarray:
    t = np.linspace(0, 1, length)
    f0, f1 = np.random.uniform(1, 3), np.random.uniform(8, 14)
    return np.sin(2 * math.pi * (f0 * t + 0.5 * (f1 - f0) * t * t))

def _mixed(length: int) -> np.ndarray:
    a, b = random.sample([_sine, _damped, _chirp], 2)
    return 0.6 * a(length) + 0.4 * b(length)

def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / (std if std > 1e-8 else 1.0)

GENERATORS = {"Sine": _sine, "Damped": _damped, "Chirp": _chirp, "Mixed": _mixed}
COLORS     = {"Sine": BLUE,  "Damped": ORANGE,  "Chirp": GREEN,  "Mixed": PURPLE}


# All 4 signal types, 3 samples each 

def plot_data_overview() -> None:
    fig, axes = plt.subplots(4, 3, figsize=(16, 10))
    fig.suptitle("Training Data — 4 Synthetic Signal Types  (3 random samples each)",
                 fontsize=15, fontweight="bold", y=1.01)

    for row, (name, gen) in enumerate(GENERATORS.items()):
        color = COLORS[name]
        for col in range(3):
            ax = axes[row, col]
            ax.plot(_zscore(gen(160)), color=color, lw=1.5, alpha=0.9)
            ax.set_ylim(-3.5, 3.5)
            ax.grid(True)
            if col == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold", color=color)
            if row == 0:
                ax.set_title(f"Sample {col + 1}", fontsize=11)
            if row < 3:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time step")

    plt.tight_layout()
    p = OUT / "A_data_overview.png"
    fig.savefig(p); plt.close(fig)
    print(f"  saved: {p}")


# Plot B: What the model actually sees and predicts
def plot_prediction_task() -> None:
    series = _zscore(_mixed(160))
    window = 24

    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle(
        "What the Models Learn\n"
        "Input: 24-step window (blue)   →   Predict: next value (orange dot)",
        fontsize=13, fontweight="bold",
    )

    for ax, start in zip(axes.flat, [0, 20, 40, 60, 80, 100]):
        end = start + window
        ax.plot(series, color=GRAY, lw=1.2, alpha=0.35)
        ax.plot(range(start, end), series[start:end], color=BLUE, lw=2.2, label="Input (24 steps)")
        ax.scatter([end], [series[end]], color=ORANGE, s=120, zorder=5, label="Predict this")
        ax.axvspan(start, end - 1, alpha=0.07, color=BLUE)
        ax.set_title(f"Window at t={start}", fontsize=10)
        ax.set_ylim(-3.5, 3.5)
        ax.grid(True)
        ax.set_xlabel("Time step")

    axes[0, 0].legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    p = OUT / "B_prediction_task.png"
    fig.savefig(p); plt.close(fig)
    print(f"  saved: {p}")


def main() -> None:
    print(f"\nGenerating data visualizations → {OUT}/\n")
    print("Signal types overview...")
    plot_data_overview()
    print("Prediction task illustration...")
    plot_prediction_task()
    print(f"\nDone. Open {OUT}/ to view the plots.")

if __name__ == "__main__":
    main()
