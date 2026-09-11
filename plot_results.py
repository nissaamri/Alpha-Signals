"""
plot_results.py — Plots the strategy vs. benchmark equity curve.

Run after backtest.py:
    python plot_results.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

if __name__ == "__main__":
    curve = pd.read_csv(DATA_DIR / "equity_curve.csv", index_col=0, parse_dates=True)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(curve.index, curve["strategy"], label="Alpha Signals (long-short)", linewidth=1.8, color="#37E6C4")
    ax.plot(curve.index, curve["benchmark"], label="Buy-and-hold benchmark", linewidth=1.8, color="#E8B75D")
    ax.set_title("Alpha Signals — Strategy vs. Benchmark Equity Curve")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = DATA_DIR / "equity_curve.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved chart -> {out_path}")
