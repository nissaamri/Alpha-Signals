"""
main.py — Runs the full Alpha Signals pipeline end to end:
data -> features -> model -> backtest.

Usage:
    python main.py --demo          # synthetic data, no internet needed
    python main.py                 # real Yahoo Finance data
"""
import argparse
from pathlib import Path

import pandas as pd

from backtest import performance_stats, run_backtest
from data_pipeline import build_dataset
from features import engineer_features
from model import train_model

DATA_DIR = Path(__file__).parent / "data"


def main(demo: bool, seed: int):
    print("== 1/4 building dataset ==")
    raw = build_dataset(demo=demo, seed=seed)

    print("\n== 2/4 engineering features ==")
    feats = engineer_features(raw)
    feats.to_parquet(DATA_DIR / "features.parquet", index=False)
    print(f"  {len(feats):,} feature rows")

    print("\n== 3/4 training model ==")
    model, test_preds = train_model(feats)
    test_preds.to_parquet(DATA_DIR / "test_predictions.parquet", index=False)
    model.save_model(str(DATA_DIR / "lgbm_model.txt"))

    print("\n== 4/4 backtesting ==")
    strat_ret, bench_ret, cost = run_backtest(test_preds)
    strat_stats = performance_stats(strat_ret, "Alpha Signals strategy (long-short)")
    bench_stats = performance_stats(bench_ret, "Buy-and-hold benchmark (equal-weight universe)")

    print(f"\ntotal transaction-cost drag: {cost:.2%}")
    print(f"Sharpe uplift vs. benchmark: {strat_stats['sharpe'] - bench_stats['sharpe']:+.2f}")

    curve = pd.DataFrame({
        "strategy": (1 + strat_ret).cumprod(),
        "benchmark": (1 + bench_ret).cumprod(),
    })
    curve.to_csv(DATA_DIR / "equity_curve.csv")
    print(f"\nsaved equity curve -> {DATA_DIR}/equity_curve.csv  (run plot_results.py to chart it)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no internet needed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(demo=args.demo, seed=args.seed)
