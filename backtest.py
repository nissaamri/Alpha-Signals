"""
backtest.py — Long-short backtest driven by model predictions,
benchmarked against an equal-weight buy-and-hold portfolio.

Run after model.py:
    python backtest.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
TXN_COST_BPS = 5          # one-way cost per unit of turnover, in basis points
REBALANCE_EVERY = 5       # trading days; matches the 5-day forward-return target
TOP_FRAC = 0.2             # long top quintile, short bottom quintile by predicted return


def build_signal_positions(preds: pd.DataFrame, top_frac: float = TOP_FRAC) -> pd.DataFrame:
    preds = preds.sort_values(["Date", "ticker"]).copy()
    dates = sorted(preds["Date"].unique())
    rebalance_dates = dates[::REBALANCE_EVERY]

    rows = []
    for d in rebalance_dates:
        day = preds[preds["Date"] == d]
        n = max(1, int(len(day) * top_frac))
        longs = day.nlargest(n, "pred")["ticker"]
        shorts = day.nsmallest(n, "pred")["ticker"]
        rows += [{"Date": d, "ticker": t, "side": 1} for t in longs]
        rows += [{"Date": d, "ticker": t, "side": -1} for t in shorts]
    return pd.DataFrame(rows)


def run_backtest(preds: pd.DataFrame):
    pos_df = build_signal_positions(preds)
    price_panel = preds.pivot(index="Date", columns="ticker", values="Close").sort_index()
    ret_panel = price_panel.pct_change()

    dates = ret_panel.index
    strat_daily_ret = pd.Series(0.0, index=dates)
    bench_daily_ret = ret_panel.mean(axis=1)  # equal-weight buy-and-hold on the same universe

    rebalance_dates = set(pos_df["Date"].unique())
    current_positions = pd.Series(dtype=float)
    total_cost = 0.0

    for d in dates:
        if d in rebalance_dates:
            new_pos = pos_df[pos_df["Date"] == d].set_index("ticker")["side"].astype(float)
            n_active = new_pos.abs().sum()
            new_pos = new_pos / max(n_active, 1)  # gross exposure normalized to 1 long + 1 short

            idx = current_positions.index.union(new_pos.index)
            turnover = (new_pos.reindex(idx, fill_value=0) - current_positions.reindex(idx, fill_value=0)).abs().sum()
            cost = turnover * (TXN_COST_BPS / 10000)
            total_cost += cost
            strat_daily_ret.loc[d] -= cost
            current_positions = new_pos

        if len(current_positions):
            day_ret = ret_panel.loc[d, current_positions.index].fillna(0)
            strat_daily_ret.loc[d] += (day_ret * current_positions).sum()

    return strat_daily_ret, bench_daily_ret, total_cost


def performance_stats(daily_ret: pd.Series, label: str) -> dict:
    daily_ret = daily_ret.dropna()
    ann_ret = (1 + daily_ret).prod() ** (252 / len(daily_ret)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + daily_ret).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()

    print(f"\n{label}")
    print(f"  annualized return: {ann_ret:>8.2%}")
    print(f"  annualized vol:    {ann_vol:>8.2%}")
    print(f"  sharpe ratio:      {sharpe:>8.2f}")
    print(f"  max drawdown:      {max_dd:>8.2%}")
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd}


if __name__ == "__main__":
    preds = pd.read_parquet(DATA_DIR / "test_predictions.parquet")
    strat_ret, bench_ret, cost = run_backtest(preds)

    strat_stats = performance_stats(strat_ret, "Alpha Signals strategy (long-short)")
    bench_stats = performance_stats(bench_ret, "Buy-and-hold benchmark (equal-weight universe)")

    print(f"\ntotal transaction-cost drag: {cost:.2%}")
    print(f"Sharpe uplift vs. benchmark: {strat_stats['sharpe'] - bench_stats['sharpe']:+.2f}")

    curve = pd.DataFrame({
        "strategy": (1 + strat_ret).cumprod(),
        "benchmark": (1 + bench_ret).cumprod(),
    })
    curve.to_csv(DATA_DIR / "equity_curve.csv")
    print(f"saved equity curve -> {DATA_DIR}/equity_curve.csv")
