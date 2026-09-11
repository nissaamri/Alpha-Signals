# Alpha Signals — Multi-Market Forecasting Pipeline

A daily-frequency, cross-market forecasting pipeline that pulls equities
data from three exchanges (NYSE, LSE, Bursa Malaysia), engineers momentum /
volatility / macro-correlation features, trains a LightGBM model to predict
5-day forward returns, and backtests a long-short strategy against an
equal-weight buy-and-hold benchmark — with realistic transaction costs.

## Quick start

```bash
pip install -r requirements.txt

# Demo mode — synthetic data, runs instantly, no internet needed.
# Good for checking the pipeline works before pointing it at real markets.
python main.py --demo

# Real mode — pulls live data from Yahoo Finance. Needs internet.
python main.py
```

Each stage can also be run individually, in order:

```bash
python data_pipeline.py --demo
python features.py
python model.py
python backtest.py
python plot_results.py     # writes data/equity_curve.png
```

## What's actually happening at each stage

1. **`data_pipeline.py`** — pulls 10 years of daily OHLCV for 30 large-cap
   tickers across three markets (10 NYSE, 10 LSE, 10 Bursa Malaysia), plus
   each market's benchmark index. `fetch_demo()` generates synthetic data
   via geometric Brownian motion so you can test everything offline;
   `fetch_real()` swaps in actual Yahoo Finance data.

2. **`features.py`** — for each ticker, computes 5-day and 20-day momentum,
   20-day realized volatility, 14-day RSI, and rolling 60-day correlation
   and beta against that ticker's own market index (the "macro" features).
   The prediction target is the forward 5-day return.

3. **`model.py`** — trains a LightGBM regressor on a **time-based**
   train/test split (never trains on the future) and reports MAE and
   information coefficient (IC) — the correlation between predicted and
   realized returns, which is the standard way quant researchers judge
   whether a signal has any real predictive power.

4. **`backtest.py`** — every 5 trading days, ranks all tickers by predicted
   return, goes long the top quintile and short the bottom quintile
   (equal-weighted, dollar-neutral), and charges 5 bps of transaction cost
   per unit of turnover. Reports annualized return, volatility, Sharpe
   ratio, and max drawdown for both the strategy and the benchmark.

## Reading the demo-mode results honestly

On synthetic random-walk data, the model correctly finds almost no signal
(IC ≈ 0.04, close to zero — there's no real pattern to find in fake data),
and the strategy **underperforms** the benchmark once transaction costs are
subtracted. That's the expected, correct outcome — a good sign the
pipeline isn't secretly leaking future information into the model. Real
market data may show a genuine signal, a weak one, or none at all — write
up whatever you actually find. A well-reasoned negative result ("the
signal decayed too fast on liquid markets to survive costs") is a more
credible portfolio piece than a suspiciously good backtest, and it's the
kind of finding real quant researchers report all the time.

## Known simplifications (worth naming explicitly in interviews)

- **No currency conversion.** Returns are used directly in local currency;
  a real cross-market strategy would need to either hedge FX or explicitly
  account for it.
- **No survivorship-bias control.** The ticker list is fixed and modern —
  a rigorous backtest would use a point-in-time index constituent list.
- **Single fixed rebalance frequency** (5 days) rather than something
  tuned or adaptive.
- **Spark/EMR vs. pandas.** This scales to pandas comfortably at ~30
  tickers x 10 years (~85k rows). The original brief specified Spark on
  AWS EMR — that becomes worth it once you're at hundreds of tickers or
  tick-level (not daily) data. `data_pipeline.py`'s structure — fetch per
  market, tag with market/index, concatenate — maps directly onto a
  PySpark job if you want to demonstrate that version too: swap the
  pandas `concat` for a Spark `union`, and the groupby-based feature
  engineering in `features.py` for a Spark window function over
  `partitionBy("ticker").orderBy("Date")`.

## Files

```
data_pipeline.py     fetch + clean raw OHLCV (real or synthetic)
features.py           momentum / volatility / macro features + target
model.py               LightGBM training + walk-forward evaluation
backtest.py            long-short backtest vs. buy-and-hold benchmark
plot_results.py        equity curve chart
main.py                 runs all four stages end to end
```
