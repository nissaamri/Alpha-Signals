# Alpha Signals — Stock Prediction & Trading Test Pipeline

A project that looks at stock prices from three different stock markets — New
York, London, and Bursa Malaysia — tries to predict where prices are heading
in the next 5 days, and then tests whether that prediction could actually
make money if you traded on it (accounting for real trading fees).

## Quick start

```bash
pip install -r requirements.txt

# Demo mode — uses made-up (fake) price data, runs instantly, no internet needed.
# Good for checking everything works before trying it on real markets.
python main.py --demo

# Real mode — pulls real, live data from Yahoo Finance. Needs internet.
python main.py
```

Each step can also be run on its own, in order:

```bash
python data_pipeline.py --demo
python features.py
python model.py
python backtest.py
python plot_results.py     # creates data/equity_curve.png (a results chart)
```

## What each file actually does

1. **`data_pipeline.py`** — Collects 10 years of daily stock prices for 30
   well-known companies (10 from New York, 10 from London, 10 from Bursa
   Malaysia), plus each market's overall index for comparison. It can either
   make up fake price data to test with (`fetch_demo()`), or pull real prices
   from Yahoo Finance (`fetch_real()`).

2. **`features.py`** — Turns raw prices into useful signals for each stock:
   how much it's moved recently (5-day and 20-day momentum), how bouncy it's
   been (volatility), a popular trading indicator called RSI, and how closely
   it moves with its own market's overall index. The goal is predicting the
   stock's return over the next 5 days.

3. **`model.py`** — Trains a machine learning model (LightGBM) to make that
   prediction. It's tested the honest way — trained only on past data,
   tested only on data that comes *after* that, so it can never "cheat" by
   peeking into the future. Two numbers measure how good it is: average
   error, and how well predictions actually line up with what really
   happened (called the Information Coefficient).

4. **`backtest.py`** — Simulates actually trading on the model's predictions:
   every 5 days, bet on the stocks predicted to do best, bet against the
   ones predicted to do worst, and subtract realistic trading fees. Then
   compares that strategy's return, risk, and biggest losing streak against
   simply buying and holding everything.

## Being honest about the demo results

When tested on made-up (fake) price data, the model correctly finds almost
no real pattern — which is exactly what should happen, since fake random
data has no real pattern to find. And once trading fees are subtracted, the
strategy actually does slightly worse than just buying and holding. That's
a good sign, not a bad one — it means the model isn't secretly cheating by
looking at "future" information it shouldn't have access to.

Real market data might show a real pattern, a weak one, or none at all —
and that's fine either way. An honest, well-explained "this didn't quite
work and here's why" is a more trustworthy result than a suspiciously
perfect one, and it's the kind of finding real trading researchers report
all the time too.

## Known simplifications

- **No currency conversion.** Each market's returns are used in that
  market's own currency, without converting everything to one common
  currency first.
- **No adjustment for companies that no longer exist.** The list of stocks
  used is fixed and made up of companies still around today — a fully
  rigorous test would need the exact list of companies that existed at each
  point in the past, including ones that later went bankrupt or got bought
  out.
- **One fixed trading schedule** (every 5 days) rather than something
  fine-tuned or automatically adjusted.
- **Runs on a regular computer, not a big-data cluster.** With about 30
  stocks over 10 years (~85,000 rows), a normal computer handles this fine.
  A much bigger version (hundreds of stocks, or prices updated every
  second) would need distributed computing tools like Spark — the code is
  structured so it could be adapted that way later if needed.

## Files

```
data_pipeline.py     collects and cleans stock price data (real or fake/demo)
features.py           turns prices into prediction signals
model.py               trains the prediction model and checks its accuracy
backtest.py            simulates trading on the predictions vs. just holding
plot_results.py        draws the results chart
main.py                 runs all four steps in order, start to finish
```
