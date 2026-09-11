"""
data_pipeline.py — Fetches and cleans multi-market daily OHLCV data.

Real mode:  pulls from Yahoo Finance via yfinance (needs internet).
Demo mode:  generates synthetic-but-realistic OHLCV data (geometric
            Brownian motion per ticker) so you can test the full
            pipeline offline before pointing it at real markets.

Usage:
    python data_pipeline.py --demo          # synthetic, no internet
    python data_pipeline.py                 # real Yahoo Finance data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MARKETS = {
    "NYSE": {
        "index": "^GSPC",
        "tickers": ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "PG", "KO", "DIS", "WMT", "GS"],
    },
    "LSE": {
        "index": "^FTSE",
        "tickers": ["HSBA.L", "BP.L", "GSK.L", "ULVR.L", "AZN.L", "VOD.L", "RIO.L", "BARC.L", "SHEL.L", "DGE.L"],
    },
    "BURSA": {
        "index": "^KLSE",
        "tickers": ["1155.KL", "1295.KL", "5347.KL", "1023.KL", "6033.KL", "5225.KL", "5183.KL", "6888.KL", "4197.KL", "6947.KL"],
    },
}

DATA_DIR = Path(__file__).parent / "data"
START = "2015-01-01"
END = "2025-01-01"


def fetch_real(tickers, start=START, end=END):
    """Pull real daily OHLCV from Yahoo Finance. Requires internet access."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False)
    frames = []
    for t in tickers:
        try:
            sub = raw[t].copy()
        except KeyError:
            print(f"  [warn] no data returned for {t}, skipping")
            continue
        sub["ticker"] = t
        frames.append(sub.reset_index())
    return pd.concat(frames, ignore_index=True)


def fetch_demo(tickers, start=START, end=END, seed=None):
    """Synthetic OHLCV via geometric Brownian motion, so the rest of the
    pipeline (features -> model -> backtest) can be built and tested
    without network access. Swap for fetch_real() when you have internet."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    frames = []
    for t in tickers:
        n = len(dates)
        mu = rng.uniform(0.00015, 0.0006)
        sigma = rng.uniform(0.012, 0.028)
        shocks = rng.normal(mu, sigma, n)
        price = 50 * np.exp(np.cumsum(shocks))
        high = price * (1 + rng.uniform(0, 0.01, n))
        low = price * (1 - rng.uniform(0, 0.01, n))
        open_ = price * (1 + rng.normal(0, 0.003, n))
        vol = rng.integers(1_000_000, 8_000_000, n)
        frames.append(pd.DataFrame({
            "Date": dates, "Open": open_, "High": high, "Low": low,
            "Close": price, "Volume": vol, "ticker": t,
        }))
    return pd.concat(frames, ignore_index=True)


def build_dataset(demo=True, seed=42):
    DATA_DIR.mkdir(exist_ok=True)
    all_rows = []
    for market, cfg in MARKETS.items():
        tickers = cfg["tickers"] + [cfg["index"]]
        print(f"  fetching {market} ({len(tickers)} symbols, demo={demo})...")
        if demo:
            raw = fetch_demo(tickers, seed=seed)
        else:
            raw = fetch_real(tickers)
        raw["market"] = market
        raw["is_index"] = raw["ticker"] == cfg["index"]
        all_rows.append(raw)

    full = pd.concat(all_rows, ignore_index=True)
    full = full.sort_values(["ticker", "Date"]).reset_index(drop=True)
    out_path = DATA_DIR / "raw_prices.parquet"
    full.to_parquet(out_path, index=False)
    print(f"  saved {len(full):,} rows -> {out_path}")
    return full


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Use synthetic data (no internet needed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(demo=args.demo, seed=args.seed)
