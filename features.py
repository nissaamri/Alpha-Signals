"""
features.py — Engineers momentum, volatility, and macro-correlation
features per ticker, plus the forward-return prediction target.

Run after data_pipeline.py:
    python features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline import MARKETS

DATA_DIR = Path(__file__).parent / "data"

FEATURE_COLS = ["mom_5d", "mom_20d", "vol_20d", "rsi_14", "macro_corr_60d", "macro_beta_60d"]


def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def engineer_features(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    df = df.sort_values(["ticker", "Date"]).copy()

    # one return series per market's index, used for macro-correlation features
    index_returns = {}
    for market, cfg in MARKETS.items():
        idx_df = df[df["ticker"] == cfg["index"]].sort_values("Date")
        index_returns[market] = idx_df.set_index("Date")["Close"].pct_change()

    out = []
    for ticker, g in df[~df["is_index"]].groupby("ticker"):
        g = g.sort_values("Date").set_index("Date")
        market = g["market"].iloc[0]

        ret1 = g["Close"].pct_change()
        g["ret_1d"] = ret1
        g["mom_5d"] = g["Close"].pct_change(5)
        g["mom_20d"] = g["Close"].pct_change(20)
        g["vol_20d"] = ret1.rolling(20).std()
        g["rsi_14"] = rsi(g["Close"])

        idx_r = index_returns[market].reindex(g.index)
        g["macro_corr_60d"] = ret1.rolling(60).corr(idx_r)
        g["macro_beta_60d"] = ret1.rolling(60).cov(idx_r) / idx_r.rolling(60).var()

        # what we're trying to predict: forward `horizon`-day return
        g["target_fwd_ret"] = g["Close"].shift(-horizon) / g["Close"] - 1

        g["ticker"] = ticker
        g["market"] = market
        out.append(g.reset_index())

    feats = pd.concat(out, ignore_index=True).dropna(subset=FEATURE_COLS + ["target_fwd_ret"])
    return feats.reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_parquet(DATA_DIR / "raw_prices.parquet")
    feats = engineer_features(raw)
    out_path = DATA_DIR / "features.parquet"
    feats.to_parquet(out_path, index=False)
    print(f"saved {len(feats):,} feature rows -> {out_path}")
    print(feats[["ticker", "Date"] + FEATURE_COLS + ["target_fwd_ret"]].head())
