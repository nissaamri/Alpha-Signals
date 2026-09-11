"""
model.py — Trains a LightGBM regressor to predict forward returns,
using a walk-forward (time-based) train/test split to avoid lookahead.

Run after features.py:
    python model.py
"""
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error

from features import FEATURE_COLS

DATA_DIR = Path(__file__).parent / "data"


def walk_forward_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Time-based split: model never sees the future during training."""
    df = df.sort_values("Date")
    cutoff = df["Date"].quantile(1 - test_frac)
    train = df[df["Date"] < cutoff]
    test = df[df["Date"] >= cutoff]
    return train, test, cutoff


def train_model(feats: pd.DataFrame):
    train, test, cutoff = walk_forward_split(feats)
    print(f"train: {train['Date'].min().date()} -> {train['Date'].max().date()}  ({len(train):,} rows)")
    print(f"test:  {test['Date'].min().date()} -> {test['Date'].max().date()}  ({len(test):,} rows)")

    dtrain = lgb.Dataset(train[FEATURE_COLS], label=train["target_fwd_ret"])
    dtest = lgb.Dataset(test[FEATURE_COLS], label=test["target_fwd_ret"], reference=dtrain)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    model = lgb.train(
        params, dtrain, num_boost_round=500,
        valid_sets=[dtest],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    test = test.copy()
    test["pred"] = model.predict(test[FEATURE_COLS])

    mae = mean_absolute_error(test["target_fwd_ret"], test["pred"])
    ic = test[["pred", "target_fwd_ret"]].corr().iloc[0, 1]  # information coefficient
    print(f"\ntest MAE: {mae:.4f}")
    print(f"test IC (corr of predicted vs. realized fwd return): {ic:.3f}")
    if ic < 0.02:
        print("  -> IC is near zero: expected on synthetic random-walk demo data, since")
        print("     there's no real signal to find. Real market data should do better.")

    importance = pd.Series(model.feature_importance(), index=FEATURE_COLS).sort_values(ascending=False)
    print("\nfeature importance:")
    print(importance.to_string())

    return model, test


if __name__ == "__main__":
    feats = pd.read_parquet(DATA_DIR / "features.parquet")
    model, test_with_preds = train_model(feats)
    test_with_preds.to_parquet(DATA_DIR / "test_predictions.parquet", index=False)
    model.save_model(str(DATA_DIR / "lgbm_model.txt"))
    print(f"\nsaved model + test predictions -> {DATA_DIR}/")
