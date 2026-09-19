"""Data loading + hygiene checks. Never fills, never back-fills, never interpolates."""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

REQUIRED = ["Open", "High", "Low", "Close", "Adj Close"]


def load_prices(ticker: str, start: str, end: str | None = None, cache_dir: str = "data_cache",
                csv: str | None = None, refresh: bool = False, drop_today: bool = True) -> pd.DataFrame:
    if csv:
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
    else:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{ticker}.csv")
        if os.path.exists(path) and not refresh:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            import yfinance as yf  # imported lazily so --csv works without it
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.DatetimeIndex(df.index).tz_localize(None)
            df.to_csv(path)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.loc[start:end] if end else df.loc[start:]
    if drop_today and len(df) and df.index[-1].date() >= date.today():
        df = df.iloc[:-1]                      # never use a partial bar
    df = df[REQUIRED].dropna()                 # drop bad rows; NEVER fill them
    return df


def validate_prices(df: pd.DataFrame, name: str = "data", max_gap_days: int = 7) -> list[str]:
    """Raise on hard problems, return list of soft warnings."""
    hard, soft = [], []
    if not isinstance(df.index, pd.DatetimeIndex):
        hard.append("index is not a DatetimeIndex")
    else:
        if not df.index.is_monotonic_increasing:
            hard.append("index not sorted")
        if df.index.has_duplicates:
            hard.append("duplicate dates")
        gap = df.index.to_series().diff().dt.days.max()
        if gap > max_gap_days:
            soft.append(f"largest calendar gap = {int(gap)} days")
    for c in REQUIRED:
        if c not in df:
            hard.append(f"missing column {c}")
    if not hard:
        if df[REQUIRED].isna().any().any():
            hard.append("NaNs present")
        if (df[REQUIRED] <= 0).any().any():
            hard.append("non-positive prices")
        big = df["Adj Close"].pct_change().abs().max()
        if big > 0.4:
            soft.append(f"max |daily return| = {big:.0%} (check splits / bad ticks)")
        if (df["High"] < df["Low"]).any():
            soft.append("rows with High < Low")
    if hard:
        raise ValueError(f"[{name}] data problems: {hard}")
    return soft


def make_synthetic(n: int = 4500, seed: int = 1) -> pd.DataFrame:
    """Regime-switching-volatility random walk. Contains NO exploitable signal by construction."""
    rng = np.random.default_rng(seed)
    vol = np.empty(n)
    v = 0.01
    for i in range(n):
        v = 0.98 * v + 0.02 * 0.01 + 0.00008 * abs(rng.normal())
        vol[i] = v
    r = rng.normal(0.0003, vol)
    close = 100 * np.exp(np.cumsum(r))
    open_ = close * np.exp(rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    idx = pd.bdate_range("2004-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                         "Adj Close": close}, index=idx)
