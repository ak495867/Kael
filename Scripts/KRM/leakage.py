"""
Automatic look-ahead / leakage tests. Each returns (passed: bool, detail: str).
`run_all()` raises if anything fails, so a leaky change can never produce a report.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from krm_esc import backtest

SignalFn = Callable[[pd.Series], pd.Series]


def _same(a: float, b: float, atol: float) -> bool:
    if np.isnan(a) and np.isnan(b):
        return True
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= atol)


def check_truncation_invariance(fn: SignalFn, price: pd.Series, n_checks: int = 30,
                                seed: int = 0, atol: float = 1e-10):
    """signal[k] computed on the FULL series must equal signal[-1] computed on price[:k+1]."""
    rng = np.random.default_rng(seed)
    full = fn(price)
    ks = rng.choice(np.arange(len(price) // 10, len(price) - 1), size=n_checks, replace=False)
    bad = [int(k) for k in ks if not _same(full.iloc[k], fn(price.iloc[: k + 1]).iloc[-1], atol)]
    return (not bad), f"truncation invariance: {n_checks - len(bad)}/{n_checks} ok" + (f", FAILED at rows {bad[:5]}" if bad else "")


def check_future_perturbation(fn: SignalFn, price: pd.Series, n_checks: int = 10,
                              seed: int = 1, atol: float = 1e-10):
    """Destroying ALL data after row k must leave signal[:k+1] untouched."""
    rng = np.random.default_rng(seed)
    full = fn(price)
    ks = rng.choice(np.arange(len(price) // 10, len(price) - 5), size=n_checks, replace=False)
    bad = []
    for k in ks:
        p2 = price.copy()
        tail = len(p2) - (k + 1)
        p2.iloc[k + 1:] = p2.iloc[k + 1:].to_numpy() * np.exp(rng.normal(0, 0.3, tail))
        s2 = fn(p2)
        a, b = full.iloc[: k + 1].to_numpy(), s2.iloc[: k + 1].to_numpy()
        ok = np.all((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= atol))
        if not ok:
            bad.append(int(k))
    return (not bad), f"future perturbation: {n_checks - len(bad)}/{n_checks} ok" + (f", FAILED at rows {bad[:5]}" if bad else "")


def check_backtest_alignment(seed: int = 2, lag: int = 1):
    """(a) exact identity gross[t] = pos[t-lag]*ret[t];  (b) peeking at same-day return earns ~nothing;
       (c) the oracle that knows tomorrow earns a lot (proves the test has power)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=5000)
    ret = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    pos = pd.Series(rng.uniform(-1, 1, len(idx)), index=idx)
    bt = backtest(ret, pos, lag=lag, cost_bps=0.0)
    ident = np.allclose(bt["gross"].iloc[lag:], (pos.shift(lag) * ret).iloc[lag:])

    def sh(x): return x.mean() / x.std(ddof=1) * np.sqrt(252)
    peek = sh(backtest(ret, np.sign(ret), lag=lag, cost_bps=0.0)["net"])
    oracle = sh(backtest(ret, np.sign(ret.shift(-lag)).fillna(0.0), lag=lag, cost_bps=0.0)["net"])
    ok = ident and abs(peek) < 1.2 and oracle > 8
    return ok, f"backtest alignment: identity={ident}, same-day-peek Sharpe={peek:+.2f} (want ~0), oracle Sharpe={oracle:+.1f} (want >>0)"


def check_lag_guard():
    idx = pd.bdate_range("2020-01-01", periods=10)
    s = pd.Series(0.0, index=idx)
    try:
        backtest(s, s, lag=0)
    except ValueError:
        return True, "lag=0 correctly refused"
    return False, "lag=0 was accepted (look-ahead!)"


def check_determinism(fn: SignalFn, price: pd.Series):
    a, b = fn(price), fn(price.copy())
    return bool(a.equals(b)), "deterministic: identical output on identical input"


def run_all(fn: SignalFn, price: pd.Series, verbose: bool = True) -> None:
    results = [check_truncation_invariance(fn, price), check_future_perturbation(fn, price),
               check_backtest_alignment(), check_lag_guard(), check_determinism(fn, price)]
    for ok, msg in results:
        if verbose:
            print(("  [PASS] " if ok else "  [FAIL] ") + msg)
    if not all(ok for ok, _ in results):
        raise RuntimeError("LEAKAGE CHECK FAILED - refusing to produce results")


# ---- negative controls: deliberately leaky signals the tests MUST catch -------------------
def leaky_centered(price: pd.Series) -> pd.Series:
    return price.pct_change().rolling(11, center=True, min_periods=11).mean()

def leaky_shift(price: pd.Series) -> pd.Series:
    return price.pct_change().shift(-1)

def leaky_fullsample_norm(price: pd.Series) -> pd.Series:
    r = price.pct_change()
    return (r - r.mean()) / r.std()
