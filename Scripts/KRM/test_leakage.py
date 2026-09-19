"""Run with `python -m pytest tests -q` or simply `python tests/test_leakage.py`."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import leakage
from data import make_synthetic
from krm_esc import KrmParams, backtest, compute_components, signal_series

PRICE = make_synthetic(3000)["Adj Close"]


def test_signal_is_causal():
    for p in (KrmParams(), KrmParams(m=20, n_short=5, n_long=30, ema_n=3), KrmParams(sign=-1), KrmParams(use_gate=False)):
        leakage.run_all(lambda s, p=p: signal_series(s, p), PRICE, verbose=False)


def test_negative_controls_are_caught():
    for fn in (leakage.leaky_centered, leakage.leaky_shift, leakage.leaky_fullsample_norm):
        ok1, _ = leakage.check_truncation_invariance(fn, PRICE)
        ok2, _ = leakage.check_future_perturbation(fn, PRICE)
        assert not (ok1 and ok2), f"{fn.__name__} slipped through"


def test_backtest_engine():
    assert leakage.check_backtest_alignment()[0]
    assert leakage.check_lag_guard()[0]


def test_bounds_and_warmup():
    p = KrmParams()
    c = compute_components(PRICE, p)
    s = c["signal"]
    assert s.iloc[: p.warmup].isna().all()
    assert s.dropna().abs().max() <= 1.0 + 1e-12
    assert c["lam"].dropna().between(0, 1).all()
    assert c["u"].dropna().between(-1, 1).all()


def test_limiting_cases():
    # perfectly flat prices -> u=0 -> zero signal, no crash
    flat = pd.Series(100.0, index=pd.bdate_range("2020-01-01", periods=400))
    sig = signal_series(flat, KrmParams()).dropna()
    assert (sig.abs() < 1e-12).all()   # NaN (zero vol) or exactly 0 -> flat position, never a crash


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
