"""
KRM_ESC - Kramers Escape-Rate Breakout Signal  (core: signal + position + backtest)

CAUSALITY CONTRACT
------------------
* Every value at row t is a function of price rows <= t ONLY.
  (only trailing rolling windows and forward-recursive EMAs are used; no centred
  windows, no shift(-k), no bfill, no full-sample statistics)
* A position decided from the signal at close t earns the return of day t+lag
  (lag >= 1). The engine enforces this in ONE place: `backtest()`.
* Both properties are verified automatically in leakage.py.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class KrmParams:
    m: int = 60            # range (well) window
    n_short: int = 10      # short-vol window  (temperature numerator)
    n_long: int = 60       # long-vol window   (temperature denominator)
    ema_n: int = 5         # output smoothing
    eps: float = 1e-4      # division guard
    sign: int = 1          # +1 breakout, -1 mean-reversion.  FIX BEFORE LOOKING AT OOS.
    use_gate: bool = True  # False = ablation (Lambda := 1, i.e. plain range position)

    def __post_init__(self):
        if self.m < 3 or self.n_short < 2 or self.n_long < 3 or self.ema_n < 1:
            raise ValueError("window too small")
        if self.n_short >= self.n_long:
            raise ValueError("n_short must be < n_long")
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")

    @property
    def warmup(self) -> int:
        """Rows at the start where the signal is forced to NaN (windows + EMA burn-in)."""
        return max(self.m, self.n_long + 1) + 5 * self.ema_n

    def key(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def compute_components(price: pd.Series, p: KrmParams) -> pd.DataFrame:
    """Return u, theta, lam, raw, signal - all strictly causal."""
    price = price.astype(float)
    if not (price > 0).all():
        raise ValueError("prices must be strictly positive")

    lp = np.log(price)                      # Step 1: log price
    r = lp.diff()                           #         log return (first row NaN)

    hi = lp.rolling(p.m, min_periods=p.m).max()   # Step 2: trailing well
    lo = lp.rolling(p.m, min_periods=p.m).min()
    rng = hi - lo
    u = 2.0 * (lp - lo) / rng - 1.0
    u = u.mask(rng == 0, 0.0).clip(-1.0, 1.0)     # flat range -> centre; NaN stays NaN

    s_short = r.rolling(p.n_short, min_periods=p.n_short).std(ddof=1)   # Step 4
    s_long = r.rolling(p.n_long, min_periods=p.n_long).std(ddof=1)
    theta = s_short / s_long.mask(s_long == 0)

    barrier = 1.0 - u ** 2                                   # Step 3 (kappa = 2)
    lam = np.exp(-barrier / (theta ** 2 + p.eps))            # Step 5
    raw = u * lam if p.use_gate else u                       # Step 6
    sig = raw.ewm(span=p.ema_n, adjust=False).mean()         # Step 7 (recursive => causal)

    sig = sig * p.sign
    sig.iloc[: p.warmup] = np.nan                            # never trade the burn-in
    return pd.DataFrame({"u": u, "theta": theta, "lam": lam, "raw": raw, "signal": sig})


def signal_series(price: pd.Series, p: KrmParams) -> pd.Series:
    return compute_components(price, p)["signal"]


def to_position(sig: pd.Series, mode: str = "linear", long_only: bool = False) -> pd.Series:
    """Signal-time position in [-1, 1]. NaN (warm-up) -> flat."""
    pos = sig.fillna(0.0)
    if mode == "sign":
        pos = np.sign(pos)
    elif mode != "linear":
        raise ValueError("mode must be 'linear' or 'sign'")
    if long_only:
        pos = pos.clip(lower=0.0)
    return pos.clip(-1.0, 1.0)


def backtest(ret: pd.Series, pos: pd.Series, lag: int = 1, cost_bps: float = 1.0) -> pd.DataFrame:
    """
    ret[t] = return from close[t-1] to close[t].
    pos[t] = position decided with information up to close[t].
    held[t] = pos[t-lag]  ->  the position that earns ret[t] was decided `lag` bars earlier.
    lag=1 : trade at the same close the signal was computed on (market-on-close).
    lag=2 : one extra day of delay (e.g. trade next open/close) - the safer assumption.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1 (lag=0 would be look-ahead)")
    if not ret.index.equals(pos.index):
        raise ValueError("ret and pos must share the same index")
    held = pos.shift(lag).fillna(0.0)
    turn = held.diff().abs()
    turn.iloc[0] = abs(held.iloc[0])
    gross = held * ret
    net = gross - turn * cost_bps / 1e4
    return pd.DataFrame({"held": held, "turnover": turn, "gross": gross, "net": net})
