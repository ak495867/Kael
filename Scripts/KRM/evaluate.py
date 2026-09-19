"""Robustness suite. Everything here is DIAGNOSTIC: nothing in this file may be used to pick parameters."""
from __future__ import annotations

from dataclasses import replace
from math import erf, sqrt

import numpy as np
import pandas as pd

from krm_esc import KrmParams, backtest, signal_series, to_position

ANN = 252


def sharpe(r) -> float:
    r = np.asarray(r, float)
    r = r[~np.isnan(r)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(ANN))


def perf(net: pd.Series, held: pd.Series | None = None, turn: pd.Series | None = None) -> dict:
    r = net.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return dict(sharpe=np.nan, sortino=np.nan, maxdd=np.nan, ret_pct=np.nan, turnover=np.nan, mpos=np.nan, n=len(r))
    dd = np.sqrt((np.minimum(r, 0) ** 2).mean())
    eq = (1 + r).cumprod()
    return dict(sharpe=sharpe(r), sortino=float(r.mean() / dd * np.sqrt(ANN)) if dd > 0 else np.nan,
                maxdd=float((1 - eq / eq.cummax()).max()), ret_pct=float((eq.iloc[-1] - 1) * 100),
                turnover=float(turn.loc[r.index].mean()) if turn is not None else np.nan,
                mpos=float(held.loc[r.index].mean()) if held is not None else np.nan, n=len(r))


def perf_bt(bt: pd.DataFrame) -> dict:
    return perf(bt["net"], bt["held"], bt["turnover"])


def _net(ret: np.ndarray, pos: np.ndarray, lag: int, cost_bps: float) -> np.ndarray:
    held = np.r_[np.zeros(lag), pos[:-lag]]
    turn = np.abs(np.diff(np.r_[0.0, held]))
    return held * ret - turn * cost_bps / 1e4


def block_bootstrap(r: np.ndarray, block: int = 20, n_sims: int = 500, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(r)
    nb = int(np.ceil(n / block))
    out = np.empty(n_sims)
    for i in range(n_sims):
        starts = rng.integers(0, n, nb)
        idx = ((starts[:, None] + np.arange(block)[None, :]).ravel()[:n]) % n
        out[i] = sharpe(r[idx])
    return dict(mean=out.mean(), std=out.std(), p5=np.percentile(out, 5), p50=np.percentile(out, 50),
                p95=np.percentile(out, 95), prob_pos=float((out > 0).mean()), n_sims=n_sims)


def placebo_test(ret: np.ndarray, pos: np.ndarray, lag: int, cost: float, n: int = 1000,
                 seed: int = 0, min_shift: int = 20) -> dict:
    """Circularly shift the position path against the returns. Keeps the position distribution and
    autocorrelation, destroys any true timing. Real Sharpe should sit in the far right tail."""
    rng = np.random.default_rng(seed)
    real = sharpe(_net(ret, pos, lag, cost))
    shifts = rng.integers(min_shift, len(ret) - min_shift, size=n)
    plac = np.array([sharpe(_net(ret, np.roll(pos, s), lag, cost)) for s in shifts])
    return dict(real=real, placebo_mean=float(np.nanmean(plac)), placebo_p95=float(np.nanpercentile(plac, 95)),
                p_value=float((1 + np.sum(plac >= real)) / (1 + n)))


def psr(r: np.ndarray, sr_bench: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado): P(true Sharpe > benchmark)."""
    r = pd.Series(r).dropna()
    n = len(r)
    if n < 10 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)                      # per-period
    g3, g4 = float(r.skew()), float(r.kurt()) + 3.0    # raw kurtosis
    denom = sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    z = (sr - sr_bench / sqrt(ANN)) * sqrt(n - 1) / denom
    return 0.5 * (1 + erf(z / sqrt(2)))


def regime_split(net: pd.Series, bench_close: pd.Series) -> pd.DataFrame:
    """bull/bear label for day t uses ONLY information through close t-1."""
    sma = bench_close.rolling(200, min_periods=200).mean()
    valid = sma.shift(1).notna().reindex(net.index, fill_value=False)
    bull = (bench_close > sma).shift(1).fillna(False).astype(bool).reindex(net.index, fill_value=False)
    rows = {}
    for name, mask in (("bull", bull & valid), ("bear", (~bull) & valid)):
        rows[name] = perf(net[mask])
    return pd.DataFrame(rows).T[["sharpe", "sortino", "maxdd", "n"]]


def _split(x: pd.Series, k: int) -> list[pd.Series]:
    """k contiguous, chronologically ordered chunks (Series, index preserved)."""
    x = x.dropna()
    edges = np.linspace(0, len(x), k + 1).astype(int)
    return [x.iloc[a:b] for a, b in zip(edges[:-1], edges[1:])]


def folds(net: pd.Series, k: int = 5) -> pd.DataFrame:
    parts = _split(net, k)
    return pd.DataFrame([dict(fold=i + 1, **{kk: v for kk, v in perf(p).items() if kk in ("sharpe", "sortino", "maxdd", "n")})
                         for i, p in enumerate(parts)]).set_index("fold")


def _probe_set(base: KrmParams, seed: int = 0) -> list[KrmParams]:
    rng = np.random.default_rng(seed)
    out, seen = [], {base.key()}

    def add(q):
        if q.key() not in seen:
            seen.add(q.key()); out.append(q)
    for name in ("m", "n_short", "n_long", "ema_n"):
        for f in (0.6, 0.8, 1.2, 1.4):
            try:
                add(replace(base, **{name: max(2, int(round(getattr(base, name) * f)))}))
            except ValueError:
                pass
    for _ in range(20):
        try:
            add(replace(base, **{n: max(2, int(round(getattr(base, n) * rng.uniform(0.7, 1.3))))
                                 for n in ("m", "n_short", "n_long", "ema_n")}))
        except ValueError:
            pass
    return out


def oos_report(price: pd.Series, ret: pd.Series, bench_close: pd.Series, params: KrmParams,
               split: pd.Timestamp, pos_mode: str, long_only: bool, lag: int, cost_bps: float,
               seed: int = 0, n_boot: int = 500, n_plac: int = 1000) -> dict:

    def run(p: KrmParams, lag_: int = lag, cost: float = cost_bps):
        sig = signal_series(price, p)
        pos = to_position(sig, pos_mode, long_only)
        return sig, pos, backtest(ret, pos, lag_, cost)

    sig, pos, bt = run(params)
    start = sig.first_valid_index()
    live = bt.index >= start
    is_oos = bt.index >= split
    tr, te = bt[live & ~is_oos], bt[is_oos]
    R: dict = {}
    R["headline"] = pd.DataFrame({"in_sample": perf_bt(tr), "out_of_sample": perf_bt(te)}).T
    R["cost"] = pd.DataFrame([dict(cost_bps=c, **{k: v for k, v in perf_bt(backtest(ret, pos, lag, c)[is_oos]).items()
                                                  if k in ("sharpe", "sortino", "maxdd", "turnover", "ret_pct")})
                              for c in (0, 1, 2, 5, 10, 20)]).set_index("cost_bps")
    R["lag"] = pd.DataFrame([dict(lag=l, sharpe=perf_bt(backtest(ret, pos, l, cost_bps)[is_oos])["sharpe"])
                             for l in (1, 2, 3, 5)]).set_index("lag")
    R["bootstrap"] = block_bootstrap(te["net"].to_numpy(), seed=seed, n_sims=n_boot)
    pos_oos = pos[is_oos].to_numpy()
    R["placebo"] = placebo_test(ret[is_oos].to_numpy(), pos_oos, lag, cost_bps, n=n_plac, seed=seed)
    R["psr_vs_0"] = psr(te["net"].to_numpy(), 0.0)

    base_sh = R["headline"].loc["out_of_sample", "sharpe"]
    shs = []
    for q in _probe_set(params, seed):
        try:
            shs.append(perf_bt(run(q)[2][is_oos])["sharpe"])
        except Exception:
            pass
    shs = np.array([s for s in shs if np.isfinite(s)])
    if len(shs):
        p10 = float(np.percentile(shs, 10))
        verdict = "ROBUST" if (base_sh > 0 and p10 >= 0.5 * base_sh and shs.min() > 0) else "FRAGILE"
        R["param_sensitivity"] = dict(baseline=base_sh, probe_mean=float(shs.mean()), probe_std=float(shs.std()),
                                      probe_min=float(shs.min()), probe_p10=p10, probe_max=float(shs.max()),
                                      n_probes=len(shs), verdict=verdict)
    R["folds_full_history"] = folds(bt.loc[live, "net"])
    R["regime_oos"] = regime_split(te["net"], bench_close)

    ab = run(replace(params, use_gate=False))[2][is_oos]
    R["ablation"] = pd.DataFrame({"with_gate": perf_bt(te), "gate_removed (Lambda=1)": perf_bt(ab)}).T[["sharpe", "sortino", "maxdd", "turnover"]]
    bh = pd.Series(ret[is_oos])
    R["benchmark_buy_hold"] = perf(bh)
    R["corr_with_buy_hold"] = float(np.corrcoef(te["net"], bh)[0, 1])
    R["_daily"] = pd.concat([sig.rename("signal"), pos.rename("pos"), bt], axis=1)
    return R


def grid_search_train(price_tr: pd.Series, ret_tr: pd.Series, base: KrmParams, pos_mode: str,
                      long_only: bool, lag: int, cost_bps: float, k: int = 3):
    """Parameter selection using TRAIN DATA ONLY (the caller slices; OOS rows never enter this function).
    Score = mean fold Sharpe - 0.5*std of fold Sharpe  (rewards stability, not one lucky stretch)."""
    rows = []
    for m in (40, 60, 90, 120):
        for ns in (5, 10, 15):
            for nl in (60, 120):
                for en in (3, 5, 10):
                    try:
                        q = replace(base, m=m, n_short=ns, n_long=nl, ema_n=en)
                    except ValueError:
                        continue
                    sig = signal_series(price_tr, q)
                    bt = backtest(ret_tr, to_position(sig, pos_mode, long_only), lag, cost_bps)
                    net = bt.loc[bt.index >= sig.first_valid_index(), "net"]
                    fs = [sharpe(p) for p in _split(net, k)]
                    rows.append(dict(m=m, n_short=ns, n_long=nl, ema_n=en, score=np.nanmean(fs) - 0.5 * np.nanstd(fs),
                                     train_sharpe=sharpe(net)))
    tab = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    b = tab.iloc[0]
    return replace(base, m=int(b.m), n_short=int(b.n_short), n_long=int(b.n_long), ema_n=int(b.ema_n)), tab
