import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_OUT = "Outputs"
os.makedirs(BASE_OUT, exist_ok=True)
plt.style.use("ggplot")

UNIVERSES = {
    "us_equities": [
        "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","V","UNH",
        "HD","PG","MA","DIS","BAC","XOM","PFE","CSCO","VZ","ADBE",
        "KO","PEP","WMT","MRK","ABT","TMO","COST","NKE","ORCL","CRM",
        "AMD","INTC","QCOM","TXN","AVGO","MU","AMAT","IBM","GE","CAT",
        "BA","MMM","HON","LMT","RTX","GS","MS","BLK","AXP","SPGI"
    ],
    "sector_etfs": [
        "XLK","XLF","XLE","XLV","XLP","XLU","XLY","XLI","XLB","XLRE","XLC",
        "SMH","XBI","KRE","XOP","ITB","XRT","IBB","XME","XSD"
    ],
    "international": [
        "EFA","EEM","FXI","EWJ","EWZ","EWG","EWU","EWC","EWA","EWY",
        "INDA","EWT","EWW","EWS","EWH","EWL","EWN","EWD","EWI","EWP"
    ],
    "crypto": [
        "BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD","ADA-USD",
        "DOGE-USD","DOT-USD","LTC-USD","LINK-USD","MATIC-USD","AVAX-USD",
        "ATOM-USD","UNI-USD","XLM-USD"
    ],
    "mixed_assets": [
        "SPY","QQQ","IWM","TLT","IEF","LQD","HYG","GLD","SLV","USO",
        "DBC","VNQ","EFA","EEM","FXI","EWJ","BTC-USD","ETH-USD",
        "AAPL","MSFT","NVDA","TSLA","JPM","XOM","PG","WMT","KO","DIS",
        "XLK","XLF","XLE","XLV","XLU","GLD","TLT","SLV","USO","DBA"
    ],
}

TIMEFRAMES = {
    "1d":  {"interval": "1d",  "period": None,    "ann": 252},
    "1h":  {"interval": "1h",  "period": "730d",  "ann": 1638},
    "1wk": {"interval": "1wk", "period": None,    "ann": 52},
}

RUNS = [
    ("us_equities",  "1d"),
    ("sector_etfs",  "1d"),
    ("international","1d"),
    ("crypto",       "1d"),
    ("mixed_assets", "1d"),
    ("sector_etfs",  "1h"),
    ("crypto",       "1h"),
    ("mixed_assets", "1h"),
    ("us_equities",  "1wk"),
    ("sector_etfs",  "1wk"),
]

START = "2018-01-01"
END   = pd.Timestamp.today().strftime("%Y-%m-%d")


def download_data(tickers, tf_cfg, start, end):
    interval = tf_cfg["interval"]
    period = tf_cfg.get("period")
    kwargs = dict(tickers=tickers, interval=interval, auto_adjust=True,
                  progress=False, group_by="column", threads=True)
    if period:
        kwargs["period"] = period
    else:
        kwargs["start"] = start
        kwargs["end"] = end
    raw = yf.download(**kwargs)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"].copy()
    else:
        data = raw.to_frame("Close")
        data.columns = list(tickers)
    data = data.loc[:, data.notna().mean() > 0.85]
    if data.index.tz is not None:
        data.index = data.index.tz_convert("UTC").tz_localize(None)
    if interval in ("1d", "1wk", "1mo"):
        data = data.loc[data.index.dayofweek < 5]
    data = data.ffill().dropna(how="any")
    return data


def normalise(w):
    denom = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(denom, axis=0).fillna(0)


def scaled_params(ann):
    s = ann / 252.0
    def sv(x, lo):
        return max(lo, int(round(x * s)))
    return {
        "momentum":        {"lookback": sv(252, 30), "skip": sv(21, 2), "holding": sv(21, 2), "quantile": 0.2},
        "q4_breakout":     {"high_window": sv(252, 30), "hold_days": sv(21, 2)},
        "turn_of_month":   {"pre_days": 2, "post_days": 3},
        "mean_reversion":  {"lookback": sv(5, 2), "holding": sv(5, 2), "quantile": 0.2},
        "low_vol":         {"lookback": sv(63, 10), "holding": sv(21, 2), "quantile": 0.3},
        "dual_momentum":   {"lookback": sv(126, 20), "skip": sv(21, 2), "holding": sv(21, 2)},
        "trend_following": {"fast": sv(20, 3), "slow": sv(100, 8), "holding": sv(5, 1)},
    }


def momentum_signal(prices, lookback, skip, holding, quantile):
    mom = prices.shift(skip) / prices.shift(lookback) - 1.0
    ranks = mom.rank(axis=1, pct=True)
    long_mask = ranks >= (1 - quantile)
    short_mask = ranks <= quantile
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    short_w = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w = long_w - short_w
    return w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def q4_breakout_signal(prices, high_window, hold_days):
    rolling_high = prices.rolling(high_window, min_periods=max(2, high_window // 4)).max()
    breakout = (prices >= rolling_high.shift(1)).astype(float)
    months = pd.Series(prices.index.month, index=prices.index)
    in_q4 = months.isin([10, 11, 12]).astype(float)
    sig = breakout.mul(in_q4, axis=0)
    w = sig.rolling(hold_days, min_periods=1).max().shift(1).fillna(0)
    active = w.sum(axis=1).replace(0, np.nan)
    return w.div(active, axis=0).fillna(0)


def turn_of_month_signal(prices, pre_days, post_days):
    idx = prices.index
    day = pd.Series(idx.day, index=idx)
    ym = pd.Series([(d.year, d.month) for d in idx], index=idx)
    max_day = day.groupby(ym).transform("max")
    mask = ((day > (max_day - pre_days)) | (day <= post_days)).astype(float)
    ncols = prices.shape[1]
    w = np.outer(mask.values, np.ones(ncols) / ncols)
    return pd.DataFrame(w, index=idx, columns=prices.columns)


def mean_reversion_signal(prices, lookback, holding, quantile):
    ret = prices.pct_change(lookback)
    ranks = ret.rank(axis=1, pct=True)
    long_mask = ranks <= quantile
    short_mask = ranks >= (1 - quantile)
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    short_w = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w = long_w - short_w
    return w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def low_vol_signal(prices, lookback, holding, quantile):
    vol = prices.pct_change().rolling(lookback).std()
    ranks = vol.rank(axis=1, pct=True)
    long_mask = ranks <= quantile
    short_mask = ranks >= (1 - quantile)
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    short_w = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w = long_w - short_w
    return w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def dual_momentum_signal(prices, lookback, skip, holding):
    mom = prices.shift(skip) / prices.shift(lookback) - 1.0
    pos = (mom > 0).astype(float)
    neg = (mom < 0).astype(float)
    long_w = pos.div(pos.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    short_w = neg.div(neg.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w = long_w - short_w
    return w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def trend_following_signal(prices, fast, slow, holding):
    ma_f = prices.rolling(fast, min_periods=max(2, fast // 2)).mean()
    ma_s = prices.rolling(slow, min_periods=max(3, slow // 2)).mean()
    sig = (ma_f > ma_s).astype(float) - (ma_f < ma_s).astype(float)
    long_mask = (sig > 0)
    short_mask = (sig < 0)
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    short_w = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    w = long_w - short_w
    return w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


STRATEGY_FUNCS = {
    "momentum":        momentum_signal,
    "q4_breakout":     q4_breakout_signal,
    "turn_of_month":   turn_of_month_signal,
    "mean_reversion":  mean_reversion_signal,
    "low_vol":         low_vol_signal,
    "dual_momentum":   dual_momentum_signal,
    "trend_following": trend_following_signal,
}

ALLOCATIONS = {
    "momentum":        0.20,
    "q4_breakout":     0.12,
    "turn_of_month":   0.10,
    "mean_reversion":  0.15,
    "low_vol":         0.13,
    "dual_momentum":   0.15,
    "trend_following": 0.15,
}


def build_all_signals(prices, params):
    out = {}
    for name, fn in STRATEGY_FUNCS.items():
        out[name] = normalise(fn(prices, **params[name]))
    return out


def combine_signals(signals, allocations):
    total = sum(allocations.values())
    combined = None
    for name, w in signals.items():
        a = allocations.get(name, 0.0) / total
        term = w * a
        combined = term if combined is None else combined + term
    return combined.fillna(0)


def backtest(prices, weights, cost_bps=0.0):
    ret = prices.pct_change().fillna(0)
    w = weights.shift(1).fillna(0)
    gross = (w * ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).shift(1).fillna(0)
    cost = turnover * cost_bps / 10000.0
    return (gross - cost), turnover


def max_streak(mask):
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_metrics(returns, ann, benchmark=None, rf=0.02):
    r = returns.dropna()
    if len(r) < 5:
        return {}, pd.Series(dtype=float)
    cum = (1 + r).cumprod()
    total = cum.iloc[-1] - 1
    n_years = len(r) / ann
    ann_ret = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(ann)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(ann)
    sortino = (ann_ret - rf) / downside if downside > 0 else np.nan
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    wins = r[r > 0]
    losses = r[r < 0]
    win_rate = len(wins) / len(r)
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    var95 = np.percentile(r, 5)
    cvar95 = r[r <= var95].mean() if (r <= var95).any() else np.nan
    skew = r.skew()
    kurt = r.kurtosis()
    ulcer = np.sqrt((dd ** 2).mean())
    omega = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    gain_to_pain = r.sum() / abs(losses.sum()) if losses.sum() != 0 else np.nan
    tail_ratio = np.percentile(r, 95) / abs(np.percentile(r, 5))
    ic = np.nan
    ir = np.nan
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        a = r.reindex(b.index)
        if len(a) > 5 and a.std() > 0 and b.std() > 0:
            ic = a.corr(b)
        te = (a - b).std() * np.sqrt(ann)
        if te > 0:
            ir = (a.mean() - b.mean()) * ann / te
    return {
        "Total Return": total,
        "Annualised Return": ann_ret,
        "Annualised Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Win Rate": win_rate,
        "Profit Factor": pf,
        "Information Coefficient (IC)": ic,
        "Information Ratio (IR)": ir,
        "VaR 95%": var95,
        "CVaR 95%": cvar95,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Ulcer Index": ulcer,
        "Omega Ratio": omega,
        "Gain-to-Pain Ratio": gain_to_pain,
        "Tail Ratio": tail_ratio,
        "Best Period": r.max(),
        "Worst Period": r.min(),
        "Max Consecutive Wins": max_streak((r > 0).values),
        "Max Consecutive Losses": max_streak((r < 0).values),
        "Number of Periods": len(r),
    }, dd


def walk_forward(prices, params, ann, train_days, test_days, lookback_grid):
    n = len(prices)
    oos = []
    i = 0
    while i + train_days + test_days <= n:
        tr_s, tr_e = i, i + train_days
        te_s, te_e = tr_e, tr_e + test_days
        best_sharpe, best_lb = -np.inf, lookback_grid[0]
        for lb in lookback_grid:
            local = dict(params)
            m = dict(local["momentum"]); m["lookback"] = lb
            local["momentum"] = m
            d = dict(local["dual_momentum"]); d["lookback"] = lb
            local["dual_momentum"] = d
            sig_tr = build_all_signals(prices.iloc[tr_s:tr_e], local)
            w_tr = combine_signals(sig_tr, ALLOCATIONS)
            r_tr, _ = backtest(prices.iloc[tr_s:tr_e], w_tr)
            r_tr = r_tr.dropna()
            if len(r_tr) < 20 or r_tr.std() == 0:
                continue
            s = r_tr.mean() / r_tr.std() * np.sqrt(ann)
            if s > best_sharpe:
                best_sharpe, best_lb = s, lb
        local = dict(params)
        m = dict(local["momentum"]); m["lookback"] = best_lb
        local["momentum"] = m
        d = dict(local["dual_momentum"]); d["lookback"] = best_lb
        local["dual_momentum"] = d
        sig_te = build_all_signals(prices.iloc[te_s:te_e], local)
        w_te = combine_signals(sig_te, ALLOCATIONS)
        r_te, _ = backtest(prices.iloc[te_s:te_e], w_te)
        oos.append(r_te)
        i += test_days
    return pd.concat(oos).sort_index() if oos else pd.Series(dtype=float)


def monte_carlo_permutation(returns, ann, n_perm=1000):
    r = returns.dropna().values
    if len(r) < 10 or r.std() == 0:
        return np.nan, np.nan, np.array([])
    obs = r.mean() / r.std() * np.sqrt(ann)
    perm = np.empty(n_perm)
    for k in range(n_perm):
        s = np.random.permutation(r)
        perm[k] = s.mean() / s.std() * np.sqrt(ann)
    return float((perm >= obs).mean()), obs, perm


def parameter_robustness(prices, params, ann, pct=0.2):
    base = params["momentum"]["lookback"]
    lbs = sorted(set([max(20, int(base * (1 - pct))), base, int(base * (1 + pct))]))
    base_h = params["momentum"]["holding"]
    holds = sorted(set([max(2, int(base_h * (1 - pct))), base_h, int(base_h * (1 + pct))]))
    res = np.zeros((len(lbs), len(holds)))
    for i, lb in enumerate(lbs):
        for j, h in enumerate(holds):
            local = dict(params)
            m = dict(local["momentum"]); m["lookback"] = lb; m["holding"] = h
            local["momentum"] = m
            sig = build_all_signals(prices, local)
            w = combine_signals(sig, ALLOCATIONS)
            r, _ = backtest(prices, w)
            r = r.dropna()
            res[i, j] = r.mean() / r.std() * np.sqrt(ann) if len(r) > 5 and r.std() > 0 else np.nan
    return lbs, holds, res


def cost_sensitivity(prices, weights, ann, bps_list=(0, 2, 5, 10, 20, 50)):
    out = {}
    for bps in bps_list:
        r, _ = backtest(prices, weights, cost_bps=bps)
        r = r.dropna()
        out[bps] = r.mean() / r.std() * np.sqrt(ann) if len(r) > 5 and r.std() > 0 else np.nan
    return out


def placebo_test(prices, weights, ann, shifts=(1, 5, 10, 21, 42, 63)):
    orig_r, _ = backtest(prices, weights)
    orig_r = orig_r.dropna()
    orig = orig_r.mean() / orig_r.std() * np.sqrt(ann) if orig_r.std() > 0 else np.nan
    out = {}
    for s in shifts:
        wp = weights.shift(s).fillna(0)
        r, _ = backtest(prices, wp)
        r = r.dropna()
        out[s] = r.mean() / r.std() * np.sqrt(ann) if len(r) > 5 and r.std() > 0 else np.nan
    return orig, out


def one_shot_generalisation(prices, params, ann, split=0.5, lookbacks=(63, 126, 189, 252, 378)):
    n = int(len(prices) * split)
    train = prices.iloc[:n]
    test = prices.iloc[n:]
    best_sharpe, best_lb = -np.inf, lookbacks[0]
    for lb in lookbacks:
        local = dict(params)
        m = dict(local["momentum"]); m["lookback"] = lb
        local["momentum"] = m
        sig = build_all_signals(train, local)
        w = combine_signals(sig, ALLOCATIONS)
        r, _ = backtest(train, w)
        r = r.dropna()
        if len(r) < 20 or r.std() == 0:
            continue
        s = r.mean() / r.std() * np.sqrt(ann)
        if s > best_sharpe:
            best_sharpe, best_lb = s, lb
    local = dict(params)
    m = dict(local["momentum"]); m["lookback"] = best_lb
    local["momentum"] = m
    sig = build_all_signals(test, local)
    w = combine_signals(sig, ALLOCATIONS)
    r, _ = backtest(test, w)
    r = r.dropna()
    test_sharpe = r.mean() / r.std() * np.sqrt(ann) if len(r) > 5 and r.std() > 0 else np.nan
    return best_lb, best_sharpe, test_sharpe


def estimate_ann_factor(index, fallback):
    if len(index) < 2:
        return fallback
    years = (index[-1] - index[0]).days / 365.25
    if years <= 0:
        return fallback
    return len(index) / years


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def equity(r):
    return (1 + r.fillna(0)).cumprod()


def run_pipeline(universe_name, tf_name, tickers):
    tag = f"{universe_name}_{tf_name}"
    out = os.path.join(BASE_OUT, tag)
    os.makedirs(out, exist_ok=True)
    tf = TIMEFRAMES[tf_name]

    print("\n" + "=" * 74)
    print(f"PIPELINE :: {tag}")
    print("=" * 74)

    prices = download_data(tickers, tf, START, END)
    if prices.empty or prices.shape[1] < 3:
        print(f"[SKIP] {tag}: insufficient data")
        return None

    ann = estimate_ann_factor(prices.index, tf["ann"])
    params = scaled_params(ann)

    print(f"Tickers   : {prices.shape[1]}")
    print(f"Bars      : {len(prices)}")
    print(f"Span      : {prices.index[0]} -> {prices.index[-1]}")
    print(f"Ann factor: {ann:.1f}")

    signals = build_all_signals(prices, params)
    weights = combine_signals(signals, ALLOCATIONS)
    returns, turnover = backtest(prices, weights)
    returns = returns.dropna()
    if len(returns) < 50:
        print(f"[SKIP] {tag}: not enough returns")
        return None

    bench_ret = prices.pct_change().mean(axis=1).reindex(returns.index).dropna()

    strat_returns = {}
    for name, w in signals.items():
        r, _ = backtest(prices, w)
        strat_returns[name] = r.reindex(returns.index).fillna(0)

    metrics, dd = compute_metrics(returns, ann, bench_ret)
    print("\n--- Full Sample ---")
    for k, v in metrics.items():
        print(f"{k:32s}: {v: .4f}")

    pd.DataFrame(metrics, index=["value"]).T.to_csv(os.path.join(out, "metrics_full.csv"))
    pd.DataFrame(strat_returns).to_csv(os.path.join(out, "strategy_returns.csv"))

    strat_metrics = {}
    for name, r in strat_returns.items():
        sm, _ = compute_metrics(r, ann, bench_ret)
        strat_metrics[name] = sm
    pd.DataFrame(strat_metrics).T.to_csv(os.path.join(out, "metrics_by_strategy.csv"))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(equity(returns), label="Combined", lw=2.4, color="black")
    for name, r in strat_returns.items():
        ax.plot(equity(r), label=name, lw=1.0, alpha=0.75)
    ax.plot(equity(bench_ret), label="EW Benchmark", lw=1.2, ls="--", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(f"[{tag}] Equity Curves (log)")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, os.path.join(out, "01_equity.png"))

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.7)
    ax.set_title(f"[{tag}] Drawdown")
    save(fig, os.path.join(out, "02_drawdown.png"))

    roll_sharpe = returns.rolling(min(126, max(20, len(returns) // 4))).mean() / \
                  returns.rolling(min(126, max(20, len(returns) // 4))).std() * np.sqrt(ann)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(roll_sharpe, color="navy", lw=1.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="green", lw=0.8, ls="--")
    ax.axhline(-1, color="red", lw=0.8, ls="--")
    ax.set_title(f"[{tag}] Rolling Sharpe")
    save(fig, os.path.join(out, "03_rolling_sharpe.png"))

    roll_vol = returns.rolling(min(63, max(10, len(returns) // 6))).std() * np.sqrt(ann)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(roll_vol, color="darkorange", lw=1.4)
    ax.set_title(f"[{tag}] Rolling Annualised Volatility")
    save(fig, os.path.join(out, "04_rolling_vol.png"))

    monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
        lambda x: (1 + x).prod() - 1)
    monthly.index.names = ["y", "m"]
    pivot = monthly.unstack(level="m")
    if len(pivot) > 0:
        fig, ax = plt.subplots(figsize=(11, max(3.5, 0.55 * len(pivot))))
        m_lim = np.nanmax(np.abs(pivot.values))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-m_lim, vmax=m_lim)
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:pivot.shape[1]])
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="Return")
        ax.set_title(f"[{tag}] Monthly Heatmap (%)")
        save(fig, os.path.join(out, "05_monthly_heatmap.png"))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(returns.values, bins=80, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(np.percentile(returns, 5), color="red", ls="--", lw=1.5, label="VaR 95%")
    ax.axvline(returns.mean(), color="black", lw=1.5, label="Mean")
    ax.set_title(f"[{tag}] Return Distribution")
    ax.legend()
    save(fig, os.path.join(out, "06_distribution.png"))

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(turnover.rolling(max(2, min(21, len(turnover) // 8))).mean(), color="purple", lw=1.4)
    ax.set_title(f"[{tag}] Rolling Avg Turnover")
    save(fig, os.path.join(out, "07_turnover.png"))

    fig, ax = plt.subplots(figsize=(13, 6))
    for name, r in strat_returns.items():
        m, _ = compute_metrics(r, ann, bench_ret)
        ax.scatter(m["Annualised Volatility"], m["Annualised Return"], s=80, label=name)
    m_comb, _ = compute_metrics(returns, ann, bench_ret)
    ax.scatter(m_comb["Annualised Volatility"], m_comb["Annualised Return"], s=180,
               color="black", marker="*", label="Combined")
    for name, r in strat_returns.items():
        m, _ = compute_metrics(r, ann, bench_ret)
        ax.annotate(name, (m["Annualised Volatility"], m["Annualised Return"]),
                    fontsize=7, alpha=0.75)
    ax.set_xlabel("Annualised Volatility")
    ax.set_ylabel("Annualised Return")
    ax.set_title(f"[{tag}] Risk-Return by Strategy")
    save(fig, os.path.join(out, "08_risk_return.png"))

    print("\n--- Walk-Forward ---")
    train_days = int(ann * 3)
    test_days = int(ann * 1)
    if len(prices) > train_days + test_days + 20:
        lb_grid = sorted(set([max(20, int(ann * x)) for x in (0.25, 0.5, 0.75, 1.0, 1.5)]))
        wf_ret = walk_forward(prices, params, ann, train_days, test_days, lb_grid)
        if len(wf_ret) > 20:
            wf_m, wf_dd = compute_metrics(wf_ret, ann, bench_ret.reindex(wf_ret.index).dropna())
            for k, v in wf_m.items():
                print(f"{k:32s}: {v: .4f}")
            pd.DataFrame(wf_m, index=["value"]).T.to_csv(os.path.join(out, "metrics_walkforward.csv"))
            fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})
            axes[0].plot(equity(wf_ret), color="black", lw=2, label="WF OOS")
            axes[0].plot(equity(bench_ret.reindex(wf_ret.index).fillna(0)), ls="--", label="Bench")
            axes[0].set_yscale("log")
            axes[0].set_title(f"[{tag}] Walk-Forward Equity")
            axes[0].legend()
            axes[1].fill_between(wf_dd.index, wf_dd.values, 0, color="crimson", alpha=0.7)
            save(fig, os.path.join(out, "09_walkforward.png"))
        else:
            print("[WF] insufficient OOS returns")
    else:
        print("[WF] history too short")

    print("\n--- Monte Carlo ---")
    p_val, obs_sharpe, perm = monte_carlo_permutation(returns, ann, 1000)
    print(f"Observed Sharpe     : {obs_sharpe:.4f}")
    print(f"p-value             : {p_val:.4f}")
    print(f"Null 95% CI         : [{np.percentile(perm, 2.5):.4f}, {np.percentile(perm, 97.5):.4f}]")
    pd.Series({"p_value": p_val, "obs_sharpe": obs_sharpe,
               "ci_lo": np.percentile(perm, 2.5), "ci_hi": np.percentile(perm, 97.5)}
              ).to_csv(os.path.join(out, "monte_carlo.csv"))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(perm, bins=60, color="lightgray", edgecolor="black", alpha=0.85, label="Null")
    ax.axvline(obs_sharpe, color="red", lw=2.2, label=f"Observed = {obs_sharpe:.2f}")
    ax.set_title(f"[{tag}] Monte Carlo (p = {p_val:.4f})")
    ax.legend()
    save(fig, os.path.join(out, "10_monte_carlo.png"))

    print("\n--- Parameter Robustness ---")
    lbs, holds, rob = parameter_robustness(prices, params, ann)
    rob_df = pd.DataFrame(rob, index=[f"lookback={x}" for x in lbs],
                          columns=[f"hold={x}" for x in holds])
    print(rob_df.round(3))
    rob_df.to_csv(os.path.join(out, "param_robustness.csv"))
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(rob, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(holds))); ax.set_xticklabels(holds)
    ax.set_yticks(range(len(lbs))); ax.set_yticklabels(lbs)
    for i in range(len(lbs)):
        for j in range(len(holds)):
            if not np.isnan(rob[i, j]):
                ax.text(j, i, f"{rob[i,j]:.2f}", ha="center", va="center",
                        color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Sharpe")
    ax.set_xlabel("Holding"); ax.set_ylabel("Lookback")
    ax.set_title(f"[{tag}] Parameter Robustness (Sharpe)")
    save(fig, os.path.join(out, "11_param_robustness.png"))

    print("\n--- Cost Sensitivity ---")
    cost = cost_sensitivity(prices, weights, ann)
    for bps, s in cost.items():
        print(f"{bps:4d} bps : Sharpe {s: .4f}")
    pd.Series(cost).to_csv(os.path.join(out, "cost_sensitivity.csv"))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(k) for k in cost.keys()], list(cost.values()), color="teal", edgecolor="black")
    for i, v in enumerate(cost.values()):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Bps"); ax.set_ylabel("Sharpe")
    ax.set_title(f"[{tag}] Cost Sensitivity")
    save(fig, os.path.join(out, "12_cost_sensitivity.png"))

    print("\n--- Placebo Test ---")
    orig_s, placebo = placebo_test(prices, weights, ann)
    print(f"Original Sharpe : {orig_s:.4f}")
    for k, v in placebo.items():
        print(f"Shift {k:3d} : Sharpe {v: .4f}")
    pd.Series(placebo).to_csv(os.path.join(out, "placebo.csv"))
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Original"] + [f"+{k}" for k in placebo.keys()]
    vals = [orig_s] + list(placebo.values())
    colors = ["black"] + ["gray"] * len(placebo)
    ax.bar(labels, vals, color=colors, edgecolor="black")
    ax.axhline(0, color="red", lw=0.8)
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"[{tag}] Placebo Test")
    save(fig, os.path.join(out, "13_placebo.png"))

    print("\n--- One-Shot Generalisation ---")
    best_lb, train_s, test_s = one_shot_generalisation(prices, params, ann)
    print(f"Best lookback on train : {best_lb}")
    print(f"Train Sharpe           : {train_s:.4f}")
    print(f"Test Sharpe (frozen)   : {test_s:.4f}")
    print(f"Degradation            : {train_s - test_s:.4f}")
    pd.Series({"best_lookback": best_lb, "train_sharpe": train_s,
               "test_sharpe": test_s, "degradation": train_s - test_s}
              ).to_csv(os.path.join(out, "one_shot.csv"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Train", "Test (frozen)"], [train_s, test_s],
           color=["steelblue", "darkorange"], edgecolor="black")
    for i, v in enumerate([train_s, test_s]):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Sharpe")
    ax.set_title(f"[{tag}] One-Shot Generalisation")
    save(fig, os.path.join(out, "14_one_shot.png"))

    summary = pd.DataFrame({
        "Full Sample": pd.Series(metrics),
    })
    summary.to_csv(os.path.join(out, "summary.csv"))

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({
            "universe": universe_name, "timeframe": tf_name,
            "tickers": list(prices.columns), "ann_factor": ann,
            "allocations": ALLOCATIONS, "params": params
        }, f, indent=2, default=str)

    result = {
        "universe": universe_name, "timeframe": tf_name,
        "n_tickers": prices.shape[1], "n_bars": len(prices),
        "ann_factor": ann,
        "sharpe": metrics.get("Sharpe Ratio", np.nan),
        "ann_return": metrics.get("Annualised Return", np.nan),
        "max_dd": metrics.get("Max Drawdown", np.nan),
        "sortino": metrics.get("Sortino Ratio", np.nan),
        "calmar": metrics.get("Calmar Ratio", np.nan),
        "mc_p": p_val,
    }
    print(f"\n[DONE] {tag} saved to {out}")
    return result


def main():
    results = []
    for universe_name, tf_name in RUNS:
        tickers = UNIVERSES.get(universe_name)
        if not tickers:
            continue
        try:
            r = run_pipeline(universe_name, tf_name, tickers)
            if r:
                results.append(r)
        except Exception as e:
            print(f"[ERROR] {universe_name}_{tf_name}: {e}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(BASE_OUT, "00_overall_summary.csv"), index=False)
        print("\n" + "=" * 74)
        print("OVERALL SUMMARY ACROSS ALL RUNS")
        print("=" * 74)
        print(df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(13, 6))
        labels = [f"{r['universe']}\n{r['timeframe']}" for r in results]
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, [r["sharpe"] for r in results], w, label="Sharpe", color="steelblue")
        ax.bar(x + w / 2, [r["sortino"] for r in results], w, label="Sortino", color="darkorange")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("Ratio")
        ax.set_title("Cross-Run Sharpe / Sortino Comparison")
        ax.legend()
        save(fig, os.path.join(BASE_OUT, "00_overall_summary.png"))

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(labels, [r["max_dd"] for r in results], color="crimson", edgecolor="black")
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Max Drawdown")
        ax.set_title("Cross-Run Max Drawdown")
        save(fig, os.path.join(BASE_OUT, "00_overall_drawdown.png"))

    print(f"\nAll outputs saved to ./{BASE_OUT}/")


if __name__ == "__main__":
    main()