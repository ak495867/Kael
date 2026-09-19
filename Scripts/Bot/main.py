import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

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
        "DOGE-USD","DOT-USD","LTC-USD","LINK-USD","AVAX-USD","ATOM-USD"
    ],
    "treasuries_and_credit": [
        "SHY","IEI","IEF","TLH","TLT","TIP","LQD","HYG","EMB","MBB","AGG"
    ],
    "mixed_assets": [
        "SPY","QQQ","IWM","TLT","IEF","LQD","HYG","GLD","SLV","USO",
        "DBC","VNQ","EFA","EEM","FXI","EWJ"
    ],
}

TIMEFRAMES = {
    "1d":  {"interval": "1d",  "period": None,   "ann": 252},
    "1h":  {"interval": "1h",  "period": "730d", "ann": 1638},
    "1wk": {"interval": "1wk", "period": None,   "ann": 52},
}

RUNS = [
    ("us_equities",           "1d"),
    ("sector_etfs",           "1d"),
    ("international",         "1d"),
    ("crypto",                "1d"),
    ("treasuries_and_credit", "1d"),
    ("mixed_assets",          "1d"),
    ("sector_etfs",           "1h"),
    ("crypto",                "1h"),
    ("mixed_assets",          "1h"),
    ("us_equities",           "1wk"),
    ("sector_etfs",           "1wk"),
]

START = "2018-01-01"
END   = pd.Timestamp.today().strftime("%Y-%m-%d")
VOL_TARGET = 0.10
MAX_LEV = 2.0


def download_data(tickers, tf_cfg, start, end):
    kwargs = dict(tickers=tickers, interval=tf_cfg["interval"], auto_adjust=True,
                  progress=False, group_by="column", threads=True)
    if tf_cfg.get("period"):
        kwargs["period"] = tf_cfg["period"]
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
    if tf_cfg["interval"] in ("1d", "1wk", "1mo"):
        data = data.loc[data.index.dayofweek < 5]
    data = data.ffill().dropna(how="any")
    return data


def normalise(w):
    denom = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(denom, axis=0).fillna(0)


def compute_regime(prices, ann):
    window = max(40, int(0.8 * ann))
    bench = prices.mean(axis=1)
    ma = bench.rolling(window, min_periods=max(10, window // 4)).mean()
    uptrend = (bench > ma).astype(float)
    return uptrend, bench


def scaled_params(ann):
    s = ann / 252.0
    def sv(x, lo):
        return max(lo, int(round(x * s)))
    return {
        "momentum": {
            "lookbacks": (sv(126, 30), sv(189, 40), sv(252, 50), sv(378, 60)),
            "skip": sv(21, 3), "holding": sv(21, 3), "quantile": 0.2
        },
        "q4_breakout":     {"high_window": sv(252, 40), "hold_days": sv(21, 3)},
        "turn_of_month":   {"pre_days": 2, "post_days": 4},
        "mean_reversion":  {"lookback": sv(5, 2), "holding": sv(5, 2), "quantile": 0.2},
        "low_vol":         {"lookback": sv(63, 15), "holding": sv(21, 3), "quantile": 0.25},
        "dual_momentum":   {"lookback": sv(126, 30), "skip": sv(21, 3), "holding": sv(21, 3)},
        "trend_following": {"fast": sv(20, 4), "slow": sv(100, 12), "holding": sv(5, 1)},
        "short_term_reversal": {
            "lookback": sv(5, 2), "holding": sv(5, 2),
            "quantile": 0.2, "trend_window": sv(200, 40),
        },
        "cs_neutral_momentum": {
            "lookback": sv(252, 50), "skip": sv(21, 3), "holding": sv(21, 3),
            "quantile": 0.1, "vol_window": sv(126, 30),
        },
        "pead_proxy": {
            "move_threshold": 0.04, "holding": sv(21, 3), "trend_window": sv(200, 40),
        },
    }


def vol_scaled_momentum(prices, lookback, skip, holding, quantile, uptrend):
    ret = prices.pct_change()
    mom = prices.shift(skip) / prices.shift(lookback) - 1.0
    vol = ret.rolling(max(20, lookback // 4), min_periods=10).std()
    score = mom / vol.replace(0, np.nan)
    ma = prices.rolling(lookback, min_periods=max(10, lookback // 3)).mean()
    above = (prices > ma).astype(float)
    ranks = score.rank(axis=1, pct=True)
    long_mask = (ranks >= (1 - quantile)) & (above > 0)
    short_mask = (ranks <= quantile) & (above < 1)
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    short_w = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if uptrend is not None:
        gate = 0.15 + 0.85 * (1.0 - uptrend)
        short_w = short_w.mul(gate, axis=0)
    w = long_w - 0.3 * short_w
    return w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def ensemble_momentum(prices, params, uptrend):
    ws = [vol_scaled_momentum(prices, lb, params["skip"], params["holding"],
                              params["quantile"], uptrend)
          for lb in params["lookbacks"]]
    return sum(ws) / len(ws)


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
    w = np.outer(mask.values, np.ones(prices.shape[1]) / prices.shape[1])
    return pd.DataFrame(w, index=idx, columns=prices.columns)


def mean_reversion_signal(prices, lookback, holding, quantile):
    ret = prices.pct_change(lookback)
    ma = prices.rolling(max(20, 3 * lookback), min_periods=10).mean()
    above = (prices > ma).astype(float)
    ranks = ret.rank(axis=1, pct=True)
    long_mask = (ranks <= quantile) & (above > 0)
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return long_w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def low_vol_signal(prices, lookback, holding, quantile):
    vol = prices.pct_change().rolling(lookback).std()
    ma = prices.rolling(max(20, lookback * 2), min_periods=10).mean()
    above = (prices > ma).astype(float)
    ranks = vol.rank(axis=1, pct=True)
    long_mask = (ranks <= quantile) & (above > 0)
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return long_w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def dual_momentum_signal(prices, lookback, skip, holding):
    mom = prices.shift(skip) / prices.shift(lookback) - 1.0
    abs_pos = (mom > 0).astype(float)
    ma = prices.rolling(lookback, min_periods=max(10, lookback // 3)).mean()
    above = (prices > ma).astype(float)
    long_mask = abs_pos * above
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return long_w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def trend_following_signal(prices, fast, slow, holding):
    ma_f = prices.rolling(fast, min_periods=max(2, fast // 2)).mean()
    ma_s = prices.rolling(slow, min_periods=max(3, slow // 2)).mean()
    sig = (ma_f > ma_s).astype(float)
    long_w = sig.div(sig.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return long_w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def short_term_reversal_signal(prices, lookback, holding, quantile, trend_window,
                               uptrend=None):
    ret = prices.pct_change(lookback)
    ma = prices.rolling(trend_window, min_periods=max(10, trend_window // 3)).mean()
    above = (prices > ma).astype(float)
    ranks = ret.rank(axis=1, pct=True)
    long_mask = (ranks <= quantile) & (above > 0)
    short_mask = (ranks >= 1 - quantile) & (above < 1)
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    short_w = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if uptrend is not None:
        gate = 0.15 + 0.85 * (1.0 - uptrend)
        short_w = short_w.mul(gate, axis=0)
    w = long_w - 0.5 * short_w
    return w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def cs_neutral_momentum_signal(prices, lookback, skip, holding, quantile, vol_window,
                               uptrend=None):
    mom = prices.shift(skip) / prices.shift(lookback) - 1.0
    mom_neutral = mom.sub(mom.mean(axis=1), axis=0)
    vol = mom_neutral.rolling(vol_window, min_periods=max(10, vol_window // 3)).std()
    z = mom_neutral / vol.replace(0, np.nan)
    ranks = z.rank(axis=1, pct=True)
    long_mask = ranks >= 1 - quantile
    short_mask = ranks <= quantile
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    short_w = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    if uptrend is not None:
        gate = 0.15 + 0.85 * (1.0 - uptrend)
        short_w = short_w.mul(gate, axis=0)
    w = long_w - short_w
    return w.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)


def pead_proxy_signal(prices, move_threshold, holding, trend_window):
    ret = prices.pct_change()
    big_pos = (ret > move_threshold).astype(float)
    trend = (prices > prices.rolling(trend_window, min_periods=max(10, trend_window // 3)).mean()).astype(float)
    active = (big_pos * trend).rolling(holding, min_periods=1).max().shift(1).fillna(0)
    denom = active.sum(axis=1).replace(0, np.nan)
    return active.div(denom, axis=0).fillna(0)


STRATEGY_FUNCS = {
    "momentum":            ensemble_momentum,
    "q4_breakout":         q4_breakout_signal,
    "turn_of_month":       turn_of_month_signal,
    "mean_reversion":      mean_reversion_signal,
    "low_vol":             low_vol_signal,
    "dual_momentum":       dual_momentum_signal,
    "trend_following":     trend_following_signal,
    "short_term_reversal": short_term_reversal_signal,
    "cs_neutral_momentum": cs_neutral_momentum_signal,
    "pead_proxy":          pead_proxy_signal,
}

BASE_ALLOC = {
    "momentum":            0.14,
    "q4_breakout":         0.06,
    "turn_of_month":       0.05,
    "mean_reversion":      0.08,
    "low_vol":             0.10,
    "dual_momentum":       0.10,
    "trend_following":     0.10,
    "short_term_reversal": 0.12,
    "cs_neutral_momentum": 0.15,
    "pead_proxy":          0.10,
}


def build_all_signals(prices, params, uptrend):
    out = {}
    for name, fn in STRATEGY_FUNCS.items():
        if name == "momentum":
            out[name] = normalise(fn(prices, params[name], uptrend))
        elif name in ("short_term_reversal", "cs_neutral_momentum"):
            out[name] = normalise(fn(prices, uptrend=uptrend, **params[name]))
        else:
            out[name] = normalise(fn(prices, **params[name]))
    return out


def cross_sectional_ic(prices, weights, horizon, step=None):
    if step is None:
        step = max(1, horizon)
    fwd = prices.shift(-horizon) / prices - 1.0
    ics = []
    for i in range(0, len(prices) - horizon, step):
        sig = weights.iloc[i]
        ret_fwd = fwd.iloc[i]
        common = sig.dropna().index.intersection(ret_fwd.dropna().index)
        common = common[sig[common].abs() > 1e-8]
        if len(common) < 5:
            continue
        try:
            ic = stats.spearmanr(sig[common].values, ret_fwd[common].values).correlation
            if not np.isnan(ic):
                ics.append(ic)
        except Exception:
            continue
    if len(ics) == 0:
        return np.nan, np.nan, 0
    ics = np.array(ics)
    mean_ic = float(np.nanmean(ics))
    ir_ic = mean_ic / (np.nanstd(ics) / np.sqrt(len(ics))) if len(ics) > 1 else np.nan
    return mean_ic, ir_ic, len(ics)


def rolling_ic_weight(signals, prices, horizon, window_ic, step, min_obs=6):
    fwd = prices.shift(-horizon) / prices - 1.0
    rebal_idx = list(range(0, len(prices) - horizon, step))
    if len(rebal_idx) == 0:
        return pd.DataFrame(0.0, index=prices.index, columns=list(signals.keys()))
    ic_records = {}
    for name, w in signals.items():
        ics = []
        for i in rebal_idx:
            sig = w.iloc[i]
            rf = fwd.iloc[i]
            common = sig.dropna().index.intersection(rf.dropna().index)
            common = common[sig[common].abs() > 1e-8]
            if len(common) < 5:
                ics.append(np.nan)
                continue
            try:
                v = stats.spearmanr(sig[common].values, rf[common].values).correlation
            except Exception:
                v = np.nan
            ics.append(v)
        ic_records[name] = pd.Series(ics, index=prices.index[rebal_idx])
    ic_df = pd.DataFrame(ic_records)
    ic_roll = ic_df.rolling(window_ic, min_periods=min_obs).mean()
    ic_roll = ic_roll.clip(lower=0.0)
    w = ic_roll.div(ic_roll.sum(axis=1).replace(0, np.nan), axis=0)
    w = w.fillna(0)
    w = w.reindex(prices.index, method="ffill").fillna(0)
    return w.shift(1)


def strategy_cs_ic_table(prices, signals, horizon, step):
    rows = {}
    for name, w in signals.items():
        ic, ir, n = cross_sectional_ic(prices, w, horizon=horizon, step=step)
        rows[name] = {"cs_ic": ic, "ic_ir": ir, "n_obs": n}
    return pd.DataFrame(rows).T.sort_values("cs_ic", ascending=False)


def combine_signals_ic(signals, ic_w):
    combined = None
    for name, w in signals.items():
        if name not in ic_w.columns:
            continue
        wt = ic_w[name].reindex(w.index).fillna(0)
        term = w.mul(wt, axis=0)
        combined = term if combined is None else combined + term
    if combined is None:
        first = next(iter(signals.values()))
        return pd.DataFrame(0.0, index=first.index, columns=first.columns)
    return combined.fillna(0)


def apply_vol_target(weights, prices, ann, target=VOL_TARGET, max_lev=MAX_LEV):
    ret = prices.pct_change().fillna(0)
    prev_w = weights.shift(1).fillna(0)
    strat_ret = (prev_w * ret).sum(axis=1)
    window = max(20, int(ann * 0.25))
    realized = strat_ret.rolling(window, min_periods=max(10, window // 3)).std() * np.sqrt(ann)
    scale = (target / realized).clip(upper=max_lev).shift(1).fillna(1.0)
    return weights.mul(scale, axis=0)


def backtest(prices, weights, cost_bps=0.0):
    ret = prices.pct_change().fillna(0)
    w = weights.shift(1).fillna(0)
    gross = (w * ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).shift(1).fillna(0)
    cost = turnover * cost_bps / 10000.0
    return gross - cost, turnover


def max_streak(mask):
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def deflated_sharpe_ratio(returns, ann, n_trials=100):
    r = returns.dropna()
    n = len(r)
    if n < 30 or r.std() == 0:
        return np.nan
    sr_ann = r.mean() / r.std() * np.sqrt(ann)
    sr_p = sr_ann / np.sqrt(ann)
    skew = r.skew()
    kurt = r.kurtosis() + 3.0
    var_sr = (1 + 0.5 * sr_p**2 - skew * sr_p + (kurt - 3) / 4 * sr_p**2) / max(1, n - 1)
    se = np.sqrt(var_sr)
    if se == 0:
        return np.nan
    emc = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    exp_max = (1 - emc) * z1 + emc * z2
    dsr = stats.norm.cdf((sr_p - exp_max * se) / se)
    return dsr


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
    beta = np.nan
    alpha = np.nan
    ir = np.nan
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        a = r.reindex(b.index)
        if len(a) > 5 and a.std() > 0 and b.std() > 0:
            beta = a.cov(b) / b.var()
            alpha = (a.mean() - beta * b.mean()) * ann
        te = (a - b).std() * np.sqrt(ann)
        if te > 0:
            ir = (a.mean() - b.mean()) * ann / te
    dsr = deflated_sharpe_ratio(r, ann, n_trials=100)
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
        "Deflated Sharpe Ratio": dsr,
        "Information Ratio (IR)": ir,
        "Alpha (ann.)": alpha,
        "Beta": beta,
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


def monte_carlo_signflip(returns, ann, n_perm=2000):
    r = returns.dropna().values
    n = len(r)
    if n < 30 or r.std() == 0:
        return np.nan, np.nan, np.array([])
    obs = r.mean() / r.std() * np.sqrt(ann)
    perm = np.empty(n_perm)
    for k in range(n_perm):
        signs = np.random.choice([-1.0, 1.0], size=n)
        rs = r * signs
        if rs.std() == 0:
            perm[k] = np.nan
            continue
        perm[k] = rs.mean() / rs.std() * np.sqrt(ann)
    perm = perm[~np.isnan(perm)]
    p = float((perm >= obs).mean()) if len(perm) > 0 else np.nan
    return p, obs, perm


def cs_shuffle_placebo(prices, weights, ann, n_perm=200):
    orig_r, _ = backtest(prices, weights)
    orig_r = orig_r.dropna()
    orig = orig_r.mean() / orig_r.std() * np.sqrt(ann) if orig_r.std() > 0 else np.nan
    perm_sharpes = []
    w_arr = weights.values.copy()
    n_rows, n_cols = w_arr.shape
    if n_cols < 2:
        return orig, np.nan, np.array([])
    for k in range(n_perm):
        w_shuf = w_arr.copy()
        for i in range(n_rows):
            np.random.shuffle(w_shuf[i])
        w_df = pd.DataFrame(w_shuf, index=weights.index, columns=weights.columns)
        r, _ = backtest(prices, w_df)
        r = r.dropna()
        if len(r) > 20 and r.std() > 0:
            perm_sharpes.append(r.mean() / r.std() * np.sqrt(ann))
    perm_sharpes = np.array(perm_sharpes)
    p = float((perm_sharpes >= orig).mean()) if len(perm_sharpes) > 0 else np.nan
    return orig, p, perm_sharpes


def time_shift_placebo(prices, weights, ann, shifts=(1, 5, 10, 21, 42, 63)):
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


def signal_decay(prices, weights, ann, horizons=(1, 2, 5, 10, 21, 42)):
    out = {}
    for h in horizons:
        w = weights.shift(h).fillna(0)
        r, _ = backtest(prices, w)
        r = r.dropna()
        out[h] = r.mean() / r.std() * np.sqrt(ann) if len(r) > 5 and r.std() > 0 else np.nan
    return out


def parameter_robustness(prices, params, ann, uptrend, pct=0.2):
    base_lbs = params["momentum"]["lookbacks"]
    base_lb = base_lbs[len(base_lbs) // 2]
    lbs = sorted(set([max(30, int(base_lb * (1 - pct))), base_lb, int(base_lb * (1 + pct))]))
    base_h = params["momentum"]["holding"]
    holds = sorted(set([max(2, int(base_h * (1 - pct))), base_h, int(base_h * (1 + pct))]))
    res = np.zeros((len(lbs), len(holds)))
    horizon_ic = max(5, int(ann / 12))
    step_ic = max(2, horizon_ic)
    window_ic = max(8, int(ann * 0.5) // step_ic)
    for i, lb in enumerate(lbs):
        for j, h in enumerate(holds):
            local = dict(params)
            local["momentum"] = {"lookbacks": (lb,), "skip": params["momentum"]["skip"],
                                 "holding": h, "quantile": params["momentum"]["quantile"]}
            sig = build_all_signals(prices, local, uptrend)
            ic_w = rolling_ic_weight(sig, prices, horizon_ic, window_ic, step_ic)
            w = apply_vol_target(combine_signals_ic(sig, ic_w), prices, ann)
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


def walk_forward_windows(n_bars, ann):
    if n_bars < int(ann * 1.5):
        return None
    if n_bars >= int(ann * 5):
        return int(ann * 3), int(ann * 1)
    if n_bars >= int(ann * 3):
        return int(ann * 1.5), int(ann * 0.5)
    return int(ann * 1.0), int(ann * 0.25)


def walk_forward(prices, params, ann):
    win = walk_forward_windows(len(prices), ann)
    if win is None:
        print("  [WF] history too short")
        return pd.Series(dtype=float)
    train_days, test_days = win
    if test_days < 5:
        test_days = 5
    uptrend_full, _ = compute_regime(prices, ann)

    horizon_ic = max(5, int(ann / 12))
    step_ic = max(2, horizon_ic)
    window_ic = max(8, int(ann * 0.5) // step_ic)

    signals_full = build_all_signals(prices, params, uptrend_full)
    ic_w_full = rolling_ic_weight(signals_full, prices, horizon_ic,
                                  window_ic, step_ic)
    raw_full = combine_signals_ic(signals_full, ic_w_full)
    weights_full = apply_vol_target(raw_full, prices, ann)

    oos = []
    i = 0
    while i + train_days + test_days <= len(prices):
        te_s = i + train_days
        te_e = te_s + test_days
        w_te = weights_full.iloc[te_s:te_e]
        r_te, _ = backtest(prices.iloc[te_s:te_e], w_te)
        r_te = r_te.dropna()
        if len(r_te) > 20:
            oos.append(r_te)
        i += test_days

    if not oos:
        print("  [WF] no valid OOS windows produced")
        return pd.Series(dtype=float)
    out = pd.concat(oos)
    out = out[~out.index.duplicated(keep="first")].sort_index()
    return out


def one_shot_generalisation(prices, params, ann, uptrend, split=0.5):
    n = int(len(prices) * split)
    train = prices.iloc[:n]
    test = prices.iloc[n:]

    horizon_ic = max(5, int(ann / 12))
    step_ic = max(2, horizon_ic)
    window_ic = max(8, int(ann * 0.5) // step_ic)

    up_tr = uptrend.iloc[:n]
    sig_tr = build_all_signals(train, params, up_tr)
    ic_w_tr = rolling_ic_weight(sig_tr, train, horizon_ic, window_ic, step_ic)
    w_tr = apply_vol_target(combine_signals_ic(sig_tr, ic_w_tr), train, ann)
    r_tr, _ = backtest(train, w_tr)
    r_tr = r_tr.dropna()
    train_sharpe = r_tr.mean() / r_tr.std() * np.sqrt(ann) if r_tr.std() > 0 else np.nan

    up_te = uptrend.iloc[n:]
    sig_te = build_all_signals(test, params, up_te)
    ic_w_te = rolling_ic_weight(sig_te, test, horizon_ic, window_ic, step_ic)
    w_te = apply_vol_target(combine_signals_ic(sig_te, ic_w_te), test, ann)
    r_te, _ = backtest(test, w_te)
    r_te = r_te.dropna()
    test_sharpe = r_te.mean() / r_te.std() * np.sqrt(ann) if r_te.std() > 0 else np.nan
    return train_sharpe, test_sharpe


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
    uptrend, bench_price = compute_regime(prices, ann)

    print(f"Tickers   : {prices.shape[1]}")
    print(f"Bars      : {len(prices)}")
    print(f"Span      : {prices.index[0]} -> {prices.index[-1]}")
    print(f"Ann factor: {ann:.1f}")

    signals = build_all_signals(prices, params, uptrend)
    ret = prices.pct_change().fillna(0)

    horizon_ic = max(5, int(ann / 12))
    step_ic = max(2, horizon_ic)
    window_ic = max(8, int(ann * 0.5) // step_ic)

    ic_w = rolling_ic_weight(signals, prices, horizon_ic, window_ic, step_ic)
    raw_w = combine_signals_ic(signals, ic_w)
    weights = apply_vol_target(raw_w, prices, ann)

    returns, turnover = backtest(prices, weights)
    returns = returns.dropna()
    if len(returns) < 50:
        print(f"[SKIP] {tag}: not enough returns")
        return None

    bench_ret = ret.mean(axis=1).reindex(returns.index).fillna(0)

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

    mean_ic, ir_ic, n_ic = cross_sectional_ic(prices, weights, horizon=horizon_ic, step=step_ic)
    print(f"\nCombined Cross-sectional IC (h={horizon_ic})  : {mean_ic: .4f}")
    print(f"Combined IC Information Ratio                 : {ir_ic: .4f}")
    print(f"IC observations                               : {n_ic}")
    pd.Series({"mean_ic": mean_ic, "ic_ir": ir_ic, "n": n_ic}).to_csv(
        os.path.join(out, "cross_sectional_ic.csv"))

    ic_table = strategy_cs_ic_table(prices, signals, horizon=horizon_ic, step=step_ic)
    print("\n--- Per-Strategy Cross-Sectional IC ---")
    print(ic_table.round(4))
    ic_table.to_csv(os.path.join(out, "strategy_cs_ic.csv"))

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
    ax.legend(loc="upper left", fontsize=7)
    save(fig, os.path.join(out, "01_equity.png"))

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.7)
    ax.set_title(f"[{tag}] Drawdown")
    save(fig, os.path.join(out, "02_drawdown.png"))

    rwin = min(126, max(20, len(returns) // 4))
    roll_sharpe = returns.rolling(rwin).mean() / returns.rolling(rwin).std() * np.sqrt(ann)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(roll_sharpe, color="navy", lw=1.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="green", lw=0.8, ls="--")
    ax.axhline(-1, color="red", lw=0.8, ls="--")
    ax.set_title(f"[{tag}] Rolling Sharpe")
    save(fig, os.path.join(out, "03_rolling_sharpe.png"))

    vwin = min(63, max(10, len(returns) // 6))
    roll_vol = returns.rolling(vwin).std() * np.sqrt(ann)
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
    ax.hist(returns.values, bins=min(80, max(10, len(returns) // 10)),
            color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(np.percentile(returns, 5), color="red", ls="--", lw=1.5, label="VaR 95%")
    ax.axvline(returns.mean(), color="black", lw=1.5, label="Mean")
    ax.set_title(f"[{tag}] Return Distribution")
    ax.legend()
    save(fig, os.path.join(out, "06_distribution.png"))

    tw = max(2, min(21, len(turnover) // 8))
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(turnover.rolling(tw).mean(), color="purple", lw=1.4)
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
    ax.set_xlabel("Annualised Volatility"); ax.set_ylabel("Annualised Return")
    ax.set_title(f"[{tag}] Risk-Return by Strategy")
    save(fig, os.path.join(out, "08_risk_return.png"))

    fig, ax = plt.subplots(figsize=(11, 5))
    ic_table["cs_ic"].plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(0.03, color="green", lw=1.0, ls="--", label="Tradable IC threshold")
    ax.set_title(f"[{tag}] Per-Strategy Cross-Sectional IC (h={horizon_ic})")
    ax.legend()
    save(fig, os.path.join(out, "16_strategy_cs_ic.png"))

    print("\n--- Walk-Forward (full-history IC weights, sliced) ---")
    wf_ret = walk_forward(prices, params, ann)
    if len(wf_ret) > 20:
        wf_m, wf_dd = compute_metrics(wf_ret, ann, bench_ret.reindex(wf_ret.index).fillna(0))
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
        wf_m = {}

    print("\n--- Monte Carlo (sign-flip, H0: mean=0) ---")
    p_val, obs_sharpe, perm = monte_carlo_signflip(returns, ann, n_perm=2000)
    print(f"Observed Sharpe     : {obs_sharpe:.4f}")
    print(f"p-value             : {p_val:.4f}")
    if len(perm) > 0:
        print(f"Null 95% CI         : [{np.percentile(perm, 2.5):.4f}, {np.percentile(perm, 97.5):.4f}]")
        pd.Series({"p_value": p_val, "obs_sharpe": obs_sharpe,
                   "ci_lo": np.percentile(perm, 2.5),
                   "ci_hi": np.percentile(perm, 97.5)}).to_csv(
            os.path.join(out, "monte_carlo.csv"))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(perm, bins=min(60, max(10, len(perm) // 10)),
                color="lightgray", edgecolor="black", alpha=0.85, label="Null (sign-flip)")
        ax.axvline(obs_sharpe, color="red", lw=2.2, label=f"Observed = {obs_sharpe:.2f}")
        ax.set_title(f"[{tag}] Monte Carlo sign-flip (p = {p_val:.4f})")
        ax.legend()
        save(fig, os.path.join(out, "10_monte_carlo.png"))

    print("\n--- Cross-sectional shuffle placebo ---")
    orig_s, cs_p, cs_perm = cs_shuffle_placebo(prices, weights, ann, n_perm=200)
    print(f"Original Sharpe     : {orig_s:.4f}")
    print(f"CS-shuffle p-value  : {cs_p:.4f}")
    if len(cs_perm) > 0:
        print(f"CS Null 95% CI      : [{np.percentile(cs_perm, 2.5):.4f}, {np.percentile(cs_perm, 97.5):.4f}]")
        pd.Series({"p_value": cs_p, "orig_sharpe": orig_s,
                   "ci_lo": np.percentile(cs_perm, 2.5),
                   "ci_hi": np.percentile(cs_perm, 97.5)}).to_csv(
            os.path.join(out, "cs_placebo.csv"))
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.hist(cs_perm, bins=min(60, max(10, len(cs_perm) // 10)),
                color="lightsteelblue", edgecolor="black", alpha=0.85,
                label="Null (cross-sectional shuffle)")
        ax.axvline(orig_s, color="red", lw=2.2, label=f"Observed = {orig_s:.2f}")
        ax.set_title(f"[{tag}] CS-shuffle placebo (p = {cs_p:.4f})")
        ax.legend()
        save(fig, os.path.join(out, "10b_cs_placebo.png"))

    print("\n--- Parameter Robustness ---")
    lbs, holds, rob = parameter_robustness(prices, params, ann, uptrend)
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

    print("\n--- Time-shift placebo ---")
    orig_s2, placebo = time_shift_placebo(prices, weights, ann)
    print(f"Original Sharpe : {orig_s2:.4f}")
    for k, v in placebo.items():
        print(f"Shift {k:3d} : Sharpe {v: .4f}")
    pd.Series(placebo).to_csv(os.path.join(out, "time_shift_placebo.csv"))

    print("\n--- Signal decay ---")
    decay = signal_decay(prices, weights, ann)
    for k, v in decay.items():
        print(f"Lag {k:3d} : Sharpe {v: .4f}")
    pd.Series(decay).to_csv(os.path.join(out, "signal_decay.csv"))

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Original"] + [f"+{k}" for k in placebo.keys()]
    vals = [orig_s2] + list(placebo.values())
    colors = ["black"] + ["gray"] * len(placebo)
    ax.bar(labels, vals, color=colors, edgecolor="black")
    ax.axhline(0, color="red", lw=0.8)
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"[{tag}] Time-shift placebo")
    save(fig, os.path.join(out, "13_time_shift_placebo.png"))

    fig, ax = plt.subplots(figsize=(9, 5))
    ks = list(decay.keys()); vs = list(decay.values())
    ax.plot(ks, vs, marker="o", color="black", lw=1.6)
    ax.axhline(0, color="red", lw=0.8)
    ax.set_xlabel("Signal->Return lag (periods)")
    ax.set_ylabel("Sharpe")
    ax.set_title(f"[{tag}] Signal Decay")
    save(fig, os.path.join(out, "14_signal_decay.png"))

    print("\n--- One-Shot Generalisation ---")
    train_s, test_s = one_shot_generalisation(prices, params, ann, uptrend)
    print(f"Train Sharpe           : {train_s:.4f}")
    print(f"Test Sharpe (frozen)   : {test_s:.4f}")
    print(f"Degradation            : {train_s - test_s:.4f}")
    pd.Series({"train_sharpe": train_s, "test_sharpe": test_s,
               "degradation": train_s - test_s}).to_csv(os.path.join(out, "one_shot.csv"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Train", "Test (frozen)"], [train_s, test_s],
           color=["steelblue", "darkorange"], edgecolor="black")
    for i, v in enumerate([train_s, test_s]):
        if not np.isnan(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Sharpe")
    ax.set_title(f"[{tag}] One-Shot Generalisation")
    save(fig, os.path.join(out, "15_one_shot.png"))

    pd.DataFrame({"Full Sample": pd.Series(metrics),
                  "Walk-Forward": pd.Series(wf_m) if wf_m else np.nan}).to_csv(
        os.path.join(out, "summary.csv"))

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({"universe": universe_name, "timeframe": tf_name,
                   "tickers": list(prices.columns), "ann_factor": ann,
                   "base_alloc": BASE_ALLOC,
                   "params": {k: (v if not isinstance(v, tuple) else list(v))
                              for k, v in params.items()},
                   "vol_target": VOL_TARGET, "max_lev": MAX_LEV},
                  f, indent=2, default=str)

    result = {
        "universe": universe_name, "timeframe": tf_name,
        "n_tickers": prices.shape[1], "n_bars": len(prices), "ann_factor": ann,
        "sharpe": metrics.get("Sharpe Ratio", np.nan),
        "ann_return": metrics.get("Annualised Return", np.nan),
        "max_dd": metrics.get("Max Drawdown", np.nan),
        "sortino": metrics.get("Sortino Ratio", np.nan),
        "calmar": metrics.get("Calmar Ratio", np.nan),
        "dsr": metrics.get("Deflated Sharpe Ratio", np.nan),
        "cs_ic": mean_ic, "ic_ir": ir_ic,
        "ir": metrics.get("Information Ratio (IR)", np.nan),
        "alpha": metrics.get("Alpha (ann.)", np.nan),
        "beta": metrics.get("Beta", np.nan),
        "wf_sharpe": wf_m.get("Sharpe Ratio", np.nan) if wf_m else np.nan,
        "mc_p": p_val, "cs_p": cs_p,
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

        labels = [f"{r['universe']}\n{r['timeframe']}" for r in results]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(14, 6))
        w = 0.25
        ax.bar(x - w, [r["sharpe"] for r in results], w, label="Sharpe", color="steelblue")
        ax.bar(x, [r["sortino"] for r in results], w, label="Sortino", color="darkorange")
        ax.bar(x + w, [r["wf_sharpe"] for r in results], w, label="WF Sharpe", color="seagreen")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(1, color="green", lw=0.8, ls="--")
        ax.set_ylabel("Ratio")
        ax.set_title("Cross-Run Sharpe / Sortino / WF-Sharpe")
        ax.legend()
        save(fig, os.path.join(BASE_OUT, "00_overall_summary.png"))

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(x - 0.2, [r["mc_p"] for r in results], 0.4,
               label="MC sign-flip p", color="crimson")
        ax.bar(x + 0.2, [r["cs_p"] for r in results], 0.4,
               label="CS-shuffle p", color="navy")
        ax.axhline(0.05, color="green", lw=1.2, ls="--", label="0.05")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("p-value")
        ax.set_title("Cross-Run Statistical Significance")
        ax.legend()
        save(fig, os.path.join(BASE_OUT, "00_overall_pvalues.png"))

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.bar(labels, [r["cs_ic"] for r in results], color="purple", edgecolor="black")
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(0.03, color="green", lw=1.0, ls="--", label="Tradable IC")
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Cross-Sectional IC")
        ax.set_title("Cross-Run CS-IC (combined signal)")
        ax.legend()
        save(fig, os.path.join(BASE_OUT, "00_overall_csic.png"))

    print(f"\nAll outputs saved to ./{BASE_OUT}/")


if __name__ == "__main__":
    main()