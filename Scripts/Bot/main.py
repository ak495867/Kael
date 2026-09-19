import os
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = "Outputs"
os.makedirs(OUT, exist_ok=True)
plt.style.use("ggplot")

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","V","UNH",
    "HD","PG","MA","DIS","BAC","XOM","PFE","CSCO","VZ","ADBE",
    "KO","PEP","WMT","MRK","ABT","TMO","COST","NKE","ORCL","CRM",
    "XLK","XLF","XLE","XLV","XLP","XLU","GLD","SLV",
    "TLT","IEF","LQD","HYG",
    "EFA","EEM","FXI",
    "USO","DBA",
    "BTC-USD","ETH-USD","LTC-USD"
]

START = "2018-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
ANN = 252


def download_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        data = raw["Close"].copy()
    else:
        data = raw.to_frame("Close")
        data.columns = tickers
    data = data.loc[:, data.notna().mean() > 0.95]
    data = data.loc[data.index.dayofweek < 5]
    data = data.ffill().dropna()
    return data


def momentum_signal(prices, lookback=252, skip=21, holding=21, quantile=0.2):
    mom = prices.shift(skip) / prices.shift(lookback) - 1.0
    ranks = mom.rank(axis=1, pct=True)
    long_mask = ranks >= (1 - quantile)
    short_mask = ranks <= quantile
    long_w = long_mask.div(long_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    short_w = short_mask.div(short_mask.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    weights = long_w - short_w
    rebal = weights.iloc[::holding].reindex(prices.index, method="ffill").fillna(0)
    return rebal


def q4_breakout_signal(prices, high_window=252, hold_days=21):
    rolling_high = prices.rolling(high_window, min_periods=20).max()
    breakout = (prices >= rolling_high.shift(1)).astype(float)
    months = pd.Series(prices.index.month, index=prices.index)
    in_q4 = months.isin([10, 11, 12]).astype(float)
    signal = breakout.mul(in_q4, axis=0)
    weights = signal.rolling(hold_days, min_periods=1).max().shift(1).fillna(0)
    active = weights.sum(axis=1).replace(0, np.nan)
    weights = weights.div(active, axis=0).fillna(0)
    return weights


def turn_of_month_signal(prices, pre_days=1, post_days=3):
    dates = prices.index
    n = len(dates)
    mask = np.zeros(n)
    periods = dates.to_period("M")
    for p in periods.unique():
        locs = np.where(periods == p)[0]
        s = max(0, locs[0] - pre_days)
        e = min(n - 1, locs[-1] + post_days)
        mask[s:e + 1] = 1.0
    ncols = prices.shape[1]
    w = np.outer(mask, np.ones(ncols) / ncols)
    return pd.DataFrame(w, index=dates, columns=prices.columns)


def normalise(w):
    denom = w.abs().sum(axis=1).replace(0, np.nan)
    return w.div(denom, axis=0).fillna(0)


def combine_strategies(prices, mom_lb=252, mom_skip=21, mom_hold=21,
                       alloc=(0.4, 0.3, 0.3)):
    w_mom = normalise(momentum_signal(prices, mom_lb, mom_skip, mom_hold))
    w_q4 = normalise(q4_breakout_signal(prices))
    w_tom = normalise(turn_of_month_signal(prices))
    w = alloc[0] * w_mom + alloc[1] * w_q4 + alloc[2] * w_tom
    return w


def backtest(prices, weights, cost_bps=0.0):
    ret = prices.pct_change().fillna(0)
    w = weights.shift(1).fillna(0)
    gross = (w * ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).shift(1).fillna(0)
    cost = turnover * cost_bps / 10000.0
    net = gross - cost
    return net, turnover


def max_streak(mask):
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def compute_metrics(returns, benchmark=None, rf=0.02):
    r = returns.dropna()
    cum = (1 + r).cumprod()
    total = cum.iloc[-1] - 1
    n_years = len(r) / ANN
    ann_ret = (1 + total) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = r.std() * np.sqrt(ANN)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    downside = r[r < 0].std() * np.sqrt(ANN)
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
    cvar95 = r[r <= var95].mean()
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
        if len(a) > 2 and a.std() > 0 and b.std() > 0:
            ic = a.corr(b)
        te = (a - b).std() * np.sqrt(ANN)
        if te > 0:
            ir = (a.mean() - b.mean()) * ANN / te
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
        "Best Day": r.max(),
        "Worst Day": r.min(),
        "Max Consecutive Wins": max_streak((r > 0).values),
        "Max Consecutive Losses": max_streak((r < 0).values),
    }, dd


def walk_forward(prices, train_days=756, test_days=252, lookbacks=(126, 189, 252)):
    dates = prices.index
    cache = {lb: combine_strategies(prices, mom_lb=lb) for lb in lookbacks}
    oos = []
    i = 0
    while i + train_days + test_days <= len(dates):
        tr_s, tr_e = i, i + train_days
        te_s, te_e = tr_e, tr_e + test_days
        best_sharpe, best_lb = -np.inf, lookbacks[0]
        for lb in lookbacks:
            w = cache[lb].iloc[tr_s:tr_e]
            r, _ = backtest(prices.iloc[tr_s:tr_e], w)
            r = r.dropna()
            if len(r) < 20 or r.std() == 0:
                continue
            s = r.mean() / r.std() * np.sqrt(ANN)
            if s > best_sharpe:
                best_sharpe, best_lb = s, lb
        w_te = cache[best_lb].iloc[te_s:te_e]
        r_te, _ = backtest(prices.iloc[te_s:te_e], w_te)
        oos.append(r_te)
        i += test_days
    return pd.concat(oos).sort_index()


def monte_carlo_permutation(returns, n_perm=1000):
    r = returns.dropna().values
    obs = r.mean() / r.std() * np.sqrt(ANN)
    perm = np.empty(n_perm)
    for k in range(n_perm):
        s = np.random.permutation(r)
        perm[k] = s.mean() / s.std() * np.sqrt(ANN)
    p = float((perm >= obs).mean())
    return p, obs, perm


def parameter_robustness(prices, base_lb=252, base_hold=21, pct=0.2):
    lbs = sorted(set([max(21, int(base_lb * (1 - pct))), base_lb, int(base_lb * (1 + pct))]))
    holds = sorted(set([max(5, int(base_hold * (1 - pct))), base_hold, int(base_hold * (1 + pct))]))
    res = np.zeros((len(lbs), len(holds)))
    for i, lb in enumerate(lbs):
        for j, h in enumerate(holds):
            w = combine_strategies(prices, mom_lb=lb, mom_hold=h)
            r, _ = backtest(prices, w)
            r = r.dropna()
            res[i, j] = r.mean() / r.std() * np.sqrt(ANN)
    return lbs, holds, res


def cost_sensitivity(prices, weights, bps_list=(0, 2, 5, 10, 20, 50)):
    out = {}
    for bps in bps_list:
        r, _ = backtest(prices, weights, cost_bps=bps)
        r = r.dropna()
        out[bps] = r.mean() / r.std() * np.sqrt(ANN)
    return out


def placebo_test(prices, weights, shifts=(5, 10, 21, 42, 63)):
    orig_r, _ = backtest(prices, weights)
    orig_r = orig_r.dropna()
    orig_sharpe = orig_r.mean() / orig_r.std() * np.sqrt(ANN)
    out = {}
    for s in shifts:
        wp = weights.shift(s).fillna(0)
        r, _ = backtest(prices, wp)
        r = r.dropna()
        out[s] = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan
    return orig_sharpe, out


def one_shot_generalisation(prices, split=0.5, lookbacks=(126, 189, 252)):
    n = int(len(prices) * split)
    train = prices.iloc[:n]
    test = prices.iloc[n:]
    cache = {lb: combine_strategies(prices, mom_lb=lb) for lb in lookbacks}
    best_sharpe, best_lb = -np.inf, lookbacks[0]
    for lb in lookbacks:
        w = cache[lb].iloc[:n]
        r, _ = backtest(train, w)
        r = r.dropna()
        if r.std() == 0:
            continue
        s = r.mean() / r.std() * np.sqrt(ANN)
        if s > best_sharpe:
            best_sharpe, best_lb = s, lb
    w_test = cache[best_lb].iloc[n:]
    r_test, _ = backtest(test, w_test)
    r_test = r_test.dropna()
    test_sharpe = r_test.mean() / r_test.std() * np.sqrt(ANN) if r_test.std() > 0 else np.nan
    return best_lb, best_sharpe, test_sharpe


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight")
    plt.close(fig)


def equity(r):
    return (1 + r.fillna(0)).cumprod()


def main():
    print("=" * 70)
    print("DOWNLOADING DATA")
    print("=" * 70)
    prices = download_data(TICKERS, START, END)
    print(f"Universe : {prices.shape[1]} tickers")
    print(f"Period   : {prices.index[0].date()} -> {prices.index[-1].date()}")
    print(f"Days     : {len(prices)}")
    print(f"Tickers  : {list(prices.columns)}")

    weights = combine_strategies(prices)
    returns, turnover = backtest(prices, weights)
    returns = returns.dropna()
    bench_ret = prices.pct_change().mean(axis=1).reindex(returns.index).dropna()

    w_mom = normalise(momentum_signal(prices))
    w_q4 = normalise(q4_breakout_signal(prices))
    w_tom = normalise(turn_of_month_signal(prices))
    r_mom, _ = backtest(prices, w_mom)
    r_q4, _ = backtest(prices, w_q4)
    r_tom, _ = backtest(prices, w_tom)

    metrics, dd = compute_metrics(returns, bench_ret)

    print("\n" + "=" * 70)
    print("FULL-SAMPLE BACKTEST METRICS")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"{k:32s}: {v: .4f}")

    pd.DataFrame(metrics, index=["value"]).T.to_csv(os.path.join(OUT, "metrics_full_sample.csv"))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(equity(returns), label="Combined", lw=2.4, color="black")
    ax.plot(equity(r_mom.reindex(returns.index).fillna(0)), label="Momentum", lw=1.2, alpha=0.85)
    ax.plot(equity(r_q4.reindex(returns.index).fillna(0)), label="Q4 Breakout", lw=1.2, alpha=0.85)
    ax.plot(equity(r_tom.reindex(returns.index).fillna(0)), label="Turn-of-Month", lw=1.2, alpha=0.85)
    ax.plot(equity(bench_ret), label="EW Benchmark", lw=1.2, ls="--", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title("Equity Curves (log scale)")
    ax.set_ylabel("Cumulative Growth")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "01_equity_curves.png")

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.7)
    ax.set_title("Drawdown")
    ax.set_ylabel("Drawdown")
    save(fig, "02_drawdown.png")

    roll_sharpe = returns.rolling(126).mean() / returns.rolling(126).std() * np.sqrt(ANN)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(roll_sharpe, color="navy", lw=1.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="green", lw=0.8, ls="--")
    ax.axhline(-1, color="red", lw=0.8, ls="--")
    ax.set_title("Rolling 6M Sharpe Ratio")
    save(fig, "03_rolling_sharpe.png")

    roll_vol = returns.rolling(63).std() * np.sqrt(ANN)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(roll_vol, color="darkorange", lw=1.4)
    ax.set_title("Rolling 3M Annualised Volatility")
    save(fig, "04_rolling_volatility.png")

    monthly = returns.groupby([returns.index.year, returns.index.month]).apply(
        lambda x: (1 + x).prod() - 1)
    monthly.index.names = ["y", "m"]
    pivot = monthly.unstack(level="m")
    fig, ax = plt.subplots(figsize=(11, max(3.5, 0.55 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-np.nanmax(np.abs(pivot.values)), vmax=np.nanmax(np.abs(pivot.values)))
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:pivot.shape[1]])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Monthly Return")
    ax.set_title("Monthly Returns Heatmap (%)")
    save(fig, "05_monthly_heatmap.png")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(returns.values, bins=80, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(np.percentile(returns, 5), color="red", ls="--", lw=1.5, label="VaR 95%")
    ax.axvline(returns.mean(), color="black", ls="-", lw=1.5, label="Mean")
    ax.set_title("Daily Return Distribution")
    ax.legend()
    save(fig, "06_return_distribution.png")

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(turnover.rolling(21).mean(), color="purple", lw=1.4)
    ax.set_title("Rolling 1M Average Daily Turnover")
    save(fig, "07_turnover.png")

    print("\n" + "=" * 70)
    print("WALK-FORWARD TEST (train 756d / test 252d)")
    print("=" * 70)
    wf_ret = walk_forward(prices)
    wf_metrics, wf_dd = compute_metrics(wf_ret, bench_ret.reindex(wf_ret.index).dropna())
    for k, v in wf_metrics.items():
        print(f"{k:32s}: {v: .4f}")
    pd.DataFrame(wf_metrics, index=["value"]).T.to_csv(os.path.join(OUT, "metrics_walkforward.csv"))

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(equity(wf_ret), color="black", lw=2, label="Walk-Forward OOS")
    axes[0].plot(equity(bench_ret.reindex(wf_ret.index).fillna(0)), ls="--", label="Benchmark")
    axes[0].set_yscale("log")
    axes[0].set_title("Walk-Forward Out-of-Sample Equity")
    axes[0].legend()
    axes[1].fill_between(wf_dd.index, wf_dd.values, 0, color="crimson", alpha=0.7)
    axes[1].set_title("Walk-Forward Drawdown")
    save(fig, "08_walkforward.png")

    print("\n" + "=" * 70)
    print("MONTE CARLO PERMUTATION TEST (1000 shuffles)")
    print("=" * 70)
    p_val, obs_sharpe, perm = monte_carlo_permutation(returns, n_perm=1000)
    print(f"Observed Sharpe       : {obs_sharpe:.4f}")
    print(f"Permutation p-value   : {p_val:.4f}")
    print(f"95% CI of null Sharpe : [{np.percentile(perm, 2.5):.4f}, {np.percentile(perm, 97.5):.4f}]")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(perm, bins=60, color="lightgray", edgecolor="black", alpha=0.85, label="Permuted")
    ax.axvline(obs_sharpe, color="red", lw=2.2, label=f"Observed = {obs_sharpe:.2f}")
    ax.set_title(f"Monte Carlo Permutation Test (p = {p_val:.4f})")
    ax.set_xlabel("Sharpe Ratio")
    ax.legend()
    save(fig, "09_monte_carlo.png")

    print("\n" + "=" * 70)
    print("PARAMETER ROBUSTNESS TEST (+/-20%)")
    print("=" * 70)
    lbs, holds, rob = parameter_robustness(prices)
    rob_df = pd.DataFrame(rob, index=[f"lookback={x}" for x in lbs], columns=[f"hold={x}" for x in holds])
    print(rob_df.round(4))
    rob_df.to_csv(os.path.join(OUT, "parameter_robustness.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(rob, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(holds)))
    ax.set_xticklabels(holds)
    ax.set_yticks(range(len(lbs)))
    ax.set_yticklabels(lbs)
    for i in range(len(lbs)):
        for j in range(len(holds)):
            ax.text(j, i, f"{rob[i, j]:.2f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="Sharpe")
    ax.set_xlabel("Momentum Holding")
    ax.set_ylabel("Momentum Lookback")
    ax.set_title("Parameter Robustness (Sharpe)")
    save(fig, "10_parameter_robustness.png")

    print("\n" + "=" * 70)
    print("TRANSACTION COST SENSITIVITY")
    print("=" * 70)
    cost = cost_sensitivity(prices, weights)
    for bps, s in cost.items():
        print(f"{bps:4d} bps : Sharpe {s: .4f}")
    pd.Series(cost).to_csv(os.path.join(OUT, "cost_sensitivity.csv"))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(k) for k in cost.keys()], list(cost.values()), color="teal", edgecolor="black")
    ax.set_xlabel("Transaction Cost (bps)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Cost Sensitivity")
    for i, v in enumerate(cost.values()):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, "11_cost_sensitivity.png")

    print("\n" + "=" * 70)
    print("PLACEBO TEST (signal shifted forward)")
    print("=" * 70)
    orig_sharpe, placebo = placebo_test(prices, weights)
    print(f"Original Sharpe : {orig_sharpe:.4f}")
    for k, v in placebo.items():
        print(f"Shift {k:3d} days : Sharpe {v: .4f}")
    pd.Series(placebo).to_csv(os.path.join(OUT, "placebo_test.csv"))

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["Original"] + [f"+{k}d" for k in placebo.keys()]
    vals = [orig_sharpe] + list(placebo.values())
    colors = ["black"] + ["gray"] * len(placebo)
    ax.bar(labels, vals, color=colors, edgecolor="black")
    ax.axhline(0, color="red", lw=0.8)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Placebo Test: Original vs Time-Shifted Signal")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    save(fig, "12_placebo.png")

    print("\n" + "=" * 70)
    print("ONE-SHOT GENERALISATION TEST")
    print("=" * 70)
    best_lb, train_s, test_s = one_shot_generalisation(prices)
    print(f"Best lookback on train : {best_lb}")
    print(f"Train Sharpe           : {train_s:.4f}")
    print(f"Test Sharpe (frozen)   : {test_s:.4f}")
    print(f"Degradation            : {train_s - test_s:.4f}")
    pd.Series({"best_lookback": best_lb, "train_sharpe": train_s,
               "test_sharpe": test_s, "degradation": train_s - test_s}).to_csv(
        os.path.join(OUT, "one_shot_generalisation.csv"))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["Train", "Test (frozen)"], [train_s, test_s],
           color=["steelblue", "darkorange"], edgecolor="black")
    for i, v in enumerate([train_s, test_s]):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("One-Shot Generalisation")
    save(fig, "13_one_shot_generalisation.png")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(equity(w_mom.mul(prices.pct_change()).sum(axis=1).dropna()), label="Momentum Only", lw=1.5)
    ax.plot(equity(w_q4.mul(prices.pct_change()).sum(axis=1).dropna()), label="Q4 Only", lw=1.5)
    ax.plot(equity(w_tom.mul(prices.pct_change()).sum(axis=1).dropna()), label="TOM Only", lw=1.5)
    ax.plot(equity(returns), label="Combined", lw=2.4, color="black")
    ax.set_yscale("log")
    ax.set_title("Strategy Contribution Comparison")
    ax.legend()
    save(fig, "14_strategy_contributions.png")

    summary = pd.DataFrame({
        "Full Sample": pd.Series(metrics),
        "Walk-Forward": pd.Series(wf_metrics),
    })
    summary.to_csv(os.path.join(OUT, "summary_all_metrics.csv"))
    print("\n" + "=" * 70)
    print(f"All outputs saved to ./{OUT}/")
    print("=" * 70)


if __name__ == "__main__":
    main()