"""
KRM_ESC validation runner.

Protocol (enforced in code):
  1. Data hygiene checks (no fill, no partial bar).
  2. Leakage tests on the signal + engine; abort if any fails.
  3. Parameters are FROZEN before OOS is touched (defaults, or grid search on TRAIN rows only).
  4. OOS is evaluated once; every OOS evaluation is appended to oos_log.jsonl so the number of
     things you tried is visible (multiple-testing honesty).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

import leakage
from data import load_prices, make_synthetic, validate_prices
from evaluate import grid_search_train, oos_report, sharpe
from krm_esc import KrmParams, signal_series

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")


def log_oos_touch(path: str, tickers, split, params: KrmParams) -> int:
    rec = dict(ts=time.strftime("%Y-%m-%d %H:%M:%S"), tickers=list(tickers), split=str(split), key=params.key(), params=asdict(params))
    prev = []
    if os.path.exists(path):
        prev = [json.loads(l) for l in open(path) if l.strip()]
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    keys = {p["key"] for p in prev if p["split"] == str(split) and p["tickers"] == list(tickers)} | {params.key()}
    return len(keys)


def show(title: str, obj) -> None:
    print(f"\n--- {title} " + "-" * max(0, 70 - len(title)))
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"  {k:>14}: {v:,.4f}" if isinstance(v, float) else f"  {k:>14}: {v}")
    else:
        print(obj.to_string())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=["SPY"])
    ap.add_argument("--csv", help="load one ticker from CSV instead of yfinance (columns: Open High Low Close 'Adj Close')")
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--split", default="2016-01-01", help="first OOS date. Decide it BEFORE looking at results.")
    ap.add_argument("--bench", default="SPY")
    ap.add_argument("--cost-bps", type=float, default=1.0)
    ap.add_argument("--lag", type=int, default=1, help="1 = trade at signal close; 2 = one extra day of delay")
    ap.add_argument("--pos-mode", default="linear", choices=["linear", "sign"])
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--signal-price", default="adj", choices=["adj", "raw"])
    ap.add_argument("--m", type=int, default=60)
    ap.add_argument("--n-short", type=int, default=10)
    ap.add_argument("--n-long", type=int, default=60)
    ap.add_argument("--ema-n", type=int, default=5)
    ap.add_argument("--sign", type=int, default=1, choices=[-1, 1])
    ap.add_argument("--grid-search", action="store_true", help="select params on TRAIN rows only")
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using cache")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true", help="run on synthetic data, no network")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    split = pd.Timestamp(args.split)
    base = KrmParams(m=args.m, n_short=args.n_short, n_long=args.n_long, ema_n=args.ema_n, sign=args.sign)

    if args.self_test:
        frames = {"SYNTH": make_synthetic()}
        args.bench = "SYNTH"
        split = frames["SYNTH"].index[int(len(frames["SYNTH"]) * 0.6)]
    elif args.csv:
        frames = {args.tickers[0]: load_prices(args.tickers[0], args.start, args.end, csv=args.csv)}
        args.bench = args.tickers[0] if args.bench not in args.tickers else args.bench
    else:
        wanted = list(dict.fromkeys(args.tickers + [args.bench]))
        frames = {t: load_prices(t, args.start, args.end, refresh=args.refresh) for t in wanted}

    bench_close = frames[args.bench]["Adj Close"] if args.bench in frames else next(iter(frames.values()))["Adj Close"]
    ew = {}
    for tk in [t for t in frames if t in args.tickers or args.self_test]:
        df = frames[tk]
        print("\n" + "=" * 78 + f"\n{tk}: {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} rows)  split={split.date()}\n" + "=" * 78)
        for w in validate_prices(df, tk):
            print("  [warn]", w)
        n_tr, n_te = int((df.index < split).sum()), int((df.index >= split).sum())
        if n_tr < 750 or n_te < 500:
            raise SystemExit(f"need >=750 train and >=500 OOS rows, have {n_tr}/{n_te}")

        price = df["Adj Close" if args.signal_price == "adj" else "Close"]
        ret = df["Adj Close"].pct_change().fillna(0.0)

        print("\nLEAKAGE TESTS")
        leakage.run_all(lambda s: signal_series(s, base), price)

        params = base
        if args.grid_search:
            tr = df.index < split                        # physically slice: OOS rows cannot enter
            params, tab = grid_search_train(price[tr], ret[tr], base, args.pos_mode, args.long_only, args.lag, args.cost_bps)
            show(f"grid search on TRAIN only ({len(tab)} trials) - top 5", tab.head(5))
        json.dump(asdict(params), open(os.path.join(args.out, f"frozen_params_{tk}.json"), "w"), indent=2)
        n_keys = log_oos_touch(os.path.join(args.out, "oos_log.jsonl"), [tk], split, params)
        print(f"\nPARAMS FROZEN: {asdict(params)}\nOOS has now been evaluated with {n_keys} distinct parameter set(s) for this ticker/split.")

        R = oos_report(price, ret, bench_close, params, split, args.pos_mode, args.long_only,
                       args.lag, args.cost_bps, seed=args.seed)
        show("HEADLINE (in-sample vs out-of-sample)", R["headline"])
        show(f"COST SENSITIVITY (OOS, lag={args.lag})", R["cost"])
        show("EXECUTION-DELAY SENSITIVITY (OOS Sharpe)", R["lag"])
        show("BLOCK BOOTSTRAP (OOS)", R["bootstrap"])
        show("PLACEBO: shifted-position permutation test (OOS)", R["placebo"])
        show("PROBABILISTIC SHARPE  P(true Sharpe > 0)", {"psr": R["psr_vs_0"]})
        if "param_sensitivity" in R:
            show("PARAMETER SENSITIVITY (diagnostic only)", R["param_sensitivity"])
        show("TIME-SLICE STABILITY (5 folds, full live history)", R["folds_full_history"])
        show("REGIME SPLIT (OOS; label uses info through t-1)", R["regime_oos"])
        show("ABLATION: does the physics gate add anything?", R["ablation"])
        show("BENCHMARK buy & hold (OOS)", R["benchmark_buy_hold"])
        print(f"  corr(strategy, buy&hold) = {R['corr_with_buy_hold']:+.3f}")

        R["_daily"].to_csv(os.path.join(args.out, f"daily_{tk}.csv"))
        ew[tk] = R["_daily"].loc[R["_daily"].index >= split, "net"]

    if len(ew) > 1:
        port = pd.concat(ew, axis=1).dropna().mean(axis=1)
        print("\n" + "=" * 78 + f"\nEQUAL-WEIGHT PORTFOLIO of {list(ew)} (OOS): Sharpe = {sharpe(port):+.3f}\n" + "=" * 78)


if __name__ == "__main__":
    main()
