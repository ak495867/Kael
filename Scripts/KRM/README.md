# KRM_ESC test lab

Leak-proof code for the Kramers escape-rate breakout signal (see `Research/KRM_ESC.md` for the maths).

## Run

```bash
pip install -r requirements.txt
python tests/test_leakage.py                      # proves the leak detectors work (incl. deliberately leaky signals)
python run.py --self-test                         # full pipeline on synthetic no-edge data (no network)
python run.py --tickers SPY QQQ IWM EFA TLT GLD --bench SPY --start 2000-01-01 --split 2016-01-01
python run.py --tickers SPY --grid-search         # params chosen on TRAIN rows only
python run.py --tickers SPY --lag 2 --cost-bps 5  # harsher execution assumptions
```

## Files

| File | Role |
|---|---|
| `krm_esc.py` | signal (Steps 1-7), position mapping, the single place where signal time is aligned to return time |
| `data.py` | loading, hygiene checks, synthetic generator |
| `leakage.py` | truncation-invariance, future-perturbation, engine-alignment, lag guard, determinism, negative controls |
| `evaluate.py` | metrics, cost/lag sensitivity, block bootstrap, placebo permutation test, PSR, parameter probes, folds, regimes, ablation, train-only grid search |
| `run.py` | protocol orchestration + reporting; writes `outputs/` (frozen params, daily CSV, OOS log) |

## How leakage is prevented

1. **Causal by construction:** only trailing `rolling()` windows and forward-recursive `ewm(adjust=False)`; no centred windows, `shift(-k)`, back-fill, or full-sample normalisation.
2. **Proven, not assumed:** `leakage.run_all` runs before every report and aborts on failure.
   - *Truncation invariance:* signal at row k on the full series must equal the last signal on `price[:k+1]`.
   - *Future perturbation:* randomising everything after row k must not change signal rows up to k.
   - *Negative controls:* centred-window, `shift(-1)` and full-sample-z-score signals are confirmed to be caught.
3. **One alignment point:** `backtest()` gives day t the position decided at close t-lag, `lag >= 1` enforced. Exact identity test plus a "peek at same-day return" test (must earn ~0) and an oracle test (must earn a lot).
4. **Train/OOS separation:** the grid search receives *sliced* train arrays, so OOS rows physically cannot enter it. Regime labels use information through t-1 only. Warm-up rows are forced flat.
5. **Parameters frozen before OOS**, written to `outputs/frozen_params_*.json`, and every OOS evaluation is appended to `outputs/oos_log.jsonl`. The runner prints how many distinct parameter sets have touched that OOS window. If that number grows, discount your result.

## What code cannot fix (your responsibility)

- **Choosing `--split`, tickers, start date or `--sign` after seeing results** is leakage of a human kind. Decide first; the default sign is +1 (breakout).
- **Survivorship / universe selection:** pick ETFs that existed for the whole sample, not today's winners.
- **Adjusted prices:** `Adj Close` is back-adjusted using future dividends, which slightly shifts old price levels (mostly harmless for a 60-day range, but not perfectly point-in-time). `--signal-price raw` uses unadjusted `Close`, which has small ex-dividend gaps instead. Check that both give similar conclusions.
- **Execution:** `--lag 1` assumes you can trade at the same close the signal uses. Treat `--lag 2` as the honest number.
- **Costs/slippage** here are a flat bps on turnover; real ETF spreads, borrow and taxes differ.
- **Multiple testing:** more variants tried means higher expected best-case Sharpe by luck. The placebo p-value and PSR help, but do not correct for how many things you tried.

## Reading the output

Trust the strategy only if **all** of these hold: OOS Sharpe positive at `--lag 2`; placebo p-value small (<0.05); PSR high; parameter verdict ROBUST; ablation shows the gate helping; positive in most folds; bear-regime Sharpe not sharply negative.
