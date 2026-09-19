# Kael Strategy Reference

Ten strategies from the Pareto frontier, each explained with formulas, a flow diagram and test results.

|#|File|Expression|Cx|OOS Sharpe|
|-|-|-|-|-|
|1|[KELT\_HMA\_RANK\_133](Strategies/KELT_HMA_RANK_133.md)|`RANK(HMA(DELAY(MAXR(KELTNER(30, 1.979), 131), 50), 9), 133)`|5|+1.060|
|2|[KELT\_HMA\_RANK\_132](Strategies/KELT_HMA_RANK_132.md)|same, rank window 132|5|+1.060|
|3|[VWAP\_ROC\_SLOPE](Strategies/VWAP_ROC_SLOPE.md)|`ROLL\_SLOPE(ROC(vwap, 193), 95)`|3|+0.566|
|4|[OPEN\_ROC](Strategies/OPEN_ROC.md)|`ROC(open, 191)`|2|+0.271|
|5|[DEMA\_XOVER\_NATR](Strategies/DEMA_XOVER_NATR.md)|`DEMA(CROSS\_OVER(DIV(close, hlc3), NATR(39)), 44)`|6|+0.232|
|6|[HL2\_ROC](Strategies/HL2_ROC.md)|`ROC(hl2, 83)`|2|+0.332|
|7|[SQRT\_ATR](Strategies/SQRT_ATR.md)|`SQRT(ATR(77))`|2|+0.378|
|8|[MACD\_37\_97\_41](Strategies/MACD_37_97_41.md)|`MACD(37, 97, 41)`|1|+0.731|
|9|[WMA\_CCI](Strategies/WMA_CCI.md)|`WMA(CCI(21), 113)`|2|+0.198|
|10|[STD\_SLOPE](Strategies/STD_SLOPE.md)|`ROLL\_SLOPE(STD(close, 24), 95)`|3|-0.071|
| 11 | [KRM_ESC](Research/KRM_ESC.md) | Kramers escape-rate breakout (physics-derived, **untested**) | n/a | not run |

## Common conventions

* Signal $S\_t$ is computed from data up to day $t$ only; the engine converts it to a position and applies it to the next bar.
* Sharpe is annualised; "OOS" means out-of-sample.
* Full diagnostics (costs, bootstrap, parameter sensitivity, folds, regimes) exist only for the top strategy; other files show whatever the log contained.
* Indicator definitions follow standard conventions because the library source was not provided.

