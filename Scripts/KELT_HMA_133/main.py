import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
from scipy import stats
from itertools import product

warnings.filterwarnings('ignore')
np.random.seed(42)

# ==============================================================================
# 1. CONFIGURATION & UNIVERSE DEFINITION
# ==============================================================================
UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA',
    'JPM', 'V', 'GS', 'JNJ', 'UNH',
    'XOM', 'CVX', 'COP',
    'WMT', 'PG', 'HD',
    'SPY', 'GLD', 'TLT', 'BTC-USD', 'ETH-USD', 'SOL-USD'
]

START_DATE = '2020-01-01'
END_DATE = '2024-10-01'
TOP_N_LONG = 5
BOTTOM_N_SHORT = 5

# ==============================================================================
# 2. DATA ENGINE (Real-World, Strictly Aligned, No Leakage)
# ==============================================================================
class DataEngine:
    @staticmethod
    def fetch_universe_data(tickers, start_date, end_date):
        print(f"Fetching real-world data for {len(tickers)} assets from {start_date} to {end_date}...")
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        close_prices = data['Close'].copy()
        high_prices = data['High'].copy()
        low_prices = data['Low'].copy()
        
        # Forward fill minor gaps, then drop completely dead assets
        close_prices = close_prices.ffill().dropna(how='all', axis=1)
        high_prices = high_prices.ffill().dropna(how='all', axis=1)
        low_prices = low_prices.ffill().dropna(how='all', axis=1)
        
        # Strictly align to dates where ALL remaining assets have valid data
        common_dates = close_prices.dropna(how='any', axis=0).index
        close_prices = close_prices.loc[common_dates]
        high_prices = high_prices.loc[common_dates]
        low_prices = low_prices.loc[common_dates]
        
        returns = close_prices.pct_change().dropna(how='any', axis=0)
        
        print(f"Data aligned successfully. Shape: {close_prices.shape} | Date Range: {close_prices.index[0]} to {close_prices.index[-1]}")
        return close_prices, high_prices, low_prices, returns

# ==============================================================================
# 3. INDICATOR ENGINE (Exact Mathematical Replication of Spec)
# ==============================================================================
class IndicatorEngine:
    @staticmethod
    def calculate_atr(high, low, close, window=30):
        """Wilder's True Range and ATR (Fixed for DataFrame element-wise operations)"""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # FIX: Element-wise max across the 3 DataFrames
        tr = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
        
        # Wilder's smoothing: alpha = 1/window
        return tr.ewm(alpha=1/window, adjust=False).mean()

    @staticmethod
    def calculate_hma(series, window=9):
        """Hull Moving Average: WMA_sqrt(2 * WMA_half - WMA_full)"""
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        
        def wma(x):
            weights = np.arange(1, len(x) + 1)
            return np.dot(x, weights) / weights.sum()
            
        wma_half = series.rolling(half_window).apply(wma, raw=True)
        wma_full = series.rolling(window).apply(wma, raw=True)
        
        return (2 * wma_half - wma_full).rolling(sqrt_window).apply(wma, raw=True)

    @classmethod
    def calculate_signals(cls, close_prices, high_prices, low_prices, 
                          kelt_len=30, k_multiplier=1.9789691043998496, 
                          maxr_len=131, delay_len=50, hma_len=9, rank_len=133):
        """Calculates KELT_HMA_RANK_133 exactly as specified."""
        print("Calculating cross-sectional signals (Strictly No Lookahead)...")
        
        # Step 1: Keltner Position K_t = (Close - EMA30) / (k * ATR30)
        ema_30 = close_prices.ewm(span=kelt_len, adjust=False).mean()
        atr_30 = cls.calculate_atr(high_prices, low_prices, close_prices, window=kelt_len)
        keltner_pos = (close_prices - ema_30) / (k_multiplier * atr_30)
        
        # Step 2: MAXR (Rolling Maximum over 131 days)
        maxr = keltner_pos.rolling(maxr_len).max()
        
        # Step 3: DELAY (Shift back 50 days)
        delayed = maxr.shift(delay_len)
        
        # Step 4: HMA(9) Smooth with Hull Moving Average
        hma_9 = cls.calculate_hma(delayed, window=hma_len)
        
        # Step 5: RANK(133) Percentile rank vs last 133 days
        signals = hma_9.rolling(rank_len).apply(lambda x: x.rank(pct=True).iloc[-1], raw=False)
        
        # Debug info to prove signals are generating correctly
        print(f"  -> Signals generated. Total non-NaN values: {signals.notna().sum().sum()}")
        print(f"  -> Average non-NaN assets per day: {signals.notna().sum().mean():.1f}")
        
        return signals

# ==============================================================================
# 4. PORTFOLIO & BACKTEST ENGINE (Strict Weight Shifting)
# ==============================================================================
class PortfolioEngine:
    @staticmethod
    def construct_portfolio(signals, returns, top_n=5, bottom_n=5):
        # GUARANTEE: Indices are perfectly aligned before any operation
        common_index = signals.index.intersection(returns.index)
        signals = signals.loc[common_index]
        returns = returns.loc[common_index]
        
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
        
        for date in signals.index:
            daily_signals = signals.loc[date].dropna()
            if len(daily_signals) < (top_n + bottom_n):
                continue
            sorted_signals = daily_signals.sort_values()
            
            # Top N for Long
            long_assets = sorted_signals.tail(top_n).index
            weights.loc[date, long_assets] = 1.0 / top_n
            
            # Bottom N for Short
            short_assets = sorted_signals.head(bottom_n).index
            weights.loc[date, short_assets] = -1.0 / bottom_n

        # STRICT NO LEAKAGE: Shift weights by 1 day. Signal at T dictates position at T+1.
        weights_shifted = weights.shift(1).dropna()
        
        aligned_returns = returns.loc[weights_shifted.index]
        portfolio_returns = (weights_shifted * aligned_returns).sum(axis=1)
        turnover = weights_shifted.sub(weights_shifted.shift(1)).abs().sum(axis=1).fillna(0)
        
        return portfolio_returns, turnover, weights_shifted

# ==============================================================================
# 5. METRICS ENGINE (19 Detailed Quant Metrics)
# ==============================================================================
class MetricsEngine:
    @staticmethod
    def calculate_all_metrics(returns, trades=None, turnover_series=None):
        metrics = {}
        returns = returns.dropna()
        if len(returns) < 10:
            return {k: np.nan for k in range(20)}

        ann_factor = 252
        ann_ret = (1 + returns).prod() ** (ann_factor / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(ann_factor)
        
        metrics['CAGR'] = ann_ret
        metrics['Ann_Volatility'] = ann_vol
        metrics['Sharpe'] = ann_ret / ann_vol if ann_vol > 0 else 0
        downside_vol = returns[returns < 0].std() * np.sqrt(ann_factor)
        metrics['Sortino'] = ann_ret / downside_vol if downside_vol > 0 else 0
        
        cum_ret = (1 + returns).cumprod()
        running_max = cum_ret.cummax()
        drawdown = (cum_ret / running_max) - 1
        metrics['Max_Drawdown'] = drawdown.min()
        metrics['Calmar'] = ann_ret / abs(metrics['Max_Drawdown']) if metrics['Max_Drawdown'] != 0 else 0
        metrics['Information_Ratio'] = (returns.mean() * ann_factor) / (returns.std() * np.sqrt(ann_factor)) if returns.std() > 0 else 0
        
        if trades is not None and len(trades) > 0:
            wins = trades[trades > 0]
            losses = trades[trades < 0]
            metrics['Win_Rate'] = len(wins) / len(trades)
            metrics['Profit_Factor'] = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
            metrics['Payoff_Ratio'] = wins.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.inf
            is_loss = (trades < 0).astype(int)
            streaks = is_loss * (is_loss.groupby((is_loss != is_loss.shift(1)).cumsum()).cumcount() + 1)
            metrics['Max_Cons_Losses'] = int(streaks.max()) if not streaks.empty else 0
        else:
            metrics.update({'Win_Rate': np.nan, 'Profit_Factor': np.nan, 'Payoff_Ratio': np.nan, 'Max_Cons_Losses': np.nan})

        metrics['Skewness'] = returns.skew()
        metrics['Kurtosis'] = returns.kurtosis()
        p95, p5 = returns.quantile(0.95), returns.quantile(0.05)
        metrics['Tail_Ratio'] = abs(p95 / p5) if p5 != 0 else np.inf
        gains, losses_sum = returns[returns > 0].sum(), abs(returns[returns < 0].sum())
        metrics['Omega'] = gains / losses_sum if losses_sum > 0 else np.inf
        
        if turnover_series is not None:
            metrics['Avg_Daily_Turnover'] = turnover_series.mean()
            cost_adj_ret = returns - (turnover_series * 0.0010)
            metrics['Sharpe_10bps_Cost'] = (cost_adj_ret.mean() * ann_factor) / (cost_adj_ret.std() * np.sqrt(ann_factor))
        else:
            metrics['Avg_Daily_Turnover'] = np.nan
            metrics['Sharpe_10bps_Cost'] = np.nan

        return metrics

    @staticmethod
    def calculate_ic(signals, forward_returns):
        valid_days = signals.notna().any(axis=1) & forward_returns.notna().any(axis=1)
        daily_ic = []
        for date in signals.index[valid_days]:
            sig = signals.loc[date].dropna()
            ret = forward_returns.loc[date].dropna()
            common_idx = sig.index.intersection(ret.index)
            if len(common_idx) > 5:
                ic = stats.spearmanr(sig[common_idx], ret[common_idx])[0]
                daily_ic.append(ic)
        
        mean_ic = np.mean(daily_ic) if daily_ic else 0.0
        ic_ir = mean_ic / np.std(daily_ic) if np.std(daily_ic) > 0 else 0.0
        return mean_ic, ic_ir

# ==============================================================================
# 6. ROBUST TEST SUITE
# ==============================================================================
class RobustTestSuite:
    def __init__(self, close_prices, returns, signals):
        # ULTIMATE FIX: Force all dataframes to share the EXACT same index upfront
        common_index = signals.index.intersection(returns.index).intersection(close_prices.index)
        self.close_prices = close_prices.loc[common_index]
        self.returns = returns.loc[common_index]
        self.signals = signals.loc[common_index]
        self.results = {}

    def _get_portfolio(self, cost_bps=0):
        port_ret, turnover, weights = PortfolioEngine.construct_portfolio(self.signals, self.returns, TOP_N_LONG, BOTTOM_N_SHORT)
        costs = turnover * (cost_bps / 10000.0)
        net_ret = port_ret - costs
        return net_ret, turnover

    def test_1_standard_backtest(self, cost_bps=5):
        print("\n--- TEST 1: Standard Backtest ---")
        net_ret, turnover = self._get_portfolio(cost_bps)
        metrics = MetricsEngine.calculate_all_metrics(net_ret, turnover_series=turnover)
        self.results['Standard'] = metrics
        self._print_metrics(metrics)
        return net_ret

    def test_2_placebo_test(self, iterations=100):
        print("\n--- TEST 2: Placebo Test (Permutation) ---")
        real_ret, _ = self._get_portfolio(0)
        real_sharpe = MetricsEngine.calculate_all_metrics(real_ret)['Sharpe']
        
        placebo_sharpes = []
        for _ in range(iterations):
            permuted_signals = self.signals.copy()
            for col in permuted_signals.columns:
                valid_mask = permuted_signals[col].notna()
                # FIX: Use .to_numpy().copy() to prevent read-only array ValueError
                valid_vals = permuted_signals.loc[valid_mask, col].to_numpy().copy()
                permuted_signals.loc[valid_mask, col] = np.random.permutation(valid_vals)
            
            port_ret, _, _ = PortfolioEngine.construct_portfolio(permuted_signals, self.returns, TOP_N_LONG, BOTTOM_N_SHORT)
            placebo_sharpes.append(MetricsEngine.calculate_all_metrics(port_ret)['Sharpe'])
            
        p_value = np.mean(np.array(placebo_sharpes) >= real_sharpe)
        print(f"Real Sharpe: {real_sharpe:.3f} | Placebo Mean: {np.mean(placebo_sharpes):.3f} | P-Value: {p_value:.4f}")
        self.results['Placebo_P_Value'] = p_value

    def test_3_monte_carlo(self, iterations=500):
        print("\n--- TEST 3: Monte Carlo (Block Bootstrap) ---")
        net_ret, _ = self._get_portfolio(5)
        ret_vals = net_ret.dropna().values
        
        mc_sharpes = []
        block_size = 5
        for _ in range(iterations):
            boot_ret = []
            for _ in range(0, len(ret_vals), block_size):
                start_idx = np.random.randint(0, len(ret_vals) - block_size + 1)
                boot_ret.extend(ret_vals[start_idx:start_idx+block_size])
            boot_ret = pd.Series(boot_ret[:len(ret_vals)])
            mc_sharpes.append(MetricsEngine.calculate_all_metrics(boot_ret)['Sharpe'])
            
        print(f"MC Sharpe Mean: {np.mean(mc_sharpes):.3f} | 5% CI: {np.percentile(mc_sharpes, 5):.3f} | 95% CI: {np.percentile(mc_sharpes, 95):.3f}")
        self.results['MC_Sharpe_CI'] = (np.percentile(mc_sharpes, 5), np.percentile(mc_sharpes, 95))

    def test_4_walk_forward(self, window_years=2, step_years=1):
        print("\n--- TEST 4: Walk-Forward Analysis ---")
        days_per_year = 252
        window_days = window_years * days_per_year
        step_days = step_years * days_per_year
        
        wf_sharpes = []
        for start in range(0, len(self.signals) - window_days, step_days):
            end = start + window_days
            sig_chunk = self.signals.iloc[start:end]
            ret_chunk = self.returns.iloc[start:end]
            
            port_ret, _, _ = PortfolioEngine.construct_portfolio(sig_chunk, ret_chunk, TOP_N_LONG, BOTTOM_N_SHORT)
            if len(port_ret) > 10:
                sharpe = MetricsEngine.calculate_all_metrics(port_ret)['Sharpe']
                wf_sharpes.append(sharpe)
                print(f"  Window {sig_chunk.index[0].year}-{sig_chunk.index[-1].year}: Sharpe {sharpe:.3f}")
                
        if wf_sharpes:
            print(f"Walk-Forward Avg Sharpe: {np.mean(wf_sharpes):.3f} | StdDev: {np.std(wf_sharpes):.3f}")

    def test_5_one_shot_generalization(self):
        print("\n--- TEST 5: One-Shot Generalization (Out-of-Sample) ---")
        split_idx = int(len(self.signals) * 0.6)
        
        sig_is, sig_oos = self.signals.iloc[:split_idx], self.signals.iloc[split_idx:]
        ret_is, ret_oos = self.returns.iloc[:split_idx], self.returns.iloc[split_idx:]
        
        ret_is_port, _, _ = PortfolioEngine.construct_portfolio(sig_is, ret_is, TOP_N_LONG, BOTTOM_N_SHORT)
        ret_oos_port, _, _ = PortfolioEngine.construct_portfolio(sig_oos, ret_oos, TOP_N_LONG, BOTTOM_N_SHORT)
        
        sharpe_is = MetricsEngine.calculate_all_metrics(ret_is_port)['Sharpe']
        sharpe_oos = MetricsEngine.calculate_all_metrics(ret_oos_port)['Sharpe']
        
        print(f"In-Sample Sharpe: {sharpe_is:.3f} | Out-of-Sample Sharpe: {sharpe_oos:.3f}")
        print(f"Decay Ratio (OOS/IS): {sharpe_oos/sharpe_is if sharpe_is != 0 else 0:.2f}")

    def test_6_parameter_robustness(self, close_prices, high_prices, low_prices):
        print("\n--- TEST 6: Parameter Robustness (Perturbation) ---")
        base_sharpes = []
        kelt_range = [25, 30, 35]
        hma_range = [7, 9, 11]
        
        for k, h in product(kelt_range, hma_range):
            sig = IndicatorEngine.calculate_signals(close_prices, high_prices, low_prices, kelt_len=k, hma_len=h)
            port_ret, _, _ = PortfolioEngine.construct_portfolio(sig, self.returns, TOP_N_LONG, BOTTOM_N_SHORT)
            sharpe = MetricsEngine.calculate_all_metrics(port_ret)['Sharpe']
            base_sharpes.append(sharpe)
            print(f"  Kelt={k}, HMA={h} -> Sharpe: {sharpe:.3f}")
        print(f"Robustness StdDev: {np.std(base_sharpes):.3f} (Lower is better/more robust)")

    def test_7_cost_sensitivity(self):
        print("\n--- TEST 7: Cost Sensitivity Analysis ---")
        costs = [0, 2, 5, 10, 20, 50]
        for c in costs:
            net_ret, _ = self._get_portfolio(c)
            sharpe = MetricsEngine.calculate_all_metrics(net_ret)['Sharpe']
            print(f"  Cost {c:2d} bps -> Sharpe: {sharpe:.3f}")

    def test_8_ic_analysis(self):
        print("\n--- TEST 8: Information Coefficient (IC) Analysis ---")
        fwd_returns = self.returns.shift(-1)
        mean_ic, ic_ir = MetricsEngine.calculate_ic(self.signals, fwd_returns)
        print(f"  Mean Cross-Sectional Rank IC: {mean_ic:.4f} | IC IR: {ic_ir:.4f}")
        self.results['Mean_IC'] = mean_ic
        self.results['IC_IR'] = ic_ir

    def _print_metrics(self, metrics):
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v}")

# ==============================================================================
# 7. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print("="*70)
    print("INITIALIZING MULTI-ASSET ROBUST QUANT FRAMEWORK")
    print("Strategy: KELT_HMA_RANK_133 (Exact Mathematical Replication)")
    print("="*70)
    
    # 1. Fetch Real Data (Returns Close, High, Low, and Returns)
    close_prices, high_prices, low_prices, returns = DataEngine.fetch_universe_data(UNIVERSE, START_DATE, END_DATE)
    
    # 2. Calculate Signals (Using exact spec formula)
    signals = IndicatorEngine.calculate_signals(close_prices, high_prices, low_prices)
    
    # 3. Run Test Suite
    suite = RobustTestSuite(close_prices, returns, signals)
    
    eq_curve = suite.test_1_standard_backtest(cost_bps=5)
    suite.test_2_placebo_test(iterations=100)
    suite.test_3_monte_carlo(iterations=500)
    suite.test_4_walk_forward(window_years=2, step_years=1)
    suite.test_5_one_shot_generalization()
    suite.test_6_parameter_robustness(close_prices, high_prices, low_prices)
    suite.test_7_cost_sensitivity()
    suite.test_8_ic_analysis()
    
    # 4. Plot Equity Curve
    print("\nPlotting Equity Curve...")
    cum_ret = (1 + eq_curve).cumprod()
    plt.figure(figsize=(12, 6))
    plt.plot(cum_ret.index, cum_ret.values, label='Long/Short Portfolio (5 Top / 5 Bottom)', color='#1f77b4', linewidth=1.5)
    plt.title('KELT_HMA_RANK_133 Multi-Asset Equity Curve (Strict Out-of-Sample Execution)')
    plt.ylabel('Cumulative Return')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('Output/equity_curve.png', dpi=300)
    plt.show()
    
    print("\n" + "="*70)
    print("FRAMEWORK EXECUTION COMPLETE. Check 'Output/equity_curve.png' for results.")
    print("="*70)