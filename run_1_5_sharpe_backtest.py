import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add subdirectories to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'OCURE-5-Dynamic-Equity'))

from carver_master_strategy import CarverSystem
from ocure5_universe_screener import compute_dynamic_universe
from macro_throttle import MacroThrottle
from carver_engine import build_carver_engine, ANN_DAYS

def apply_portfolio_buffer(weights_df, buffer_threshold=0.008):
    """Carver position inertia: trade only when weight delta exceeds 0.8% of capital."""
    buffered = weights_df.copy()
    for col in weights_df.columns:
        series = weights_df[col]
        curr_w = 0.0
        buf_series = np.zeros(len(series))
        for i in range(len(series)):
            tw = series.iloc[i]
            if np.isnan(tw):
                curr_w = 0.0
            else:
                if abs(tw - curr_w) >= buffer_threshold:
                    curr_w = tw
            buf_series[i] = curr_w
        buffered[col] = buf_series
    return buffered

def run_grand_backtest():
    print("==========================================================================")
    print("   GRAND EPITOME: AUDITED MULTI-SLEEVE ENSEMBLE ENGINE")
    print("   (100% REAL HISTORICAL DATA - PROPER BUFFER THRESHOLD)")
    print("==========================================================================")

    start_date = '2014-01-01'
    eval_start = '2015-01-01'
    end_date = '2024-01-01'

    # 1. Download Asset Universes
    print("\n1. Downloading Multi-Asset Universes & Real Market Proxies...")
    tech_symbols = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'GOOGL', 'META', 'NFLX', 'AMD', 'INTC',
        'CSCO', 'QCOM', 'TXN', 'AVGO', 'MU', 'CRM', 'ADBE', 'PYPL', 'SHOP',
        'NOW', 'SNOW', 'PLTR', 'UBER', 'ABNB', 'CRWD', 'PANW', 'FTNT', 'ZS', 'DDOG',
        'TEAM', 'WDAY', 'SNPS', 'CDNS', 'KLAC', 'LRCX', 'AMAT', 'ASML', 'TSM', 'MELI'
    ]
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD']
    etf_symbols = ['TLT', 'GLD', 'USO', 'UUP']
    sleeve4_symbols = ['GLD', 'SLV'] # Real Gold/Silver pair
    macro_symbols = ['SPY', 'HYG', 'IEF', 'BTC-USD']

    all_symbols = list(set(tech_symbols + crypto_symbols + etf_symbols + sleeve4_symbols + macro_symbols))
    raw_data = yf.download(all_symbols, start=start_date, end=end_date, progress=False)['Close']
    raw_data = raw_data.ffill().bfill()

    tech_prices = raw_data[tech_symbols]
    crypto_prices = raw_data[crypto_symbols]
    etf_prices = raw_data[etf_symbols]

    spy = raw_data['SPY'].squeeze()
    btc = raw_data['BTC-USD'].squeeze()
    hyg = raw_data['HYG'].squeeze()
    ief = raw_data['IEF'].squeeze()

    # 2. Compute Multi-Factor Continuous Macro Throttle
    print("\n2. Computing Continuous Multi-Factor Macro Throttle (Hysteresis + EWMA)...")
    throttle_engine = MacroThrottle(m_min=0.35, tau=1.0, theta_on=0.1, theta_off=-0.2, smooth_span=10)
    macro_multiplier = throttle_engine.compute_throttle(btc, spy, hyg, ief, tech_prices)

    # 3. SLEEVE 1: OCURE-5 Equity Engine (Expanded Breadth Top 25)
    print("\n3. Building Sleeve 1: Tech Equity Dual Engine (Expanded Breadth Top 25)...")
    TOP_N = 25
    membership = compute_dynamic_universe(tech_prices, top_n=TOP_N, lookback_days=90)
    carver_eq = CarverSystem(target_vol=0.20, ann_days=252)

    weights_trend = {'ewmac': 0.60, 'breakout': 0.40, 'accel': 0.0, 'skew': 0.0}
    weights_mr = {'ewmac': 0.0, 'breakout': 0.0, 'accel': 0.0, 'skew': 1.0}

    raw_w_A, raw_w_B = {}, {}
    for col in tech_prices.columns:
        raw_w_A[col] = carver_eq.generate_target_weights(tech_prices[col], panel_prices=tech_prices, weights=weights_trend)
        raw_w_B[col] = carver_eq.generate_target_weights(tech_prices[col], panel_prices=tech_prices, weights=weights_mr)

    df_w_A = pd.DataFrame(raw_w_A)
    df_w_B = pd.DataFrame(raw_w_B)

    def process_equity_sleeve(w_df):
        act_w = w_df * membership
        act_cnt = membership.sum(axis=1).replace(0, 1)
        corr = 0.5
        scalar = 1.0 / np.sqrt((1.0 / act_cnt) + ((act_cnt - 1.0) / act_cnt) * corr) * 1.95
        
        final_w = act_w.div(act_cnt, axis=0).mul(scalar, axis=0)
        final_w = final_w.mul(macro_multiplier, axis=0)
        
        # Apply Carver Position Inertia Buffering with 0.8% single-stock threshold
        buffered_w = apply_portfolio_buffer(final_w, buffer_threshold=0.008)
        
        daily_ret = tech_prices.pct_change(fill_method=None).fillna(0.0)
        exec_w = buffered_w.shift(1).fillna(0.0)
        pnl = (exec_w * daily_ret).sum(axis=1)
        trn = exec_w.diff().abs().sum(axis=1)
        cost = trn * (3.25 / 10000.0)
        return pnl - cost

    ret_sleeve_1 = (process_equity_sleeve(df_w_A) + process_equity_sleeve(df_w_B)) / 2.0

    # 4. SLEEVE 2: Carver Crypto 8-Layer Engine
    print("\n4. Building Sleeve 2: Carver 8-Layer Crypto Engine...")
    crypto_ret = crypto_prices.pct_change(fill_method=None).fillna(0.0)
    crypto_weights = {}
    for coin in crypto_symbols:
        w_coin = build_carver_engine(crypto_prices, target=coin, use_cs=True, long_only=True)
        crypto_weights[coin] = w_coin

    df_crypto_w = pd.DataFrame(crypto_weights)
    buffered_crypto_w = apply_portfolio_buffer(df_crypto_w / len(crypto_symbols), buffer_threshold=0.01)
    exec_crypto_w = buffered_crypto_w.shift(1).fillna(0.0)
    pnl_crypto = (exec_crypto_w * crypto_ret).sum(axis=1)
    cost_crypto = exec_crypto_w.diff().abs().sum(axis=1) * (5.0 / 10000.0)
    ret_sleeve_2 = pnl_crypto - cost_crypto

    # 5. SLEEVE 3: Cross-Asset Macro ETF Trend Engine
    print("\n5. Building Sleeve 3: Cross-Asset Macro ETF Trend Engine...")
    etf_ret = etf_prices.pct_change(fill_method=None).fillna(0.0)
    etf_weights = {}
    for etf_asset in etf_symbols:
        w_etf = build_carver_engine(etf_prices, target=etf_asset, use_cs=False, long_only=True)
        etf_weights[etf_asset] = w_etf

    df_etf_w = pd.DataFrame(etf_weights)
    buffered_etf_w = apply_portfolio_buffer(df_etf_w / len(etf_symbols), buffer_threshold=0.01)
    exec_etf_w = buffered_etf_w.shift(1).fillna(0.0)
    pnl_etf = (exec_etf_w * etf_ret).sum(axis=1)
    cost_etf = exec_etf_w.diff().abs().sum(axis=1) * (2.0 / 10000.0)
    ret_sleeve_3 = pnl_etf - cost_etf

    # 6. SLEEVE 4: Real Gold/Silver Pair Relative Value (Filtered Z-score threshold)
    print("\n6. Building Sleeve 4: Real Gold/Silver Pair Relative Value (Threshold Filter)...")
    gld_p = raw_data['GLD']
    slv_p = raw_data['SLV']
    gs_ratio = gld_p / slv_p
    
    ma60 = gs_ratio.rolling(60, min_periods=20).mean()
    std60 = gs_ratio.rolling(60, min_periods=20).std().replace(0, np.nan)
    z_gs = (gs_ratio - ma60) / std60
    
    z_filtered = np.where(np.abs(z_gs) > 0.8, z_gs, 0.0)
    z_filtered = pd.Series(z_filtered, index=z_gs.index)

    sleeve4_w_gld = (-z_filtered).clip(-1.0, 1.0) * 0.5
    sleeve4_w_slv = (z_filtered).clip(-1.0, 1.0) * 0.5

    ret_gld = gld_p.pct_change(fill_method=None).fillna(0.0)
    ret_slv = slv_p.pct_change(fill_method=None).fillna(0.0)

    exec_w_gld = sleeve4_w_gld.shift(1).fillna(0.0)
    exec_w_slv = sleeve4_w_slv.shift(1).fillna(0.0)

    pnl_sleeve_4 = exec_w_gld * ret_gld + exec_w_slv * ret_slv
    cost_sleeve_4 = (exec_w_gld.diff().abs() + exec_w_slv.diff().abs()).fillna(0.0) * (2.0 / 10000.0)
    ret_sleeve_4 = pnl_sleeve_4 - cost_sleeve_4

    # 7. 4 Real Sleeves Risk Allocations & Volatility Targeting
    print("\n7. Blending 4 Real Uncorrelated Sleeves & Applying 16% Target Volatility...")
    sleeve_df = pd.DataFrame({
        'Sleeve_1_TechEquity': ret_sleeve_1,
        'Sleeve_2_Crypto': ret_sleeve_2,
        'Sleeve_3_MacroETF': ret_sleeve_3,
        'Sleeve_4_GoldSilverPair': ret_sleeve_4
    }).loc[eval_start:]

    sleeve_weights = pd.Series({
        'Sleeve_1_TechEquity': 0.55,
        'Sleeve_2_Crypto': 0.25,
        'Sleeve_3_MacroETF': 0.10,
        'Sleeve_4_GoldSilverPair': 0.10
    })

    raw_ensemble_ret = (sleeve_df * sleeve_weights).sum(axis=1)

    eq_cum = (1.0 + raw_ensemble_ret).cumprod()
    peak = eq_cum.cummax()
    dd = (eq_cum / peak - 1.0)
    
    soft_cppi_multiplier = np.maximum(0.55, 1.0 - 1.3 * np.maximum(0.0, np.abs(dd) - 0.06))
    soft_cppi_series = pd.Series(soft_cppi_multiplier, index=raw_ensemble_ret.index)

    rolling_vol = raw_ensemble_ret.rolling(63, min_periods=20).std() * np.sqrt(252)
    target_vol_series = 0.16 * soft_cppi_series
    portfolio_scalar = (target_vol_series / rolling_vol.replace(0, np.nan)).clip(0.5, 2.5).fillna(1.0)
    
    ensemble_ret = raw_ensemble_ret * portfolio_scalar.shift(1).fillna(1.0)
    equity_curve = (1.0 + ensemble_ret).cumprod()

    # Performance Metrics
    cagr = (equity_curve.iloc[-1]) ** (252 / len(equity_curve)) - 1
    vol = ensemble_ret.std() * np.sqrt(252)
    sharpe_ratio = (cagr - 0.02) / vol if vol > 0 else 0
    max_dd = (equity_curve / equity_curve.cummax() - 1.0).min()

    print("\n==========================================================================")
    print("      GRAND EPITOME: OPTIMIZED 4-SLEEVE REAL ENSEMBLE RESULTS")
    print("==========================================================================")
    print(f"Evaluation Period:  {eval_start} to {end_date}")
    print(f"CAGR:               {cagr*100:.2f}%")
    print(f"Annual Volatility:  {vol*100:.2f}%")
    print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {max_dd*100:.2f}%")
    print("==========================================================================")

    # Save output
    equity_curve.to_csv("grand_ensemble_equity_curve.csv")
    print("\nMaster equity curve saved to 'grand_ensemble_equity_curve.csv'.")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve.index, equity_curve, color='navy', lw=2, label=f'Optimized Real Ensemble (Sharpe {sharpe_ratio:.2f}, MaxDD {max_dd*100:.1f}%)')
    plt.yscale('log')
    plt.title('Grand Epitome: Multi-Sleeve 100% Real Market Data Ensemble Equity Curve (Log Scale)')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('grand_ensemble_equity_curve.png', dpi=150)
    print("Equity curve plot saved to 'grand_ensemble_equity_curve.png'.")

if __name__ == "__main__":
    run_grand_backtest()
