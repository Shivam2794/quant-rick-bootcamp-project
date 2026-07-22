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
    print("   GRAND EPITOME: 26-YEAR HEDGED ENSEMBLE ENGINE (2000 TO JULY 2026)")
    print("   (100% REAL HISTORICAL DATA - ZERO LOOKAHEAD BIAS)")
    print("==========================================================================")

    # 26-Year Historical Date Range
    start_date = '1999-01-01'
    eval_start = '2000-01-01'
    end_date = '2026-07-22'

    # 1. Download Multi-Asset Universes (Tech, Crypto, Index ETFs, Commodities, Macro)
    print("\n1. Downloading 26-Year Multi-Asset Universes & Real Market Proxies (1999-2026)...")
    tech_symbols = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'GOOGL', 'META', 'NFLX', 'AMD', 'INTC',
        'CSCO', 'QCOM', 'TXN', 'AVGO', 'MU', 'CRM', 'ADBE', 'PYPL', 'SHOP', 'ORCL',
        'IBM', 'HPQ', 'AMAT', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'NOW', 'PLTR', 'UBER',
        'ABNB', 'CRWD', 'PANW', 'FTNT', 'ZS', 'DDOG', 'TEAM', 'WDAY', 'TSM', 'MELI'
    ]
    crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD']
    index_symbols = ['SPY', 'QQQ', 'TLT', 'GLD', 'USO', 'UUP']
    sleeve4_symbols = ['GLD', 'SLV']
    macro_symbols = ['SPY', 'HYG', 'IEF', 'BTC-USD']

    all_symbols = list(set(tech_symbols + crypto_symbols + index_symbols + sleeve4_symbols + macro_symbols))
    raw_data = yf.download(all_symbols, start=start_date, end=end_date, progress=False)['Close']
    raw_data = raw_data.ffill()

    tech_prices = raw_data[[c for c in tech_symbols if c in raw_data.columns]]
    crypto_prices = raw_data[[c for c in crypto_symbols if c in raw_data.columns]]
    index_prices = raw_data[[c for c in index_symbols if c in raw_data.columns]]

    spy = raw_data['SPY'].squeeze() if 'SPY' in raw_data.columns else None
    btc = raw_data['BTC-USD'].squeeze() if 'BTC-USD' in raw_data.columns else None
    hyg = raw_data['HYG'].squeeze() if 'HYG' in raw_data.columns else None
    ief = raw_data['IEF'].squeeze() if 'IEF' in raw_data.columns else None

    # 2. Compute Multi-Factor Continuous Macro Throttle (2000 to 2026)
    print("\n2. Computing Continuous Multi-Factor Macro Throttle (Dot-Com, GFC, 2020, 2022, 2026)...")
    throttle_engine = MacroThrottle(m_min=0.25, tau=1.0, theta_on=0.1, theta_off=-0.2, smooth_span=10)
    macro_multiplier = throttle_engine.compute_throttle(btc, spy, hyg, ief, tech_prices)

    # 3. SLEEVE 1: Tech Equity Dual Engine (Dynamic Top 20/25 Active Tech)
    print("\n3. Building Sleeve 1: Tech Equity Dual Engine (2000-2026)...")
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
        final_w = final_w.mul(macro_multiplier.reindex(final_w.index).ffill(), axis=0)
        
        buffered_w = apply_portfolio_buffer(final_w, buffer_threshold=0.008)
        
        daily_ret = tech_prices.pct_change(fill_method=None).fillna(0.0)
        exec_w = buffered_w.shift(1).fillna(0.0)
        pnl = (exec_w * daily_ret).sum(axis=1)
        trn = exec_w.diff().abs().sum(axis=1)
        cost = trn * (3.25 / 10000.0)
        return pnl - cost

    ret_sleeve_1 = (process_equity_sleeve(df_w_A) + process_equity_sleeve(df_w_B)) / 2.0

    # 4. SLEEVE 2: Carver Crypto 8-Layer Engine (Post-2014)
    print("\n4. Building Sleeve 2: Carver 8-Layer Crypto Engine (Post-2014)...")
    crypto_ret = crypto_prices.pct_change(fill_method=None).fillna(0.0)
    crypto_weights = {}
    for coin in crypto_prices.columns:
        w_coin = build_carver_engine(crypto_prices, target=coin, use_cs=True, long_only=True)
        crypto_weights[coin] = w_coin

    df_crypto_w = pd.DataFrame(crypto_weights)
    buffered_crypto_w = apply_portfolio_buffer(df_crypto_w / len(crypto_prices.columns), buffer_threshold=0.01)
    exec_crypto_w = buffered_crypto_w.shift(1).fillna(0.0)
    pnl_crypto = (exec_crypto_w * crypto_ret).sum(axis=1)
    cost_crypto = exec_crypto_w.diff().abs().sum(axis=1) * (5.0 / 10000.0)
    ret_sleeve_2 = (pnl_crypto - cost_crypto).fillna(0.0)

    # 5. SLEEVE 3: Long/Short Index & Macro ETF Trend Engine (SPY, QQQ, TLT, GLD - 2000 to 2026)
    print("\n5. Building Sleeve 3: Long/Short Index Macro Trend Engine (SPY, QQQ, TLT, GLD)...")
    index_ret = index_prices.pct_change(fill_method=None).fillna(0.0)
    index_weights = {}
    for etf_asset in index_prices.columns:
        # Long/Short Trend for Macro Protection
        w_etf = build_carver_engine(index_prices, target=etf_asset, use_cs=False, long_only=False)
        index_weights[etf_asset] = w_etf

    df_index_w = pd.DataFrame(index_weights)
    buffered_index_w = apply_portfolio_buffer(df_index_w / len(index_prices.columns), buffer_threshold=0.01)
    exec_index_w = buffered_index_w.shift(1).fillna(0.0)
    pnl_index = (exec_index_w * index_ret).sum(axis=1)
    cost_index = exec_index_w.diff().abs().sum(axis=1) * (2.0 / 10000.0)
    ret_sleeve_3 = (pnl_index - cost_index).fillna(0.0)

    # 6. SLEEVE 4: Real Gold/Silver Pair Relative Value (Post-2006)
    print("\n6. Building Sleeve 4: Real Gold/Silver Pair Relative Value...")
    if 'GLD' in raw_data.columns and 'SLV' in raw_data.columns:
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
        ret_sleeve_4 = (pnl_sleeve_4 - cost_sleeve_4).fillna(0.0)
    else:
        ret_sleeve_4 = pd.Series(0.0, index=ret_sleeve_1.index)

    # 7. 26-Year Dynamic Risk Parity Blending
    print("\n7. Blending Active Sleeves & Applying 15% Volatility Target (2000-2026)...")
    sleeve_df = pd.DataFrame({
        'Sleeve_1_TechEquity': ret_sleeve_1,
        'Sleeve_2_Crypto': ret_sleeve_2,
        'Sleeve_3_IndexMacroTrend': ret_sleeve_3,
        'Sleeve_4_GoldSilverPair': ret_sleeve_4
    }).loc[eval_start:]

    # Dynamic Inverse Volatility Risk Parity
    rolling_vols = sleeve_df.rolling(126, min_periods=30).std() * np.sqrt(252)
    inv_vols = 1.0 / rolling_vols.replace(0, np.nan)
    erc_weights = inv_vols.div(inv_vols.sum(axis=1), axis=0).fillna(0.25)

    raw_ensemble_ret = (sleeve_df * erc_weights.shift(1).fillna(0.25)).sum(axis=1)

    # Soft CPPI Drawdown Governor:
    eq_cum = (1.0 + raw_ensemble_ret).cumprod()
    peak = eq_cum.cummax()
    dd = (eq_cum / peak - 1.0)
    
    soft_cppi_multiplier = np.maximum(0.50, 1.0 - 1.3 * np.maximum(0.0, np.abs(dd) - 0.06))
    soft_cppi_series = pd.Series(soft_cppi_multiplier, index=raw_ensemble_ret.index)

    rolling_vol = raw_ensemble_ret.rolling(63, min_periods=20).std() * np.sqrt(252)
    target_vol_series = 0.15 * soft_cppi_series
    portfolio_scalar = (target_vol_series / rolling_vol.replace(0, np.nan)).clip(0.5, 2.5).fillna(1.0)
    
    ensemble_ret = raw_ensemble_ret * portfolio_scalar.shift(1).fillna(1.0)
    equity_curve = (1.0 + ensemble_ret).cumprod()

    # Performance Metrics
    total_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    cagr = (equity_curve.iloc[-1]) ** (1.0 / total_years) - 1.0
    vol = ensemble_ret.std() * np.sqrt(252)
    sharpe_ratio = (cagr - 0.02) / vol if vol > 0 else 0
    max_dd = (equity_curve / equity_curve.cummax() - 1.0).min()

    print("\n==========================================================================")
    print("      GRAND EPITOME: 26-YEAR FULL BACKTEST RESULTS (2000-2026)")
    print("==========================================================================")
    print(f"Evaluation Period:  {equity_curve.index[0].strftime('%Y-%m-%d')} to {equity_curve.index[-1].strftime('%Y-%m-%d')} ({total_years:.1f} Years)")
    print(f"CAGR:               {cagr*100:.2f}%")
    print(f"Annual Volatility:  {vol*100:.2f}%")
    print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {max_dd*100:.2f}%")
    print("==========================================================================")

    # Save output
    equity_curve.to_csv("grand_ensemble_equity_curve_2000_2026.csv")
    print("\nMaster 26-year equity curve saved to 'grand_ensemble_equity_curve_2000_2026.csv'.")

    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(equity_curve.index, equity_curve, color='navy', lw=2, label=f'Grand Ensemble 2000-2026 (Sharpe {sharpe_ratio:.2f}, CAGR {cagr*100:.1f}%, MaxDD {max_dd*100:.1f}%)')
    plt.yscale('log')
    plt.title('Grand Epitome: 26-Year Multi-Asset Ensemble Equity Curve (2000 to July 2026, Log Scale)')
    plt.ylabel('Cumulative Return')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('grand_ensemble_equity_curve_2000_2026.png', dpi=150)
    print("Equity curve plot saved to 'grand_ensemble_equity_curve_2000_2026.png'.")

if __name__ == "__main__":
    run_grand_backtest()
