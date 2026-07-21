import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from carver_master_strategy import CarverSystem
from ocure5_universe_screener import compute_dynamic_universe

def run_ocure5_backtest():
    print("1. Downloading Data for Tech/Growth Universe...")
    # A sample of highly volatile, historically significant tech/growth stocks
    tech_universe = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'GOOGL', 'META', 'NFLX', 'AMD', 'INTC',
        'CSCO', 'QCOM', 'TXN', 'AVGO', 'MU', 'CRM', 'ADBE', 'PYPL', 'SHOP',
        'NOW', 'SNOW', 'PLTR', 'UBER', 'ABNB', 'CRWD', 'PANW', 'FTNT', 'ZS', 'DDOG',
        'TEAM', 'WDAY', 'SNPS', 'CDNS', 'KLAC', 'LRCX', 'AMAT', 'ASML', 'TSM', 'MELI',
        'SE', 'JD', 'PDD', 'BABA', 'BIDU', 'NTES', 'TCEHY', 'SPOT', 'ROKU', 'ZM'
    ]
    
    # Use 2014 to build up 200MA for the macro filter and initial vol buffers
    start_date = '2014-01-01'
    eval_start = '2015-01-01'
    end_date = '2024-01-01'
    
    # Download universe
    data = yf.download(tech_universe, start=start_date, end=end_date)['Close']
    data = data.ffill().bfill()
    
    # Download Macro regime signals
    spy = yf.download('SPY', start=start_date, end=end_date)['Close'].ffill().bfill()
    btc = yf.download('BTC-USD', start=start_date, end=end_date)['Close'].ffill().bfill()

    # Squeeze the series
    spy = spy.squeeze()
    btc = btc.squeeze()

    print("2. Running Dynamic Universe Screener (Layer 0)...")
    TOP_N = 20
    membership = compute_dynamic_universe(data, top_n=TOP_N, lookback_days=90)
    
    print("3. Generating Carver Forecasts for 2 Separate Portfolios...")
    engine = CarverSystem(target_vol=0.20, ann_days=252)
    
    # Portfolio A: Pure Trend
    weights_trend = {'ewmac': 0.60, 'breakout': 0.40, 'accel': 0.0, 'skew': 0.0}
    # Portfolio B: Pure Mean Reversion
    weights_mr = {'ewmac': 0.0, 'breakout': 0.0, 'accel': 0.0, 'skew': 1.0}
    
    raw_weights_A = {}
    raw_weights_B = {}
    
    for ticker in data.columns:
        w_A = engine.generate_target_weights(
            close_prices=data[ticker], 
            panel_prices=data, 
            target_name=None,
            btc_prices=btc,
            spy_prices=spy,
            weights=weights_trend
        )
        w_B = engine.generate_target_weights(
            close_prices=data[ticker], 
            panel_prices=data, 
            target_name=None,
            btc_prices=btc,
            spy_prices=spy,
            weights=weights_mr
        )
        raw_weights_A[ticker] = w_A
        raw_weights_B[ticker] = w_B

    raw_weights_df_A = pd.DataFrame(raw_weights_A)
    raw_weights_df_B = pd.DataFrame(raw_weights_B)
    
    def process_portfolio(raw_weights_df):
        # 4. Applying Dynamic Membership & Portfolio Sizing
        active_weights = raw_weights_df * membership
        
        # Calculate dynamic active count
        active_count = membership.sum(axis=1)
        active_count = active_count.replace(0, 1)  # prevent division by zero
        
        corr = 0.5
        dynamic_scalar = 1.0 / np.sqrt((1.0 / active_count) + ((active_count - 1.0) / active_count) * corr)
        empirical_boost = 1.95
        
        weight_divisor = active_count.values[:, np.newaxis]
        scalar_multiplier = (dynamic_scalar * empirical_boost).values[:, np.newaxis]
        
        final_weights = active_weights / weight_divisor * scalar_multiplier
        
        # 5. Backtesting Portfolio
        daily_returns = data.pct_change(fill_method=None).fillna(0.0)
        executed_weights = final_weights.shift(1).fillna(0.0)
        
        portfolio_returns = (executed_weights * daily_returns).sum(axis=1)
        turnover = executed_weights.diff().abs().sum(axis=1)
        cost = turnover * (3.25 / 10000.0)
        
        net_returns = portfolio_returns - cost
        return net_returns

    print("4. Processing Portfolios and Blending (50/50)...")
    net_A = process_portfolio(raw_weights_df_A)
    net_B = process_portfolio(raw_weights_df_B)
    
    # Blend 50/50
    blended_net = (net_A + net_B) / 2.0
    
    # Clip to evaluation period to match previous benchmark
    eval_net = blended_net.loc[eval_start:]
    
    equity_curve = (1.0 + eval_net).cumprod()
    
    # 6. Performance Metrics
    cagr = (equity_curve.iloc[-1]) ** (252 / len(equity_curve)) - 1
    vol = eval_net.std() * np.sqrt(252)
    sr = (cagr - 0.02) / vol if vol > 0 else 0
    drawdown = (equity_curve / equity_curve.cummax() - 1).min()
    
    print("\n==========================================")
    print("      OCURE-5: DUAL EQUITY ENGINE         ")
    print("      (BTC/SPY 200MA MACRO EXIT)          ")
    print("==========================================")
    print(f"CAGR:               {cagr*100:.2f}%")
    print(f"Volatility:         {vol*100:.2f}%")
    print(f"Sharpe Ratio:       {sr:.2f}")
    print(f"Max Drawdown:       {drawdown*100:.2f}%")
    print("==========================================")
    
    # Save the equity curve
    equity_curve.to_csv("ocure5_equity_curve_dual.csv")
    print("Equity curve saved to 'ocure5_equity_curve_dual.csv'.")

if __name__ == "__main__":
    run_ocure5_backtest()
