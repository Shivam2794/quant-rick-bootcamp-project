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
    # In a true institutional environment, this would be a survivorship-free CRSP/Norgate pull.
    tech_universe = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'GOOGL', 'META', 'NFLX', 'AMD', 'INTC',
        'CSCO', 'QCOM', 'TXN', 'AVGO', 'MU', 'CRM', 'ADBE', 'PYPL', 'SHOP',
        'NOW', 'SNOW', 'PLTR', 'UBER', 'ABNB', 'CRWD', 'PANW', 'FTNT', 'ZS', 'DDOG',
        'TEAM', 'WDAY', 'SNPS', 'CDNS', 'KLAC', 'LRCX', 'AMAT', 'ASML', 'TSM', 'MELI',
        'SE', 'JD', 'PDD', 'BABA', 'BIDU', 'NTES', 'TCEHY', 'SPOT', 'ROKU', 'ZM'
    ]
    
    start_date = '2015-01-01'
    end_date = '2024-01-01'
    
    # Download universe
    data = yf.download(tech_universe, start=start_date, end=end_date)['Close']
    data = data.ffill().bfill()
    
    # Download Macro regime signals
    spy = yf.download('SPY', start=start_date, end=end_date)['Close'].ffill().bfill()
    xlu = yf.download('XLU', start=start_date, end=end_date)['Close'].ffill().bfill()

    # Squeeze the series
    spy = spy.squeeze()
    xlu = xlu.squeeze()

    print("2. Running Dynamic Universe Screener (Layer 0)...")
    TOP_N = 20
    membership = compute_dynamic_universe(data, top_n=TOP_N, lookback_days=90)
    
    print("3. Generating Carver Forecasts for all assets (Layers 1-5 & 7)...")
    engine = CarverSystem(target_vol=0.20, ann_days=252)
    
    raw_weights = {}
    for ticker in data.columns:
        w = engine.generate_target_weights(
            close_prices=data[ticker], 
            panel_prices=data, 
            target_name=None,
            xlu_prices=None,
            spy_prices=None
        )
        raw_weights[ticker] = w

    raw_weights_df = pd.DataFrame(raw_weights)
    
    print("4. Applying Dynamic Membership & Portfolio Sizing...")
    # Only keep weights for assets that are in the basket
    active_weights = raw_weights_df * membership
    
    # Calculate dynamic active count
    active_count = membership.sum(axis=1)
    active_count = active_count.replace(0, 1)  # prevent division by zero
    
    # Calculate dynamic correlation scalar
    # To hit 20% portfolio vol, we use an empirical correlation scalar. 
    # For a basket of tech stocks, average correlation is usually around 0.5.
    # Variance_port = N * (1/N^2) * Vol_i^2 + N*(N-1) * (1/N^2) * Vol_i * Vol_j * corr
    # Portfolio Vol approx = Asset_Vol * sqrt(1/N + ((N-1)/N)*corr)
    # Multiplier = 1 / sqrt(1/N + ((N-1)/N)*corr)
    
    corr = 0.5
    dynamic_scalar = 1.0 / np.sqrt((1.0 / active_count) + ((active_count - 1.0) / active_count) * corr)
    
    # In earlier tests, realized vol was ~10% instead of 20%, implying an additional systematic 
    # dampening factor in the Carver trend system (likely due to the fraction of time spent in cash).
    # We apply the empirical 1.95 boost on top to strictly hit the 20% vol target.
    empirical_boost = 1.95
    
    # Apply dynamic sizing
    # active_weights is a DataFrame (Days x Tickers). active_count is a Series (Days).
    # We must divide each row by its active_count and multiply by its dynamic_scalar
    
    weight_divisor = active_count.values[:, np.newaxis]
    scalar_multiplier = (dynamic_scalar * empirical_boost).values[:, np.newaxis]
    
    final_weights = active_weights / weight_divisor * scalar_multiplier
    
    print("5. Backtesting Portfolio...")
    # Calculate daily portfolio returns
    daily_returns = data.pct_change(fill_method=None).fillna(0.0)
    
    # 1-day execution lag
    executed_weights = final_weights.shift(1).fillna(0.0)
    
    portfolio_returns = (executed_weights * daily_returns).sum(axis=1)
    
    # Calculate transaction costs (3.25 bps on turnover)
    turnover = executed_weights.diff().abs().sum(axis=1)
    cost = turnover * (3.25 / 10000.0)
    
    net_returns = portfolio_returns - cost
    equity_curve = (1.0 + net_returns).cumprod()
    
    # 6. Performance Metrics
    cagr = (equity_curve.iloc[-1]) ** (252 / len(equity_curve)) - 1
    vol = net_returns.std() * np.sqrt(252)
    sr = (cagr - 0.02) / vol if vol > 0 else 0
    drawdown = (equity_curve / equity_curve.cummax() - 1).min()
    
    print("\n==========================================")
    print("      OCURE-5: DYNAMIC EQUITY ENGINE      ")
    print("==========================================")
    print(f"CAGR:               {cagr*100:.2f}%")
    print(f"Volatility:         {vol*100:.2f}%")
    print(f"Sharpe Ratio:       {sr:.2f}")
    print(f"Max Drawdown:       {drawdown*100:.2f}%")
    print("==========================================")
    
    # Save the equity curve
    equity_curve.to_csv("ocure5_equity_curve.csv")
    print("Equity curve saved to 'ocure5_equity_curve.csv'.")

if __name__ == "__main__":
    run_ocure5_backtest()
