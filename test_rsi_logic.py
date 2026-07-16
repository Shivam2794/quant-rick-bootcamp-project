import pandas as pd
import numpy as np
import pandas_ta as ta

def test_cumulative_rsi():
    # Create a synthetic price series
    prices = pd.Series([10, 11, 10.5, 12, 13, 12.5, 14, 15, 14.5, 16, 17, 18, 17.5, 19, 20])
    
    # 1. Standard Price RSI
    rsi_price = ta.rsi(prices, length=5)
    
    # 2. Cumulative Return RSI (what the author actually meant)
    returns = prices.pct_change()
    cum_returns = returns.cumsum()
    rsi_cum_ret = ta.rsi(cum_returns.fillna(0), length=5)
    
    # 3. Log Return Cumulative RSI
    log_returns = np.log(prices / prices.shift(1))
    cum_log_returns = log_returns.cumsum()
    rsi_log_ret = ta.rsi(cum_log_returns.fillna(0), length=5)
    
    df = pd.DataFrame({
        'Price': prices,
        'RSI_Price': rsi_price,
        'RSI_CumPct': rsi_cum_ret,
        'RSI_CumLog': rsi_log_ret
    })
    
    print(df.tail())
    
    # Check if they are different
    diff = (rsi_price - rsi_cum_ret).abs().mean()
    print(f"Mean Absolute Difference between Price RSI and CumPct RSI: {diff:.4f}")

if __name__ == '__main__':
    test_cumulative_rsi()
