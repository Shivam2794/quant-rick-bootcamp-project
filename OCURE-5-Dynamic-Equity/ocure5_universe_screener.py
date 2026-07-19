import pandas as pd
import numpy as np

def compute_dynamic_universe(close_prices: pd.DataFrame, top_n: int = 20, lookback_days: int = 90) -> pd.DataFrame:
    """
    Computes a dynamic universe membership matrix based on highest realized volatility.
    Rebalances quarterly.
    
    Args:
        close_prices (pd.DataFrame): Daily close prices for the full universe of stocks.
        top_n (int): Number of stocks to select each quarter.
        lookback_days (int): Lookback window for volatility calculation.
        
    Returns:
        pd.DataFrame: A binary matrix (same shape as close_prices) where 1 indicates membership in the universe for that day.
    """
    # 1. Compute daily returns and 90-day realized volatility
    returns = close_prices.pct_change(fill_method=None)
    volatility = returns.rolling(window=lookback_days, min_periods=lookback_days // 2).std() * np.sqrt(252)
    
    # 2. Identify quarterly rebalance dates (Last trading day of Mar, Jun, Sep, Dec)
    # Resample to business month end (BM), then filter for quarters (months 3, 6, 9, 12)
    bm_end = close_prices.resample('BME').last().index
    quarter_ends = [d for d in bm_end if d.month in [3, 6, 9, 12]]
    
    # Also add the very first available day as a starting point if we have data
    if len(close_prices) > lookback_days:
        first_valid = close_prices.index[lookback_days]
        if first_valid not in quarter_ends:
            quarter_ends = [first_valid] + quarter_ends
            quarter_ends.sort()

    # 3. Build the membership matrix
    membership = pd.DataFrame(0, index=close_prices.index, columns=close_prices.columns)
    
    current_basket = []
    
    for i, date in enumerate(close_prices.index):
        if date in quarter_ends:
            # Rebalance!
            # Get the volatility row for this date
            if date in volatility.index:
                vol_row = volatility.loc[date].dropna()
                # Sort by highest volatility
                if len(vol_row) > 0:
                    top_vols = vol_row.sort_values(ascending=False)
                    current_basket = top_vols.head(top_n).index.tolist()
        
        # Assign current basket to the membership matrix
        if current_basket:
            membership.loc[date, current_basket] = 1
            
    return membership
