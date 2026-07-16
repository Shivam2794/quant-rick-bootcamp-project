import numpy as np
import pandas as pd
import warnings

# -----------------------------------------------------------------------------
# BT2 Video 7: Data Pipeline & Regime Selection
#
# Genius Coder / Institutional Quant Implementation
# 
# Key Architectural Features:
# 1. THE TRENDINESS FILTER: Mathematically codifies the author's "soft scan". 
#    Uses the Average Directional Index (ADX) to ruthlessly reject choppy assets 
#    (e.g., historical Gold, Dow Jones) and accept healthy trending assets.
# 2. INSTITUTIONAL SPLITTER: Precisely slices data into In-Sample (IS) and 
#    Out-of-Sample (OOS) periods based on a strictly chronological split (e.g., 70/30)
#    ensuring no temporal leakage.
# -----------------------------------------------------------------------------

def calculate_adx(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, lookback: int = 14) -> pd.DataFrame:
    """
    Calculates the Average Directional Index (ADX) for multiple assets.
    ADX measures trend strength, regardless of direction.
    """
    if isinstance(close, pd.Series):
        close = close.to_frame()
        high = high.to_frame()
        low = low.to_frame()
        
    plus_dm = high.diff()
    minus_dm = low.diff()
    
    # +DM and -DM are only valid if they are positive and larger than the other
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = minus_dm.abs()
    
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    # Modern pandas compatible max over level 1 (Transpose, groupby, transpose back)
    concat_tr = pd.concat([tr1, tr2, tr3], axis=1, keys=['tr1', 'tr2', 'tr3'])
    tr = concat_tr.T.groupby(level=1).max().T
    
    # Smoothed using Wilder's Smoothing (roughly equivalent to EWMA with alpha=1/lookback)
    alpha = 1.0 / lookback
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    
    dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    
    return adx.fillna(0.0)

def filter_trending_assets(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, threshold: float = 20.0) -> list:
    """
    The mathematical manifestation of the "soft eye scan".
    Analyzes the entire asset history and rejects assets that spend too much time 
    in a choppy, sideways regime (Mean ADX < threshold).
    
    Returns a list of column names (assets) that passed the rigorous trend test.
    """
    adx_df = calculate_adx(high, low, close)
    
    # We want to measure the proportion of time the asset is actually trending
    # An ADX > threshold indicates a strong trend.
    # If the asset spends less than 30% of its life trending, we reject it.
    trending_mask = adx_df > threshold
    percent_time_trending = trending_mask.mean() # Mean of boolean is the percentage
    
    passing_assets = []
    for asset in percent_time_trending.index:
        if percent_time_trending[asset] > 0.30:
            passing_assets.append(asset)
            
    return passing_assets

def institutional_train_test_split(df: pd.DataFrame, train_ratio: float = 0.70, burn_in: int = 256):
    """
    Splits the dataframe chronologically to prevent temporal data leakage.
    Ensures that the Backtester evaluates In-Sample (training) vs Out-of-Sample (testing).
    
    CRITICAL FIX: Appends a 'burn_in' window to the beginning of the Out-of-Sample data.
    Without this, rolling calculations (like 256-day EWMA) will output NaNs for the first 
    year of OOS trading, resulting in Boundary Window Loss.
    """
    if df.empty:
        raise ValueError("Cannot split an empty DataFrame.")
        
    split_index = int(len(df) * train_ratio)
    
    # The split MUST be perfectly chronological.
    in_sample = df.iloc[:split_index]
    
    # Safely calculate the start of the OOS burn-in window
    oos_start_idx = max(0, split_index - burn_in)
    out_of_sample = df.iloc[oos_start_idx:]
    
    return in_sample, out_of_sample

def run_data_orchestration_pipeline(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, split_ratio: float = 0.70):
    """
    The Video 6 & 7 Master Pipeline.
    1. Ingests raw universe data.
    2. Runs the Brutal Trendiness Filter, destroying chopped/sideways assets.
    3. Slices the surviving data into rigorous IS/OOS splits.
    
    Returns: IS_dict, OOS_dict containing the high, low, close for the surviving assets.
    """
    # 1. Filter Assets
    surviving_assets = filter_trending_assets(high, low, close, threshold=20.0)
    
    if len(surviving_assets) == 0:
        raise ValueError("ALL ASSETS REJECTED. The universe provided is entirely choppy.")
        
    filtered_high = high[surviving_assets]
    filtered_low = low[surviving_assets]
    filtered_close = close[surviving_assets]
    
    # 2. Temporal Split
    high_is, high_oos = institutional_train_test_split(filtered_high, split_ratio)
    low_is, low_oos = institutional_train_test_split(filtered_low, split_ratio)
    close_is, close_oos = institutional_train_test_split(filtered_close, split_ratio)
    
    is_data = {"high": high_is, "low": low_is, "close": close_is}
    oos_data = {"high": high_oos, "low": low_oos, "close": close_oos}
    
    return is_data, oos_data, surviving_assets
