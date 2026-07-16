import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# BT2 Video 10 & 11: Cross-Sectional Momentum Ranking
#
# Genius Coder / Institutional Quant Implementation
# 
# Key Architectural Features:
# 1. Solves the single-asset "Cash Drag" flaw during mean-reverting regimes.
# 2. Uses a robust Cross-Sectional Ranking algorithm across an N-asset matrix.
# 3. Allocates binary weights (1.0 or 0.0) to the Top-K assets based on Momentum.
# 4. Strict NaN handling for dead/halted/new assets.
# -----------------------------------------------------------------------------

def calculate_momentum_score(close: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    """
    Calculates the standard Rate of Change (ROC) over the lookback window.
    This serves as our baseline momentum metric for ranking.
    """
    if not isinstance(close, pd.DataFrame):
        raise TypeError("Input must be a 2D pandas DataFrame of asset prices.")
        
    # Rate of change: (Price_today / Price_N_days_ago) - 1
    # We use fill_method=None to comply with modern Pandas and avoid silent forward-filling
    roc = close.pct_change(periods=lookback, fill_method=None)
    
    # 5. Infinity Cap (Attack 4 Fix)
    # If the denominator was exactly 0.0 (e.g. data error, penny stock), pandas returns np.inf.
    # An np.inf momentum will permanently steal the #1 rank. 
    # We must explicitly cast infinities to NaN so the na_option='bottom' logic handles them safely.
    roc = roc.replace([np.inf, -np.inf], np.nan)
    
    return roc

def calculate_cross_sectional_rank_mask(momentum_df: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
    """
    Applies daily cross-sectional ranking to the momentum dataframe.
    
    Returns a binary allocation mask (1.0 for invested, 0.0 for cash/ignored)
    where exactly 'top_k' assets are selected per day.
    """
    # 1. Rank across columns (axis=1). 
    # method='first' mathematically guarantees that perfect ties are broken deterministically.
    # This prevents Capital Overflow attacks where method='min' would assign rank 1 to all assets.
    # ascending=False means the highest momentum gets rank 1.0.
    # na_option='bottom' ensures NaNs (dead assets) get the lowest possible rank.
    daily_ranks = momentum_df.rank(axis=1, method='first', ascending=False, na_option='bottom')
    
    # 2. Generate Binary Mask
    # If the rank is <= top_k, it is chosen.
    allocation_mask = (daily_ranks <= top_k).astype(float)
    
    # 3. Post-Process Security Check
    # Ensure that if the ENTIRE row is NaN, it doesn't accidentally allocate
    is_all_nan = momentum_df.isna().all(axis=1)
    allocation_mask.loc[is_all_nan, :] = 0.0
    
    # 4. Enforce that NaNs in the source data CANNOT receive an allocation, 
    # even if there are fewer than K total assets alive.
    allocation_mask[momentum_df.isna()] = 0.0
    
    return allocation_mask
