import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# BT2 Video 22: Multi-Factor Ranked Asset Allocation Model (RAAM)
#
# Genius Coder / Institutional Quant Implementation
# 
# Key Architectural Features:
# 1. ORTHOGONAL Z-SCORING: All 4 factors (Momentum, Volatility, Correlation, 
#    Entropy) are calculated, standardized via Z-Scores cross-sectionally, and 
#    summed to create a singular dynamic alpha score.
# 2. PENALTY INVERSION: Volatility and Entropy are strictly penalized (lower is better), 
#    so their Z-scores are inverted (-1 multiplier).
# -----------------------------------------------------------------------------

def calculate_momentum_factor(close: pd.DataFrame, lookback: int = 90) -> pd.DataFrame:
    """Calculates the absolute rate of change over the lookback window."""
    # Replace zeros to prevent np.inf
    safe_close = close.replace(0.0, np.nan)
    momentum = safe_close.pct_change(periods=lookback, fill_method=None)
    return momentum

def calculate_volatility_factor(close: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    """Calculates the rolling standard deviation of daily returns (annualized)."""
    returns = close.pct_change(fill_method=None)
    vol = returns.rolling(window=lookback, min_periods=10).std() * np.sqrt(252)
    return vol

def calculate_correlation_factor(close: pd.DataFrame, lookback: int = 36) -> pd.DataFrame:
    """
    Calculates the average rolling correlation of each asset against the rest 
    of the portfolio. Higher average correlation means the asset moves with the herd.
    """
    returns = close.pct_change(fill_method=None)
    rolling_corr = returns.rolling(window=lookback, min_periods=10).corr()
    
    # We want to unstack the rolling corr, calculate the mean correlation per asset per day
    try:
        # Mean correlation across columns for each (Date, Asset) pair
        # rolling_corr has MultiIndex (Date, Asset1) as rows, Asset2 as columns
        mean_corr = rolling_corr.mean(axis=1) # Mean of each row
        # Now unstack so Date is index and Asset is column
        corr_factor = mean_corr.unstack()
    except Exception:
        # Fallback if matrix algebra fails
        corr_factor = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        
    return corr_factor

def calculate_entropy_factor(close: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Approximates Shannon Entropy / Noise level. 
    Here we use a simplified proxy: Efficiency Ratio (Directional Move / Sum of Path).
    A high efficiency ratio means low entropy (straight line).
    A low efficiency ratio means high entropy (choppy mess).
    We INVERT it so high value = high entropy.
    """
    directional_move = (close - close.shift(lookback)).abs()
    sum_of_path = close.diff().abs().rolling(window=lookback, min_periods=lookback//2).sum()
    
    # Efficiency ratio (ER) = directional_move / sum_of_path
    er = directional_move / sum_of_path.replace(0.0, np.nan)
    
    # High entropy = 1 - ER
    entropy = 1.0 - er.fillna(0.0)
    return entropy

def cross_sectional_zscore(factor_df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes a factor score across the assets for each day."""
    # (x - mean) / std across columns (axis=1)
    mean = factor_df.mean(axis=1)
    std = factor_df.std(axis=1).replace(0.0, np.nan) # prevent div by zero
    
    z_scores = factor_df.sub(mean, axis=0).div(std, axis=0)
    return z_scores.fillna(0.0)

def generate_multifactor_raam_score(close: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the final unified RAAM score by combining all 4 orthogonal factors.
    """
    mom = calculate_momentum_factor(close)
    vol = calculate_volatility_factor(close)
    corr = calculate_correlation_factor(close)
    entropy = calculate_entropy_factor(close)
    
    z_mom = cross_sectional_zscore(mom)
    z_vol = cross_sectional_zscore(vol)
    z_corr = cross_sectional_zscore(corr)
    z_entropy = cross_sectional_zscore(entropy)
    
    # Equal Weighting Logic:
    # 1. Momentum: + (We want high momentum)
    # 2. Volatility: - (We want low volatility)
    # 3. Correlation: - (We want low correlation to the herd for diversification)
    # 4. Entropy: - (We want low noise)
    
    final_alpha_score = z_mom - z_vol - z_corr - z_entropy
    
    return final_alpha_score

def apply_raam_ranking(alpha_score: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
    """
    Ranks the final alpha score and allocates equal capital to the Top K assets.
    """
    # method='first' prevents perfect tie capital overflows (Video 10 fix)
    ranks = alpha_score.rank(axis=1, ascending=False, method='first', na_option='bottom')
    
    allocation_mask = pd.DataFrame(0.0, index=alpha_score.index, columns=alpha_score.columns)
    allocation_mask[ranks <= top_k] = 1.0 / top_k
    
    # Ensure NaNs get 0
    allocation_mask[alpha_score.isna()] = 0.0
    
    # Prevent Lookahead Bias
    return allocation_mask.shift(1).fillna(0.0)
