"""
bt2_video3_macro_canaries.py — BootCamp 2.0 Macroeconomic Beta Rotation
========================================================================
Implements the macro layer described in Video 3:
1. Four-Canary Voting System (XLU/SPY, XLP/XLY, LQD/HYG, SPY/QQQ).
2. Defensive Bucket Allocation (0% to 100% in 25% increments).
3. VIX Term Structure Flag (VIX > VIX3M).

BRUTAL QA REQUIREMENTS:
1. Uses yfinance to dynamically pull the macro ETFs independent of the micro universe.
2. 100% Vectorized.
3. No lookahead bias (Z-score calculation is shifted to prevent today's ratio from influencing its own historical mean before signaling).
"""

import pandas as pd
import numpy as np
import yfinance as yf
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

MACRO_TICKERS = ['SPY', 'XLU', 'XLY', 'XLP', 'HYG', 'LQD', 'QQQ', '^VIX', '^VIX3M']
OUTPUT_PATH = Path(__file__).parent / "macro_regime.parquet"
Z_LOOKBACK = 120

def calculate_canary(df: pd.DataFrame, num_ticker: str, den_ticker: str, lookback: int = Z_LOOKBACK) -> pd.Series:
    """
    Calculates a macroeconomic risk-off canary flag based on the relative momentum of Defensive vs. Risk-On assets.
    
    The function calculates the ratio of a defensive ticker (e.g. Utilities) to a risk-on ticker (e.g. S&P 500),
    and computes the Z-score of this ratio against its trailing exponential moving average. If the Z-score 
    is positive (i.e. defensive assets are accelerating faster than risk-on), the canary fires (1.0).

    Args:
        df (pd.DataFrame): Vectorized historical prices containing macro ETF columns.
        num_ticker (str): The ticker symbol for the defensive asset (numerator).
        den_ticker (str): The ticker symbol for the risk-on asset (denominator).
        lookback (int, optional): The exponential moving average window span. Defaults to Z_LOOKBACK.

    Returns:
        pd.Series: A binary time-series flag (1.0 for Risk-Off, 0.0 for Risk-On) mapping the canary state.
    """
    logger.info(f"Calculating Canary: {num_ticker} / {den_ticker}")
    ratio = df[num_ticker] / df[den_ticker]
    
    # INSTITUTIONAL QA FIX 8 (Pass 8): The Phantom Time Halt
    # EWMA calculations must be chronologically continuous. If an ETF stops trading 
    # for 3 days, we MUST ffill() to force the EWMA weights to decay properly through time.
    filled_ratio = ratio.ffill()
    
    # SHIFT 1: Prevent lookahead.
    shifted_ratio = filled_ratio.shift(1)
    
    # INSTITUTIONAL QA FIX 6 (Pass 6): Ghost Data Drop-Off Effect
    # Hard rolling windows (.rolling(120)) cause violent, false Z-score spikes when a 121-day old 
    # outlier suddenly falls out of the window. We MUST use EWMA to smoothly decay old data.
    roll_mean = shifted_ratio.ewm(span=lookback, adjust=False).mean()
    roll_std = shifted_ratio.ewm(span=lookback, adjust=False).std()
    

    # Enforce exact 120-day warmup (measured against the original un-filled ratio to prevent false active days)
    valid_count = ratio.shift(1).notna().cumsum()
    roll_mean = roll_mean.where(valid_count >= lookback, np.nan)
    roll_std = roll_std.where(valid_count >= lookback, np.nan)
    
    # Avoid division by zero if ETF flatlines
    roll_std = roll_std.replace(0.0, np.nan)
    z_score = (ratio - roll_mean) / roll_std
    
    # Flag is 1 if Z-score > 0 (Defensive is outperforming)
    # INSTITUTIONAL QA FIX: pandas evaluates (NaN > 0) as False (0.0), causing data drops to default to Risk-On!
    # We MUST propagate NaNs explicitly.
    flag = (z_score > 0.0).astype(float)
    flag = flag.where(z_score.notna(), np.nan)
    
    # Mask warmup period
    valid_count = ratio.notna().cumsum()
    flag = flag.where(valid_count > lookback, np.nan)
    
    return flag

def main():
    logger.info("Starting BT2 Video 3 Macro Canary Engine...")
    
    logger.info(f"Downloading Macro Tickers: {MACRO_TICKERS}")
    # Download data
    data = yf.download(MACRO_TICKERS, start='2005-01-01', auto_adjust=True, progress=False)
    
    if data.empty:
        logger.error("Failed to download macro data.")
        sys.exit(1)
        
    # INSTITUTIONAL QA FIX 12 (Pass 12): YFinance Dimensionality Collapse
    # If yfinance catastrophically drops 8 out of 9 macro tickers, it may return a flat DataFrame 
    # instead of a MultiIndex, causing an AttributeError on data['Close'].columns.
    # We must explicitly check the index depth to guarantee structural integrity.
    if isinstance(data.columns, pd.MultiIndex):
        # Extract available tickers from the second level of the MultiIndex
        available_tickers = data.columns.get_level_values(1).unique()
        missing_tickers = [t for t in MACRO_TICKERS if t not in available_tickers]
    else:
        # Dimensional collapse occurred: it returned a flat dataframe for 1 ticker.
        missing_tickers = MACRO_TICKERS
        
    if missing_tickers:
        logger.error(f"CRITICAL: Failed to download required macro tickers: {missing_tickers}")
        sys.exit(1)
        
    if 'Adj Close' in data:
        prices = data['Adj Close'].sort_index().ffill(limit=5)
    elif 'Close' in data:
        prices = data['Close'].sort_index().ffill(limit=5)
    else:
        logger.error("CRITICAL: Neither 'Adj Close' nor 'Close' found in data.")
        sys.exit(1)
    
    # Calculate Canaries (Numerator must be Defensive, Denominator Risk-On)
    # Canary 1: Utilities vs Equities
    c1 = calculate_canary(prices, 'XLU', 'SPY')
    # Canary 2: Staples vs Discretionary
    c2 = calculate_canary(prices, 'XLP', 'XLY')
    # Canary 3: Investment Grade vs High Yield
    c3 = calculate_canary(prices, 'LQD', 'HYG')
    # Canary 4: Broad Market vs Tech (SPY acts defensive relative to QQQ)
    c4 = calculate_canary(prices, 'SPY', 'QQQ')
    
    # INSTITUTIONAL QA FIX 5 (Pass 5): Asset Inception Blindness
    # HYG was launched in 2007. LQD was launched in 2002.
    # We MUST use .mean(axis=1) to dynamically average ONLY the active canaries.
    # INSTITUTIONAL QA FIX 6 (Pass 6): Genesis Blackout
    # Before the first ETF launched (e.g. 1990s), all canaries are NaN.
    # This causes the defensive_weight to become NaN, crashing the allocator.
    # We must explicitly default to 0.0 (Max Risk-On) during the Genesis era.
    canaries = pd.DataFrame({'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4})
    defensive_weight = canaries.mean(axis=1).fillna(0.0)
    
    # VIX Stress Flag (Backwardation)
    logger.info("Calculating VIX Term Structure...")
    vix_stress = (prices['^VIX'] > prices['^VIX3M']).astype(float)
    # Both need to be valid to calculate, and ensure no zero-feed API glitches
    valid_vix = prices['^VIX'].notna() & prices['^VIX3M'].notna() & (prices['^VIX'] > 0) & (prices['^VIX3M'] > 0)
    vix_stress = vix_stress.where(valid_vix, np.nan)
    
    # Combine into regime DataFrame
    # INSTITUTIONAL QA FIX 15 (Pass 14): VIX NaN Leakage / Panic Liquidation
    # Using .where(vix_stress < 1.0, 1.0) evaluates to False if vix_stress is NaN.
    # This means if Yahoo Finance drops a single day of VIX data, the engine defaults 
    # to 1.0 (Max Risk-Off) and liquidates the entire equity portfolio to cash!
    # We MUST use .mask(vix_stress == 1.0, 1.0) to safely pass-through the Canary weight if VIX is missing.
    final_defensive_weight = defensive_weight.mask(vix_stress == 1.0, 1.0)
    regime = pd.DataFrame({
        'Defensive_Weight': final_defensive_weight,
        'VIX_Stress': vix_stress,
        'XLU_SPY_Flag': c1,
        'XLP_XLY_Flag': c2,
        'LQD_HYG_Flag': c3,
        'SPY_QQQ_Flag': c4
    })
    
    # Drop rows where all are NaN (usually weekends or very early dates before ETFs existed)
    regime = regime.dropna(how='all')
    
    logger.info(f"Saving Macro Regime to {OUTPUT_PATH}")
    regime.to_parquet(OUTPUT_PATH)
    
    # Brutal QA
    logger.info("Running Brutal QA Assertions...")
    assert (regime['Defensive_Weight'].dropna() >= 0.0).all() and (regime['Defensive_Weight'].dropna() <= 1.0).all(), "Defensive Weight out of bounds"
    assert (regime['VIX_Stress'].dropna().isin([0.0, 1.0]).all()), "VIX Stress is not binary"
    
    logger.info("BT2 Video 3 Macro Canary Engine Complete. ALL QA PASSED.")

if __name__ == "__main__":
    main()
