import numpy as np
import pandas as pd
import warnings

# -----------------------------------------------------------------------------
# BT2 Video 21: The Final Exam - Triple EMA Ensemble
#
# Genius Coder / Institutional Quant Implementation
# 
# Key Architectural Features:
# 1. TERNARY LOGIC STACK: Fast > Medium > Slow determines strict LONG/FLAT/SHORT 
#    conditions natively across 2D DataFrames without loop overhead.
# 2. EWMA HYBRID BLENDING: Inherits the Video 5 continuous shock-absorption gating
#    so signals don't binary whip-saw capital into margins.
# -----------------------------------------------------------------------------

def calculate_triple_ema_signal(close: pd.DataFrame, fast: int = 8, med: int = 21, slow: int = 55) -> pd.DataFrame:
    """
    Calculates a ternary signal (-1, 0, 1) based on Triple EMA alignment.
    1.0 = Fast > Med > Slow (Full Bull)
    -1.0 = Fast < Med < Slow (Full Bear)
    0.0 = Misaligned (Chop / Cash)
    """
    if isinstance(close, pd.Series):
        close = close.to_frame(name="Asset")
        
    fast_ema = close.ewm(span=fast, adjust=False).mean()
    med_ema = close.ewm(span=med, adjust=False).mean()
    slow_ema = close.ewm(span=slow, adjust=False).mean()
    
    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
    
    # Bullish Alignment
    bull_mask = (fast_ema > med_ema) & (med_ema > slow_ema)
    signal[bull_mask] = 1.0
    
    # Bearish Alignment
    bear_mask = (fast_ema < med_ema) & (med_ema < slow_ema)
    signal[bear_mask] = -1.0
    
    # Re-inject NaNs for missing data to prevent false "Cash" signals
    signal[close.isna()] = np.nan
    
    return signal

def apply_ewma_hybrid_smoothing(signal: pd.DataFrame, span: int = 3) -> pd.DataFrame:
    """
    Applies the Video 5 EWMA smoothing to binary/ternary signals to prevent 
    sudden 100% -> 0% volatility shocks in the target position.
    """
    return signal.ewm(span=span, adjust=False).mean()

def generate_triple_ema_ensemble(close: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a blended ensemble of multiple Triple EMA speeds.
    """
    speeds = [
        (4, 12, 32),
        (8, 21, 55),
        (16, 42, 110),
        (32, 84, 220)
    ]
    
    ensemble = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    
    for fast, med, slow in speeds:
        sig = calculate_triple_ema_signal(close, fast, med, slow)
        ensemble += sig
        
    ensemble = ensemble / len(speeds)
    
    # Smooth the final ensemble to absorb entry/exit shock
    smoothed_ensemble = apply_ewma_hybrid_smoothing(ensemble, span=3)
    
    return smoothed_ensemble
