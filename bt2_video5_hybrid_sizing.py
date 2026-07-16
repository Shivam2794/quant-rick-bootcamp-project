import numpy as np
import pandas as pd
import warnings

# Import the Carver engine components from Video 4
from bt2_video4_math_derived import calculate_annualized_volatility, calculate_dynamic_fdm_blended_forecast, master_sizer

# -----------------------------------------------------------------------------
# BT2 Video 5: The Hybrid Sizing Engine (Carver vs. Ensembles)
#
# Genius Coder / Institutional Quant Implementation
# 
# Key Architectural Fixes:
# 1. MULTI-ASSET VECTORIZATION: Supports 2D DataFrames for portfolio gating.
# 2. GATING SMOOTHING (ANTI-SHOCK): When the ensemble disagrees, risk is not 
#    violently dropped to 0 instantly. The forecast is smoothed using an EWMA
#    filter to gracefully fade in and out of the market.
# 3. TEMPORAL SYNCHRONIZATION: Target positions explicitly shifted to day T+1.
# -----------------------------------------------------------------------------

def donchian_breakout_flag(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Classic binary ensemble signal. 
    1 = Long, -1 = Short, 0 = Flat
    Vectorized for 2D DataFrames.
    """
    if isinstance(close, pd.Series):
        close = close.to_frame()
        high = high.to_frame()
        low = low.to_frame()
        
    rolling_high = high.rolling(window=lookback, min_periods=lookback//2).max().shift(1)
    rolling_low = low.rolling(window=lookback, min_periods=lookback//2).min().shift(1)
    
    signal = pd.DataFrame(0.0, index=close.index, columns=close.columns, dtype=float)
    
    signal[close > rolling_high] = 1.0
    signal[close < rolling_low] = -1.0
    
    signal = signal.replace(0.0, np.nan).ffill().fillna(0.0)
    return signal

def run_hybrid_prop_firm_engine(
    close: pd.DataFrame, 
    high: pd.DataFrame, 
    low: pd.DataFrame, 
    capital: float = 100000.0, 
    mode: str = "eval_passing",
    trading_days: float = 256.0
):
    """
    Implements the hybrid architecture, optimized for Prop-Firm limits, 
    with institutional lookahead prevention and Multi-Asset support.
    """
    if isinstance(close, pd.Series):
        close = close.to_frame(name="Asset")
        high = high.to_frame(name="Asset")
        low = low.to_frame(name="Asset")
        
    # 1. Target Volatility Dials
    if mode == "eval_passing":
        target_vol = 0.35
    elif mode == "funded_protect":
        target_vol = 0.12
    else:
        target_vol = 0.15
        
    # 2. The Timing Flag (Ensemble)
    ensemble_direction = donchian_breakout_flag(high, low, close, lookback=20)
    
    # 3. The Carver Components
    annual_vol = calculate_annualized_volatility(close, trading_days=trading_days)
    blended_forecast = calculate_dynamic_fdm_blended_forecast(close, trading_days=trading_days)
    
    # 4. Hybrid Gating with EWMA Smoothing (Shock Absorption)
    carver_direction = np.sign(blended_forecast)
    gated_forecast = blended_forecast.copy()
    
    disagreement_mask = (ensemble_direction != carver_direction) & (ensemble_direction != 0.0)
    gated_forecast[disagreement_mask | (ensemble_direction == 0.0)] = 0.0
    
    # NEW INSTITUTIONAL FIX: Smooth the gated forecast to prevent violent volatility shocks
    # e.g., dropping from 20 to 0 overnight, then back to 20.
    # An EWMA span of 3 days gracefully fades the signal over half a week.
    smoothed_gated_forecast = gated_forecast.ewm(span=3, adjust=False).mean()
    
    # 5. Final Sizing (Continuous Control + Turnover Buffer)
    target_positions = master_sizer(
        combined_forecast=smoothed_gated_forecast,
        close=close,
        annual_vol=annual_vol,
        capital=capital,
        target_vol=target_vol,
        use_buffer=True # Essential for institutional trading
    )
    
    # 6. INSTITUTIONAL FIX: PREVENT LOOKAHEAD BIAS
    executable_hybrid_positions = target_positions.shift(1).fillna(0.0)
    
    return executable_hybrid_positions, smoothed_gated_forecast, ensemble_direction
