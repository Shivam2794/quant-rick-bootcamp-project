import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# BT2 Video 9: Ensemble Math Engine (TEMA + MACD)
#
# Genius Coder / Institutional Quant Implementation
# 
# Key Architectural Features:
# 1. TEMA (Triple Exponential Moving Average): Ultra-fast reaction, minimal lag.
#    Formula: 3*EMA1 - 3*EMA2 + EMA3
# 2. MACD (Moving Average Convergence Divergence): Slow, deep trend conviction.
#    Standard: 12, 26, 9.
# 3. ENSEMBLE GATES: "Twins that correct each other's flaws".
#    Combines indicators using logical OR / AND to suppress noise/whipsaw.
# -----------------------------------------------------------------------------

def calculate_tema(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Calculates the Triple Exponential Moving Average (TEMA).
    Reduces the lag of a traditional EMA while reacting aggressively to snaps.
    """
    # Ensure correct alignment/dimension
    if isinstance(close, pd.Series):
        close = close.to_frame()
        
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    
    tema = (3 * ema1) - (3 * ema2) + ema3
    return tema

def calculate_macd(close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Calculates the MACD line and Signal line.
    Returns: macd_line, signal_line
    """
    if isinstance(close, pd.Series):
        close = close.to_frame()
        
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line, signal_line

def calculate_ensemble_signals(close: pd.DataFrame, tema_window: int = 20, logic: str = "OR") -> pd.DataFrame:
    """
    The Core Ensemble Gating Engine.
    
    Generates binary states (1.0 for Long, 0.0 for Cash).
    
    TEMA Signal: Price > TEMA
    MACD Signal: MACD Line > Signal Line
    
    Logic = "OR": Aggressive. Enters if EITHER indicator catches the trend.
    Logic = "AND": Conservative. Enters only if BOTH agree (maximum noise suppression).
    """
    
    if isinstance(close, pd.Series):
        close = close.to_frame()
        
    # 1. TEMA State
    tema = calculate_tema(close, window=tema_window)
    tema_bullish = (close > tema).astype(float)
    
    # 2. MACD State (Deep Macro Trend: MACD Line > 0)
    # The zero-crossover is much slower and immune to fast whipsaws.
    macd_line, signal_line = calculate_macd(close)
    macd_bullish = (macd_line > 0.0).astype(float)
    
    # 3. Macro Trend Filter (Video 8: 200 SMA)
    sma_200 = close.rolling(window=200, min_periods=100).mean()
    macro_bullish = (close > sma_200).astype(float)
    
    # 4. Ensemble Gate
    if logic.upper() == "OR":
        # If either short-term indicator is bullish, be long
        short_term_state = ((tema_bullish + macd_bullish) > 0).astype(float)
    elif logic.upper() == "AND":
        # Both must be bullish
        short_term_state = ((tema_bullish + macd_bullish) == 2).astype(float)
    else:
        raise ValueError(f"Unknown logic gate: {logic}. Must be 'OR' or 'AND'.")
        
    # The ultimate institutional gate: Short-term momentum MUST be confirmed by Macro Trend (Video 8)
    ensemble_state = (short_term_state * macro_bullish)
    
    # 5. Institutional Null-Handling (Attack 1 Fix)
    # If the underlying price is missing (NaN), we MUST output NaN, not 0.0.
    # A 0.0 implies a conscious decision to go to Cash. NaN implies "Data does not exist".
    is_missing = close.isna()
    ensemble_state[is_missing] = np.nan
    tema_bullish[is_missing] = np.nan
    macd_bullish[is_missing] = np.nan
    macro_bullish[is_missing] = np.nan
    
    return ensemble_state, tema_bullish, macd_bullish, macro_bullish
