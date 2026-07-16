import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# BT2 Video 24: Beta Rotations Meta-Layer (Regime Awareness)
#
# Genius Coder / Institutional Quant Implementation
# 
# Key Architectural Features:
# 1. ORTHOGONAL META-LAYER: Acts as a supreme regime gate overriding standard RSI/Momentum.
# 2. SECTOR RELATIVE STRENGTH: Calculates the ratio of Risk-On (e.g., Tech/XLK) 
#    vs Risk-Off (e.g., Utilities/XLU) against the broader market (SPY).
# 3. FAST EMA HYBRID: Uses short-term EMAs on the sector ratios to detect 
#    sudden liquidity shifts instantly.
# -----------------------------------------------------------------------------

def calculate_relative_strength_ratio(numerator_asset: pd.Series, denominator_asset: pd.Series) -> pd.Series:
    """
    Calculates the normalized ratio between two assets. 
    Protects against division by zero.
    """
    safe_denominator = denominator_asset.replace(0.0, np.nan)
    return numerator_asset / safe_denominator

def detect_regime_flow(risk_on_asset: pd.Series, risk_off_asset: pd.Series, spy_benchmark: pd.Series, lookback: int = 21) -> pd.Series:
    """
    Identifies if market liquidity is flowing into Risk-On or Risk-Off sectors.
    Returns 1.0 (Risk-On / Bull Regime) or 0.0 (Risk-Off / Defensive Regime).
    """
    # Calculate Ratios vs SPY
    risk_on_ratio = calculate_relative_strength_ratio(risk_on_asset, spy_benchmark)
    risk_off_ratio = calculate_relative_strength_ratio(risk_off_asset, spy_benchmark)
    
    # Smooth the ratios to prevent daily noise whipsaw
    smooth_on = risk_on_ratio.ewm(span=lookback, adjust=False).mean()
    smooth_off = risk_off_ratio.ewm(span=lookback, adjust=False).mean()
    
    # Determine the Meta-Regime
    # If Risk-On sector is outperforming the Risk-Off sector relative to SPY
    # Alternatively, you can just compare smooth_on > smooth_on.shift(X), but Sector vs Sector is more robust.
    
    # To compare them directly, we can check the momentum of the ratio
    # If Tech is accelerating faster than Utilities, Risk-On is True.
    on_mom = smooth_on.pct_change(periods=5).fillna(0.0)
    off_mom = smooth_off.pct_change(periods=5).fillna(0.0)
    
    regime_mask = pd.Series(0.0, index=spy_benchmark.index)
    regime_mask[on_mom > off_mom] = 1.0
    
    # Safety Check: If SPY data is missing or corrupted (<= 0), default to Risk-Off (0.0)
    regime_mask[spy_benchmark.isna() | (spy_benchmark <= 0.0)] = 0.0
    
    return regime_mask

def apply_beta_rotation_gate(base_signals: pd.DataFrame, regime_mask: pd.Series) -> pd.DataFrame:
    """
    Applies the meta-layer gate to the underlying strategy signals (e.g. RSI rotation).
    If Regime is Risk-Off (0.0), forces all long allocations to Cash (0.0).
    """
    # Align the 1D regime mask across the 2D signal DataFrame
    gated_signals = base_signals.multiply(regime_mask, axis=0)
    
    # Prevent lookahead bias - shift the gate 1 day forward
    return gated_signals.shift(1).fillna(0.0)
