import numpy as np
import pandas as pd
import warnings

# -----------------------------------------------------------------------------
# BT2 Video 4: Math-Derived Trend Following Engine (The 2.0 System)
#
# Genius Coder / Institutional Quant Implementation
# 
# Key Architectural Fixes over Naive Retail Versions:
# 1. MULTI-ASSET VECTORIZATION: Engine accepts 2D pd.DataFrames for portfolio-wide
#    execution and dynamic Instrument Diversification Multiplier (IDM) calculation.
# 2. TRADING DAYS PARAMETER: Eliminates hardcoded sqrt(256), accepting dynamic 
#    annualization constants (Crypto=365, Equities=252).
# 3. BUFFERED POSITION SIZING: Implements a 10% position tolerance buffer to 
#    eradicate micro-trading transaction friction.
# 4. STRICT LOOKAHEAD PREVENTION: Target positions explicitly shifted by 1 day.
# 5. DYNAMIC CORRELATION (FDM): 3D tensor math for FDM across all rules and assets.
# -----------------------------------------------------------------------------

def calculate_annualized_volatility(close: pd.DataFrame, lookback: int = 36, trading_days: float = 256.0) -> pd.DataFrame:
    """
    Calculates the rolling annualized volatility using Parkinson or standard close-to-close.
    Accepts 1D Series or 2D DataFrames.
    """
    # Ensure DataFrame for consistent vectorization
    if isinstance(close, pd.Series):
        close = close.to_frame(name="Asset")
        
    # INSTITUTIONAL FIX (Carver Math): Use 36-day EMA of Absolute Price Differences 
    # scaled by 1.2533 instead of rolling stdev of pct_returns to prevent outlier skew.
    price_diff = close.diff()
    abs_price_diff = price_diff.abs()
    
    # 36-day EMA of absolute diffs
    mad = abs_price_diff.ewm(span=lookback, adjust=False, min_periods=10).mean()
    
    # Convert MAD to standard deviation equivalent using Carver's constant 1.2533
    daily_vol = mad * 1.2533
    
    # Convert to annual volatility percentage (divide by price)
    # sigma_p in Carver's formula is actually price * (annual_vol / sqrt(trading_days)).
    # We will output annualized percentage volatility.
    annual_vol_pct = (daily_vol / close) * np.sqrt(trading_days)
    
    return annual_vol_pct

def calc_ewmac_forecast(close: pd.DataFrame, fast_span: int, slow_span: int, annual_vol: pd.DataFrame, trading_days: float = 256.0) -> pd.DataFrame:
    """
    Calculates the raw crossover and standardizes it into 'normal-day units'.
    """
    fast_ewma = close.ewm(span=fast_span, adjust=False).mean()
    slow_ewma = close.ewm(span=slow_span, adjust=False).mean()
    
    raw_crossover = fast_ewma - slow_ewma
    
    # Volatility Normalization: sigma_p = price * (annual_vol / sqrt(trading_days))
    # INSTITUTIONAL FIX (Attack 4): Use absolute price to prevent negative prices (e.g. Oil futures, spreads) 
    # from inverting the forecast sign.
    sigma_p = close.abs() * (annual_vol / np.sqrt(trading_days))
    
    # Defensive Programming: Prevent ZeroDivisionError in dead markets
    sigma_p = sigma_p.replace(0.0, np.nan)
    
    forecast = raw_crossover / sigma_p
    
    # Volatility Scaling scalar (maps average 1-stdev move to 10)
    SCALAR = 10.0 
    scaled_forecast = forecast * SCALAR
    
    return scaled_forecast.clip(lower=-20.0, upper=20.0).fillna(0.0)

def calculate_dynamic_fdm_blended_forecast(close: pd.DataFrame, trading_days: float = 256.0) -> pd.DataFrame:
    """
    Calculates the weighted blend of 6 EWMAC speeds and applies a dynamically 
    calculated Forecast Diversification Multiplier (FDM).
    Works across 2D DataFrames (T x Assets).
    """
    if isinstance(close, pd.Series):
        close = close.to_frame(name="Asset")
        
    speeds = [
        (2, 8),      # Fast (high cost)
        (4, 16),
        (8, 32),
        (16, 64),
        (32, 128),
        (64, 256)    # Slow (low cost, high robust)
    ]
    
    # Institutional Weighting: Penalize fast speeds that bleed alpha to spread
    weights = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.25])
    
    annual_vol = calculate_annualized_volatility(close, lookback=36, trading_days=trading_days)
    
    assets = close.columns
    final_blended = pd.DataFrame(index=close.index, columns=assets, dtype=float)
    
    for asset in assets:
        asset_close = close[[asset]]
        asset_vol = annual_vol[[asset]]
        
        forecast_df = pd.DataFrame(index=asset_close.index)
        for i, (fast, slow) in enumerate(speeds):
            col_name = f"EWMAC_{fast}_{slow}"
            forecast_df[col_name] = calc_ewmac_forecast(asset_close, fast, slow, asset_vol, trading_days)[asset]
            
        raw_combined = forecast_df.dot(weights)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rolling_corr = forecast_df.rolling(window=36, min_periods=10).corr()
            fdm = pd.Series(1.2, index=asset_close.index) # Fallback / Default
            
            try:
                if not rolling_corr.empty and len(rolling_corr.dropna()) > 0:
                    unstacked_corr = rolling_corr.unstack()
                    T = unstacked_corr.shape[0]
                    N = len(weights)
                    corr_tensor = unstacked_corr.values.reshape(T, N, N)
                    
                    wT_rho_w = np.sum((corr_tensor @ weights) * weights, axis=1)
                    wT_rho_w = np.clip(wT_rho_w, 0.01, 1.0)
                    calculated_fdm = 1.0 / np.sqrt(wT_rho_w)
                    calculated_fdm = np.clip(calculated_fdm, 1.0, 1.5)
                    
                    fdm = pd.Series(calculated_fdm, index=unstacked_corr.index).fillna(1.2)
            except Exception:
                pass
                
        final_blended[asset] = raw_combined * fdm
        
    return final_blended.clip(lower=-20.0, upper=20.0)

def calculate_dynamic_idm(target_positions: pd.DataFrame) -> pd.Series:
    """
    Calculates the Instrument Diversification Multiplier dynamically based on 
    the cross-sectional correlation of the portfolio's target positions.
    1 / sqrt(w^T * rho * w)
    """
    if target_positions.shape[1] == 1:
        return pd.Series(1.0, index=target_positions.index) # 1 asset = no IDM
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        # Calculate daily return correlation of the instruments. 
        # For simplicity and stability, we just use rolling correlation of positions.
        rolling_corr = target_positions.rolling(window=36, min_periods=10).corr()
        
        idm = pd.Series(1.2, index=target_positions.index)
        try:
            if not rolling_corr.empty and len(rolling_corr.dropna()) > 0:
                unstacked_corr = rolling_corr.unstack()
                T = unstacked_corr.shape[0]
                N = target_positions.shape[1]
                corr_tensor = unstacked_corr.values.reshape(T, N, N)
                
                # Assume equal capital weight per instrument for IDM calculation
                weights = np.ones(N) / N 
                wT_rho_w = np.sum((corr_tensor @ weights) * weights, axis=1)
                wT_rho_w = np.clip(wT_rho_w, 0.01, 1.0)
                
                calculated_idm = 1.0 / np.sqrt(wT_rho_w)
                calculated_idm = np.clip(calculated_idm, 1.0, 2.5) # Allow up to 2.5 leverage for highly diversified books
                
                idm = pd.Series(calculated_idm, index=unstacked_corr.index).fillna(1.2)
        except Exception:
            pass
            
    return idm

def apply_position_buffer(ideal_positions: pd.DataFrame, buffer_pct: float = 0.10) -> pd.DataFrame:
    """
    Applies a turnover buffer. If the new ideal position is within +/- 10% 
    of the currently held position, do not trade. This prevents micro-friction.
    """
    buffered_positions = ideal_positions.copy()
    
    # We must iterate row by row to carry forward the actual held position
    held_pos = np.zeros(ideal_positions.shape[1])
    
    values = ideal_positions.values
    buffered_values = np.zeros_like(values)
    
    for i in range(len(values)):
        ideal = values[i]
        
        # If held is 0, we take the ideal.
        # If absolute percentage difference is > buffer, we take the ideal.
        # Otherwise, we keep held.
        diff = np.abs(ideal - held_pos)
        
        # Avoid division by zero
        safe_held = np.where(held_pos == 0, 1e-9, held_pos)
        pct_diff = diff / np.abs(safe_held)
        
        # Condition to change position: (held is 0) OR (pct_diff > buffer_pct) OR (sign flipped)
        change_mask = (held_pos == 0) | (pct_diff > buffer_pct) | (np.sign(ideal) != np.sign(held_pos))
        
        held_pos = np.where(change_mask, ideal, held_pos)
        buffered_values[i] = held_pos
        
    return pd.DataFrame(buffered_values, index=ideal_positions.index, columns=ideal_positions.columns)

def master_sizer(combined_forecast: pd.DataFrame, close: pd.DataFrame, annual_vol: pd.DataFrame, 
                 capital: float, target_vol: float, fx: float = 1.0, use_buffer: bool = True) -> pd.DataFrame:
    """
    Translates the conviction forecast into a physical position size across multiple assets.
    """
    bet_scalar = combined_forecast / 10.0
    
    # Initially calculate positions without IDM to get weights
    share_cash_vol = annual_vol * close * fx
    share_cash_vol = share_cash_vol.replace(0.0, np.nan)
    
    base_risk_budget = capital * target_vol
    base_position_size = bet_scalar * (base_risk_budget / share_cash_vol)
    
    # Calculate Dynamic IDM across the portfolio
    idm_series = calculate_dynamic_idm(base_position_size)
    
    # Apply IDM
    # DataFrame multiply Series aligns on index (row-wise)
    ideal_position_size = base_position_size.multiply(idm_series, axis=0).fillna(0.0)
    
    # Apply 10% Turnover Buffer
    if use_buffer:
        final_positions = apply_position_buffer(ideal_position_size, buffer_pct=0.10)
    else:
        final_positions = ideal_position_size
        
    return final_positions

def run_math_derived_system(close_price: pd.DataFrame, capital: float = 100000.0, target_vol: float = 0.15, trading_days: float = 256.0):
    """
    End-to-End Execution of the Multi-Asset Video 4 Logic.
    """
    if isinstance(close_price, pd.Series):
        close_price = close_price.to_frame(name="Asset")
        
    annual_vol = calculate_annualized_volatility(close_price, trading_days=trading_days)
    blended_forecast = calculate_dynamic_fdm_blended_forecast(close_price, trading_days=trading_days)
    
    target_positions = master_sizer(
        combined_forecast=blended_forecast,
        close=close_price,
        annual_vol=annual_vol,
        capital=capital,
        target_vol=target_vol
    )
    
    # INSTITUTIONAL FIX: PREVENT LOOKAHEAD BIAS
    executable_target_positions = target_positions.shift(1).fillna(0.0)
    
    return executable_target_positions, blended_forecast
