"""
master_pipeline_v1_to_v13.py
=========================================================
PURPOSE: The Master Integration Pipeline that unifies the entire
Carver + Cross-Sectional stack (Videos 1-13).

This pipeline takes a raw price matrix and processes it through:
- V4 Instrument Risk (Volatility)
- V1 EWMAC Rule Generation
- V12 Forecast Diversification Multiplier (FDM)
- V12 Volatility Attenuation
- V13 Cross-Sectional Ranking & Z-Scoring
- V12 Position Sizing & Turnover Buffers
- Lookahead Prevention (Shift)
"""

import sys
import numpy as np
import pandas as pd
import traceback

import bt2_video12_final_assembly as v12
import bt2_video13_cross_sectional as v13

def run_master_carver_pipeline(
    close: pd.DataFrame,
    capital: float = 100_000.0,
    target_annual_vol: float = 0.20,
    trading_days: float = 256.0,
    blend_weight_cs: float = 0.5,
    use_buffer: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the full Carver + CS pipeline.
    """
    if isinstance(close, pd.Series):
        close = close.to_frame(name="Asset")

    close = close.sort_index()

    # --- L1: Instrument Risk (V4/12) ---
    sigma_price = v12.calculate_instrument_risk(close, lookback=v12.VOL_LOOKBACK)

    # --- L2/L3: Rule Forecast Stack (V1/12) ---
    forecasts = v12.build_rule_forecast_stack(close, sigma_price, v12.EWMAC_SPECS)

    # --- L5: Weights ---
    rule_weights = {f"EWMAC{f}_{s}": w for (f, s, _), w in
                    zip(v12.EWMAC_SPECS, [1.0 / len(v12.EWMAC_SPECS)] * len(v12.EWMAC_SPECS))}
    rule_weights = v12.normalise_weights(rule_weights)

    # --- L6: FDM (Time-Series) ---
    fdm_df = v12.calculate_rolling_fdm(forecasts, rule_weights, lookback=v12.CORR_LOOKBACK)

    # --- L7: Combined Time-Series Forecast ---
    combined = v12.combine_forecasts_with_fdm(forecasts, rule_weights, fdm_df)

    # --- L8: Volatility Attenuation ---
    attenuation = v12.compute_volatility_attenuation(sigma_price)
    combined = combined * attenuation

    # --- L9: FTI Overlay (Stubbed) ---
    fti = v12.fti_multiplier_stub(close)
    combined = combined * fti

    # Cap Time-Series Forecast
    combined = combined.clip(lower=-v12.FORECAST_CAP, upper=v12.FORECAST_CAP)

    # --- L10: Cross-Sectional Layer (V13) ---
    # Fuses the time-series forecast with cross-sectional relative momentum.
    adjusted_forecasts = v13.full_carver_cross_sectional(
        combined_forecasts=combined,
        blend_weight_cs=blend_weight_cs,
        min_assets_for_cs=2
    )

    # --- L11: Position Sizing (Volatility Targeting) ---
    raw_positions = v12.size_positions(
        combined_forecast=adjusted_forecasts,
        close=close,
        sigma_price=sigma_price,
        capital=capital,
        target_annual_vol=target_annual_vol,
        trading_days=trading_days
    )

    # --- L12: Turnover Buffer ---
    if use_buffer:
        raw_positions = v12.apply_position_buffer(raw_positions)

    # --- L13: Lookahead Prevention ---
    executable_positions = raw_positions.shift(1).fillna(0.0)

    return executable_positions, adjusted_forecasts, sigma_price


def brutal_master_integration_inspector():
    print("="*70)
    print("  BRUTAL MASTER INTEGRATION PIPELINE INSPECTOR (V1 - V13)")
    print("="*70)
    
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=1000, freq="B")
    
    # Generate Synthetic Universe
    # Asset A: Strong Uptrend
    asset_a = np.exp(np.random.randn(1000) * 0.02 + 0.002).cumprod() * 100
    # Asset B: Strong Downtrend
    asset_b = np.exp(np.random.randn(1000) * 0.02 - 0.002).cumprod() * 100
    # Asset C: Choppy/Flat
    asset_c = 100 + 10 * np.sin(np.arange(1000) / 10.0) + np.random.randn(1000) * 2
    # Asset D: Missing Data (Trading Halt in the middle)
    asset_d = np.exp(np.random.randn(1000) * 0.02 + 0.001).cumprod() * 100
    asset_d[400:450] = np.nan
    # Asset E: Penny Stock crash
    asset_e = np.exp(np.random.randn(1000) * 0.03 - 0.005).cumprod() * 10
    asset_e[800:] = 0.0 # goes to zero
    
    close_df = pd.DataFrame({
        "A_UP": asset_a,
        "B_DOWN": asset_b,
        "C_CHOP": asset_c,
        "D_HALT": asset_d,
        "E_ZERO": asset_e
    }, index=dates)

    print("\n[PHASE 1] Executing Master Pipeline on 5-Asset Universe...")
    try:
        positions, forecasts, vol = run_master_carver_pipeline(close_df)
        print("  [PASS] Pipeline executed without crashing.")
    except Exception as e:
        print(f"  [FAIL] Pipeline crashed: {e}")
        traceback.print_exc()
        return False
        
    print("\n[PHASE 2] Validating Cross-Sectional Integration...")
    try:
        # A_UP should have positive forecasts, B_DOWN negative
        # CS layer should boost A_UP and penalize B_DOWN relatively.
        last_f = forecasts.iloc[-1]
        
        if last_f["A_UP"] > 0 and last_f["B_DOWN"] < 0:
            print(f"  [PASS] Directional Integrity Maintained (A: {last_f['A_UP']:.2f}, B: {last_f['B_DOWN']:.2f})")
        else:
            print(f"  [FAIL] Directional Integrity Failed (A: {last_f['A_UP']:.2f}, B: {last_f['B_DOWN']:.2f})")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Phase 2 crashed: {e}")
        return False

    print("\n[PHASE 3] Validating Safety Bounds & Halt Resistance...")
    try:
        # Halt asset should have 0 position during the halt (accounting for 1-day shift)
        halt_positions = positions["D_HALT"].iloc[401:450]
        if (halt_positions == 0.0).all():
            print("  [PASS] Trading Halt successfully forced positions to 0.")
        else:
            print(f"  [FAIL] Positions were taken during a trading halt! Unique values: {halt_positions.unique()}")
            return False
            
        # Zero asset should not cause inf
        if np.isfinite(positions["E_ZERO"].iloc[850]):
            print("  [PASS] Zero-price asset (bankruptcy) handled gracefully without inf/NaN propagation.")
        else:
            print("  [FAIL] Zero-price asset generated inf/NaN positions.")
            return False
            
        # Global Lookahead Check
        if (positions.iloc[0] == 0.0).all():
            print("  [PASS] Day 1 Lookahead Prevention active (0.0 positions).")
        else:
            print("  [FAIL] Lookahead detected on Day 1.")
            return False
            
    except Exception as e:
        print(f"  [FAIL] Phase 3 crashed: {e}")
        return False

    print("\n--- MASTER PIPELINE: ZERO BUGS DETECTED ---")
    return True

if __name__ == "__main__":
    success = brutal_master_integration_inspector()
    sys.exit(0 if success else 1)
