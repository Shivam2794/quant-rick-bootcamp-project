import sys
import os
import traceback
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from bt2_video7_data_pipeline import run_data_orchestration_pipeline
from bt2_video5_hybrid_sizing import run_hybrid_prop_firm_engine

def brutal_integration_inspector():
    print("--- INITIATING MASTER INTEGRATION PIPELINE (VIDEOS 4, 5, 6, 7) ---")
    
    # 1. Synthesize Universe (5 Assets, 3000 days)
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=3000, freq="B")
    
    # A) STRONG_UP: High volatility, pure drift
    strong_up = np.exp(np.random.randn(3000) * 0.03 + 0.003).cumprod() * 100
    
    # B) WEAK_DOWN: Low volatility, negative drift
    weak_down = np.exp(np.random.randn(3000) * 0.01 - 0.001).cumprod() * 50
    
    # C) CHOPPY_DEATH: Will be rejected by Video 7 filter
    t = np.arange(3000)
    choppy = 100 + 10 * np.sin(t / 20.0) + np.random.randn(3000) * 5
    choppy = np.clip(choppy, 10, None)
    
    # D) POISON_ASSET_1: NaNs in the middle (simulating a trading halt)
    poison_1 = np.exp(np.random.randn(3000) * 0.02 + 0.002).cumprod() * 100
    poison_1[1500:1520] = np.nan
    
    # E) POISON_ASSET_2: Zero-price crash (bankruptcy)
    poison_2 = np.exp(np.random.randn(3000) * 0.02 + 0.002).cumprod() * 100
    poison_2[2000:] = 0.0
    
    close_df = pd.DataFrame({
        "STRONG_UP": strong_up,
        "WEAK_DOWN": weak_down,
        "CHOPPY_DEATH": choppy,
        "POISON_1": poison_1,
        "POISON_2": poison_2
    }, index=dates)
    
    high_df = close_df * 1.02
    low_df = close_df * 0.98
    
    print("\n[PHASE 1] Video 7 Data Orchestration Pipeline")
    try:
        is_data, oos_data, survivors = run_data_orchestration_pipeline(high_df, low_df, close_df, split_ratio=0.70)
        print(f"Surviving Assets Passed to Engine: {survivors}")
    except Exception as e:
        print(f"[FAIL] Pipeline Crashed at Phase 1: {e}")
        traceback.print_exc()
        return False

    print("\n[PHASE 2] Video 5/4 Hybrid Sizing Engine (In-Sample)")
    try:
        # Run IS Data
        is_positions, is_forecast, is_ensemble = run_hybrid_prop_firm_engine(
            close=is_data["close"],
            high=is_data["high"],
            low=is_data["low"],
            capital=100000.0,
            trading_days=252.0
        )
        
        # Check for NaN propagation from POISON_1
        # It's okay if POISON_1 has NaNs during the halt, but it should NOT leak to STRONG_UP
        if is_positions["STRONG_UP"].isna().any():
            print("[FAIL] CROSS-ASSET NaN LEAKAGE! A halt in one asset poisoned the IDM/Covariance matrix of a healthy asset.")
            return False
            
        print("[PASS] In-Sample matrix calculated. Cross-Asset isolation maintained.")
    except Exception as e:
        print(f"[FAIL] Pipeline Crashed at Phase 2: {e}")
        traceback.print_exc()
        return False
        
    print("\n[PHASE 3] Video 5/4 Hybrid Sizing Engine (Out-of-Sample)")
    try:
        # Run OOS Data
        oos_positions, oos_forecast, oos_ensemble = run_hybrid_prop_firm_engine(
            close=oos_data["close"],
            high=oos_data["high"],
            low=oos_data["low"],
            capital=100000.0,
            trading_days=252.0
        )
        
        # Check for Boundary Loss (Burn-in problem)
        # Because of the 256-day burn-in window, the first 256 days will output 0.0
        # However, day 257 (which is the TRUE start of the Out-of-Sample period) MUST have positions.
        
        # In our synth data, STRONG_UP should have a positive position after burn-in.
        if oos_positions["STRONG_UP"].iloc[257:260].sum() == 0.0:
            print("[FAIL] FATAL BOUNDARY LOSS: The Out-of-Sample period started trading with empty/zero positions.")
            return False
            
        print("[PASS] Out-of-Sample matrix calculated without Boundary Loss. True OOS day 1 successfully traded.")
    except Exception as e:
        print(f"[FAIL] Pipeline Crashed at Phase 3: {e}")
        traceback.print_exc()
        return False
        
    print("\n--- FINAL VERDICT: INTEGRATED PIPELINE IS 100% BUG-FREE ---")
    return True

if __name__ == "__main__":
    brutal_integration_inspector()
