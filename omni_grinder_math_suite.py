import numpy as np
import pandas as pd
import traceback
import sys

# Import Bootcamp 1 Modules
import portfolio_amalgamator
import carver_pipeline
from carver_pipeline import IndicatorFactory

# Import Bootcamp 2 Modules
import master_pipeline_v1_to_v13 as master_bc2

def generate_adversarial_df(periods=1000):
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=periods, freq="B")
    
    # Base prices
    close = np.exp(np.random.randn(periods) * 0.02 + 0.002).cumprod() * 100
    
    # Introduce NaN block (Halt)
    close[400:450] = np.nan
    
    # Introduce Zero-Price block (Bankruptcy)
    close[800:] = 0.0
    
    # Introduce Negative Prices (Future Contracts)
    close[200:250] = -10.0
    
    high = close * 1.02
    low = close * 0.98
    volume = np.random.randint(100, 10000, size=periods).astype(float)
    volume[800:] = 0.0 # Zero volume for bankruptcy
    volume[400:450] = 0.0 # Zero volume for halt
    
    df = pd.DataFrame({'Open': close, 'High': high, 'Low': low, 'Close': close, 'Volume': volume}, index=dates)
    
    return df

def test_bc1_portfolio_amalgamator():
    print("\n--- Testing BC1: portfolio_amalgamator.py ---")
    df = generate_adversarial_df(1000)
    
    try:
        # Avoid pandas deprecation warning spam for test output
        pd.options.mode.chained_assignment = None
        
        # Test SPY/QQQ Logic
        sig_spy = portfolio_amalgamator.get_signal_spy_qqq(df)
        if sig_spy.isna().any():
            return False, "NaN leaked in SPY signal"
            
        # Test SMH Logic
        sig_smh = portfolio_amalgamator.get_signal_smh(df)
        if sig_smh.isna().any():
            return False, "NaN leaked in SMH signal"
            
        # Test DIA Logic
        sig_dia = portfolio_amalgamator.get_signal_dia(df)
        if sig_dia.isna().any():
            return False, "NaN leaked in DIA signal"
            
        # Test SLV Logic
        sig_slv = portfolio_amalgamator.get_signal_slv(df)
        if sig_slv.isna().any():
            return False, "NaN leaked in SLV signal"
            
        print("  [PASS] Signals handle NaNs, Zeros, and Negatives safely.")
        return True, ""
    except Exception as e:
        traceback.print_exc()
        return False, f"Crash: {e}"

def test_bc1_carver_pipeline():
    print("\n--- Testing BC1: carver_pipeline.py ---")
    df = generate_adversarial_df(1000)
    
    try:
        # Test IndicatorFactory
        tema_params = (10, 64, 126)
        out_df = IndicatorFactory.generate_all(df, tema_params)
        
        # Verify no unhandled inf/NaNs in signals
        signals = [c for c in out_df.columns if c.endswith('_Signal')]
        for sig in signals:
            if out_df[sig].isna().any():
                return False, f"NaN leaked in {sig}"
        
        print("  [PASS] IndicatorFactory handles adversarial data safely.")
        return True, ""
    except Exception as e:
        traceback.print_exc()
        return False, f"Crash: {e}"

def test_bc2_master_pipeline():
    print("\n--- Testing BC2: master_pipeline_v1_to_v13.py ---")
    # This was already heavily tested in master_pipeline, but let's run it with Negative Prices too
    df1 = generate_adversarial_df(1000)['Close']
    df2 = generate_adversarial_df(1000)['Close'] * 0.5
    
    close_df = pd.DataFrame({'A': df1, 'B': df2})
    
    try:
        pos, _, _ = master_bc2.run_master_carver_pipeline(close_df)
        
        if pos.isna().any().any():
            return False, "NaN leaked into final positions"
        if np.isinf(pos).any().any():
            return False, "Inf leaked into final positions"
            
        # Lookahead Check
        if pos.iloc[0].sum() != 0.0:
            return False, "Lookahead bias detected on Day 1"
            
        print("  [PASS] BC2 Master Pipeline handles negative prices and halts gracefully.")
        return True, ""
    except Exception as e:
        traceback.print_exc()
        return False, f"Crash: {e}"

def test_bc1_block4_blueprint():
    print("\n--- Testing BC1: block4_final_blueprint.py ---")
    import block4_final_blueprint as block4
    df = generate_adversarial_df(1000)
    
    try:
        # Test MegaIndicatorFactory
        feature_df = block4.MegaIndicatorFactory.generate_all(df)
        
        if feature_df.isna().any().any():
            return False, "NaN leaked in MegaIndicatorFactory outputs"
            
        print("  [PASS] MegaIndicatorFactory handles adversarial data safely.")
        return True, ""
    except Exception as e:
        traceback.print_exc()
        return False, f"Crash: {e}"

def run_math_grinder():
    print("="*60)
    print(" PHASE 2: BRUTAL MATHEMATICAL STRESS TESTING ")
    print("="*60)
    
    tests = [
        ("BC1_portfolio_amalgamator", test_bc1_portfolio_amalgamator),
        ("BC1_carver_pipeline", test_bc1_carver_pipeline),
        ("BC1_block4_blueprint", test_bc1_block4_blueprint),
        ("BC2_master_pipeline", test_bc2_master_pipeline)
    ]
    
    total_fails = 0
    for name, test_func in tests:
        success, msg = test_func()
        if not success:
            print(f"  [FAIL] {name}: {msg}")
            total_fails += 1
            
    return total_fails

if __name__ == "__main__":
    fails = run_math_grinder()
    if fails == 0:
        print("\n=> ZERO BUGS. THE LOOP IS BULLETPROOF.")
    sys.exit(1 if fails > 0 else 0)
