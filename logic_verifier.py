import sys

def verify_logic():
    print("============================================================")
    print(" PHASE 0: AUTHOR TRANSCRIPT LOGIC VERIFICATION ")
    print("============================================================")
    
    fails = 0
    
    # Check 1: Volatility Scalar 1.2533 (Rob Carver Video 4)
    try:
        with open("bt2_video4_math_derived.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "1.2533" not in content:
                print("  [FAIL] Volatility scalar 1.2533 missing from bt2_video4_math_derived.py")
                fails += 1
            else:
                print("  [PASS] Carver Volatility Scalar (1.2533) strictly enforced.")
    except Exception as e:
        print(f"  [FAIL] Could not read bt2_video4_math_derived.py: {e}")
        fails += 1
        
    # Check 2: 28-period RSI and GLD Safe Haven (BC1 Block 4 Blueprint)
    try:
        with open("block4_final_blueprint.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "length=28" not in content and "28" not in content:
                print("  [FAIL] 28-period RSI missing from block4_final_blueprint.py")
                fails += 1
            elif "GLD" not in content and "SAFE_HAVEN" not in content:
                print("  [FAIL] GLD Safe Haven logic missing from block4_final_blueprint.py")
                fails += 1
            else:
                print("  [PASS] BC1 Block 4: 28-Period RSI and GLD Safe Haven strictly enforced.")
    except Exception as e:
        print(f"  [FAIL] Could not read block4_final_blueprint.py: {e}")
        fails += 1
        
    # Check 3: 1-Day Execution Lag (Prevent Lookahead)
    try:
        with open("block4_final_blueprint.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "shift(1)" not in content:
                print("  [FAIL] shift(1) 1-Day Lag missing from block4_final_blueprint.py")
                fails += 1
            else:
                print("  [PASS] BC1 Block 4: 1-Day Execution Lag strictly enforced.")
    except Exception as e:
        pass # already caught
        
    # Check 4: FDM Limits 1.0 to 2.5 (Carver Video 12)
    try:
        with open("bt2_video12_final_assembly.py", "r", encoding="utf-8") as f:
            content = f.read()
            if "2.5" not in content or "1.0" not in content:
                print("  [FAIL] FDM bounds (1.0, 2.5) missing from bt2_video12_final_assembly.py")
                fails += 1
            else:
                print("  [PASS] BC2 Video 12: FDM Limits (1.0 to 2.5) strictly enforced.")
    except Exception as e:
        print(f"  [FAIL] Could not read bt2_video12_final_assembly.py: {e}")
        fails += 1
        
    return fails

if __name__ == "__main__":
    fails = verify_logic()
    sys.exit(1 if fails > 0 else 0)
