import subprocess
import sys

def run_script(script_name):
    print(f"\n--- Running {script_name} ---")
    result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"!!! CRITICAL FAILURE IN {script_name} !!!")
        print(result.stdout)
        print(result.stderr)
        return False
    print(result.stdout)
    return True

def main():
    print("============================================================")
    print(" INITIATING 20x BURN-IN OMNI-GRINDER LOOP ")
    print("============================================================")
    
    scripts_to_run = [
        "logic_verifier.py",
        "omni_grinder_master_suite.py",
        "omni_grinder_math_suite.py"
    ]
    
    ITERATIONS = 20
    
    for i in range(1, ITERATIONS + 1):
        print(f"\n\n>>>>>>>>>> STARTING LOOP {i}/{ITERATIONS} <<<<<<<<<<")
        
        for script in scripts_to_run:
            success = run_script(script)
            if not success:
                print(f"\n[FATAL] The Burn-In loop crashed on Iteration {i} during {script}.")
                sys.exit(1)
                
        print(f">>>>>>>>>> LOOP {i}/{ITERATIONS} COMPLETED SUCCESSFULLY <<<<<<<<<<")
        
    print("\n\n============================================================")
    print(" 20x BURN-IN COMPLETE. ZERO BUGS. ABSOLUTE SURRENDER ACHIEVED.")
    print("============================================================")

if __name__ == "__main__":
    main()
