import logging
import sys
import time
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Omni-Engine-Orchestrator")

def run_step(script_name: str, step_name: str):
    logger.info(f"\n--- {step_name} ---")
    logger.info(f"Executing {script_name}...")
    
    script_path = Path(__file__).parent / script_name
    python_exe = sys.executable
    
    result = subprocess.run([python_exe, "-u", str(script_path)], capture_output=True, text=True)
    
    # Print the output sequentially
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
        
    if result.returncode != 0:
        logger.error(f"CRITICAL: {script_name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def run_pipeline():
    logger.info("=" * 70)
    logger.info("INITIATING FTMO OMNI-ENGINE PIPELINE EXECUTION")
    logger.info("=" * 70)
    
    start_time = time.time()

    run_step("data_ingestion.py", "STEP 1: DATA INGESTION")
    run_step("meta_regime_filter.py", "STEP 2: META-REGIME FILTER (BETA CANARY)")
    run_step("rsi_cross_sectional_engine.py", "STEP 3: RSI CROSS-SECTIONAL ENGINE")
    run_step("momentum_prefilter.py", "STEP 4: MOMENTUM PREFILTER (3-EMA + MACD)")
    run_step("raam_scorer.py", "STEP 5: RAAM 4-FACTOR SCORER")
    run_step("position_sizer.py", "STEP 6: VOLATILITY-TARGETED POSITION SIZER")
    run_step("backtest_engine.py", "STEP 7: BACKTEST ENGINE")
    run_step("advanced_tearsheet.py", "STEP 8: ADVANCED TEARSHEET BENCHMARKING")
    
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"FTMO OMNI-ENGINE EXECUTION COMPLETE IN {elapsed:.2f} SECONDS.")
    logger.info("=" * 70)

if __name__ == "__main__":
    run_pipeline()
