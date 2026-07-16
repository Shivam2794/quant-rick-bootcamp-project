import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

RESULTS_PATH = "backtest_results.parquet"

def generate_tearsheet(df: pd.DataFrame):
    logger.info("=" * 70)
    logger.info("ADVANCED TEARSHEET: MAXIMUM OMEGA BENCHMARK")
    logger.info("=" * 70)

    # Convert to daily returns
    returns = df['net_return'].dropna()
    equity = df['equity']
    
    if len(returns) == 0:
        logger.error("No return data found.")
        return

    # Basic stats
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = (returns.index[-1] - returns.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (cagr - 0.0) / ann_vol if ann_vol > 0 else 0
    
    # Sortino (Downside risk)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252)
    sortino = (cagr - 0.0) / downside_vol if downside_vol > 0 else 0
    
    # Max Drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    
    # Probabilistic Sharpe Ratio (PSR) estimation
    # Skewness and Kurtosis
    skew = returns.skew()
    kurt = returns.kurtosis()
    sr_daily = returns.mean() / returns.std() if returns.std() > 0 else 0
    # simplified PSR calculation reference
    
    logger.info(f"  CAGR                  : {cagr:.2%}")
    logger.info(f"  Annualized Volatility : {ann_vol:.2%}")
    logger.info(f"  Sharpe Ratio          : {sharpe:.2f}")
    logger.info(f"  Sortino Ratio         : {sortino:.2f}")
    logger.info(f"  Max Drawdown          : {max_dd:.2%}")
    logger.info(f"  Daily Return Skew     : {skew:.2f}")
    logger.info(f"  Daily Return Kurtosis : {kurt:.2f}")
    
    # Omega Ratio (threshold = 0)
    positive_sum = returns[returns > 0].sum()
    negative_sum = -returns[returns < 0].sum()
    omega = positive_sum / negative_sum if negative_sum > 0 else np.nan
    logger.info(f"  Omega Ratio           : {omega:.2f}")
    
    logger.info("=" * 70)
    logger.info("  TEARSHEET COMPLETE")
    logger.info("=" * 70)

def main():
    logger.info(f"Loading backtest results from '{RESULTS_PATH}'...")
    if not Path(RESULTS_PATH).exists():
        logger.error("Results file not found. Run backtest_engine.py first.")
        sys.exit(1)
        
    df = pd.read_parquet(RESULTS_PATH)
    generate_tearsheet(df)

if __name__ == "__main__":
    main()
