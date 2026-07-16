import pandas as pd
import numpy as np
import pandas_ta as ta
import logging

logger = logging.getLogger(__name__)

class RSICrossSectionalEngine:
    def __init__(self, rsi_lookback=90, rsi_threshold=50):
        self.rsi_lookback = rsi_lookback
        self.rsi_threshold = rsi_threshold

    def calculate_cumulative_rsi(self, prices: pd.DataFrame, regimes: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculates RSI on the cumulative percentage return series rather than raw price.
        This normalizes volatility across assets with different absolute prices.
        If regimes is provided, dynamically stitches 21-day and 90-day RSI based on VIX.
        """
        logger.info(f"Calculating Cumulative RSI (Adaptive Lookback: {'Dynamic' if regimes is not None else self.rsi_lookback})...")
        
        # 1. Calculate daily percentage returns (kept for reference but NOT used for RSI)
        returns = prices.pct_change()
        
        # 2. Calculate CUMULATIVE LOG-RETURNS (equity curve trend).
        # BRUTAL QA FIX: pct_change().cumsum() is MATHEMATICALLY WRONG for compounded returns.
        # Arithmetic returns are NOT additive. Log returns ARE additive.
        # log(P_t / P_0) = sum of log(P_t / P_{t-1}) — this is the correct formulation.
        # For BTC going from $1k to $60k, arithmetic cumsum would give nonsense.
        # log(price / price.iloc[0]) is the gold standard.
        # We compute it as the cumulative sum of log daily returns.
        log_returns = np.log(prices / prices.shift(1))  # log(P_t / P_{t-1})
        cum_returns = log_returns.cumsum()               # = log(P_t / P_0)
        
        # 3. Calculate RSI on the cumulative equity curve
        # BRUTAL QA FIX: Initialize with dtype=float to ensure consistent numeric dtype.
        # An untyped DataFrame init creates object dtype, which causes silent type errors.
        rsi_scores = pd.DataFrame(
            np.nan,
            index=cum_returns.index,
            columns=cum_returns.columns,
            dtype=float
        )
        
        for col in cum_returns.columns:
            # BRUTAL QA FIX: DO NOT fillna(0) for leading NaNs (e.g. IPOs). 
            # This causes 0.0 flatlines which artificially anchor RSI to 50.
            # Instead, drop NaNs, calculate RSI strictly on active trading days, and reindex.
            active_series = cum_returns[col].dropna()
            
            if len(active_series) > self.rsi_lookback:
                # Calculate both 90-day and 21-day if regime is provided
                if regimes is not None:
                    rsi_90 = ta.rsi(active_series, length=90)
                    rsi_21 = ta.rsi(active_series, length=21)
                    
                    if rsi_90 is not None and rsi_21 is not None:
                        # Reindex to match the main index
                        rsi_90 = rsi_90.reindex(cum_returns.index)
                        rsi_21 = rsi_21.reindex(cum_returns.index)
                        
                        # Stitch together based on regime
                        vix_is_21 = (regimes['rsi_lookback'] == 21)
                        combined_rsi = np.where(vix_is_21, rsi_21, rsi_90)
                        rsi_scores[col] = pd.Series(combined_rsi, index=cum_returns.index)
                else:
                    # Static lookback fallback
                    rsi = ta.rsi(active_series, length=self.rsi_lookback)
                    if rsi is not None:
                        rsi_scores[col] = rsi.reindex(cum_returns.index)
                
        return rsi_scores

    def apply_threshold(self, rsi_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the absolute threshold filter. If an asset's RSI is <= threshold, 
        it is disqualified (score becomes NaN).
        """
        # Threshold 50 Rule: Abandon traditional 70/30 oscillator thresholds
        logger.info(f"Applying RSI > {self.rsi_threshold} threshold rule...")
        
        # True if qualified, False otherwise
        qualified_mask = rsi_scores > self.rsi_threshold
        
        # Disqualify assets by setting score to NaN
        filtered_scores = rsi_scores.where(qualified_mask, np.nan)
        return filtered_scores

    def rank_assets(self, rsi_scores: pd.DataFrame) -> pd.DataFrame:
        """
        Ranks the assets cross-sectionally for each day.
        Higher RSI gets a higher rank (e.g., 1 is best).
        Returns a DataFrame of ranks.
        """
        logger.info("Ranking assets cross-sectionally...")
        # rank(ascending=False) means highest score gets rank 1
        ranks = rsi_scores.rank(axis=1, ascending=False, method='min')
        return ranks

    def run(self, data_path: str, regime_path: str = 'macro_regime.parquet', output_path: str = 'rsi_scores.parquet'):
        import sys
        from pathlib import Path
        
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        
        regimes = None
        regime_file = Path(regime_path)
        if regime_file.exists():
            logger.info(f"Loading regime signals from {regime_path}")
            regimes = pd.read_parquet(regime_file)
        
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.levels[0]:
                prices = df['Adj Close']
            else:
                prices = df.xs('Adj Close', axis=1, level=0, drop_level=True) if 'Adj Close' in df.columns else df
        else:
            prices = df
        
        # BRUTAL QA: Validate minimum date range for 12-month lookback to be meaningful
        n_rows = len(prices)
        assert n_rows >= 252, (
            f"CRITICAL: Only {n_rows} rows of price data. "
            f"Need at least 252 trading days (1 year) for RSI to be meaningful."
        )
        logger.info(f"  Date range: {prices.index[0].date()} -> {prices.index[-1].date()} ({n_rows} rows)")
        
        rsi_scores = self.calculate_cumulative_rsi(prices, regimes)
        filtered_scores = self.apply_threshold(rsi_scores)
        ranks = self.rank_assets(filtered_scores)
        
        # Persist RSI scores to parquet for downstream pipeline consumers
        rsi_scores.to_parquet(output_path)
        logger.info(f"  RSI scores saved to '{output_path}'")
        
        return rsi_scores, filtered_scores, ranks

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    engine = RSICrossSectionalEngine()
    try:
        raw_rsi, filtered_rsi, ranks = engine.run('universe_data.parquet')
        print("\nTop 5 assets on the last available day:")
        last_day_ranks = ranks.iloc[-1].dropna().sort_values()
        print(last_day_ranks.head(5))
    except Exception as e:
        logger.error(f"Execution failed: {e}")
