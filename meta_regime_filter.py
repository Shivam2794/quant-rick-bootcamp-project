import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class MetaRegimeFilter:
    def __init__(self, data_path='macro_data.parquet'):
        self.data_path = Path(__file__).parent / data_path
        
        # Canary pairs (Risk-On / Defensive)
        self.canary_pairs = [
            ('SPY', 'XLU'),  # Broad Equity vs Utilities
            ('XLY', 'XLP'),  # Discretionary vs Staples
            ('XLB', 'GLD'),  # Materials vs Gold
            ('HYG', 'IEF'),  # High Yield Credit vs Treasuries
            ('XLK', 'XLV'),  # Tech vs Healthcare
            ('XLF', 'XLU')   # Financials vs Utilities
        ]
        
        self.z_score_lookback = 252
        self.vix_threshold = 30.0

    def calculate_regimes(self) -> pd.DataFrame:
        logger.info(f"Loading macro data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        if isinstance(df.columns, pd.MultiIndex):
            if 'Adj Close' in df.columns.levels[0]:
                prices = df['Adj Close']
            else:
                prices = df.xs('Adj Close', axis=1, level=0, drop_level=True) if 'Adj Close' in df.columns else df
        else:
            prices = df
        
        # 1. VIX Regime
        vix = prices['^VIX']
        # Forward fill VIX specifically in case of missing dates compared to equities
        vix = vix.ffill()
        
        # If VIX > 30, use 21-day lookback, else 90-day
        vix_lookback = pd.Series(90, index=prices.index)
        vix_lookback.loc[vix > self.vix_threshold] = 21
        logger.info(f"VIX > {self.vix_threshold} occurred on {(vix > self.vix_threshold).sum()} days.")
        
        # 2. Canary Regime
        composite_z_scores = pd.Series(0.0, index=prices.index)
        valid_pairs_count = 0
        
        for risk_on, defensive in self.canary_pairs:
            if risk_on in prices.columns and defensive in prices.columns:
                ratio = prices[risk_on] / prices[defensive]
                
                # Calculate 252-day rolling Z-Score of the ratio
                rolling_mean = ratio.rolling(window=self.z_score_lookback).mean()
                rolling_std = ratio.rolling(window=self.z_score_lookback).std()
                
                # Z-Score = (Current - Mean) / Std
                z_score = (ratio - rolling_mean) / rolling_std
                
                # Fill initial NaNs with 0 (neutral) so it doesn't break the sum
                z_score = z_score.fillna(0)
                
                composite_z_scores += z_score
                valid_pairs_count += 1
            else:
                logger.warning(f"Canary pair {risk_on}/{defensive} missing from data!")
                
        if valid_pairs_count == 0:
            raise ValueError("No Canary pairs were available in the data.")
            
        composite_z_scores /= valid_pairs_count
        
        # Regime definition: 1 for RISK_ON (Composite > 0), 0 for DEFENSIVE (Composite <= 0)
        # Note: We enforce a Risk-On default during the 252-day warmup period.
        regime_is_risk_on = pd.Series(1, index=prices.index)
        # Apply defensive signal only where composite is <= 0 AND we have passed the warmup
        regime_is_risk_on.loc[(composite_z_scores <= 0) & (np.arange(len(prices)) >= self.z_score_lookback)] = 0
        
        logger.info(f"DEFENSIVE Canary regime triggered on {(regime_is_risk_on == 0).sum()} days.")
        
        # Combine regimes into a single DataFrame
        regimes = pd.DataFrame({
            'rsi_lookback': vix_lookback,
            'composite_z_score': composite_z_scores,
            'is_risk_on': regime_is_risk_on
        }, index=prices.index)
        
        return regimes

    def run(self, output_path='macro_regime.parquet'):
        regimes = self.calculate_regimes()
        out_file = Path(__file__).parent / output_path
        regimes.to_parquet(out_file)
        logger.info(f"Saved regime signals to {out_file}")
        
        return regimes

if __name__ == '__main__':
    engine = MetaRegimeFilter()
    regimes = engine.run()
    print("\nRecent Regime Status:")
    print(regimes.tail(10))
