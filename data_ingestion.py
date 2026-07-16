import yfinance as yf
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import requests
from urllib3.util import Retry
from requests.adapters import HTTPAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FTMOUniverseIngestion:
    def __init__(self, output_file=None):
        if output_file is None:
            self.output_file = Path(__file__).parent / "universe_data.parquet"
        else:
            self.output_file = Path(output_file)
            
        # FTMO Universe Mapping to Yahoo Finance Tickers
        self.us_equities = [
            "MSTR", "NVDA", "AVGO", "PLTR", "GOOG", "GE", "JPM", "AAPL", "RACE", 
            "BRK-B", "META", "XOM", "IBM", "WMT", "V", "CVX", "RTX", "MSFT", 
            "NFLX", "MCD", "KO", "TSLA", "QCOM", "LMT", "AMD", "CSCO", "BAC", 
            "AMZN", "AZN", "ASML", "JNJ", "T", "ARM", "PFE", "SBUX", "FDX", 
            "BA", "NKE", "SNOW", "DIS", "BABA", "INTC", "GME", "ZM"
        ]
        
        self.eu_equities = [
            "SAN.MC", "TTE.PA", "DBK.DE", "ALV.DE", "SIE.DE", "IBE.MC", 
            "AF.PA", "ADS.DE", "MC.PA", "BMW.DE", "MBG.DE", "VOW3.DE", "BAYN.DE"
        ]
        
        self.indices = [
            "NQ=F",     # US100
            "ES=F",     # US500
            "YM=F",     # US30
            "^FTSE",    # UK100
            "^GDAXI"    # GER40
        ]
        
        self.crypto = [
            "BTC-USD", 
            "ETH-USD"
        ]
        
        self.metals = [
            "GC=F",     # XAUUSD (Gold)
            "SI=F"      # XAGUSD (Silver)
        ]
        
        self.forex = [
            "EURUSD=X", 
            "GBPUSD=X", 
            "JPY=X", 
            "CHF=X", 
            "CAD=X"
        ]

        self.bonds = [
            "TLT",      # 20+ Year Treasuries
            "IEF",      # 7-10 Year Treasuries
            "SHY"       # 1-3 Year Treasuries
        ]
        
        self.currencies = [
            "UUP"       # Invesco DB US Dollar Index
        ]

        self.tradable_tickers = (
            self.us_equities + self.eu_equities + self.indices + 
            self.crypto + self.metals + self.forex + self.bonds + self.currencies
        )

        # Macro Indicators for Regime Canary & VIX-Adaptive Lookback
        self.macro_tickers = [
            "^VIX", "SPY", "XLU", "XLY", "XLP", "XLB", "GLD", "HYG", "IEF", "XLK", "XLV", "XLF"
        ]

        self.all_tickers = self.tradable_tickers + self.macro_tickers

    def download_data(self, start_date="2015-01-01", end_date=None):
        logger.info(f"Initiating download for {len(self.all_tickers)} assets...")
        
        # Calculate a 10-day buffer start date before start_date
        requested_start_datetime = pd.to_datetime(start_date)
        if requested_start_datetime.tzinfo is not None:
            requested_start_datetime = requested_start_datetime.tz_localize(None)
            
        buffer_start_datetime = requested_start_datetime - pd.Timedelta(days=10)
        buffer_start_str = buffer_start_datetime.strftime("%Y-%m-%d")
        
        # Setup a retrying requests Session
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        })
        retry = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        df_raw = yf.download(
            self.all_tickers,
            start=buffer_start_str,
            end=end_date,
            auto_adjust=False,
            session=session
        )
        
        if df_raw.empty:
            raise ValueError("Downloaded DataFrame is empty.")
            
        # Convert index to datetime and localize it to timezone-naive
        df_raw.index = pd.to_datetime(df_raw.index).tz_localize(None)
        
        # Extract 'Adj Close' and 'Volume' data
        prices = pd.DataFrame(index=df_raw.index)
        volumes = pd.DataFrame(index=df_raw.index)
        
        for ticker in self.all_tickers:
            if ('Adj Close', ticker) in df_raw.columns:
                prices[ticker] = df_raw[('Adj Close', ticker)]
                volumes[ticker] = df_raw[('Volume', ticker)]
            else:
                logger.warning(f"Ticker {ticker} not found in yf download columns.")
                
        return prices, volumes, requested_start_datetime

    def clean_and_validate(self, prices, volumes, requested_start):
        logger.info("Applying Data Cleaning Rules...")
        
        # 1. Replace np.inf with np.nan on prices and volumes
        prices = prices.replace([np.inf, -np.inf], np.nan)
        volumes = volumes.replace([np.inf, -np.inf], np.nan)
        
        # 2. Clip negative prices to 0.0001
        prices = prices.clip(lower=0.0001)
        
        # 3. Nullify prices on zero volume days for non-forex and non-indices assets
        # BRUTAL QA FIX: Guard against column mismatch between prices and volumes.
        # Only process columns that exist in BOTH DataFrames.
        common_cols = prices.columns.intersection(volumes.columns)
        for col in common_cols:
            if col not in self.forex and col not in self.indices and col not in self.macro_tickers:
                is_zero_vol = (volumes[col] == 0) | (volumes[col].isna())
                prices.loc[is_zero_vol, col] = np.nan
        
        # 4. Implement the 30% missing data check BEFORE forward-filling
        retained_assets = []
        for col in prices.columns:
            first_valid_idx = prices[col].first_valid_index()
            if first_valid_idx is None:
                logger.warning(f"Asset {col} has no valid price data. Dropping.")
                continue
            
            active_prices = prices.loc[first_valid_idx:, col]
            
            if col in self.crypto:
                # For crypto assets, calculate missingness over active period using all days
                missing_ratio = active_prices.isna().mean()
            else:
                # For non-crypto assets, calculate missingness over active period using weekdays only
                weekday_mask = active_prices.index.dayofweek < 5
                active_prices_weekdays = active_prices[weekday_mask]
                if len(active_prices_weekdays) == 0:
                    missing_ratio = 1.0
                else:
                    missing_ratio = active_prices_weekdays.isna().mean()
            
            if missing_ratio <= 0.30:
                retained_assets.append(col)
            else:
                logger.warning(f"Asset {col} has {missing_ratio:.2%} missing data in active period. Dropping.")
                
        prices = prices[retained_assets]
        volumes = volumes[retained_assets]
        
        # 5. Forward-fill prices (max 5 days) and fill volumes with 0
        prices = prices.ffill(limit=5)
        volumes = volumes.fillna(0)
        
        # 6. Slice both prices and volumes DataFrames to start from requested_start
        clean_prices = prices.loc[requested_start:]
        clean_volumes = volumes.loc[requested_start:]
        
        logger.info(f"Final Data Shape: {clean_prices.shape}. Retained {len(clean_prices.columns)} assets.")
        return clean_prices, clean_volumes

    def run(self):
        prices, volumes, requested_start = self.download_data()
        clean_prices, clean_volumes = self.clean_and_validate(prices, volumes, requested_start)
        
        # Combine into a single MultiIndex DataFrame for parquet
        combined = pd.concat({'Adj Close': clean_prices, 'Volume': clean_volumes}, axis=1)
        
        # Run programmatic assertions
        logger.info("Running programmatic assertions...")
        
        # 1. Check exactly 0 infinite values
        assert not np.isinf(combined).any().any(), "Assertion failed: combined DataFrame contains infinite values!"
        
        # 2. Check exactly 0 negative price values (<= 0.0001 is clipped)
        # Note: combined['Adj Close'] might contain NaNs (pre-IPO), so check all non-NaNs are >= 0.0001
        assert not (combined['Adj Close'] < 0.0001).any().any(), "Assertion failed: combined DataFrame contains negative or zero-clipped prices!"
        
        # 3. Check at least 50 valid assets survived
        num_assets = len(combined['Adj Close'].columns)
        assert num_assets >= 50, f"Assertion failed: only {num_assets} assets survived, expected at least 50!"
        
        # 4. BRUTAL QA: Validate minimum date range for downstream 12-month lookback
        n_trading_rows = combined['Adj Close'].shape[0]
        assert n_trading_rows >= 252, (
            f"Assertion failed: only {n_trading_rows} rows in final data. "
            f"Need at least 252 trading days for 12-month momentum lookbacks to function."
        )
        
        logger.info("All assertions passed successfully.")
        
        # Split into tradable universe and macro indicators
        valid_tradable = [col for col in combined['Adj Close'].columns if col in self.tradable_tickers]
        valid_macro = [col for col in combined['Adj Close'].columns if col in self.macro_tickers]
        
        tradable_df = combined.loc[:, (slice(None), valid_tradable)]
        macro_df = combined.loc[:, (slice(None), valid_macro)]

        # Save to parquet
        tradable_df.to_parquet(self.output_file)
        macro_file = self.output_file.parent / "macro_data.parquet"
        macro_df.to_parquet(macro_file)
        
        logger.info(f"Successfully saved tradable universe to {self.output_file}")
        logger.info(f"Successfully saved macro data to {macro_file}")
        
if __name__ == '__main__':
    ingestor = FTMOUniverseIngestion()
    ingestor.run()
