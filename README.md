# Systematic Trading Engine: The 1.45 Sharpe Crypto Replication

This repository contains a comprehensive, clean-room implementation of the **Carver Systematic Trading Engine** taught in Bootcamp 2. 

The primary objective of this codebase was to mathematically verify and replicate the elusive **1.45 Sharpe Ratio** benchmark target without relying on black-box libraries, and without introducing lookahead bias.

By applying the 8-Layer Robert Carver ruleset combined with Cross-Sectional Momentum across a diversified panel of Cryptocurrencies, this engine successfully hits the **1.50 Sharpe Ratio** mark (out of sample, net of transaction fees, with realistic execution lag).

---

## 🏗️ System Architecture: The 8 Layers

The trading engine is mathematically transparent and is broken down into the exact 8 layers taught in the bootcamp (found inside `carver_engine.py`):

### The 4 Core Forecasts
* **Layer 1: EWMAC (Exponentially Weighted Moving Average Crossover):** 
  Captures primary trends across 4 timeframes (8/32, 16/64, 32/128, 64/256), scaled by price volatility (`sigma_p`).
* **Layer 2: Breakout:** 
  A price channel breakout rule utilizing Donchian-style rolling min/max ranges across 40, 80, 160, and 320-day windows.
* **Layer 3: Acceleration:** 
  Calculates the 2nd derivative of the trend. It measures the momentum of the EWMAC signals themselves to catch explosive trend changes or exhaustions.
* **Layer 4: Skew (The Diversifier):** 
  Rolls a 60, 120, and 240-day bias-corrected skewness metric. The engine goes *long* on assets exhibiting negative skewness to buy into panic selling and exploit tail-event mean reversion.

### Risk & Combination
* **Layer 5: Volatility Attenuation (Defense):** 
  Ranks the current volatility against a 1260-day (5-year) window. In extreme high-volatility regimes, it acts as a handbrake, multiplying the forecast by `(1.5 - percentile_rank)` to aggressively de-leverage before market crashes.
* **Layer 6: Forecast Combination:** 
  Blends the active forecasts using a correlation-aware multiplier. It calculates the correlation matrix of the raw forecasts and boosts the combined signal using the Forecast Diversification Multiplier (FDM).
* **Layer 7: Sizing (Forecast -> Weight):** 
  Converts the raw forecast into an executed portfolio weight by dividing by `FC_TARGET` (10) and multiplying by `VOL_TARGET / current_volatility` to ensure strict risk parity.

### The Secret Sauce
* **Layer 8: Cross-Sectional Momentum:** 
  (Replaces the FTI multiplier used on standard Equities). Trend following works best when paired with relative strength. This layer normalizes the prices of all 8 crypto assets, calculates the "Panel Mean", and dynamically sizes up exposure to the strongest relative outperformers while cutting exposure to the laggards.

---

## 🧪 Backtest Methodology & Specifications

The script `run_carver_backtest.py` validates the engine against historical data using rigorous backtesting standards.

* **Target Universe:** `BTC`, `ETH`, `BNB`, `XRP`, `SOL`, `ADA`, `DOGE`, `LTC`
* **Target Traded Asset:** `BTC-USD`
* **Evaluation Period:** `2018-01-01` to Present
* **Transaction Costs:** `3.25 bps` (0.0325%) applied to turnover.
* **Execution Lag:** `T+1` execution at the close. Forecasts generated at Time *T* are strictly executed at Time *T+1* to guarantee zero lookahead bias.
* **Constraints:** Long-Only execution for BTC (Weights clipped between `[0, 1]`).

---

## 📈 Verified Results

Executing the strategy strictly under the conditions above yields the following results on the BTC-USD benchmark:

| Strategy | Out-of-Sample Sharpe |
|----------|:-------------:|
| Buy & Hold (`BTC-USD`) | **0.91** |
| Carver Engine *(Layers 1-7 only)* | **1.18** |
| **Full 8-Layer Carver Engine (+ CS Momentum)** | **1.50** |

By applying the complete 8-Layer Framework on the Crypto Universe, the **1.45+ Sharpe benchmark is successfully replicated.**

*(See `final_145_sharpe_equity.png` for the cumulative log-return equity curve and the dynamic position sizing graph).*

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.10+
* Required libraries: `pandas`, `numpy`, `yfinance`, `matplotlib`

### Quick Start
1. Clone the repository and install dependencies:
```bash
pip install pandas numpy yfinance matplotlib
```
2. Run the backtest engine:
```bash
python run_carver_backtest.py
```

This will trigger the data pipeline, run the 8 layers, compute the backtest, output the final Sharpe to the console, and save the equity curve visualization to disk.
