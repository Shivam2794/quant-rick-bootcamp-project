# Institutional Quant Engine (Bootcamp 1 & 2)

Welcome to the **Omni-Grinder Verified** 7-Layer Quantitative Execution Engine. 

This repository contains a full, 100% vectorized Python/Pandas implementation of the systematic trading architecture taught across Bootcamp 1 and Bootcamp 2. The core philosophy of this engine relies on rigorous structural math rather than predictive "black boxes." It is designed to handle adversarial market conditions safely without relying on recursive loops or forward-looking data.

---

## 🏗 The 7-Layer Architecture

The execution pipeline processes a raw price matrix into physical trading positions by passing data through a sequence of mathematically isolated layers.

### Layer 1: Instrument Risk (Volstack)
Retail systems often calculate volatility using the standard deviation of logarithmic returns. However, log returns are highly susceptible to fat-tail outlier events, causing sudden, dramatic scaling errors. 
This engine uses the **Carver Institutional Volatility Scalar**:
- It computes the 36-day Exponential Moving Average of **Absolute Price Differences (MAD)**.
- It then scales this MAD by a constant of **1.2533** to approximate an outlier-resistant standard deviation ($\sigma$).
- *Source File:* `bt2_video4_math_derived.py`

### Layers 2 & 3: The Forecast Ensemble
Rather than betting on a single timeframe, the engine deploys a diversified stack of momentum and mean-reversion rules:
- **EWMAC (Exponentially Weighted Moving Average Crossover):** Uses standardized pairs (8/32, 16/64, 32/128, 64/256). Each crossover is normalized by the Layer 1 Volatility to ensure that "one unit of normal volatility" equals roughly 1.0 unit of raw signal. Empirical forecast scalars (e.g., 5.3, 7.5, 10.6, 15.0) are then applied to force the average absolute forecast to a target magnitude of ~10.
- **Breakout Systems:** Horizon mapping from 40 to 320 days.
- **Skew & Acceleration:** Implements EWMA-smoothed negated offsets and strict 1.55 FDM acceleration overrides.

### Layer 4: Volatility Attenuation (Tail-Risk Management)
Position sizes are not static. When current market volatility spikes into the extreme 95th percentile relative to the last 5 years, the engine actively suppresses the forecast magnitude. 
- A continuous multiplier is generated to scale down forecasts smoothly from `1.0` down to `0.5` (or lower, bounded by `0.1`), ensuring the portfolio naturally deleverages during violent market dislocations.

### Layers 5 & 6: Rule Weights & FDM
If multiple highly-correlated rules (e.g., EWMAC 16/64 and 32/128) agree, a naive system will blindly double down on the bet, faking a higher-confidence signal. 
- **FDM (Forecast Diversification Multiplier):** We compute a rolling 52-day correlation matrix across all active rules. The multiplier is $1 / \sqrt{w^T C w}$.
- **Singular Matrix Safe:** If the correlation matrix becomes singular/rank-deficient, the FDM relies on `np.linalg.lstsq` (pseudo-inverse) to prevent a system crash. 
- The FDM is strictly clipped between `1.0` and `2.5`.

### Layer 7: Combined Forecast & Position Sizing
The combined, FDM-boosted forecast (capped at a hard maximum of $\pm 20$) is translated into cash-denominated shares targeting a specific annualized portfolio volatility (e.g., 20%). 
- Calculation uses absolute closing prices to guard against **Negative Pricing Events** (such as WTI crude in 2020 or certain calendar spreads) which would otherwise invert the sign of the physical position.

---

## 🛡 Regime & Macro Gates

Above the continuous time-series forecasts, the engine applies discrete binary and regime filters:
- **Beta Rotations:** Employs 21-day Z-Score rotations filtering for Risk-Off environments using leading macro indicators (like Consumer Confidence metrics).
- **RSI Equity Filter:** Generates a synthetic equity curve for the strategy itself. If the strategy's 28-period RSI drops below `50`, the system identifies a structural drawdown and halts trading until the equity curve regains momentum.
- **Strict In-Sample Boruta Permutation:** Ensures no forward-looking bias exists during the feature selection process.
- **Turnover Buffers:** Demands a 10% minimal deviation threshold before a rebalance order is submitted, ensuring trades occur strictly between 2 to 15 times a year. This eliminates high-frequency noise and prevents the strategy from devolving into a buy-and-hold proxy.

---

## 🗂 File Navigation & Manifest

The repository contains numerous modularized files representing the progression of the curriculum. The most critical files for the final engine are:

- **`master_pipeline_v1_to_v13.py`** 
  The top-level execution layer. Imports all individual modules and orchestrates the data flow from raw prices to executable `T+1` positions.
- **`bt2_video12_final_assembly.py`** 
  The core mathematical assembly layer housing the Volatility Attenuation, FDM correlation matrices, and Volatility Targeting code.
- **`bt2_video13_cross_sectional.py`** 
  Houses the Cross-Sectional Ranking and Z-scoring mechanics, allowing relative strength sorting across the asset universe.
- **`bt2_video4_math_derived.py`** 
  The bedrock math formulas, notably the Carver Volatility derivation.
- **`block4_final_blueprint.py` & `portfolio_amalgamator.py`** 
  The earlier BC1 architecture files dealing with indicator generation, regime gates, and portfolio combining.
- **`omni_grinder_*.py`** 
  The brutal continuous-testing QA suites (see below).

---

## 🗺 Curriculum Map: From Bootcamp 1 to Bootcamp 2

This codebase directly implements the exact logic and mechanics taught across the entire Bootcamp 1 and Bootcamp 2 series.

### Bootcamp 1: Single Asset Regimes & Features (Videos 1 - 25)
*   **Videos 1 - 5 (Feature Engineering):** Implementation of momentum pre-filters, SMA/EMA crosses, and foundational Donchian channels.
*   **Videos 6 - 10 (Regime & Macro Gates):** Processing external macro data (Consumer Confidence, Yield Curves), constructing regime environments, and engineering Z-Score Beta Rotations.
*   **Videos 11 - 18 (Volatility & Execution Limits):** Implementation of the 5-year Volatility Attenuation framework to survive black swans, and strict Turnover Buffers constraining strategies to 2-15 trades per year.
*   **Videos 19 - 25 (Validation & The Block 4 Blueprint):** Execution of strict In-Sample Boruta Permutation to eradicate forward-looking bias, and the final construction of the `block4_final_blueprint.py` portfolio amalgamator. This also covers the 8 micro-steps from videos 20-25 detailing multi-factor RAAM scoring.

### Bootcamp 2: The Institutional Carver Engine (Videos 1 - 13)
*   **Videos 1 - 3 (Carver Principles):** Refactoring retail math into institutional mechanics, integrating Macro Canaries into continuous forecasts.
*   **Videos 4 - 6 (MAD Derivation & Sizing):** The critical mathematical derivation of the Carver Volatility Scalar (1.2533 × 36-day EMA of Absolute Differences), and hybrid position sizing calculations.
*   **Videos 7 - 9 (Robust Data & Ensemble Math):** Bulletproofing the pandas data pipelines against NaNs, and applying the exact empirically-derived scalars to normalize EWMAC forecast strength to a target of ~10.
*   **Videos 10 - 11 (Cross-Sectional Ranking):** Implementation of cross-sectional Z-scoring to force assets to compete against each other for capital.
*   **Videos 12 - 13 (Final Assembly & FDM):** The final unification of the stack. Introducing the Forecast Diversification Multiplier (FDM) matrix math to accurately blend correlated rules without synthetically overloading risk.

---

## 🧪 Omni-Grinder Safety Protocols

This codebase has passed a **20x Continuous Burn-In Loop** via the internal Omni-Grinder QA suites. 

1. **AST Structural Verification (`omni_grinder_master_suite.py`)**
   - The Abstract Syntax Trees of all 26 core files were scanned. 
   - **0 Non-Vectorized Loops:** No `iterrows` or `itertuples` exist. 
   - **0 Negative Lookahead:** Zero instances of `shift(-1)` were found. All execution arrays strictly enforce `shift(1)` to ensure `T+1` execution.
2. **Mathematical Immortality (`omni_grinder_math_suite.py`)**
   - The arrays were aggressively stressed with synthetic **Bankruptcies** (assets instantly dropping to $0.00), **Trading Halts** (massive contiguous blocks of `NaN`), and **Negative Pricing**.
   - The engine successfully processed the adversarial data without cascading `inf` or `NaN` outputs.

---

## 🚀 Execution & Usage

Because the system is 100% vectorized in Pandas, generating 10 years of multi-asset backtest data takes milliseconds. 

```python
import pandas as pd
from master_pipeline_v1_to_v13 import run_master_carver_pipeline

# 1. Load your raw (T x N) pricing matrix
# Rows = DateTimeIndex, Columns = Asset Tickers
prices = pd.read_parquet("my_universe.parquet")

# 2. Execute the 7-Layer Stack
# - executable_positions: The final share counts (safely shifted to T+1)
# - forecasts: The continuous FDM-boosted rule scores
# - sigma_price: The daily risk in price-units per asset
executable_positions, forecasts, sigma_price = run_master_carver_pipeline(
    close=prices,
    capital=100_000.0,
    target_annual_vol=0.20,
    blend_weight_cs=0.5,
    use_buffer=True
)

print(executable_positions.tail())
```
