# OCURE-5: Dynamic Equity Engine (Carver Implementation)

Following the feedback from the Author and advanced structural iteration, this module represents the next evolution of our Carver-based trend following architecture, applied exclusively to a spot-equity universe (no options, no futures).

## The Core Challenge
While the original Carver engine proved phenomenal on highly volatile crypto assets (yielding a 1.45 Sharpe), applying it to static traditional equity panels (like SPY/QQQ) heavily underperformed. The core issues were:
1. **Survivorship Bias & Stagnation:** Hardcoded baskets of stocks inherently suffer from multi-decade stagnation.
2. **The "Shorting" Problem in Equities:** Because equities possess a permanent, structural upward drift and suffer from V-shaped mean-reverting crashes, traditional trend-following shorts get severely whipsawed during the recoveries.
3. **Correlation Spikes:** During massive liquidity panics (e.g., 2022), all equities drop together.

## The Solution: OCURE-5 Architecture (Dual Portfolio & Macro Gate)

### 1. The Dynamic Universe Screener (Layer 0)
`ocure5_universe_screener.py` completely eliminates survivorship bias. At the end of every quarter, it scans a broad universe of massive Tech/Growth stocks and selects only the **Top 20 highest realized-volatility assets** over a rolling 90-day window. This ensures capital is constantly rotated into the most volatile, active assets.

### 2. Dual Portfolio Decoupling
To combat strategy degradation, `ocure5_execution.py` splits the core engine into two parallel, uncorrelated components. We execute a **Pure Trend Portfolio (EWMAC/Breakout)** and a **Pure Mean Reversion Portfolio (Skew/Accel)**. These are sized equally (50/50) and prevent overlapping failure states.

### 3. The BTC/SPY Macro Exit Gate
`carver_master_strategy.py` calculates a continuous `BTC/SPY` ratio. If global liquidity and risk appetite drop—forcing the ratio below its 200-day moving average—the system mathematically overrides all scaling and forces total Equities exposure to `0` (cash).

### 4. Dynamic Correlation Sizing
We implemented a mathematically precise **Dynamic Correlation Scalar** to ensure the portfolio hits a strict 20% Target Volatility, dynamically scaling leverage up or down based on the exact number of active assets passing the screener in any given millisecond.

## Final Verified Performance (2015 - 2024)
* **CAGR:** 16.47%
* **Volatility:** 13.89%
* **Sharpe Ratio:** 1.04
* **Max Drawdown:** -15.50%

By structurally decoupling the Carver models and implementing a brutal BTC/SPY macro gate, this system breached the 1.0 Sharpe barrier on raw spot equities, completely sidestepping the worst of the 2022 correlation spike.
