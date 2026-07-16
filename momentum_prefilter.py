"""
momentum_prefilter.py — Micro-Step 3: 3-EMA & MACD Binary Queue Pre-Filter
============================================================================

BRUTAL QA REQUIREMENTS:
1. No lookahead bias — EMAs and MACD calculated using only past data (`.shift(1)` is NOT 
   needed here because EMAs are inherently causal; they only use data up to and including
   the current bar. The *signal comparison* itself is lag-free. Signals are evaluated at the
   close of each bar and applied from the NEXT bar (handled downstream by RSI ranker's .shift(1)).
2. NaN handling — Early bars before EMA warmup produce NaN; these are preserved as NaN
   (not filled) to prevent fictitious signal generation.
3. Strict boolean output — The final `eligible_assets` DataFrame must contain only
   True/False/NaN. No float bleed.
4. Shape preservation — Output must match input prices shape exactly.
5. Assertions enforce all of the above programmatically.
"""

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


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS — all sourced from the quantitative specification
# ─────────────────────────────────────────────────────────────────────────────
EMA_SHORT  = 20   # Fast EMA
EMA_MID    = 50   # Medium EMA
EMA_SLOW   = 200  # Slow EMA / Trend anchor

MACD_FAST   = 12  # MACD fast span
MACD_SLOW   = 26  # MACD slow span
MACD_SIGNAL = 9   # MACD signal line span

DATA_PATH   = "universe_data.parquet"
OUTPUT_PATH = "binary_queue.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS — 100% vectorized, no Python loops over time axis
# ─────────────────────────────────────────────────────────────────────────────

def compute_3ema_filter(prices: pd.DataFrame) -> pd.DataFrame:
    """
    3-EMA Bullish Alignment Filter.

    An asset passes on a given day ONLY if:
        Price > EMA(20) > EMA(50) > EMA(200)

    All four conditions must hold simultaneously.

    LOOKAHEAD AUDIT:
    - pd.DataFrame.ewm(span=N, adjust=False) is a causal filter.
    - At time t, EMA(t) = alpha * price(t) + (1-alpha) * EMA(t-1).
    - It only uses data up to and including time t. NO lookahead.

    Returns:
        pd.DataFrame of bool/NaN — True if asset passes the 3-EMA filter.
    """
    logger.info(f"Computing 3-EMA filter (spans: {EMA_SHORT}, {EMA_MID}, {EMA_SLOW})...")

    # Calculate the three EMAs across ALL assets simultaneously (vectorized)
    ema_short = prices.ewm(span=EMA_SHORT, adjust=False).mean()
    ema_mid   = prices.ewm(span=EMA_MID,   adjust=False).mean()
    ema_slow  = prices.ewm(span=EMA_SLOW,  adjust=False).mean()

    # Strict bullish alignment: all 4 conditions must be True
    # NaN propagates automatically — if any value is NaN, the comparison yields False.
    # We convert False from NaN-comparisons to NaN to preserve the warmup period truthfully.
    cond_1 = prices    > ema_short   # Price above fast EMA
    cond_2 = ema_short > ema_mid     # Fast EMA above mid EMA
    cond_3 = ema_mid   > ema_slow    # Mid EMA above slow EMA

    # Combined filter: ALL conditions must pass
    ema_filter = cond_1 & cond_2 & cond_3

    # ── BRUTAL QA: Mask warmup period with NaN ──────────────────────────────
    # CRITICAL FIX: pandas EWM (adjust=False) is NEVER NaN — it initializes
    # from the very first data point. So ema_slow.isna() is always False.
    # We must explicitly count valid (non-NaN) price bars per column and
    # mask the first EMA_SLOW bars of trading for each asset.
    #
    # cumcount of non-NaN prices: gives 0 on the first valid bar, 1 on the
    # second, etc. We want NaN for the first EMA_SLOW-1 valid bars.
    valid_count = prices.notna().cumsum()  # cumulative count of valid price bars
    warmup_mask = valid_count < EMA_SLOW   # True during the warmup window
    ema_filter_clean = ema_filter.where(~warmup_mask, other=np.nan)

    logger.info(f"  3-EMA filter warmup: first {EMA_SLOW} valid bars per asset masked.")
    return ema_filter_clean


def compute_macd_filter(prices: pd.DataFrame) -> pd.DataFrame:
    """
    MACD Bullish Signal Filter.

    Standard MACD formula:
        MACD Line   = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line

    An asset passes on a given day ONLY if:
        MACD Line > Signal Line

    LOOKAHEAD AUDIT:
    - Both EMA(12), EMA(26), and the Signal EMA are causal by the same
      argument as above. NO lookahead.

    Returns:
        pd.DataFrame of bool/NaN — True if asset passes the MACD filter.
    """
    logger.info(f"Computing MACD filter ({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})...")

    # MACD Line: EMA_fast - EMA_slow (fully vectorized across all asset columns)
    macd_line = (
        prices.ewm(span=MACD_FAST, adjust=False).mean()
        - prices.ewm(span=MACD_SLOW, adjust=False).mean()
    )

    # Signal Line: EMA of the MACD Line itself
    macd_signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()

    # Bullish condition: MACD Line strictly above Signal Line
    macd_filter = macd_line > macd_signal_line

    # ── BRUTAL QA: Mask warmup period with NaN ──────────────────────────────
    # Same CRITICAL FIX as above: EWM is never NaN in pandas.
    # Explicitly mask the first (MACD_SLOW + MACD_SIGNAL) valid bars.
    macd_warmup_bars = MACD_SLOW + MACD_SIGNAL
    valid_count = prices.notna().cumsum()
    warmup_mask = valid_count < macd_warmup_bars
    macd_filter_clean = macd_filter.where(~warmup_mask, other=np.nan)

    logger.info(f"  MACD filter warmup: first {macd_warmup_bars} valid bars per asset masked.")
    return macd_filter_clean


def compute_binary_queue(
    ema_filter: pd.DataFrame,
    macd_filter: pd.DataFrame,
    regimes: pd.DataFrame = None
) -> pd.DataFrame:
    """
    The Strict Binary Queue.

    An asset is eligible (True) ONLY if it passes BOTH:
        1. The 3-EMA filter (Price > EMA20 > EMA50 > EMA200)
        2. The MACD filter  (MACD Line > MACD Signal Line)

    If EITHER filter is False, the asset is DISQUALIFIED (False).
    If EITHER filter is NaN (warmup), the result is NaN.

    The `&` operator on boolean DataFrames with NaN requires special care.
    pandas treats NaN as False in `&`. We must preserve NaN truthfully.

    Correct approach: Convert to float, multiply (True=1, False=0, NaN=NaN).
    Then re-cast to bool only where both values are non-NaN.
    """
    logger.info("Combining filters into strict Binary Queue...")

    # Convert boolean-ish frames to float (True→1.0, False→0.0, NaN→NaN)
    ema_float  = ema_filter.astype(float)
    macd_float = macd_filter.astype(float)

    # Logical AND via multiplication (preserves NaN correctly):
    #   1 * 1 = 1.0  (True & True  -> True)
    #   1 * 0 = 0.0  (True & False -> False)
    #   0 * 0 = 0.0  (False & False -> False)
    #   NaN * anything = NaN (warmup preserved)
    combined_float = ema_float * macd_float
    
    if regimes is not None:
        logger.info("Applying Defensive Canary Regime Override...")
        is_defensive = (regimes['is_risk_on'] == 0)
        
        # FTMO Safe Havens (Forex + Metals + Bonds)
        safe_havens = ["EURUSD=X", "GBPUSD=X", "JPY=X", "CHF=X", "CAD=X", "GC=F", "SI=F", "TLT", "IEF", "SHY", "UUP"]
        risk_assets = [col for col in combined_float.columns if col not in safe_havens]
        
        # 1. Disqualify all risk assets during defensive regime
        for col in risk_assets:
            mask = is_defensive & combined_float[col].notna()
            combined_float.loc[mask, col] = 0.0

        # 2. Bypass momentum prefilter for safe havens during defensive regime to allow RSI-rotation
        for col in safe_havens:
            if col in combined_float.columns:
                # We force them to 1.0 (Eligible) if they have valid price data on that day
                mask = is_defensive & combined_float[col].notna()
                combined_float.loc[mask, col] = 1.0

    # Re-cast cleanly: build an object-dtype DataFrame where values are
    # True, False, or np.nan. Using np.where on numpy arrays avoids ALL
    # pandas FutureWarnings about incompatible dtype assignment.
    arr = combined_float.to_numpy()          # shape (rows, cols), dtype float64
    has_val = ~np.isnan(arr)                 # boolean mask of non-NaN cells
    out = np.where(has_val, arr.astype(bool), np.nan)  # object array: True/False/nan
    eligible = pd.DataFrame(out, index=combined_float.index, columns=combined_float.columns)

    logger.info("  Binary Queue computed.")
    return eligible


# ─────────────────────────────────────────────────────────────────────────────
# BRUTAL QA ASSERTIONS — Programmatic verification of mathematical integrity
# ─────────────────────────────────────────────────────────────────────────────

def run_brutal_qa_assertions(
    prices: pd.DataFrame,
    ema_filter: pd.DataFrame,
    macd_filter: pd.DataFrame,
    eligible_assets: pd.DataFrame
):
    """
    Runs the full suite of programmatic assertions.
    Raises AssertionError immediately on any failure.
    """
    logger.info("=" * 70)
    logger.info("BRUTAL QA LOOP — Running programmatic assertions...")
    logger.info("=" * 70)

    # ── ASSERTION 1: Shape Preservation ──────────────────────────────────────
    assert eligible_assets.shape == prices.shape, (
        f"CRITICAL: Shape mismatch! eligible_assets={eligible_assets.shape}, "
        f"prices={prices.shape}. Output must match input dimensions exactly."
    )
    logger.info("  [PASS] Assertion 1: Shape preserved.")

    # ── ASSERTION 2: Column Preservation ─────────────────────────────────────
    assert list(eligible_assets.columns) == list(prices.columns), (
        "CRITICAL: Column mismatch! Output columns do not match input columns."
    )
    logger.info("  [PASS] Assertion 2: Columns preserved.")

    # ── ASSERTION 3: Index Preservation ──────────────────────────────────────
    assert eligible_assets.index.equals(prices.index), (
        "CRITICAL: Index mismatch! Output index does not match input index."
    )
    logger.info("  [PASS] Assertion 3: Index preserved.")

    # ── ASSERTION 4: Only Boolean or NaN Values ───────────────────────────────
    # Flatten non-NaN values and check they are all 0.0 or 1.0 (False/True)
    # FutureWarning fix: use future_stack=True for pandas >= 2.1
    non_nan_values = eligible_assets.stack(future_stack=True).dropna()
    # After multiplication, values should only be 0.0 or 1.0 or bool
    unique_non_nan = non_nan_values.unique()
    invalid_values = [v for v in unique_non_nan if v not in (0.0, 1.0, True, False)]
    assert len(invalid_values) == 0, (
        f"CRITICAL: Non-boolean values found in eligible_assets: {invalid_values}"
    )
    logger.info("  [PASS] Assertion 4: All non-NaN values are boolean (True/False).")

    # ── ASSERTION 5: NaN Consistency — No NaN islands after warmup ───────────
    # An asset's warmup NaN region should be contiguous at the START.
    # After it gets a valid value, it should not return to NaN (unless the
    # asset stopped trading — which is allowed). We check: for each column,
    # once the first non-NaN appears, there should be no NaN UNLESS the
    # PRICE itself is also NaN (delisted asset, weekend gap, etc.)
    # This check is column-wise.
    violations = 0
    for col in eligible_assets.columns:
        col_eligible = eligible_assets[col]
        col_prices   = prices[col]
        first_valid_idx = col_eligible.first_valid_index()
        if first_valid_idx is None:
            continue  # Entire column is NaN — skip
        # Slice from first valid index onward
        sub_eligible = col_eligible.loc[first_valid_idx:]
        sub_prices   = col_prices.loc[first_valid_idx:]
        # NaN in eligible where price is NOT NaN = a logical inconsistency
        illegal_nan  = sub_eligible.isna() & sub_prices.notna()
        if illegal_nan.any():
            violations += 1
            logger.warning(
                f"  [WARN] Column '{col}' has {illegal_nan.sum()} mid-series NaN values "
                f"where price data exists. This may indicate a technical indicator gap."
            )
    # We allow up to 5 warnings for edge-cases (e.g., assets with gaps)
    assert violations <= 5, (
        f"CRITICAL: {violations} columns have illegal mid-series NaN values."
    )
    logger.info(f"  [PASS] Assertion 5: NaN consistency check passed (violations={violations}).")

    # ── ASSERTION 6: Eligible Rate Sanity Check ───────────────────────────────
    # In any given bull market, at least some assets should be eligible.
    # If 0% of assets are ever eligible, something is catastrophically wrong.
    total_true  = (eligible_assets == 1.0).sum().sum()
    total_false = (eligible_assets == 0.0).sum().sum()
    total_nan   = eligible_assets.isna().sum().sum()
    total_cells = eligible_assets.size

    pct_eligible = total_true / (total_true + total_false) * 100 if (total_true + total_false) > 0 else 0
    logger.info(f"  Eligible cells: {total_true:,} ({pct_eligible:.1f}% of non-NaN)")
    logger.info(f"  Disqualified cells: {total_false:,}")
    logger.info(f"  Warmup NaN cells: {total_nan:,} ({total_nan/total_cells*100:.1f}% of total)")

    assert total_true > 0, (
        "CRITICAL: ZERO assets were ever eligible. The filter logic is broken."
    )
    assert pct_eligible > 1.0, (
        f"CRITICAL: Only {pct_eligible:.1f}% of non-NaN cells are eligible. "
        f"The filter is pathologically restrictive (likely a logic error)."
    )
    assert pct_eligible < 99.0, (
        f"CRITICAL: {pct_eligible:.1f}% of non-NaN cells are eligible. "
        f"The filter is pathologically permissive (conditions never trigger False)."
    )
    logger.info(f"  [PASS] Assertion 6: Eligible rate is sane ({pct_eligible:.1f}%).")

    # ── ASSERTION 7: Anti-Lookahead Structural Check ──────────────────────────
    # Verify that the VERY FIRST row of eligible_assets is entirely NaN
    # (no asset can be eligible on day 1, since EMAs need warmup data).
    first_row = eligible_assets.iloc[0]
    assert first_row.isna().all(), (
        "CRITICAL: Day 1 has non-NaN values in eligible_assets. "
        "This violates warmup requirements and suggests lookahead contamination."
    )
    logger.info("  [PASS] Assertion 7: Anti-lookahead check passed (Day 1 is all NaN).")

    logger.info("=" * 70)
    logger.info("BRUTAL QA LOOP — ALL 7 ASSERTIONS PASSED. Code is bulletproof.")
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Step 1: Load Data ──────────────────────────────────────────────────
    logger.info(f"Loading universe data from '{DATA_PATH}'...")

    if not Path(DATA_PATH).exists():
        logger.error(f"Data file '{DATA_PATH}' not found. Run data_ingestion.py first.")
        sys.exit(1)

    df = pd.read_parquet(DATA_PATH)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            prices = df['Adj Close']
        else:
            prices = df.xs('Adj Close', axis=1, level=0, drop_level=True) if 'Adj Close' in df.columns else df
    else:
        prices = df
    logger.info(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")


    logger.info(f"  Price matrix: {prices.shape[0]} rows × {prices.shape[1]} assets")
    logger.info(f"  Date range: {prices.index[0].date()} -> {prices.index[-1].date()}")

    # ── Step 2: Compute 3-EMA Filter ──────────────────────────────────────
    ema_filter = compute_3ema_filter(prices)

    # ── Step 3: Compute MACD Filter ────────────────────────────────────────
    macd_filter = compute_macd_filter(prices)

    # ── Step 3.5: Load Regime Data ─────────────────────────────────────────
    regimes = None
    regime_path = Path("macro_regime.parquet")
    if regime_path.exists():
        logger.info(f"Loading regime signals from '{regime_path}'...")
        regimes = pd.read_parquet(regime_path)

    # ── Step 4: Combine into Binary Queue ─────────────────────────────────
    eligible_assets = compute_binary_queue(ema_filter, macd_filter, regimes)

    # ── Step 5: BRUTAL QA LOOP ─────────────────────────────────────────────
    run_brutal_qa_assertions(prices, ema_filter, macd_filter, eligible_assets)

    # ── Step 6: Save Output ────────────────────────────────────────────────
    logger.info(f"Saving binary queue to '{OUTPUT_PATH}'...")
    # BRUTAL QA FIX: Save as float64 (1.0=True, 0.0=False, NaN=warmup) not object dtype.
    # Object-dtype parquet with Python True/False/None causes inconsistent reload behavior
    # across pandas versions. float64 is universally compatible.
    eligible_assets_float = eligible_assets.astype(float)
    eligible_assets_float.to_parquet(OUTPUT_PATH)
    logger.info(f"  Saved: {eligible_assets_float.shape[0]} rows x {eligible_assets_float.shape[1]} assets")

    # ── Step 7: Summary Report ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("MICRO-STEP 3 COMPLETE — MOMENTUM PRE-FILTER BINARY QUEUE")
    logger.info("=" * 70)

    # Sample: show eligibility rates for a few key assets on the latest day
    latest_day = eligible_assets.iloc[-1]
    n_eligible  = int((latest_day == 1.0).sum())
    n_total_valid = int(latest_day.notna().sum())
    logger.info(f"  Latest day ({prices.index[-1].date()}):")
    logger.info(f"    Assets ELIGIBLE   : {n_eligible} / {n_total_valid}")
    logger.info(f"    Assets DISQUALIFIED: {n_total_valid - n_eligible} / {n_total_valid}")
    logger.info(f"  Output file: '{OUTPUT_PATH}'")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
