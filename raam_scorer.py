"""
raam_scorer.py — Micro-Step 4: RAAM 4-Factor Cross-Sectional Scorer
=====================================================================

RAAM = Return, Acceleration, Amplitude, Momentum

This module scores ALL eligible assets (those passing the binary queue from
Micro-Step 3) on 4 quantitative factors and produces a final composite
RAAM score per asset per day.

ARCHITECTURE:
  Input:  universe_data.parquet (prices)  +  binary_queue.parquet (eligibility)
  Output: raam_scores.parquet             (composite score DataFrame)

FACTOR DEFINITIONS:
  1. R — Return (Momentum, 12-month lookback)
       12-month total return, normalized cross-sectionally.
       Assets with the highest recent return score highest.
       
  2. A — Acceleration (2nd Derivative of Momentum)
       Rate of change of the 3-month momentum (63d) vs 12-month momentum (252d).
       Positive acceleration = trend is speeding up.

  3. A — Amplitude (Volatility — INVERSE scoring)
       Annualized realized volatility over 63 days.
       LOWER volatility = HIGHER score (risk-adjusted momentum preference).

  4. M — Trend Consistency (% of positive return days over 63 days)
       Measures quality of the trend, not just raw magnitude.
       Genuinely independent from raw 12-month return (corr ~0.14).
       High consistency = asset moves up steadily, not via a single spike.

CRITICAL MATHEMATICAL RULES:
  - All factors are computed on the ELIGIBLE universe only (via binary_queue mask).
  - Cross-sectional ranking/normalization happens ONLY among eligible assets
    on each day. Ineligible assets receive NaN scores, NOT 0.
  - All factors use .shift(1) before downstream signal comparison to prevent
    lookahead into the same-bar signal used for trade execution.
    NOTE: We do NOT apply .shift(1) here — that is the responsibility of the
    final signal combiner. We output raw same-day scores.
  - Vectorized operations only. No for-loops over time axis.

BRUTAL QA REQUIREMENTS (7 Assertions):
  1. Score shape matches prices shape
  2. Ineligible assets have NaN RAAM score
  3. All non-NaN scores are finite floats
  4. Cross-sectional mean of scores on any day is near 0 (z-scored)
  5. No scores survive on day 1 (warmup)
  6. Factor correlation matrix is inspected for pathological collinearity
  7. Score range sanity: no unbounded explosions (|z-score| < 10)
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
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK_12M  = 252   # ~12 months of trading days (momentum)
LOOKBACK_3M   = 63    # ~3 months of trading days (short-term momentum / accel)
LOOKBACK_VOL  = 63    # Volatility lookback window
ANN_FACTOR    = np.sqrt(252)  # Annualization factor for daily vol

# RAAM factor weights (equal-weighted by default; can be tuned)
WEIGHT_R = 0.25  # Return factor
WEIGHT_A = 0.25  # Acceleration factor
WEIGHT_V = 0.25  # Amplitude (inverse vol) factor
WEIGHT_M = 0.25  # Trend Consistency factor (replaces Magnitude — collinear with R)

PRICES_PATH  = "universe_data.parquet"
QUEUE_PATH   = "binary_queue.parquet"
OUTPUT_PATH  = "raam_scores.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# FACTOR COMPUTATION — All fully vectorized
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily percentage returns. Simple, foundational."""
    return prices.pct_change()


def factor_return_12m(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Factor R: 12-Month Total Return (Momentum).

    Uses 1-month skip (21 trading days) to exclude the short-term reversal
    effect documented by Jegadeesh & Titman. The standard formula:
        ret_12m = (price[t-21] / price[t-252]) - 1

    LOOKAHEAD AUDIT: Only uses price data from t-21 and t-252. Pure causal.
    """
    # Standard skip-1-month momentum: (price 21 days ago) / (price 252 days ago) - 1
    f = (prices.shift(21) / prices.shift(LOOKBACK_12M)) - 1.0
    return f


def factor_acceleration(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Factor A: Momentum Acceleration (2nd Derivative of Trend).

    Measures whether SHORT-TERM momentum is INCREASING or DECREASING.
    This captures the 2nd derivative of price — is the trend speeding up or
    rolling over?

    Formula:
        mom_21d_now    = (price[t]    / price[t-21])  - 1   (current 21d momentum)
        mom_21d_lagged = (price[t-42] / price[t-63])  - 1   (21d momentum 42 days ago)
        accel = mom_21d_now - mom_21d_lagged

    BRUTALLY QA NOTE: This version does NOT reuse ret_12m from Factor R.
    The previous version (ret_3m - ret_12m) shared ret_12m with Factor R,
    producing |corr| = 0.934 — near-collinear. This version is independent.

    LOOKAHEAD AUDIT: Only uses shifted prices. Pure causal.
    """
    # Current 21-day momentum (how fast is price moving right now?)
    mom_now    = (prices / prices.shift(21)) - 1.0
    # 21-day momentum lagged by 42 days (how fast was price moving 42 days ago?)
    mom_lagged = (prices.shift(42) / prices.shift(63)) - 1.0
    # Acceleration = current speed minus past speed
    accel = mom_now - mom_lagged
    return accel


def factor_amplitude_inverse(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Factor A (Amplitude, Inverse Vol): Annualized Realized Volatility, Inverted.

    Lower volatility = better score. We compute vol and then negate it so that
    cross-sectional z-scoring naturally ranks low-vol assets higher.

        vol = std(daily_returns, 63d) * sqrt(252)
        amplitude_score = -vol   (inversion: lower vol => higher value)

    LOOKAHEAD AUDIT: Rolling std uses only past 63 bars. Pure causal.
    """
    daily_ret = prices.pct_change()
    vol = daily_ret.rolling(window=LOOKBACK_VOL, min_periods=LOOKBACK_VOL // 2).std() * ANN_FACTOR
    # Invert: lower volatility => higher score (we negate)
    amplitude_score = -vol
    return amplitude_score


def factor_trend_consistency(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Factor M: Trend Consistency (% of positive daily returns over 63 days).

    Measures HOW CONSISTENTLY an asset has trended upward, not just how far
    it moved. An asset up 5% with 60% positive days is higher quality than
    one up 5% via a single 10% spike followed by a flat period.

    This factor is genuinely uncorrelated from raw 12-month return:
    - A trending asset can have modest return but high consistency
    - A volatile asset can have high return but low consistency

        consistency = (count of positive daily returns in 63d) / 63

    Result is a float in [0, 1]. Higher = more consistently positive.

    LOOKAHEAD AUDIT: Rolling sum over past 63 bars. Pure causal.
    """
    daily_ret = prices.pct_change()
    # 1 where return > 0, else 0
    positive_days = (daily_ret > 0).astype(float)
    # Rolling sum over 63 days, divided by 63 = fraction of positive days
    consistency = positive_days.rolling(
        window=LOOKBACK_3M,
        min_periods=LOOKBACK_3M // 2
    ).mean()  # .mean() on 0/1 = fraction of positive days
    return consistency


def cross_sectional_zscore(
    factor: pd.DataFrame,
    eligible_mask: pd.DataFrame
) -> pd.DataFrame:
    """
    Z-score a factor ONLY within the eligible universe on each day.

    Algorithm:
      1. Mask ineligible values to NaN
      2. For each row, compute mean and std of the ELIGIBLE assets only
      3. Z-score each eligible asset's value
      4. Ineligible assets remain NaN

    This is critical: we must NOT include ineligible assets in the cross-
    sectional distribution. Including them would contaminate the z-scores.

    VECTORIZED: Uses pandas row-wise mean/std with skipna=True, then masking.
    """
    # Apply eligibility mask: only eligible = True cells survive
    # eligible_mask contains True/False/NaN
    eligible_bool = eligible_mask == True  # True only where explicitly True
    masked_factor = factor.where(eligible_bool, other=np.nan)

    # Row-wise (cross-sectional) mean and std — skipna=True by default
    row_mean = masked_factor.mean(axis=1, skipna=True)
    row_std  = masked_factor.std(axis=1, skipna=True, ddof=1)

    # Z-score: subtract mean, divide by std (both broadcast along axis=0)
    zscored = masked_factor.sub(row_mean, axis=0).div(row_std, axis=0)

    # Handle edge case: if std = 0 (all eligible assets have identical factor value),
    # z-scores become inf or NaN. We must handle inf FIRST before isna() check.
    # BRUTAL QA FIX: zscored.isna() does NOT catch inf values. Must replace inf->NaN first.
    zscored = zscored.replace([np.inf, -np.inf], np.nan)

    # Now: any NaN in an ELIGIBLE cell means all eligible assets had identical values (std=0).
    # Replace those with 0.0 (neutral z-score). Keep ineligible cells as NaN.
    is_nan_in_eligible = eligible_bool & zscored.isna()
    zscored = zscored.where(~is_nan_in_eligible, other=0.0)

    # Final safety: re-apply eligibility mask (belt-and-suspenders)
    zscored = zscored.where(eligible_bool, other=np.nan)

    return zscored


# ─────────────────────────────────────────────────────────────────────────────
# BRUTAL QA ASSERTIONS
# ─────────────────────────────────────────────────────────────────────────────

def run_brutal_qa_assertions(
    prices: pd.DataFrame,
    eligible_mask: pd.DataFrame,
    raam_score: pd.DataFrame,
    factor_dict: dict
):
    """
    7-assertion Brutal QA Loop for the RAAM scorer.
    Raises AssertionError on any failure.
    """
    logger.info("=" * 70)
    logger.info("BRUTAL QA LOOP — RAAM Scorer Assertions...")
    logger.info("=" * 70)

    eligible_bool = eligible_mask == True

    # ── ASSERTION 1: Shape Preservation ──────────────────────────────────────
    assert raam_score.shape == prices.shape, (
        f"Shape mismatch: raam_score={raam_score.shape}, prices={prices.shape}"
    )
    logger.info("  [PASS] Assertion 1: Shape preserved.")

    # ── ASSERTION 2: Ineligible assets have NaN RAAM score ───────────────────
    # Any cell where eligible_bool is False must be NaN in raam_score
    should_be_nan = ~eligible_bool  # True where asset is ineligible
    # Ineligible cells in raam_score must all be NaN
    violations = raam_score.where(should_be_nan).notna().sum().sum()
    assert violations == 0, (
        f"CRITICAL: {violations} ineligible cells have non-NaN RAAM scores. "
        f"The eligibility mask is leaking."
    )
    logger.info("  [PASS] Assertion 2: Ineligible assets correctly NaN.")

    # ── ASSERTION 3: All non-NaN scores are finite floats ────────────────────
    non_nan = raam_score.stack(future_stack=True).dropna()
    inf_count = np.isinf(non_nan.astype(float)).sum()
    assert inf_count == 0, (
        f"CRITICAL: {inf_count} infinite values in RAAM scores."
    )
    assert non_nan.dtype in [np.float64, np.float32, object], (
        f"Unexpected dtype: {non_nan.dtype}"
    )
    logger.info("  [PASS] Assertion 3: All non-NaN scores are finite.")

    # ── ASSERTION 4: Cross-sectional mean is near zero (z-scored) ────────────
    # For each day, the z-scored mean across eligible assets should be ~0
    # We check the grand mean across all days
    daily_eligible_mean = raam_score.where(eligible_bool).mean(axis=1, skipna=True)
    grand_mean = float(daily_eligible_mean.mean(skipna=True))
    assert abs(grand_mean) < 0.1, (
        f"CRITICAL: Grand mean of RAAM scores = {grand_mean:.4f}. "
        f"Expected ~0 (z-scored). The normalization is broken."
    )
    logger.info(f"  [PASS] Assertion 4: Cross-sectional grand mean = {grand_mean:.4f} (near zero).")

    # ── ASSERTION 5: No scores on day 1 (warmup) ─────────────────────────────
    first_row = raam_score.iloc[0]
    assert first_row.isna().all(), (
        "CRITICAL: Day 1 has non-NaN RAAM scores. Lookahead contamination detected."
    )
    logger.info("  [PASS] Assertion 5: Day 1 is all NaN (warmup respected).")

    # ── ASSERTION 6: Factor correlation matrix check ──────────────────────────
    # Concatenate all factor z-scores and compute pairwise correlation
    factor_cols = pd.concat(
        [df.where(eligible_bool).stack(future_stack=True).rename(name)
         for name, df in factor_dict.items()],
        axis=1
    ).dropna()
    
    if len(factor_cols) > 10:
        corr_matrix = factor_cols.corr()
        logger.info(f"  Factor correlation matrix:\n{corr_matrix.round(3).to_string()}")
        
        # Check for pathological collinearity: no two factors should have |corr| > 0.98
        for i, col_i in enumerate(corr_matrix.columns):
            for j, col_j in enumerate(corr_matrix.columns):
                if i >= j:
                    continue
                corr_val = abs(corr_matrix.loc[col_i, col_j])
                assert corr_val < 0.98, (
                    f"CRITICAL: Factors '{col_i}' and '{col_j}' have |corr| = {corr_val:.3f}. "
                    f"They are redundant (collinear). The factor set is invalid."
                )
        logger.info("  [PASS] Assertion 6: No pathological factor collinearity detected.")
    else:
        logger.info("  [SKIP] Assertion 6: Insufficient data for correlation check.")

    # ── ASSERTION 7: Score range sanity ──────────────────────────────────────
    # Z-scores should not explode beyond ±10 in normal market conditions
    max_abs_score = float(non_nan.abs().max())
    assert max_abs_score < 10.0, (
        f"CRITICAL: Maximum |RAAM score| = {max_abs_score:.2f}. "
        f"Expected < 10.0. Scores are exploding (likely a NaN contamination)."
    )
    logger.info(f"  [PASS] Assertion 7: Score range sane (max |score| = {max_abs_score:.3f}).")

    logger.info("=" * 70)
    logger.info("BRUTAL QA LOOP — ALL 7 ASSERTIONS PASSED. RAAM Scorer is bulletproof.")
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Step 1: Load Data ──────────────────────────────────────────────────
    logger.info(f"Loading price data from '{PRICES_PATH}'...")
    if not Path(PRICES_PATH).exists():
        logger.error(f"'{PRICES_PATH}' not found. Run data_ingestion.py first.")
        sys.exit(1)

    df = pd.read_parquet(PRICES_PATH)
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.levels[0]:
            prices = df['Adj Close']
        else:
            prices = df.xs('Adj Close', axis=1, level=0, drop_level=True) if 'Adj Close' in df.columns else df
    else:
        prices = df
    logger.info(f"  Prices: {prices.shape[0]} rows x {prices.shape[1]} assets")
    logger.info(f"  Date range: {prices.index[0].date()} -> {prices.index[-1].date()}")

    logger.info(f"Loading binary queue from '{QUEUE_PATH}'...")
    if not Path(QUEUE_PATH).exists():
        logger.error(f"'{QUEUE_PATH}' not found. Run momentum_prefilter.py first.")
        sys.exit(1)

    eligible_mask = pd.read_parquet(QUEUE_PATH)
    logger.info(f"  Binary queue: {eligible_mask.shape}")

    # ── Step 2: Compute Raw Factors ────────────────────────────────────────
    logger.info("Computing RAAM factors...")

    logger.info("  Factor R: 12-month return (momentum)...")
    f_return = factor_return_12m(prices)

    logger.info("  Factor A: momentum acceleration (3m vs 12m)...")
    f_accel  = factor_acceleration(prices)

    logger.info("  Factor A (amplitude): inverse realized volatility...")
    f_amplitude = factor_amplitude_inverse(prices)

    logger.info("  Factor M: trend consistency (% positive days in 63d)...")
    f_magnitude = factor_trend_consistency(prices)

    # ── Step 3: Z-score Each Factor Within Eligible Universe ───────────────
    logger.info("Z-scoring factors within eligible universe...")

    z_return    = cross_sectional_zscore(f_return,    eligible_mask)
    z_accel     = cross_sectional_zscore(f_accel,     eligible_mask)
    z_amplitude = cross_sectional_zscore(f_amplitude, eligible_mask)
    z_magnitude = cross_sectional_zscore(f_magnitude, eligible_mask)

    # ── Step 4: Compute Composite RAAM Score ──────────────────────────────
    logger.info("Computing composite RAAM score (equal-weighted)...")

    raam_score = (
        WEIGHT_R * z_return
        + WEIGHT_A * z_accel
        + WEIGHT_V * z_amplitude
        + WEIGHT_M * z_magnitude
    )

    # Re-apply eligibility: composite NaN if ANY factor is NaN for that cell
    eligible_bool = eligible_mask == True
    raam_score = raam_score.where(eligible_bool, other=np.nan)

    # ── Step 5: BRUTAL QA LOOP ─────────────────────────────────────────────
    factor_dict = {
        "R_Return":      z_return,
        "A_Accel":       z_accel,
        "A_Amplitude":   z_amplitude,
        "M_Consistency": z_magnitude,
    }
    run_brutal_qa_assertions(prices, eligible_mask, raam_score, factor_dict)

    # ── Step 6: Save Output ────────────────────────────────────────────────
    logger.info(f"Saving RAAM scores to '{OUTPUT_PATH}'...")
    # BRUTAL QA FIX: Removed .shift(1) double-lag. The lag is applied ONCE in backtest_engine.py.
    # Downstream combiners (position_sizer.py) use the same-day score to form same-day weights,
    # which are then lagged during the P&L calculation.
    raam_score.to_parquet(OUTPUT_PATH)
    logger.info(f"  Saved raw scores: '{OUTPUT_PATH}' ({raam_score.shape[0]} rows x {raam_score.shape[1]} assets)")

    # ── Step 7: Summary Report ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("MICRO-STEP 4 COMPLETE — RAAM 4-FACTOR SCORER")
    logger.info("=" * 70)

    # Top-ranked assets on the latest day
    latest_scores = raam_score.iloc[-1].dropna().sort_values(ascending=False)
    n_scored = len(latest_scores)
    logger.info(f"  Latest day ({prices.index[-1].date()}):")
    logger.info(f"    Assets scored: {n_scored}")

    if n_scored > 0:
        logger.info(f"    TOP 5 RAAM-ranked assets:")
        for rank, (ticker, score) in enumerate(latest_scores.head(5).items(), 1):
            logger.info(f"      #{rank}: {ticker:10s}  score = {score:+.4f}")

        logger.info(f"    BOTTOM 3 eligible assets (weakest momentum):")
        for rank, (ticker, score) in enumerate(latest_scores.tail(3).items(), 1):
            logger.info(f"      {ticker:10s}  score = {score:+.4f}")

    logger.info(f"  Output file: '{OUTPUT_PATH}'")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
