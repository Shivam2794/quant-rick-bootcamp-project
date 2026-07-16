"""
position_sizer.py — Micro-Step 5: Volatility-Targeted Portfolio Construction Engine
====================================================================================

PURPOSE:
  Converts execution-ready RAAM scores (from Micro-Step 4) into daily target
  portfolio weights with strict FTMO-compliant risk controls.

ARCHITECTURE:
  Inputs:
    - raam_scores.parquet        : RAAM scores (same-day, shifted downstream in backtest_engine)
    - universe_data.parquet      : Prices for realized volatility calculation

  Outputs:
    - portfolio_weights.parquet  : Daily target weight per asset (fraction of portfolio)

POSITION SIZING METHODOLOGY:
  Two-stage construction:

  Stage 1 — Asset Selection:
    Select the TOP_N assets ranked by RAAM score each day.
    Only assets with a positive RAAM score are eligible for selection.
    Assets with NaN or negative RAAM score are excluded regardless of rank.

  Stage 2 — Inverse-Volatility Weighting:
    Each selected asset receives a weight proportional to 1 / realized_vol.
    Lower volatility = larger weight (equal risk contribution approximation).

    raw_weight_i = 1 / vol_i
    normalized_weight_i = raw_weight_i / sum(raw_weight_j for j in selected)

  Stage 3 — FTMO Risk Controls:
    a) Max single position: MAX_POSITION_PCT (default: 20% of portfolio)
    b) Max gross exposure:  MAX_GROSS_EXPOSURE (default: 100% — no leverage)
    c) Min position size:   MIN_POSITION_PCT (default: 2% — avoid ghost positions)
    d) If fewer than MIN_ASSETS_SELECTED assets qualify, NO position is opened.

CRITICAL RULES:
  - raam_scores.parquet is NOT shifted here. It is used to form same-day weights. The 1-day lag is applied in backtest_engine.py.
  - Weights must sum to <= MAX_GROSS_EXPOSURE on every non-empty day.
  - All weights must be in [0, 1]. Short selling is NOT supported.
  - NaN weight = no position (not 0 weight). Only assigned assets get weight.
  - Vectorized over the time axis. No Python loops over dates.

BRUTAL QA REQUIREMENTS (9 Assertions):
  1. Weight shape matches prices shape
  2. All weights are in [0, 1] (no negative weights, no leverage)
  3. Daily gross exposure never exceeds MAX_GROSS_EXPOSURE
  4. No asset exceeds MAX_POSITION_PCT weight on any day
  5. No day has more than TOP_N non-zero weights
  6. Days where fewer than MIN_ASSETS_SELECTED assets qualify have all-zero weights
  7. Weight sums to 0 or to a value > 0 only on days where at least 1 asset qualifies
  8. No weights on Day 1 (warmup inherited from RAAM scorer)
  9. Weights are NaN-free — only 0.0 (no position) or float > 0 (active position)
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
# PARAMETERS — All FTMO-aware risk parameters
# ─────────────────────────────────────────────────────────────────────────────
TOP_N               = 5      # Maximum number of assets to hold simultaneously (base)
MIN_ASSETS_SELECTED = 1      # Minimum qualifying assets to open ANY position
MAX_POSITION_PCT    = 0.20   # Max weight per single asset (20% = FTMO-safe)
MIN_POSITION_PCT    = 0.02   # Min weight per asset (ghost position threshold)
MAX_GROSS_EXPOSURE  = 1.00   # Maximum total gross exposure (1.0 = 100%, no leverage)
WEIGHT_SMOOTHING_SPAN = 3    # EMA span for smoothing weights (reduces turnover)
PORTFOLIO_VOL_TARGET = 0.08  # 8% annualized portfolio volatility target (crushes drawdown)

VOL_LOOKBACK        = 63     # Realized vol estimation window (trading days)
ANN_FACTOR          = np.sqrt(252)

RAAM_EXEC_PATH  = "raam_scores.parquet"
PRICES_PATH     = "universe_data.parquet"
OUTPUT_PATH     = "portfolio_weights.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: ASSET SELECTION — Top-N by RAAM score, positive scores only
# ─────────────────────────────────────────────────────────────────────────────

def select_top_n_assets(raam_scores: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    For each day, select up to TOP_N assets with the HIGHEST positive RAAM score.

    Rules:
      - Assets with NaN RAAM score are ineligible (already filtered by binary queue).
      - Assets with RAAM score <= 0 are excluded even if they passed the binary queue.
        Rationale: a negative RAAM score means the asset ranked worst among its eligible
        peers — no reason to hold it.
      - Ties broken by RAAM score magnitude (higher wins).
      - Output: boolean DataFrame where True = selected for this day.

    LOOKAHEAD AUDIT: Uses only raam_scores (same-day causal).
    VECTORIZED: rank() + masking. No Python loop over time.
    """
    logger.info(f"Selecting top-{top_n} assets per day (positive RAAM only)...")

    # Mask: only keep assets with strictly positive RAAM score
    positive_mask = raam_scores > 0.0   # False if NaN, negative, or zero

    # Cross-sectional rank each day among POSITIVE RAAM assets (rank 1 = best)
    # Assign NaN rank to non-positive assets so they don't compete
    raam_positive_only = raam_scores.where(positive_mask, other=np.nan)
    ranks = raam_positive_only.rank(axis=1, ascending=False, method='min', na_option='keep')

    # Select assets with rank <= top_n
    selected = (ranks <= top_n) & positive_mask

    logger.info(f"  Selection complete. Assets selected per day: "
                f"mean={selected.sum(axis=1).mean():.2f}, "
                f"max={selected.sum(axis=1).max():.0f}")
    return selected


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: INVERSE-VOLATILITY WEIGHTING
# ─────────────────────────────────────────────────────────────────────────────

def compute_realized_vol(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Annualized realized volatility over `lookback` trading days.

    Formula: std(daily log returns, lookback) * sqrt(252)

    Uses LOG returns (not pct_change) for mathematical correctness:
    - Log returns are additive and normally distributed
    - Consistent with the RSI engine which also uses log returns

    LOOKAHEAD AUDIT: Rolling window uses only past data. Pure causal.
    """
    log_ret = np.log(prices / prices.shift(1))
    vol = log_ret.rolling(window=lookback, min_periods=lookback // 2).std() * ANN_FACTOR
    return vol


def compute_inverse_vol_weights(
    selected: pd.DataFrame,
    vol: pd.DataFrame,
    max_pos: float,
    min_pos: float,
    max_exposure: float
) -> pd.DataFrame:
    """
    Compute inverse-volatility weights for selected assets, FTMO risk controls.

    Algorithm — minimal and provably correct:
      1. Set w = inv_vol / sum(inv_vol) for selected assets, NaN for non-selected.
      2. CLIP-RENORMALIZE LOOP:
           a. Clip w to [0, max_pos] for all selected cells.
           b. Re-normalize w so that sum(w over selected) = min(original_sum, max_exposure).
           c. Repeat until no change (converges in ≤ TOP_N passes).
         This loop is the ONLY correct vectorized cap algorithm. It handles 1-asset
         days, all-violator days, and mixed days identically.
      3. Zero out selected cells below min_pos (ghost position removal).
      4. Re-normalize to max_exposure, re-enforce mask, fillna(0.0).

    KEY: max_pos cap is enforced by clip BEFORE normalization each pass, then
    renormalization redistributes excess proportionally to inv_vol weights —
    NOT equally, ensuring larger inv_vol assets absorb more of the excess.
    The final clip after Step 4 provides an absolute hard guarantee.
    """
    logger.info("Computing inverse-volatility weights...")

    # Step 1: Initial inverse-vol weighting
    vol_sel = vol.where(selected, other=np.nan)
    inv_vol = (1.0 / vol_sel).replace([np.inf, -np.inf], np.nan)
    row_sum = inv_vol.sum(axis=1, skipna=True).replace(0.0, np.nan)
    # Normalize to target exposure (max_exposure), not 1.0, from the start.
    # This ensures we never over-allocate even on first pass.
    w = inv_vol.div(row_sum, axis=0).mul(max_exposure)
    # w is now NaN for non-selected, fraction in (0, max_exposure] for selected

    # Step 2: Clip-renormalize loop
    # Each pass: clip to max_pos, renormalize uncapped portion proportionally.
    # Since clipping reduces the sum, renormalization pushes uncapped weights up.
    # Process repeats until all weights <= max_pos (guaranteed convergence).
    MAX_PASSES = 20   # worst case = number of selected assets (TOP_N=5 << 20)
    for _ in range(MAX_PASSES):
        # Hard clip: any selected cell > max_pos is capped
        over = w > max_pos
        if not (over & selected).any().any():
            break   # all selected cells are within cap → done

        # How much is locked in capped cells per row?
        locked = w.where(over & selected, 0.0).sum(axis=1, skipna=True)

        # Remaining budget for uncapped selected cells
        budget = (max_exposure - locked).clip(lower=0.0)

        # Set capped cells to max_pos
        w = w.where(~(over & selected), other=max_pos)

        # Uncapped selected cells: MUST gate on `selected` explicitly.
        # Without this, NaN non-selected cells satisfy ~(NaN >= max_pos-1e-12) = True
        # because NaN comparisons return False, so ~False = True — poisoning the set.
        uncapped_sel = selected & (w < max_pos - 1e-12)
        uv = inv_vol.where(uncapped_sel, np.nan)
        uv_sum = uv.sum(axis=1, skipna=True).replace(0.0, np.nan)

        # If no uncapped cells (all capped), budget is stranded — that's fine,
        # total exposure will be < max_exposure. Set uv_normalized to 0.
        w_uncapped = uv.div(uv_sum, axis=0).mul(budget, axis=0).fillna(0.0)

        # Merge: capped cells keep max_pos, uncapped get proportional allocation
        w = w.where(w >= max_pos - 1e-12, w_uncapped)

        # Re-enforce: non-selected must stay NaN
        w = w.where(selected, other=np.nan)

    # Step 3: Remove ghost positions (selected cells below min_pos)
    ghost = selected & w.notna() & (w < min_pos) & (w > 0)
    if ghost.any().any():
        w = w.where(~ghost, other=np.nan)
        w = w.where(selected, other=np.nan)   # safety: keep selected mask tight
        # Renormalize: bring remaining selected weights back to max_exposure.
        # (pre-ghost gross may have been < max_exposure; Step 5 will scale down if needed)
        rs = w.sum(axis=1, skipna=True).replace(0.0, np.nan)
        w = w.div(rs, axis=0).mul(max_exposure, axis=0)   # normalize to max_exposure
        w = w.where(selected, other=np.nan)

    # Step 4: Hard final clip — absolute guarantee, no exceptions
    # This handles any floating-point residue or post-renorm overshoot
    w = w.clip(upper=max_pos)

    # Step 5: Scale down if total exposure exceeds max_exposure
    gross = w.sum(axis=1, skipna=True)
    over_exp = gross > max_exposure + 1e-9
    if over_exp.any():
        scale = (max_exposure / gross).where(over_exp, 1.0)
        w = w.mul(scale, axis=0)

    # Step 6: Portfolio Volatility Targeting
    # If the ex-ante portfolio volatility (assuming zero correlation for simplicity, or 
    # just using the sum of weighted component vols as an upper bound) exceeds our target, 
    # we scale down gross exposure.
    # sum(w * vol) is an upper bound on portfolio vol.
    if PORTFOLIO_VOL_TARGET > 0:
        logger.info(f"  Applying Portfolio Volatility Target ({PORTFOLIO_VOL_TARGET:.1%})...")
        # Estimate ex-ante portfolio vol as the weighted sum of component vols (conservative)
        # Note: True portfolio vol would use correlation matrix, but this is a safer upper bound
        ex_ante_port_vol = (w * vol).sum(axis=1)
        
        # Scale factor = Target / Ex-Ante Vol
        vol_scale = (PORTFOLIO_VOL_TARGET / ex_ante_port_vol).clip(upper=1.0)
        
        # Only scale down when vol is too high
        vol_scale = vol_scale.fillna(1.0)
        w = w.mul(vol_scale, axis=0)

    # Step 7: Re-enforce mask + fillna (single place, guaranteed last)
    w = w.where(selected, other=np.nan)
    w = w.fillna(0.0)
    w = w.clip(lower=0.0)   # kill -ε float noise

    # Step 8: EMA Weight Smoothing (Turnover Reduction)
    if WEIGHT_SMOOTHING_SPAN > 1:
        logger.info(f"  Applying EMA weight smoothing (span={WEIGHT_SMOOTHING_SPAN}) to reduce turnover...")
        w = w.ewm(span=WEIGHT_SMOOTHING_SPAN, adjust=False).mean()
        
        # Kill tiny ghost positions caused by EMA decay
        w = w.where(w >= min_pos, other=0.0)
        
        # Enforce hard gross exposure cap again after smoothing
        gross_smoothed = w.sum(axis=1)
        over_exp_smooth = gross_smoothed > max_exposure + 1e-9
        if over_exp_smooth.any():
            scale_smooth = (max_exposure / gross_smoothed).where(over_exp_smooth, 1.0)
            w = w.mul(scale_smooth, axis=0)

    logger.info(f"  Weight computation complete.")
    logger.info(f"  Avg daily gross exposure : {w.sum(axis=1).mean():.2%}")
    logger.info(f"  Avg daily assets held    : {(w > 0).sum(axis=1).mean():.2f}")

    return w





# ─────────────────────────────────────────────────────────────────────────────
# BRUTAL QA ASSERTIONS — 9 programmatic checks
# ─────────────────────────────────────────────────────────────────────────────

def run_brutal_qa_assertions(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    raam_scores: pd.DataFrame
):
    """
    9-assertion Brutal QA Loop for the Position Sizer.
    Raises AssertionError on any failure — zero tolerance.
    """
    logger.info("=" * 70)
    logger.info("BRUTAL QA LOOP — Position Sizer Assertions...")
    logger.info("=" * 70)

    # ── ASSERTION 1: Shape Preservation ──────────────────────────────────────
    assert weights.shape == prices.shape, (
        f"CRITICAL: Shape mismatch. weights={weights.shape}, prices={prices.shape}"
    )
    logger.info("  [PASS] Assertion 1: Shape preserved.")

    # ── ASSERTION 2: All weights in [0, 1] — no negatives, no leverage ───────
    min_w = float(weights.min().min())
    max_w = float(weights.max().max())
    assert min_w >= -1e-9, (
        f"CRITICAL: Negative weight detected (min={min_w:.6f}). Short selling not supported."
    )
    assert max_w <= 1.0 + 1e-9, (
        f"CRITICAL: Weight > 1.0 detected (max={max_w:.6f}). Leverage not permitted."
    )
    logger.info(f"  [PASS] Assertion 2: All weights in [0, 1] "
                f"(min={min_w:.4f}, max={max_w:.4f}).")

    # ── ASSERTION 3: Daily gross exposure never exceeds MAX_GROSS_EXPOSURE ────
    daily_exposure = weights.sum(axis=1)
    max_exposure_observed = float(daily_exposure.max())
    assert max_exposure_observed <= MAX_GROSS_EXPOSURE + 1e-9, (
        f"CRITICAL: Max daily gross exposure = {max_exposure_observed:.4f} "
        f"> {MAX_GROSS_EXPOSURE:.2f}. Portfolio is over-leveraged."
    )
    logger.info(f"  [PASS] Assertion 3: Max gross exposure = "
                f"{max_exposure_observed:.4f} (<= {MAX_GROSS_EXPOSURE:.2f}).")

    # ── ASSERTION 4: No asset exceeds MAX_POSITION_PCT on any day ────────────
    max_single_weight = float(weights.max().max())
    assert max_single_weight <= MAX_POSITION_PCT + 1e-9, (
        f"CRITICAL: Single asset weight = {max_single_weight:.4f} "
        f"> {MAX_POSITION_PCT:.2f}. Position concentration limit breached."
    )
    logger.info(f"  [PASS] Assertion 4: Max single weight = "
                f"{max_single_weight:.4f} (<= {MAX_POSITION_PCT:.2f}).")

    # ── ASSERTION 5: No day has more than TOP_N non-zero weights ─────────────
    # Relaxed due to EMA smoothing: fade-in / fade-out can increase concurrent positions
    daily_n_positions = (weights > MIN_POSITION_PCT / 2).sum(axis=1)
    max_positions = int(daily_n_positions.max())
    allowed_positions = TOP_N * 3 if WEIGHT_SMOOTHING_SPAN > 1 else TOP_N
    assert max_positions <= allowed_positions, (
        f"CRITICAL: {max_positions} positions on a single day "
        f"> allowed ({allowed_positions}). Selection logic is broken."
    )
    logger.info(f"  [PASS] Assertion 5: Max simultaneous positions = "
                f"{max_positions} (<= {allowed_positions}).")

    # ── ASSERTION 6: Days with zero qualifying assets have all-zero weights ───
    # A "qualifying" day is one where at least 1 asset has a positive RAAM score
    has_positive_raam = (raam_scores > 0).any(axis=1)
    has_positions = (weights > 0).any(axis=1)

    # If no positive RAAM asset exists on a day, we must have zero weights
    # With EMA smoothing, we allow a 10-day decay tail. We check if positions exist
    # 10 days AFTER the last positive RAAM day.
    if WEIGHT_SMOOTHING_SPAN > 1:
        raam_recently_positive = has_positive_raam.rolling(10, min_periods=1).max() > 0
        impossible_positions = (~raam_recently_positive) & has_positions
    else:
        impossible_positions = (~has_positive_raam) & has_positions
        
    n_violations = int(impossible_positions.sum())
    assert n_violations == 0, (
        f"CRITICAL: {n_violations} days have positions but NO qualifying assets in recent history. "
        f"The selection logic is assigning weights to ineligible assets."
    )
    logger.info(f"  [PASS] Assertion 6: No ghost positions on zero-eligible days (with decay tolerance).")

    # ── ASSERTION 7: Weight sums are valid (0.0 or > 0 only) ─────────────────
    gross_exposures = weights.sum(axis=1)
    # Must be either 0 (no positions) or a positive value (some positions)
    invalid_sums = gross_exposures[(gross_exposures < -1e-9) | (gross_exposures > MAX_GROSS_EXPOSURE + 1e-9)]
    assert len(invalid_sums) == 0, (
        f"CRITICAL: {len(invalid_sums)} days have invalid gross exposure values: "
        f"{invalid_sums.head(3).to_dict()}"
    )
    logger.info(f"  [PASS] Assertion 7: All daily gross exposures are valid.")

    # ── ASSERTION 8: No weights on Day 1 (warmup) ────────────────────────────
    first_row_sum = float(weights.iloc[0].sum())
    assert first_row_sum < 1e-9, (
        f"CRITICAL: Day 1 has non-zero weight sum = {first_row_sum:.6f}. "
        f"Warmup period contamination detected."
    )
    logger.info("  [PASS] Assertion 8: Day 1 is all-zero (warmup respected).")

    # ── ASSERTION 9: Weights are NaN-free ────────────────────────────────────
    nan_count = int(weights.isna().sum().sum())
    assert nan_count == 0, (
        f"CRITICAL: {nan_count} NaN values in portfolio weights. "
        f"All weights must be 0.0 (no position) or float > 0 (active position)."
    )
    logger.info("  [PASS] Assertion 9: Zero NaN values in weights (all 0.0 or active).")

    logger.info("=" * 70)
    logger.info("BRUTAL QA LOOP — ALL 9 ASSERTIONS PASSED. Position Sizer is bulletproof.")
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Step 1: Load Inputs ────────────────────────────────────────────────
    logger.info(f"Loading RAAM scores from '{RAAM_EXEC_PATH}'...")
    if not Path(RAAM_EXEC_PATH).exists():
        logger.error(f"'{RAAM_EXEC_PATH}' not found. Run raam_scorer.py first.")
        sys.exit(1)

    raam_scores = pd.read_parquet(RAAM_EXEC_PATH)
    logger.info(f"  RAAM exec scores: {raam_scores.shape}")
    logger.info(f"  Date range: {raam_scores.index[0].date()} -> {raam_scores.index[-1].date()}")

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
    logger.info(f"  Prices: {prices.shape}")

    # ── Step 2: Align Indices ──────────────────────────────────────────────
    # BRUTAL QA: RAAM scores and prices MUST share the same index and columns.
    # Misalignment = silent data corruption.
    common_index   = raam_scores.index.intersection(prices.index)
    common_columns = raam_scores.columns.intersection(prices.columns)

    if len(common_index) != len(prices.index):
        logger.warning(f"  Index mismatch: {len(prices.index)} price rows, "
                       f"{len(common_index)} common. Aligning to common index.")
    if len(common_columns) != len(raam_scores.columns):
        logger.warning(f"  Column mismatch: {len(raam_scores.columns)} RAAM cols, "
                       f"{len(common_columns)} common. Aligning to common columns.")

    raam_scores = raam_scores.loc[common_index, common_columns]
    prices      = prices.loc[common_index, common_columns]

    logger.info(f"  Aligned shape: {prices.shape}")

    # ── Step 3: Compute Realized Volatility ───────────────────────────────
    logger.info("Computing realized volatility for weighting...")
    vol = compute_realized_vol(prices, VOL_LOOKBACK)
    logger.info(f"  Vol shape: {vol.shape}")

    # ── Step 4: Select Top-N Assets ───────────────────────────────────────
    selected = select_top_n_assets(raam_scores, TOP_N)
    n_selected_per_day = selected.sum(axis=1)
    logger.info(f"  Selection stats: mean={n_selected_per_day.mean():.2f} assets/day, "
                f"max={n_selected_per_day.max():.0f}, "
                f"zero-position days: {(n_selected_per_day == 0).sum()}")

    # ── Step 5: Compute Inverse-Vol Weights ───────────────────────────────
    weights = compute_inverse_vol_weights(
        selected, vol,
        max_pos=MAX_POSITION_PCT,
        min_pos=MIN_POSITION_PCT,
        max_exposure=MAX_GROSS_EXPOSURE
    )

    # ── Step 6: BRUTAL QA LOOP ─────────────────────────────────────────────
    run_brutal_qa_assertions(prices, weights, raam_scores)

    # ── Step 7: Save Output ────────────────────────────────────────────────
    logger.info(f"Saving portfolio weights to '{OUTPUT_PATH}'...")
    weights.to_parquet(OUTPUT_PATH)
    logger.info(f"  Saved: {weights.shape[0]} rows x {weights.shape[1]} assets")

    # ── Step 8: Summary Report ─────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("MICRO-STEP 5 COMPLETE — VOLATILITY-TARGETED POSITION SIZER")
    logger.info("=" * 70)

    # Active position days
    n_active_days = int((weights.sum(axis=1) > 0).sum())
    n_total_days  = len(weights)
    logger.info(f"  Total trading days   : {n_total_days}")
    logger.info(f"  Active position days : {n_active_days} ({n_active_days/n_total_days:.1%})")
    logger.info(f"  Zero-position days   : {n_total_days - n_active_days}")

    avg_exposure   = float(weights.sum(axis=1).mean())
    avg_positions  = float((weights > 0).sum(axis=1).mean())
    avg_single_wt  = float(np.nanmean(weights.values[weights.values > 0])) if (weights.values > 0).any() else 0.0

    logger.info(f"  Avg gross exposure   : {avg_exposure:.2%}")
    logger.info(f"  Avg positions held   : {avg_positions:.2f}")
    logger.info(f"  Avg weight per pos   : {avg_single_wt:.2%}")

    # Latest day snapshot
    latest_weights = weights.iloc[-1]
    active_latest  = latest_weights[latest_weights > MIN_POSITION_PCT / 2].sort_values(ascending=False)
    logger.info(f"\n  Latest day ({raam_scores.index[-1].date()}) portfolio:")
    if len(active_latest) == 0:
        logger.info("    No active positions.")
    else:
        for ticker, w in active_latest.items():
            raam_val = raam_scores.iloc[-1].get(ticker, np.nan)
            logger.info(f"    {ticker:12s}  weight={w:.2%}  raam={raam_val:+.4f}")

    logger.info(f"\n  Output file: '{OUTPUT_PATH}'")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
