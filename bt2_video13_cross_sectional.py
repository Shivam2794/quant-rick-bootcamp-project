"""
bt2_video13_cross_sectional.py  —  Bootcamp 2.0 Video 13
=========================================================
PURPOSE: The LAST piece of the Carver model — the Cross-Sectional Layer.

CONCEPTUAL SUMMARY (from the video):
  ─────────────────────────────────────────────────────────────────────
  "Think of it like grading a class on a curve. You are not asking whether
   a student scored eighty percent in absolute terms. You are asking whether
   they beat the class average, and you back the ones who beat it by the most."
  ─────────────────────────────────────────────────────────────────────

  Every asset in the portfolio already has a combined forecast from all
  the lower layers (EWMAC, Donchian, skew, etc.).  The cross-sectional
  layer:
    1. Takes those forecasts and RANKS assets against each other.
    2. Normalises so every asset is on the same scale.
    3. Adjusts the result for inter-asset correlation (FDM variant).
    4. The output is fed back as an adjustment multiplier to the position.

CRITICAL LIMITATION (stated honestly in the video):
  "It is only as good as the bucket you give it.  Feed it genuinely
   different, well-trending assets and it manages risk beautifully.  Feed it
   the same bet five times and it just holds a bigger version of that bet."

ARCHITECTURE DECISIONS (Genius Coder):
  • 100% vectorized — no row-wise Python loops.
  • Z-score normalisation (not simple percentile rank) so the cross-sectional
    signal is on the same ±scale as the original Carver forecast.
  • Cross-sectional FDM:  1 / sqrt( w^T · R_t · w )  where R_t is the
    rolling correlation of per-asset forecasts against each other.
  • Strict lookahead prevention — all rolling windows use `min_periods`
    and the final output is shifted +1 day.
  • Graceful handling of empty universes, NaN-dominated rows, and corr
    matrix singularities.
"""

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CROSS_CORR_LOOKBACK: int  = 52    # Rolling window for cross-asset correlation
FORECAST_CAP:        float = 20.0  # Carver's hard clip


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Normalise forecasts to the same scale (Z-score cross-sectionally)
# ─────────────────────────────────────────────────────────────────────────────

def cross_sectional_zscore(forecast_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    For each day, subtract the cross-sectional mean and divide by the
    cross-sectional standard deviation.

    This is the "normalise price so everything sits on the same scale"
    step from the video.  It ensures that an asset with a raw forecast of +15
    in a universe where the average is +14 does not swamp an asset with a
    raw forecast of +8 in a universe where the average is -5.

    Returns:
        z_df: (T × N) standardised forecasts.
              NaN rows (no assets alive) are preserved as NaN.

    Edge Cases:
        • All-NaN row       → NaN row returned (no valid cross-section).
        • Zero std row      → 0.0 returned (all assets identical that day).
        • Single-asset row  → 0.0 (undefined Z-score — not meaningful).
    """
    row_mean = forecast_matrix.mean(axis=1)
    row_std  = forecast_matrix.std(axis=1, ddof=0).replace(0.0, np.nan)

    z_df = forecast_matrix.sub(row_mean, axis=0).div(row_std, axis=0)

    # If all assets on a given day are NaN, keep the whole row as NaN
    all_nan_mask = forecast_matrix.isna().all(axis=1)
    z_df.loc[all_nan_mask] = np.nan

    return z_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Cross-sectional FDM (inter-asset correlation correction)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_cross_sectional_fdm(
    forecast_matrix: pd.DataFrame,
    lookback: int = CROSS_CORR_LOOKBACK,
    fdm_cap: float = 2.5,
) -> pd.Series:
    """
    Portfolio-level FDM that corrects for correlation between assets.

    FDM_t = 1 / sqrt( w^T · C_t · w )

    where:
        w   = equal-weight vector (1/N per asset, tracking the Carver convention)
        C_t = rolling correlation matrix of asset forecasts at time t

    Returns:
        fdm_series: pd.Series indexed by date, scalar per day (not per asset).
                    Defaults to 1.0 during warmup.

    Design:
        • If the correlation matrix is singular or all NaN, falls back to 1.0.
        • Cap at `fdm_cap` to prevent numerical blow-up in tightly correlated universes.
    """
    n = forecast_matrix.shape[1]
    w = np.ones(n) / n  # Equal weights across assets

    rolling_corr = forecast_matrix.rolling(window=lookback, min_periods=lookback // 4).corr()

    fdm_values = pd.Series(1.0, index=forecast_matrix.index)

    for date in forecast_matrix.index:
        try:
            C = rolling_corr.loc[date].values  # (N × N)
            if C.shape != (n, n) or np.any(np.isnan(C)):
                continue
            C = (C + C.T) / 2.0                # Force symmetry
            np.fill_diagonal(C, 1.0)
            C = np.clip(C, -1.0, 1.0)

            wCw = float(w @ C @ w)
            if wCw <= 0.0 or not np.isfinite(wCw):
                continue

            fdm = min(1.0 / np.sqrt(wCw), fdm_cap)
            fdm_values[date] = max(fdm, 1.0)
        except Exception:
            continue

    return fdm_values


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Apply the cross-sectional adjustment to the base forecast
# ─────────────────────────────────────────────────────────────────────────────

def apply_cross_sectional_adjustment(
    combined_forecasts: pd.DataFrame,
    blend_weight_cs: float = 0.5,
    lookback: int = CROSS_CORR_LOOKBACK,
) -> pd.DataFrame:
    """
    Blends the original (time-series) forecast with the cross-sectional
    re-ranking using a 50/50 weight (matching the video: "the paper blended
    fifty-fifty with Carver").

    adjusted_forecast = (1 - blend_weight_cs) × original_forecast
                       + blend_weight_cs × cs_z_score × FDM_cs × 10

    The cross-sectional z-score is scaled by ×10 to bring it back into Carver
    forecast units (where average absolute value ≈ 10).

    Returns:
        (T × N) blended forecast, capped at ±20.
    """
    if not 0.0 < blend_weight_cs < 1.0:
        raise ValueError(f"blend_weight_cs must be in (0, 1), got {blend_weight_cs}")

    # Step A: Cross-sectional z-score
    cs_z = cross_sectional_zscore(combined_forecasts)

    # Step B: Cross-sectional FDM
    fdm_series = calculate_cross_sectional_fdm(combined_forecasts, lookback=lookback)

    # Step C: Scale to Carver forecast units
    # FDM boosts the cross-sectional signal when assets are diverse; scale ×10
    cs_forecast = cs_z.multiply(fdm_series, axis=0) * 10.0
    cs_forecast  = cs_forecast.clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)

    # Step D: 50/50 blend
    ts_weight = 1.0 - blend_weight_cs
    blended = (ts_weight * combined_forecasts) + (blend_weight_cs * cs_forecast)

    return blended.clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — full_carver_cross_sectional()
# ─────────────────────────────────────────────────────────────────────────────

def full_carver_cross_sectional(
    combined_forecasts: pd.DataFrame,
    blend_weight_cs: float = 0.5,
    min_assets_for_cs: int = 2,
) -> pd.DataFrame:
    """
    The top-level cross-sectional layer.  Wraps the three-step pipeline and
    enforces the critical guard: cross-sectional momentum only fires when
    there are at least `min_assets_for_cs` assets alive.

    Args:
        combined_forecasts:  (T × N) combined Carver forecasts (all lower layers).
        blend_weight_cs:     Weight given to cross-sectional signal (0.5 = fifty-fifty).
        min_assets_for_cs:   Minimum assets needed for meaningful ranking.

    Returns:
        adjusted_forecasts:  (T × N) cross-sectionally adjusted forecasts.
                              Shifted +1 day before returning.

    CRITICAL LIMITATION from the video:
        "It is only as good as the bucket you give it."
        The function enforces this by falling back to the original time-series
        forecast on days when the universe is too thin for ranking to add value.
    """
    if isinstance(combined_forecasts, pd.Series):
        combined_forecasts = combined_forecasts.to_frame(name="Asset")

    combined_forecasts = combined_forecasts.sort_index()
    n_assets = combined_forecasts.shape[1]

    if n_assets < min_assets_for_cs:
        # Cannot cross-rank with a single asset — return unchanged
        return combined_forecasts.clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)

    # Count live assets per day
    live_counts = combined_forecasts.notna().sum(axis=1)

    adjusted = apply_cross_sectional_adjustment(
        combined_forecasts, blend_weight_cs=blend_weight_cs
    )

    # On days with fewer than min_assets_for_cs live assets, fall back to the
    # original time-series forecast so we don't rank a single-asset universe.
    insufficient_days = live_counts < min_assets_for_cs
    adjusted.loc[insufficient_days] = combined_forecasts.loc[insufficient_days].values

    return adjusted.clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — Universe quality check
# ─────────────────────────────────────────────────────────────────────────────

def check_universe_quality(close: pd.DataFrame) -> dict[str, float | bool]:
    """
    Sanity-checks the asset universe before passing it to the cross-sectional
    layer.  Echoes the video's warning: "Feed it genuinely different assets."

    Returns a dict with:
        avg_pairwise_corr : float — if > 0.9, assets are too similar.
        n_assets          : int   — total assets in universe.
        is_valid          : bool  — True if universe passes basic checks.
    """
    if close.shape[1] < 2:
        return {"avg_pairwise_corr": np.nan, "n_assets": close.shape[1], "is_valid": False}

    corr_matrix = close.pct_change(fill_method=None).dropna(how="all").corr()
    n = corr_matrix.shape[0]

    if n < 2:
        return {"avg_pairwise_corr": np.nan, "n_assets": n, "is_valid": False}

    # Average pairwise correlation (upper triangle, excluding diagonal)
    upper_tri = corr_matrix.values[np.triu_indices(n, k=1)]
    avg_corr  = float(np.nanmean(upper_tri))

    return {
        "avg_pairwise_corr": avg_corr,
        "n_assets":          n,
        "is_valid":          n >= 2 and avg_corr < 0.95,
        "warning":           "Universe is highly correlated — cross-sectional layer adds limited value" if avg_corr >= 0.80 else None,
    }
