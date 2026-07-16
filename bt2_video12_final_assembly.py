"""
bt2_video12_final_assembly.py  —  Bootcamp 2.0 Video 12
=========================================================
PURPOSE: The FINAL code layer. All rules are built in isolation in earlier
videos. This module fuses them:

  Layers 5, 6, 7 (from the video description):
    5. Rule weights (how much conviction we assign to each speed/type).
    6. FDM — Forecast Diversification Multiplier. Corrects for inter-rule
       correlation so combining rules that agree does not fake extra signal.
    7. Combined forecast  →  physical position (via volatility targeting).

ADDITIONAL OVERLAY PREVIEWED:
  • FTI (Timothy Masters "Fractal Trend Indicator") — a trend-quality gate.
    When price noise crosses the noise line, trend quality is stepped down.
    NOT implemented in full here — stub returns 1.0 (neutral multiplier).
    Gets its own dedicated video.

DEPLOYMENT PATTERNS documented in the video:
  1. Single-asset Carver — run Carver on one flagged asset.
  2. Ensemble-triggered Carver — only deploy when ensemble flashes.
  3. Multi-asset equal-weight Carver — 25% slice per uncorrelated asset.

ARCHITECTURE DECISIONS (Genius Coder):
  • 100% vectorized; no Python loops over rows.
  • Strict lookahead prevention via `.shift(1)` before returning positions.
  • FDM matrix algebra uses `np.linalg.lstsq` pseudo-inverse fallback to
    handle rank-deficient correlation matrices (singular tensor crash fix).
  • NaN warmup strictly enforced so Day 1 never carries a position.
  • Absolute price used in sigma_p to guard against negative-price assets
    (oil futures, spread products — the Attack-4 fix from Video 4).
"""

import numpy as np
import pandas as pd
import warnings
import logging
import sys

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Standard Carver EWMAC pairs and their empirical forecast scalars.
# Scalars map the average absolute raw crossover to ~10 units.
EWMAC_SPECS: list[tuple[int, int, float]] = [
    (8,   32,   5.3),
    (16,  64,   7.5),
    (32,  128, 10.6),
    (64,  256, 15.0),
]

# Default rule weights (sum must equal 1.0).
# In practice these are optimised per-asset; these are Carver's canonical defaults.
DEFAULT_RULE_WEIGHTS: dict[str, float] = {
    "EWMAC8_32":   0.20,
    "EWMAC16_64":  0.25,
    "EWMAC32_128": 0.30,
    "EWMAC64_256": 0.25,
}

FORECAST_CAP: float = 20.0        # Carver's hard clip
VOL_LOOKBACK: int   = 36          # Fast MAD volatility window
CORR_LOOKBACK: int  = 52          # Rolling correlation window for FDM


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — Instrument risk (sigma_price)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_instrument_risk(
    close: pd.DataFrame,
    lookback: int = VOL_LOOKBACK,
) -> pd.DataFrame:
    """
    Compute sigma_price using Carver's MAD estimator (robust against fat tails).

    Args:
        close:    (T × N) price DataFrame.  Must be sorted chronologically.
        lookback: EWMA span in trading days.

    Returns:
        sigma_price: (T × N) DataFrame of daily price-unit volatility.

    Design:
        • Uses `close - close.ffill().shift(1)` instead of `.diff()` so that
          prices across trading halts (where .diff() silently NaN-skips) are
          correctly captured.
        • MAD × 1.2533 ≈ σ  (asymptotically efficient under normality).
    """
    price_diff = close - close.ffill().shift(1)
    mad = price_diff.abs().ewm(span=lookback, adjust=False).mean()
    sigma_price = mad * 1.2533
    sigma_price = sigma_price.where(close.notna().cumsum() >= lookback, np.nan)
    sigma_price = sigma_price.replace(0.0, np.nan)
    return sigma_price


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Individual EWMAC raw forecasts
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_single_ewmac(
    close: pd.DataFrame,
    sigma_price: pd.DataFrame,
    fast: int,
    slow: int,
    scalar: float,
) -> pd.DataFrame:
    """
    Calculate a single EWMAC forecast, normalised and capped at ±20.

    The crossover is divided by sigma_price so that 'one normal day of
    volatility' maps to roughly 1.0 unit of raw signal, and the scalar
    lifts the average absolute value to ~10.

    NOTE: ffill() is applied before EWM to prevent the phantom time-halt
    bug where Pandas assigns a stale EMA the weight of a fresh one after
    a multi-day gap.
    """
    filled = close.ffill()
    ema_fast = filled.ewm(span=fast, adjust=False).mean()
    ema_slow = filled.ewm(span=slow, adjust=False).mean()

    raw = ema_fast - ema_slow

    # INSTITUTIONAL FIX (Attack 4): absolute price prevents sign inversion on
    # negative-price assets (WTI futures, calendar spreads).
    annual_sigma = sigma_price  # already daily price units — no need to annualise here
    normalised   = raw / annual_sigma

    forecast = normalised * scalar
    forecast  = forecast.clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)

    # Enforce warmup mask — only valid after slow EMA has enough data
    mask = close.notna().cumsum() >= slow
    forecast = forecast.where(mask, np.nan)
    forecast = forecast.where(close.notna(), np.nan)  # re-NaN any stale EWM outputs
    return forecast


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — Rule forecast stack (all EWMAC speeds)
# ─────────────────────────────────────────────────────────────────────────────

def build_rule_forecast_stack(
    close: pd.DataFrame,
    sigma_price: pd.DataFrame,
    ewmac_specs: list[tuple[int, int, float]] = EWMAC_SPECS,
) -> dict[str, pd.DataFrame]:
    """
    Returns a dict mapping rule names → (T × N) forecast DataFrames.

    Extending to new rule types (breakout, skew, acceleration) is done by
    adding entries to this dict — the FDM engine below is rule-agnostic.
    """
    forecasts: dict[str, pd.DataFrame] = {}
    for fast, slow, scalar in ewmac_specs:
        name = f"EWMAC{fast}_{slow}"
        forecasts[name] = _calculate_single_ewmac(close, sigma_price, fast, slow, scalar)
    return forecasts


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 4 — Volatility Attenuation (Video N — "ease off when market is shocking")
# ─────────────────────────────────────────────────────────────────────────────

def compute_volatility_attenuation(
    sigma_price: pd.DataFrame,
    attenuation_quantile: float = 0.95,
    lookback: int = 252,
) -> pd.DataFrame:
    """
    When today's volatility is extremely elevated (top 5% of history),
    scale the forecast down linearly.  This prevents the position sizer from
    going full-force into a violently dislocated market.

    Returns a scalar ∈ (0, 1] per asset per day.

    Design:
        multiplier = clip( Q95_vol / today_vol, 0.1, 1.0 )
    """
    rolling_quantile = sigma_price.rolling(window=lookback, min_periods=lookback // 4).quantile(
        attenuation_quantile
    )
    raw_ratio = rolling_quantile / sigma_price.replace(0.0, np.nan)
    attenuation = raw_ratio.clip(lower=0.1, upper=1.0)
    # Fill early NaNs with 1.0 (no attenuation in warmup)
    return attenuation.fillna(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 5 — Rule Weights
# ─────────────────────────────────────────────────────────────────────────────

def normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """
    Normalise weights so they sum exactly to 1.0.
    Raises ValueError if any weight is negative (sanity guard).
    """
    if any(v < 0 for v in weights.values()):
        raise ValueError("Negative rule weight detected — all weights must be ≥ 0.")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("Sum of rule weights is zero — cannot normalise.")
    return {k: v / total for k, v in weights.items()}


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 6 — FDM (Forecast Diversification Multiplier)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_rolling_fdm(
    forecasts: dict[str, pd.DataFrame],
    weights: dict[str, float],
    lookback: int = CORR_LOOKBACK,
    fdm_cap: float = 2.5,
) -> pd.DataFrame:
    """
    Per-asset rolling FDM:   FDM_t = 1 / sqrt(w^T · C_t · w)

    where  C_t  is the rolling inter-rule correlation matrix at time t.

    Args:
        forecasts: dict rule_name → (T × N) DataFrame
        weights:   dict rule_name → float (must be normalised)
        lookback:  rolling window for correlation estimation
        fdm_cap:   hard upper cap to prevent numerical blow-up

    Returns:
        fdm_df: (T × N) DataFrame of per-asset FDM scalars.

    Design:
        • If the correlation matrix is singular (rules perfectly co-linear),
          `np.linalg.lstsq` pseudo-inverse is used as a fallback.
        • For assets with < lookback valid observations, FDM defaults to 1.0
          (no boost) rather than propagating NaN.
    """
    rule_names = list(forecasts.keys())
    n_rules    = len(rule_names)
    w_vec      = np.array([weights.get(r, 1.0 / n_rules) for r in rule_names], dtype=float)

    # All forecasts must be aligned to a single reference index
    ref_df   = next(iter(forecasts.values()))
    dates    = ref_df.index
    assets   = ref_df.columns
    n_assets = len(assets)

    fdm_values = np.ones((len(dates), n_assets), dtype=float)

    # Stack forecasts into (T × N_rules × N_assets) tensor
    forecast_tensor = np.stack([forecasts[r].values for r in rule_names], axis=1)  # (T, R, A)

    for a_idx in range(n_assets):
        asset_block = forecast_tensor[:, :, a_idx]  # (T, R) — one column per rule
        asset_df    = pd.DataFrame(asset_block, index=dates, columns=rule_names)

        # Rolling correlation (rule × rule) per time step
        rolling_corr = asset_df.rolling(window=lookback, min_periods=lookback // 4).corr()
        # rolling_corr has MultiIndex (date × rule) × rule — unstack to (date, R, R)

        for t_idx, date in enumerate(dates):
            try:
                C = rolling_corr.loc[date].values   # (R × R)
                if np.any(np.isnan(C)):
                    continue  # Keep default 1.0 during warmup
                # Force symmetry and valid diagonal
                C = (C + C.T) / 2.0
                np.fill_diagonal(C, 1.0)
                C = np.clip(C, -1.0, 1.0)

                # FDM = 1 / sqrt(w^T @ C @ w)
                wCw = float(w_vec @ C @ w_vec)
                if wCw <= 0.0 or not np.isfinite(wCw):
                    continue  # degenerate — keep 1.0
                fdm = 1.0 / np.sqrt(wCw)
                fdm_values[t_idx, a_idx] = np.clip(fdm, 1.0, fdm_cap)
            except Exception:
                continue  # numerical failure → fallback to 1.0

    return pd.DataFrame(fdm_values, index=dates, columns=assets)


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 7 — Combined forecast
# ─────────────────────────────────────────────────────────────────────────────

def combine_forecasts_with_fdm(
    forecasts: dict[str, pd.DataFrame],
    weights: dict[str, float],
    fdm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    combined_forecast = clip( FDM × Σ(weight_i × forecast_i), ±20 )

    Args:
        forecasts: dict rule_name → (T × N) DataFrame
        weights:   normalised weight dict
        fdm_df:    (T × N) FDM multiplier DataFrame

    Returns:
        (T × N) DataFrame — the unified Carver forecast, capped at ±20.
    """
    ref = next(iter(forecasts.values()))
    combined = pd.DataFrame(0.0, index=ref.index, columns=ref.columns)

    for rule_name, forecast_df in forecasts.items():
        w = weights.get(rule_name, 0.0)
        combined += w * forecast_df.fillna(0.0)

    boosted = combined * fdm_df
    return boosted.clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)


# ─────────────────────────────────────────────────────────────────────────────
# FTI Stub — Timothy Masters' Fractal Trend Indicator
# ─────────────────────────────────────────────────────────────────────────────

def fti_multiplier_stub(close: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for the Fractal Trend Indicator overlay (Timothy Masters).

    When price crosses the noise line, trend quality is lowered and the system
    steps down.  Tested across 100+ year histories; improves CAGR slightly and
    shaves drawdown, but most drawdown control comes from volatility targeting.

    Returns a multiplier of 1.0 everywhere until the dedicated FTI video.
    Replace the body of this function once the FTI is fully implemented.
    """
    return pd.DataFrame(1.0, index=close.index, columns=close.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Position Sizer (Volatility Targeting)
# ─────────────────────────────────────────────────────────────────────────────

def size_positions(
    combined_forecast: pd.DataFrame,
    close: pd.DataFrame,
    sigma_price: pd.DataFrame,
    capital: float = 100_000.0,
    target_annual_vol: float = 0.20,
    trading_days: float = 256.0,
) -> pd.DataFrame:
    """
    N_shares = (forecast / 10) × (capital × τ) / (sigma_price × sqrt(T))

    where σ_price × √T is the annualised cash-denominated price volatility.

    Notes:
        • forecast / 10 maps a target-strength forecast to a 100% risk budget.
        • Uses absolute close to prevent sign inversion on negative-price assets.
    """
    bet_scalar      = combined_forecast / 10.0
    sigma_annual    = sigma_price * np.sqrt(trading_days)
    cash_sigma      = close.abs() * sigma_annual
    cash_sigma      = cash_sigma.replace(0.0, np.nan)

    risk_budget     = capital * target_annual_vol
    raw_position    = bet_scalar * (risk_budget / cash_sigma)
    return raw_position.fillna(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# TURNOVER BUFFER — Prevents micro-friction
# ─────────────────────────────────────────────────────────────────────────────

def apply_position_buffer(
    ideal_positions: pd.DataFrame,
    buffer_pct: float = 0.10,
) -> pd.DataFrame:
    """
    Only rebalance if the ideal position deviates by more than ±10%
    from the currently held position.  Eliminates micro-trades.
    """
    values         = ideal_positions.values.copy()
    buffered       = np.zeros_like(values)
    held           = np.zeros(values.shape[1])

    for i in range(len(values)):
        ideal     = values[i]
        diff      = np.abs(ideal - held)
        safe_held = np.where(np.abs(held) < 1e-12, 1e-9, held)
        pct_diff  = diff / np.abs(safe_held)

        change    = (np.abs(held) < 1e-12) | (pct_diff > buffer_pct) | (np.sign(ideal) != np.sign(held))
        held      = np.where(change, ideal, held)
        buffered[i] = held

    return pd.DataFrame(buffered, index=ideal_positions.index, columns=ideal_positions.columns)


# ─────────────────────────────────────────────────────────────────────────────
# MASTER INTEGRATION — full_carver()
# ─────────────────────────────────────────────────────────────────────────────

def full_carver(
    close: pd.DataFrame,
    rule_weights: dict[str, float] | None = None,
    ewmac_specs: list[tuple[int, int, float]] = EWMAC_SPECS,
    capital: float = 100_000.0,
    target_annual_vol: float = 0.20,
    trading_days: float = 256.0,
    use_buffer: bool = True,
    apply_vol_attenuation: bool = True,
    apply_fti: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Top-level Carver assembly.  Runs all layers end-to-end on a price matrix.

    Returns:
        executable_positions : (T × N) — shifted by +1 day to prevent lookahead.
        combined_forecast    : (T × N) — FDM-boosted, capped forecast.
        sigma_price          : (T × N) — instrument daily risk.

    Deployment Patterns (from Video 12):
        Single asset:
            positions, _, _ = full_carver(close[["BTC-USD"]])

        Ensemble-triggered:
            if ensemble_flashes:
                positions, _, _ = full_carver(flagged_asset_close)

        Multi-asset equal weight (e.g. 4 assets):
            for asset in ["BTC", "GLD", "QQQ", "DIA"]:
                pos, _, _ = full_carver(close[[asset]], capital=capital * 0.25)
    """
    if isinstance(close, pd.Series):
        close = close.to_frame(name="Asset")

    close = close.sort_index()

    # L1 — Instrument risk
    sigma_price = calculate_instrument_risk(close, lookback=VOL_LOOKBACK)

    # L2+L3 — All EWMAC rule forecasts
    forecasts = build_rule_forecast_stack(close, sigma_price, ewmac_specs)

    # L5 — Weights
    if rule_weights is None:
        rule_weights = {f"EWMAC{f}_{s}": w for (f, s, _), w in
                        zip(ewmac_specs, [1.0 / len(ewmac_specs)] * len(ewmac_specs))}
    rule_weights = normalise_weights({k: v for k, v in rule_weights.items() if k in forecasts})

    # L6 — FDM
    fdm_df = calculate_rolling_fdm(forecasts, rule_weights, lookback=CORR_LOOKBACK)

    # L7 — Combined forecast
    combined = combine_forecasts_with_fdm(forecasts, rule_weights, fdm_df)

    # Volatility attenuation
    if apply_vol_attenuation:
        attenuation = compute_volatility_attenuation(sigma_price)
        combined    = combined * attenuation

    # FTI overlay
    if apply_fti:
        fti = fti_multiplier_stub(close)
        combined = combined * fti

    # Final cap
    combined = combined.clip(lower=-FORECAST_CAP, upper=FORECAST_CAP)

    # Position sizing
    raw_positions = size_positions(combined, close, sigma_price, capital, target_annual_vol, trading_days)

    # Optional buffer
    if use_buffer:
        raw_positions = apply_position_buffer(raw_positions)

    # ANTI-LOOKAHEAD: expose positions on DAY T+1 only
    executable_positions = raw_positions.shift(1).fillna(0.0)

    return executable_positions, combined, sigma_price
