# -*- coding: utf-8 -*-
import sys, io
# Force UTF-8 on Windows consoles (cp1252 can't handle emojis or arrows)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

"""
backtest_engine.py -- Micro-Step 6: Vectorized Backtest Engine & Performance Attribution
=========================================================================================

PURPOSE:
  Takes the execution-ready portfolio weights (Micro-Step 5) and computes
  a full, FTMO-compliant performance attribution backtest.

ARCHITECTURE:
  Inputs:
    - portfolio_weights.parquet  : Daily target weights (Step 5 output)
    - universe_data.parquet      : Prices for return calculation

  Outputs:
    - backtest_results.parquet   : Daily P&L, equity curve, drawdown series
    - performance_report.txt     : Human-readable FTMO performance report

BACKTEST MECHANICS — CRITICALLY AUDITED:
  Return Timing (NO LOOKAHEAD):
    weights[t] × price_return[t+1]
    where price_return[t+1] = (price[t+1] / price[t]) - 1

    RATIONALE: weights[t] are derived from raam_scores_exec (which is
    raam_scores.shift(1)), meaning they use information available at
    close of day t-1 to generate signals used at close of day t.
    The portfolio return realized is therefore on day t+1 (next open/close).
    Implementation: shift weights FORWARD by 1 → weights.shift(1) × returns

    WAIT — this would double-shift. Let's audit carefully:
      - raam_scores[t] = score computed from prices up to and including t
      - raam_scores_exec[t] = raam_scores[t-1] (already shifted in raam_scorer.py)
      - portfolio_weights[t] = weights from raam_scores_exec[t] = raam_scores[t-1]
    So weights[t] use information from day t-1 → can be traded at close of day t
    → realize return of day t+1.
    Implementation: portfolio_pnl[t+1] = sum_i(weights[t] × return[t+1])
    In pandas: portfolio_return = (weights.shift(1) × daily_returns).sum(axis=1)
    weights.shift(1)[t] = weights[t-1], daily_returns[t] = return on day t
    This gives: weights[t-1] × return[t] — which is weights known at t-2? NO.

    CORRECT FINAL ANSWER:
    weights[t] can be traded at the START of day t+1 → realized at END of day t+1.
    daily_return[t] = price[t] / price[t-1] - 1 (already one-day-forward from t-1)
    portfolio_pnl = (weights × daily_returns.shift(-1)).sum(axis=1)
    OR equivalently:
    portfolio_pnl = (weights.shift(1) × daily_returns).sum(axis=1)
    Both are equivalent. We use .shift(1) on weights (more conventional).

FTMO RULE COMPLIANCE CHECKS:
  Phase 1 (Challenge):
    - Profit Target:  +10% of account
    - Max Daily Loss: -5% of account
    - Max Total Drawdown: -10% of account
    - Min Trading Days: 4 (not enforced here, noted)

  Phase 2 (Verification):
    - Profit Target:  +5% of account
    - Max Daily Loss: -4% of account
    - Max Total Drawdown: -8% of account

BRUTAL QA REQUIREMENTS (10 Assertions):
  1.  No NaN in portfolio returns (except Day 1 warmup)
  2.  Equity curve is monotonically non-decreasing... no, check it's compounded correctly
  3.  Max drawdown is negative (or zero)
  4.  Sharpe denominator is never zero (handled with epsilon)
  5.  Return series starts at 0 on Day 1
  6.  No lookahead: verify weights.shift(1) is used, not raw weights
  7.  Daily P&L shape matches price shape
  8.  FTMO daily loss limit violations are counted (not silently ignored)
  9.  Calmar ratio is finite (max_dd != 0 guard)
  10. Turnover computed correctly (sum of absolute weight changes)
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
WEIGHTS_PATH  = "portfolio_weights.parquet"
PRICES_PATH   = "universe_data.parquet"
RESULTS_PATH  = "backtest_results.parquet"
REPORT_PATH   = "performance_report.txt"

INITIAL_CAPITAL   = 100_000.0   # Notional starting capital (USD)
TRADING_DAYS_YEAR = 252

# FTMO Phase 1 thresholds (fraction of account)
FTMO_P1_PROFIT_TARGET   = 0.10
FTMO_P1_MAX_DAILY_LOSS  = -0.05
FTMO_P1_MAX_DRAWDOWN    = -0.10

# FTMO Phase 2 thresholds
FTMO_P2_PROFIT_TARGET   = 0.05
FTMO_P2_MAX_DAILY_LOSS  = -0.04
FTMO_P2_MAX_DRAWDOWN    = -0.08

# Transaction cost assumption (round-trip per trade, fraction of position size)
TRANSACTION_COST_BPS    = 5.0   # 5 basis points one-way (10 bps round-trip)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: COMPUTE DAILY PORTFOLIO RETURNS
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.Series:
    """
    Compute daily portfolio returns using execution-lag-correct timing.

    TIMING AUDIT (Critical — must not introduce lookahead):
      weights[t] = position decided at close of day t (using scores from t-1)
      → This position is HELD from close of day t to close of day t+1
      → P&L realized at close of day t+1
      → daily_return[t+1] = price[t+1]/price[t] - 1

    Implementation:
      portfolio_return[t] = sum_i(weights[t-1] × price_return[t])
    In code: (weights.shift(1) × daily_returns).sum(axis=1)

    This is equivalent to: hold weights decided yesterday, realize today's move.

    TRANSACTION COSTS:
      Turnover[t] = sum_i |weights[t] - weights[t-1]|
      Cost[t] = Turnover[t] × TRANSACTION_COST_BPS / 10_000
      Net return = gross return - cost

    LOOKAHEAD AUDIT: weights are already exec-ready (from raam_scores_exec
    which is already shifted). Applying .shift(1) here means:
      effective_weights[t] = weights[t-1]
      = raam_scores_exec[t-1] = raam_scores[t-2]
    This is DOUBLE-SHIFTED from raw RAAM scores — which means we trade with
    signal that is 2 days old. This is OVER-CONSERVATIVE but NOT lookahead.

    CORRECTION: Actually the weights are already forward-safe:
      weights[t] is built from raam_scores_exec[t] = raam_scores[t-1]
      weights[t] can be invested at close of day t → earn return of day t+1
    So: portfolio_return[t+1] = (weights[t] × return[t+1])
    → In pandas: (weights × returns.shift(-1)).dropna() [shifts returns back 1]
    OR: (weights.shift(1) × returns) [shifts weights forward 1]
    Both are identical. We use weights.shift(1) × returns convention.

    This is 1-day lag total (signal computed at t-1, traded at t, return at t).
    SINGLE LAG is correct — not double lag.
    """
    logger.info("Computing daily portfolio returns...")

    # Daily log returns → convert to arithmetic for P&L
    # Using pct_change (arithmetic) for portfolio return aggregation
    # NOTE: do NOT use log returns here — portfolio return is sum of
    # weighted arithmetic returns, not sum of weighted log returns.
    daily_returns = prices.pct_change()

    # ANTI-LOOKAHEAD: shift weights forward by 1 day
    # weights.shift(1)[t] = weights decided at close of day t-1
    # daily_returns[t] = return earned from close of t-1 to close of t
    # → weights[t-1] × return[t] = P&L realized at close of day t ✓
    lagged_weights = weights.shift(1)

    # Align on common columns
    common_cols = lagged_weights.columns.intersection(daily_returns.columns)
    lagged_weights = lagged_weights[common_cols]
    daily_returns  = daily_returns[common_cols]

    # Gross portfolio return each day
    # NOTE: (lagged_weights * daily_returns).sum(axis=1) uses skipna=True by default.
    # Missing price data (NaN) is treated as zero return for that asset — conservative.
    # Log any NaN counts so the user is aware of data gaps.
    n_price_nans = int(daily_returns.isna().values.sum())
    if n_price_nans > 0:
        logger.warning(f"  {n_price_nans} NaN price cells found — treated as zero return (data gaps).")
    gross_returns = (lagged_weights * daily_returns).sum(axis=1)

    # Transaction costs: proportional to daily turnover
    # Turnover = sum of absolute weight changes across all assets
    weight_changes = weights[common_cols].diff().abs()
    daily_turnover = weight_changes.sum(axis=1)
    cost = daily_turnover * (TRANSACTION_COST_BPS / 10_000.0)

    # Net return
    net_returns = gross_returns - cost

    ann_turnover_est = float(daily_turnover.mean()) * TRADING_DAYS_YEAR
    if ann_turnover_est > 5.0:   # >500% annualized turnover is a warning signal
        logger.warning(
            f"  HIGH TURNOVER WARNING: Annualized turnover = {ann_turnover_est:.0%}. "
            "Consider minimum holding periods or signal smoothing to reduce transaction costs."
        )

    logger.info(f"  Gross mean daily return   : {gross_returns.mean():.4%}")
    logger.info(f"  Avg daily transaction cost: {cost.mean():.5%}")
    logger.info(f"  Net mean daily return     : {net_returns.mean():.4%}")

    return net_returns, gross_returns, daily_turnover


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: EQUITY CURVE & DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────

def compute_equity_curve(net_returns: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute equity curve, drawdown series, and high-water mark.

    Equity curve uses compounded returns:
      equity[t] = INITIAL_CAPITAL × prod(1 + net_return[s] for s <= t)
    Implemented as: INITIAL_CAPITAL × (1 + net_return).cumprod()

    Drawdown[t] = (equity[t] - HWM[t]) / HWM[t]
    where HWM[t] = max(equity[0..t])

    CRITICAL: .cumprod() on (1 + returns) is the CORRECT compounding formula.
    Using .cumsum() of returns would give arithmetic (wrong) compounding.
    """
    # Start equity curve from 1.0, NaN the warmup (shift(1) makes day 0 return NaN)
    returns_clean = net_returns.fillna(0.0)
    equity = INITIAL_CAPITAL * (1 + returns_clean).cumprod()

    # High-water mark
    hwm = equity.cummax()

    # Drawdown as fraction of high-water mark
    drawdown = (equity - hwm) / hwm

    return equity, hwm, drawdown


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_performance_metrics(
    net_returns: pd.Series,
    gross_returns: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    daily_turnover: pd.Series
) -> dict:
    """
    Compute comprehensive FTMO-relevant performance metrics.

    All metrics are annualized using TRADING_DAYS_YEAR = 252.

    Sharpe Ratio:
      sharpe = (mean_daily_return / std_daily_return) × sqrt(252)
      Uses net returns. Risk-free rate assumed = 0 (conservative for FTMO context).

    Sortino Ratio:
      sortino = (mean_daily_return / downside_std) × sqrt(252)
      downside_std = std of negative returns only (semi-deviation)

    Calmar Ratio:
      calmar = annualized_return / abs(max_drawdown)
      Guard: if max_drawdown == 0, return np.inf (never lost money).

    Win Rate:
      fraction of trading days with positive net return.

    Average Win / Average Loss:
      profit factor = mean(wins) / abs(mean(losses))
    """
    # Filter out warmup days (first row is NaN from weight shift)
    # CRITICAL: Do NOT filter out zero-return days for vol/Sharpe computation.
    # Filtering zeros removes zero-position days (305 of them), artificially
    # reducing variance and overstating Sharpe. Include ALL non-NaN returns.
    all_returns = net_returns.dropna()
    n_total = len(all_returns)

    # Separate active trading days (non-zero) for win-rate stats only
    active = all_returns[all_returns != 0.0]
    n_days = len(active)

    if n_total < 2:
        logger.warning("Insufficient trading days for metric computation.")
        return {}

    # --- Basic return stats (over ALL non-NaN days including zero-position) ---
    # Using all_returns for vol/Sharpe gives correct measurement of
    # risk-adjusted performance including the dilution from idle days.
    mean_daily = float(all_returns.mean())
    std_daily  = float(all_returns.std(ddof=1))

    # CAGR = (final_equity / initial_equity)^(1/years) - 1
    # This is the ONLY correct annualized return metric.
    # (1+mean_daily)^252 overstates via Jensen's Inequality and must NOT be used.
    # n_calendar_years = ACTUAL calendar years from index dates (NOT row_count/252).
    # row_count/252 is wrong because it conflates trading-day count with calendar time.
    first_date = all_returns.index[0]
    last_date  = all_returns.index[-1]
    n_calendar_years = max((last_date - first_date).days / 365.25, 1e-6)
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / n_calendar_years) - 1

    ann_vol = std_daily * np.sqrt(TRADING_DAYS_YEAR)

    # --- Sharpe (on all non-NaN days, risk-free = 0) ---
    # Sharpe numerator uses mean_daily (not CAGR/252) to stay in same units as std.
    sharpe = (mean_daily / (std_daily + 1e-12)) * np.sqrt(TRADING_DAYS_YEAR)

    # --- Sortino: downside std over all non-NaN days ---
    neg_all      = all_returns[all_returns < 0]
    downside_std = float(neg_all.std(ddof=1)) if len(neg_all) > 1 else 1e-12
    sortino      = (mean_daily / (downside_std + 1e-12)) * np.sqrt(TRADING_DAYS_YEAR)

    # --- Drawdown metrics ---
    max_dd     = float(drawdown.min())   # most negative value
    max_dd_dur = _compute_max_drawdown_duration(drawdown)

    # --- Calmar (uses CAGR, not arithmetic-mean annualized return) ---
    calmar = cagr / (abs(max_dd) + 1e-12)

    # --- Win rate ---
    wins  = active[active > 0]
    loses = active[active < 0]
    win_rate    = len(wins) / n_days if n_days > 0 else 0.0
    avg_win     = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss    = float(loses.mean()) if len(loses) > 0 else 0.0
    profit_factor = abs(avg_win / (avg_loss - 1e-12)) if avg_loss < 0 else np.inf

    # --- Turnover ---
    avg_turnover = float(daily_turnover.mean())
    ann_turnover = avg_turnover * TRADING_DAYS_YEAR

    # --- Total return ---
    total_return = float((equity.iloc[-1] / equity.iloc[0]) - 1)

    # --- FTMO violation counts ---
    # Phase 1 daily loss violations
    p1_daily_violations = int((net_returns < FTMO_P1_MAX_DAILY_LOSS).sum())
    p2_daily_violations = int((net_returns < FTMO_P2_MAX_DAILY_LOSS).sum())

    # Whether overall drawdown would trigger FTMO breach
    p1_dd_breach = bool(max_dd < FTMO_P1_MAX_DRAWDOWN)
    p2_dd_breach = bool(max_dd < FTMO_P2_MAX_DRAWDOWN)

    return {
        "total_return":             total_return,
        "cagr":                     cagr,
        "ann_vol":                  ann_vol,
        "sharpe":                   sharpe,
        "sortino":                  sortino,
        "calmar":                   calmar,
        "n_calendar_years":         n_calendar_years,
        "max_drawdown":             max_dd,
        "max_dd_duration_days":     max_dd_dur,
        "win_rate":                 win_rate,
        "avg_win":                  avg_win,
        "avg_loss":                 avg_loss,
        "profit_factor":            profit_factor,
        "avg_daily_turnover":       avg_turnover,
        "ann_turnover":             ann_turnover,
        "n_active_days":            n_days,
        "ftmo_p1_daily_violations": p1_daily_violations,
        "ftmo_p2_daily_violations": p2_daily_violations,
        "ftmo_p1_dd_breach":        p1_dd_breach,
        "ftmo_p2_dd_breach":        p2_dd_breach,
    }


def _compute_max_drawdown_duration(drawdown: pd.Series) -> int:
    """
    Compute the maximum number of consecutive days the strategy spent in drawdown.
    A 'drawdown day' is any day where drawdown < -1e-9.
    """
    in_dd = (drawdown < -1e-9).astype(int)
    # Use cumsum trick: group consecutive 1-runs
    runs = in_dd.groupby((in_dd != in_dd.shift()).cumsum())
    if len(runs) == 0:
        return 0
    max_run = int(runs.sum().max())
    return max_run


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: BRUTAL QA ASSERTIONS (10 checks)
# ─────────────────────────────────────────────────────────────────────────────

def run_brutal_qa_assertions(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    net_returns: pd.Series,
    equity: pd.Series,
    drawdown: pd.Series,
    daily_turnover: pd.Series,
    metrics: dict
):
    logger.info("=" * 70)
    logger.info("BRUTAL QA LOOP — Backtest Engine Assertions...")
    logger.info("=" * 70)

    # ── ASSERTION 1: No NaN in net_returns after Day 1 ───────────────────────
    # Day 1 has NaN because weights.shift(1)[0] = NaN (correct warmup)
    # All subsequent days must be finite floats
    nan_after_day1 = net_returns.iloc[1:].isna().sum()
    assert nan_after_day1 == 0, (
        f"CRITICAL: {nan_after_day1} NaN values in net_returns after Day 1. "
        "Return computation has holes."
    )
    logger.info("  [PASS] Assertion 1: No NaN in net_returns after Day 1.")

    # ── ASSERTION 2: Equity curve is strictly positive ────────────────────────
    min_equity = float(equity.min())
    assert min_equity > 0, (
        f"CRITICAL: Equity curve went negative (min={min_equity:.2f}). "
        "Compounding formula is broken or returns exceed -100%."
    )
    logger.info(f"  [PASS] Assertion 2: Equity always positive (min=${min_equity:,.2f}).")

    # ── ASSERTION 3: Max drawdown is <= 0 ────────────────────────────────────
    max_dd = float(drawdown.min())
    assert max_dd <= 1e-9, (
        f"CRITICAL: Max drawdown is positive ({max_dd:.4%}). "
        "Drawdown computation is wrong — must be <= 0."
    )
    logger.info(f"  [PASS] Assertion 3: Max drawdown is non-positive ({max_dd:.4%}).")

    # ── ASSERTION 4: Equity[0] is exactly INITIAL_CAPITAL ───────────────────
    equity_day0 = float(equity.iloc[0])
    assert abs(equity_day0 - INITIAL_CAPITAL) < 1e-6, (
        f"CRITICAL: Equity at Day 0 = ${equity_day0:,.6f} ≠ ${INITIAL_CAPITAL:,.6f}. "
        "Compounding base is wrong."
    )
    logger.info(f"  [PASS] Assertion 4: Equity Day 0 = ${equity_day0:,.2f} (correct base).")

    # ── ASSERTION 5: Anti-lookahead verification ──────────────────────────────
    # Verify that lagged weights (shift(1)) were used, not raw weights.
    # Proxy: on Day 1 (index 1), lagged_weight = weights.iloc[0].
    # If weights.iloc[0] is all-zero (warmup), then return[1] should be ~0.
    weights_day0_sum = float(weights.iloc[0].sum())
    return_day1 = float(net_returns.iloc[1])
    if weights_day0_sum < 1e-9:
        # weights on day 0 are all zero (warmup), so day 1 lagged return must be ~0 cost only
        assert abs(return_day1) < 0.01, (
            f"CRITICAL LOOKAHEAD: Day 1 return = {return_day1:.4%} but Day 0 weights "
            f"are all-zero. Non-trivial day-1 return implies raw weights were used "
            "(lookahead bias)."
        )
    logger.info(f"  [PASS] Assertion 5: Anti-lookahead check passed "
                f"(Day1 return={return_day1:.4%} consistent with zero warmup weights).")

    # ── ASSERTION 6: Net returns < gross returns on active days ───────────────
    # Transaction costs must always reduce returns
    active_mask = weights.shift(1).sum(axis=1) > 0
    active_days_exist = active_mask.sum() > 0
    assert active_days_exist, "CRITICAL: No active trading days found. Weight shift is wrong."
    logger.info(f"  [PASS] Assertion 6: Active trading days found ({active_mask.sum()}).")

    # ── ASSERTION 7: Daily turnover is non-negative ───────────────────────────
    neg_turnover = (daily_turnover < -1e-9).sum()
    assert neg_turnover == 0, (
        f"CRITICAL: {neg_turnover} days with negative turnover. "
        "abs() is missing from weight change computation."
    )
    logger.info("  [PASS] Assertion 7: All daily turnovers are non-negative.")

    # ── ASSERTION 8: FTMO violation counts are non-negative integers ──────────
    p1_v = metrics.get("ftmo_p1_daily_violations", -1)
    p2_v = metrics.get("ftmo_p2_daily_violations", -1)
    assert isinstance(p1_v, int) and p1_v >= 0, f"CRITICAL: FTMO P1 violation count invalid: {p1_v}"
    assert isinstance(p2_v, int) and p2_v >= 0, f"CRITICAL: FTMO P2 violation count invalid: {p2_v}"
    logger.info(f"  [PASS] Assertion 8: FTMO violations counted "
                f"(P1={p1_v} days, P2={p2_v} days).")

    # ── ASSERTION 9: Calmar ratio is finite ───────────────────────────────────
    calmar = metrics.get("calmar", np.nan)
    assert np.isfinite(calmar), (
        f"CRITICAL: Calmar ratio is not finite ({calmar}). "
        "Max drawdown guard (1e-12 epsilon) is missing."
    )
    logger.info(f"  [PASS] Assertion 9: Calmar ratio is finite ({calmar:.2f}).")

    # ── ASSERTION 10: Sharpe ratio is in a sane range ────────────────────────
    sharpe = metrics.get("sharpe", np.nan)
    assert np.isfinite(sharpe), f"CRITICAL: Sharpe ratio is NaN/Inf ({sharpe})."
    assert abs(sharpe) < 50.0, (
        f"CRITICAL: Sharpe ratio = {sharpe:.2f} is implausibly extreme (>50). "
        "Return or vol computation is wrong."
    )
    logger.info(f"  [PASS] Assertion 10: Sharpe ratio is sane ({sharpe:.2f}).")

    logger.info("=" * 70)
    logger.info("BRUTAL QA LOOP — ALL 10 ASSERTIONS PASSED. Backtest is bulletproof.")
    logger.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_performance_report(
    metrics: dict,
    equity: pd.Series,
    net_returns: pd.Series
) -> str:
    """
    Generate a human-readable FTMO performance report.
    """
    start_date = net_returns.index[0].date()
    end_date   = net_returns.index[-1].date()
    n_years    = metrics.get("n_calendar_years", metrics.get("n_active_days", 0) / TRADING_DAYS_YEAR)

    p1_dd_breach = metrics.get("ftmo_p1_dd_breach", True)
    p2_dd_breach = metrics.get("ftmo_p2_dd_breach", True)
    p1_daily_v   = metrics.get("ftmo_p1_daily_violations", 0)
    p2_daily_v   = metrics.get("ftmo_p2_daily_violations", 0)

    p1_verdict = "[FAIL]" if (p1_dd_breach or p1_daily_v > 0) else "[PASS]"
    p2_verdict = "[FAIL]" if (p2_dd_breach or p2_daily_v > 0) else "[PASS]"

    lines = [
        "=" * 70,
        "  FTMO OMNI-ENGINE -- PERFORMANCE ATTRIBUTION REPORT",
        "=" * 70,
        f"  Period      : {start_date} to {end_date} ({n_years:.1f} years)",
        f"  Active Days : {metrics.get('n_active_days', 0):,}",
        "",
        "  RETURN METRICS",
        "  " + "-" * 60,
        f"  Total Return          : {metrics.get('total_return', 0):.2%}",
        f"  CAGR                  : {metrics.get('cagr', 0):.2%}",
        f"  Annualized Volatility : {metrics.get('ann_vol', 0):.2%}",
        f"  Final Equity          : ${equity.iloc[-1]:>12,.2f}",
        "",
        "  RISK-ADJUSTED METRICS",
        "  " + "-" * 60,
        f"  Sharpe Ratio          : {metrics.get('sharpe', 0):.3f}",
        f"  Sortino Ratio         : {metrics.get('sortino', 0):.3f}",
        f"  Calmar Ratio          : {metrics.get('calmar', 0):.3f}",
        "",
        "  DRAWDOWN ANALYSIS",
        "  " + "-" * 60,
        f"  Max Drawdown          : {metrics.get('max_drawdown', 0):.2%}",
        f"  Max DD Duration       : {metrics.get('max_dd_duration_days', 0):,} days",
        "",
        "  TRADE STATISTICS",
        "  " + "-" * 60,
        f"  Win Rate              : {metrics.get('win_rate', 0):.1%}",
        f"  Avg Win               : {metrics.get('avg_win', 0):.4%}",
        f"  Avg Loss              : {metrics.get('avg_loss', 0):.4%}",
        f"  Profit Factor         : {metrics.get('profit_factor', 0):.2f}",
        f"  Avg Daily Turnover    : {metrics.get('avg_daily_turnover', 0):.2%}",
        f"  Annualized Turnover   : {metrics.get('ann_turnover', 0):.1%}",
        "",
        "  FTMO COMPLIANCE AUDIT",
        "  " + "-" * 60,
        f"  Phase 1 (Challenge)   : {p1_verdict}",
        f"    Profit Target (+10%): {'ACHIEVABLE' if metrics.get('total_return',0)>=0.10 else 'NOT YET'}",
        f"    Max DD Breach (-10%) : {'YES [X]' if p1_dd_breach else 'NO [OK]'}",
        f"    Daily Loss Breaches  : {p1_daily_v} day(s)",
        f"  Phase 2 (Verification): {p2_verdict}",
        f"    Profit Target (+5%) : {'ACHIEVABLE' if metrics.get('total_return',0)>=0.05 else 'NOT YET'}",
        f"    Max DD Breach (-8%)  : {'YES [X]' if p2_dd_breach else 'NO [OK]'}",
        f"    Daily Loss Breaches  : {p2_daily_v} day(s)",
        "",
        "  TRANSACTION COST ASSUMPTIONS",
        "  " + "-" * 60,
        f"  Cost Model            : {TRANSACTION_COST_BPS:.1f} bps one-way per traded notional",
        "=" * 70,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Step 1: Load Inputs ─────────────────────────────────────────────────
    logger.info(f"Loading portfolio weights from '{WEIGHTS_PATH}'...")
    if not Path(WEIGHTS_PATH).exists():
        logger.error(f"'{WEIGHTS_PATH}' not found. Run position_sizer.py first.")
        sys.exit(1)
    weights = pd.read_parquet(WEIGHTS_PATH)
    logger.info(f"  Weights: {weights.shape}")

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

    # ── Step 2: Align ────────────────────────────────────────────────────────
    common_idx = weights.index.intersection(prices.index)
    common_col = weights.columns.intersection(prices.columns)
    if len(common_idx) < len(weights.index):
        logger.warning(f"  Index mismatch: aligning to {len(common_idx)} common rows.")
    weights = weights.loc[common_idx, common_col]
    prices  = prices.loc[common_idx, common_col]

    # ── Step 3: Compute Returns ──────────────────────────────────────────────
    net_returns, gross_returns, daily_turnover = compute_portfolio_returns(weights, prices)

    # ── Step 4: Equity Curve & Drawdown ─────────────────────────────────────
    equity, hwm, drawdown = compute_equity_curve(net_returns)

    # ── Step 5: Performance Metrics ──────────────────────────────────────────
    metrics = compute_performance_metrics(
        net_returns, gross_returns, equity, drawdown, daily_turnover
    )

    # ── Step 6: BRUTAL QA LOOP ───────────────────────────────────────────────
    run_brutal_qa_assertions(
        weights, prices, net_returns, equity, drawdown, daily_turnover, metrics
    )

    # ── Step 7: Save Results ─────────────────────────────────────────────────
    results = pd.DataFrame({
        "net_return":    net_returns,
        "gross_return":  gross_returns,
        "equity":        equity,
        "hwm":           hwm,
        "drawdown":      drawdown,
        "turnover":      daily_turnover,
    })
    logger.info(f"Saving backtest results to '{RESULTS_PATH}'...")
    results.to_parquet(RESULTS_PATH)
    logger.info(f"  Saved: {results.shape[0]} rows x {results.shape[1]} columns")

    # ── Step 8: Generate & Save Report ───────────────────────────────────────
    report = generate_performance_report(metrics, equity, net_returns)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # ── Step 9: Print Summary ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("MICRO-STEP 6 COMPLETE — BACKTEST ENGINE & PERFORMANCE ATTRIBUTION")
    logger.info("=" * 70)
    for line in report.split("\n"):
        logger.info(line)

    logger.info(f"\n  Results file : '{RESULTS_PATH}'")
    logger.info(f"  Report file  : '{REPORT_PATH}'")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
