import numpy as np
import pandas as pd

# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
ANN_DAYS   = 365      # Crypto trades 365 days a year
FC_TARGET  = 10.0     
FC_CAP     = 20.0     
VOL_TARGET = 0.20     
EWMAC_PAIRS = [(8, 32, 5.95), (16, 64, 4.10), (32, 128, 2.79), (64, 256, 1.91)]

# =========================================================
# UTILITIES: VOLATILITY ESTIMATION (Prerequisite)
# =========================================================
def ewm_std(x, span):
    """Exponentially weighted standard deviation."""
    a = 2.0 / (span + 1.0)
    m  = x.ewm(alpha=a, adjust=False).mean()
    m2 = (x * x).ewm(alpha=a, adjust=False).mean()
    var = (m2 - m * m).clip(lower=0.0)
    bc = (2.0 - 2.0 * a) / (2.0 - a)
    return np.sqrt(var / bc)

def vol_stack(close, ann_days=ANN_DAYS, short_span=30, anchor_years=5,
              anchor_min_years=1, short_w=0.70, long_w=0.30):
    """Combines short-term EWM vol and long-term average vol."""
    ann = np.sqrt(ann_days)
    ret = close.pct_change().clip(-0.9, 9.0)
    vol_short = ewm_std(ret, short_span) * ann
    vol_long  = vol_short.rolling(anchor_years * ann_days,
                                  min_periods=anchor_min_years * ann_days).mean()
    
    vol = pd.Series(
        np.where(vol_long.isna(), vol_short, short_w * vol_short + long_w * vol_long),
        index=close.index, 
        name="vol"
    )
    sigma_p = (close * vol / ann).rename("sigma_p")
    
    return pd.DataFrame({
        "ret": ret, 
        "vol_short": vol_short, 
        "vol_long": vol_long,
        "vol": vol, 
        "sigma_p": sigma_p
    })

# =========================================================
# LAYER 1: EWMAC (Exponentially Weighted Moving Average Crossover)
# =========================================================
def ewmac_forecast(close, sigma_p, pairs=EWMAC_PAIRS, fdm=1.13, cap=FC_CAP):
    """Captures primary trends across 4 timeframes."""
    subs = []
    for f, s, sc in pairs:
        raw = (close.ewm(span=f, adjust=False).mean() - close.ewm(span=s, adjust=False).mean()) / sigma_p
        subs.append((raw * sc).clip(-cap, cap))
    return (sum(subs) / len(subs) * fdm).clip(-cap, cap)

# =========================================================
# LAYER 2: BREAKOUT
# =========================================================
def breakout_single(close, n, sc, sm, cap=FC_CAP):
    mn, mx = close.rolling(n).min(), close.rolling(n).max()
    rng = mx - mn
    spir = ((close - mn) / rng).where(rng > 0, 0.5)
    return ((spir - 0.5) * 40.0).ewm(span=sm, adjust=False).mean().mul(sc).clip(-cap, cap)

def breakout_forecast(close, fdm=1.17, cap=FC_CAP):
    """Price channel breakout normalized by volatility."""
    P = [(40, 0.70, 10), (80, 0.73, 20), (160, 0.74, 40), (320, 0.74, 80)]
    subs = [breakout_single(close, n, sc, sm) for n, sc, sm in P]
    return (sum(subs) / len(subs) * fdm).clip(-cap, cap)

# =========================================================
# LAYER 3: ACCELERATION (2nd Derivative)
# =========================================================
def ewmac_base(close, sigma_p, f, s, sc, cap=FC_CAP):
    return ((close.ewm(span=f, adjust=False).mean()
             - close.ewm(span=s, adjust=False).mean()) / sigma_p * sc).clip(-cap, cap)

def accel_forecast(close, sigma_p, fdm=1.55, cap=FC_CAP):
    """Measures the momentum of the EWMAC signals themselves."""
    bases = [ewmac_base(close, sigma_p, f, s, sc) for f, s, sc in EWMAC_PAIRS]
    AP, SC = [8, 16, 32, 64], [1.87, 1.90, 1.98, 2.05]
    subs = [((bases[i] - bases[i].shift(AP[i])) * SC[i]).clip(-cap, cap) for i in range(4)]
    return (sum(subs) / len(subs) * fdm).clip(-cap, cap)

# =========================================================
# LAYER 4: SKEW (The Diversifier)
# =========================================================
def skew_forecast(ret, fdm=1.18, cap=FC_CAP):
    """Goes long assets with negative skew to exploit mean reversion in panic selling."""
    P = [(60, 33.3, 15), (120, 37.2, 30), (240, 39.2, 60)]
    subs = []
    for w, sc, sm in P:
        g = ret.rolling(w, min_periods=w // 2).skew()
        subs.append((-g).ewm(span=sm, adjust=False).mean().mul(sc).clip(-cap, cap))
    return (sum(subs) / len(subs) * fdm).clip(-cap, cap)

# =========================================================
# LAYER 5: VOLATILITY ATTENUATION (Defense)
# =========================================================
def vol_attenuation(vol, window=1260, smooth=10):
    """Reduces position size during extreme high-volatility regimes."""
    p = vol.rolling(window, min_periods=ANN_DAYS).rank(pct=True)
    p = p.ewm(span=smooth, adjust=False).mean()
    return (1.5 - p).clip(0.5, 1.5).fillna(1.0).rename("vol_att")

# =========================================================
# LAYER 6: FORECAST COMBINATION (FDM)
# =========================================================
def fdm_from_corr(w, C, fdm_max=2.5):
    """Calculates Forecast Diversification Multiplier."""
    w = np.asarray(w, float)
    w = w / w.sum()
    var = float(w @ C @ w)
    return float(min(fdm_max, 1.0 / np.sqrt(var))) if var > 0 else 1.0

def combine_forecasts(forecasts, weights, cap=FC_CAP):
    """Blends the active forecasts using a correlation-aware multiplier."""
    names = list(forecasts.keys())
    F = pd.concat([forecasts[n].rename(n) for n in names], axis=1)
    
    w = np.array([weights[n] for n in names], float)
    w = w / w.sum()
    
    C = F.dropna().corr().reindex(index=names, columns=names).values
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    
    fdm = fdm_from_corr(w, C)
    combined = (F.mul(w, axis=1).sum(axis=1) * fdm).clip(-cap, cap)
    return combined, fdm, F.dropna().corr()

# =========================================================
# LAYER 7: SIZING (Forecast -> Weight)
# =========================================================
def position_from_forecast(forecast, vol, vol_target=VOL_TARGET, fc_target=FC_TARGET, long_only=True):
    """Converts a forecast into a portfolio weight based on volatility targeting."""
    w = (forecast / fc_target) * (vol_target / vol)
    w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return w.clip(0.0 if long_only else -1.0, 1.0).rename("weight")

# =========================================================
# LAYER 8: CROSS-SECTIONAL MOMENTUM (The Secret Sauce)
# =========================================================
def normalised_price(close):
    v = vol_stack(close); ann = np.sqrt(ANN_DAYS)
    rn = (v["ret"] / (v["vol"] / ann)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return (100.0 * rn.cumsum()).rename("PN")

def cs_momentum_forecast(panel, target, horizons=(40, 80), fam_fdm=1.10, cap=FC_CAP):
    """Cross Sectional Momentum against a panel of peers."""
    PN = pd.DataFrame({c: normalised_price(panel[c]) for c in panel.columns})
    A  = PN.mean(axis=1)
    R  = PN[target] - A
    
    subs, scs = [], []
    for H in horizons:
        raw = (R - R.shift(H)) / H
        sm  = raw.ewm(span=max(2, H // 4), adjust=False).mean()
        sc  = 10.0 / sm.abs().mean() if sm.abs().mean() > 0 else 1.0
        scs.append(sc)
        subs.append((sm * sc).clip(-cap, cap))
        
    cs = (sum(subs) / len(subs) * fam_fdm).clip(-cap, cap)
    return cs.reindex(panel.index).rename("CSmom")

# =========================================================
# FULL 8-LAYER ENGINE ORCHESTRATION
# =========================================================
def build_carver_engine(panel, target, use_cs=True, long_only=True):
    """
    Orchestrates the 8-Layer Carver ruleset to generate target position weights.
    
    Args:
        panel (pd.DataFrame): DataFrame containing closing prices for the universe.
        target (str): Column name of the asset to trade.
        use_cs (bool): Whether to include Layer 8 (Cross-Sectional Momentum).
        long_only (bool): If True, clips weights to [0, 1]. If False, clips to [-1, 1].
        
    Returns:
        pd.Series: The daily target weights for the asset.
    """
    close = panel[target].dropna()
    vs = vol_stack(close)
    
    # Layers 1-4: Generate individual forecasts
    forecasts = {
        "EWMAC": ewmac_forecast(close, vs.sigma_p),        # Layer 1
        "Breakout": breakout_forecast(close),              # Layer 2
        "Accel": accel_forecast(close, vs.sigma_p),        # Layer 3
        "Skew": skew_forecast(vs.ret)                      # Layer 4
    }
    
    weights = {
        "EWMAC": 0.15, 
        "Breakout": 0.15, 
        "Accel": 0.15, 
        "Skew": 0.20
    }
    
    # Layer 8: Cross-Sectional Momentum
    if use_cs:
        forecasts["CSmom"] = cs_momentum_forecast(panel, target).reindex(close.index)
        weights["CSmom"] = 0.15
        
    # Layer 6: Combine Forecasts (Correlation-aware)
    combined, _, _ = combine_forecasts(forecasts, weights)
    
    # Layer 5: Volatility Attenuation (Defense)
    fc_att = combined * vol_attenuation(vs.vol)
    fc_att = fc_att.clip(-FC_CAP, FC_CAP)
    
    # Layer 7: Volatility Targeting to Position Size (Forecast -> Weight)
    w = position_from_forecast(fc_att, vs.vol, long_only=long_only)
    return w
