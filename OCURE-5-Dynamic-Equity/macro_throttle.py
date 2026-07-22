import numpy as np
import pandas as pd

class MacroThrottle:
    """
    Multi-Factor Continuous Macro Throttle with Hysteresis.
    Replaces binary 200MA exit switches with a smooth, continuous exposure multiplier m_t in [0.25, 1.0].
    Works robustly across full 26-year dataset (2000-2026).
    """
    def __init__(self, m_min=0.35, tau=1.0, theta_on=0.1, theta_off=-0.2, smooth_span=10):
        self.m_min = m_min
        self.tau = tau
        self.theta_on = theta_on
        self.theta_off = theta_off
        self.smooth_span = smooth_span

    def compute_throttle(self, btc_prices, spy_prices, hyg_prices=None, ief_prices=None, panel_prices=None):
        safe_spy = spy_prices.copy()
        if safe_spy.index.tz is not None:
            safe_spy.index = safe_spy.index.tz_localize(None)
        safe_spy = safe_spy.replace(0, np.nan).ffill()

        # 1. SPY 200MA Distance Z-score (Continuous from 2000 to 2026)
        ma200_spy = safe_spy.rolling(200, min_periods=50).mean()
        std200_spy = safe_spy.rolling(200, min_periods=50).std().replace(0, np.nan)
        z_spy = (safe_spy - ma200_spy) / std200_spy
        scores = [z_spy.fillna(0.0)]

        # 2. BTC/SPY Relative Distance Z-score (When available post-2014)
        if btc_prices is not None:
            safe_btc = btc_prices.copy()
            if safe_btc.index.tz is not None:
                safe_btc.index = safe_btc.index.tz_localize(None)
            safe_btc = safe_btc.reindex(safe_spy.index).ffill()
            
            ratio_btc_spy = safe_btc / safe_spy
            ma200_btc = ratio_btc_spy.rolling(200, min_periods=50).mean()
            std200_btc = ratio_btc_spy.rolling(200, min_periods=50).std().replace(0, np.nan)
            z_btc = (ratio_btc_spy - ma200_btc) / std200_btc
            scores.append(z_btc.fillna(0.0))

        # 3. Credit Risk (HYG / IEF, when available post-2007)
        if hyg_prices is not None and ief_prices is not None:
            safe_hyg = hyg_prices.copy()
            safe_ief = ief_prices.copy()
            if safe_hyg.index.tz is not None:
                safe_hyg.index = safe_hyg.index.tz_localize(None)
            if safe_ief.index.tz is not None:
                safe_ief.index = safe_ief.index.tz_localize(None)

            safe_hyg = safe_hyg.reindex(safe_spy.index).ffill()
            safe_ief = safe_ief.reindex(safe_spy.index).ffill()

            credit_ratio = safe_hyg / safe_ief.replace(0, np.nan)
            ma50_credit = credit_ratio.rolling(50, min_periods=20).mean()
            std50_credit = credit_ratio.rolling(50, min_periods=20).std().replace(0, np.nan)
            z_credit = (credit_ratio - ma50_credit) / std50_credit
            scores.append(z_credit.fillna(0.0))

        # 4. Market Breadth (% assets > 50MA)
        if panel_prices is not None:
            safe_panel = panel_prices.copy()
            if safe_panel.index.tz is not None:
                safe_panel.index = safe_panel.index.tz_localize(None)
            safe_panel = safe_panel.reindex(safe_spy.index).ffill()

            ma50_panel = safe_panel.rolling(50, min_periods=20).mean()
            above_50 = (safe_panel > ma50_panel).astype(float)
            breadth = above_50.mean(axis=1) # fraction 0.0 to 1.0
            z_breadth = (breadth - 0.5) * 4.0 # scale to approx [-2, +2]
            scores.append(z_breadth.fillna(0.0))

        # Composite Score Z_t
        score_df = pd.concat(scores, axis=1).ffill().fillna(0.0)
        Z = score_df.mean(axis=1)

        # Hysteresis Filter
        regime_state = np.zeros(len(Z))
        curr_state = 1.0 # Start Risk-On
        for i in range(len(Z)):
            val = Z.iloc[i]
            if curr_state == 1.0:
                if val < self.theta_off:
                    curr_state = 0.0
            else:
                if val > self.theta_on:
                    curr_state = 1.0
            regime_state[i] = curr_state

        regime_series = pd.Series(regime_state, index=Z.index)

        # Continuous Sigmoid Mapping
        raw_m = self.m_min + (1.0 - self.m_min) / (1.0 + np.exp(-Z / self.tau))
        
        # Apply hysteresis bound
        m_t = np.where(regime_series == 1.0, raw_m, self.m_min)
        m_series = pd.Series(m_t, index=Z.index)

        # EWMA Smoothing to kill turnover spikes
        smoothed_m = m_series.ewm(span=self.smooth_span, adjust=False).mean()
        return smoothed_m.clip(self.m_min, 1.0)
