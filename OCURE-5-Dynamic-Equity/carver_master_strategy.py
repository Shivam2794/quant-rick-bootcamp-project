import numpy as np
import pandas as pd
from scipy.signal import savgol_coeffs
import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# CARVER MASTER STRATEGY: LAYERED SYSTEM ARCHITECTURE
# Translated from Quant Bootcamp 2.0
# ==============================================================================

class CarverSystem:
    def __init__(self, target_vol=0.20, capital=100000, ann_days=252):
        """
        Initializes the Carver Trading System.
        target_vol: The annualized portfolio volatility target.
        capital: Total portfolio capital.
        ann_days: Annualization factor (e.g., 252 for equities, 365 for crypto).
        """
        self.target_vol = target_vol
        self.capital = capital
        self.ann_days = ann_days
        self.fc_target = 10.0
        self.fc_cap = 20.0
        
        # Layer parameters
        self.ewmac_pairs = [(8, 32, 5.95), (16, 64, 4.10), (32, 128, 2.79), (64, 256, 1.91)]
        self.ewmac_fdm = 1.13
        
        self.breakout_params = [(40, 0.70, 10), (80, 0.73, 20), (160, 0.74, 40), (320, 0.74, 80)]
        self.breakout_fdm = 1.17
        
        self.accel_fdm = 1.55
        
        self.skew_params = [(60, 33.3, 15), (120, 37.0, 20), (240, 39.2, 60)]
        self.skew_fdm = 1.18

    def _clip_fc(self, fc):
        """Clips forecast to the max cap (-20 to +20)."""
        return fc.clip(-self.fc_cap, self.fc_cap)

    # --------------------------------------------------------------------------
    # LAYER 0: VOLATILITY STACK & SIGMA P
    # --------------------------------------------------------------------------
    def _ewm_std(self, x, span):
        """Carver bias-corrected EWM std."""
        a = 2.0 / (span + 1.0)
        m = x.ewm(alpha=a, adjust=False).mean()
        m2 = (x ** 2).ewm(alpha=a, adjust=False).mean()
        var = (m2 - m ** 2).clip(lower=0.0)
        bc = (2.0 - 2.0 * a) / (2.0 - a)
        return np.sqrt(var / bc)

    def vol_stack(self, close, short_span=30, anchor_years=5):
        """
        Calculates the blended volatility denominator (Sigma P) and annualized vol.
        """
        ann = np.sqrt(self.ann_days)
        ret = close.pct_change().clip(-0.9, 9.0) # winsorized
        
        vol_short = self._ewm_std(ret, short_span) * ann
        vol_long = vol_short.rolling(anchor_years * self.ann_days, 
                                     min_periods=anchor_years * self.ann_days).mean()
        
        # Blend 70% short, 30% long
        vol = pd.Series(np.where(vol_long.isna(), vol_short, 0.70 * vol_short + 0.30 * vol_long),
                        index=close.index, name="vol")
        
        sigma_p = (close * vol / ann).rename("sigma_p")
        return vol, sigma_p

    # --------------------------------------------------------------------------
    # LAYER 1: EWMAC (Exponentially Weighted Moving Average Crossover)
    # --------------------------------------------------------------------------
    def _ewmac_base(self, close, sigma_p, f, s, sc):
        """Raw scaled difference for EWMAC and Acceleration base."""
        diff = close.ewm(span=f, adjust=False).mean() - close.ewm(span=s, adjust=False).mean()
        return self._clip_fc((diff / sigma_p) * sc)

    def layer1_ewmac_forecast(self, close, sigma_p):
        subs = []
        for f, s, sc in self.ewmac_pairs:
            subs.append(self._ewmac_base(close, sigma_p, f, s, sc))
        return self._clip_fc((sum(subs) / len(subs)) * self.ewmac_fdm)

    # --------------------------------------------------------------------------
    # LAYER 2: BREAKOUT ENGINE
    # --------------------------------------------------------------------------
    def _breakout_single(self, close, n, sc, sm):
        mn, mx = close.rolling(n).min(), close.rolling(n).max()
        rng = mx - mn
        pdir = ((close - mn) / rng).where(rng > 0, 0.5)
        return self._clip_fc(((pdir - 0.5) * 40.0).ewm(span=sm, adjust=False).mean().mul(sc))

    def layer2_breakout_forecast(self, close):
        subs = [self._breakout_single(close, n, sc, sm) for n, sc, sm in self.breakout_params]
        return self._clip_fc((sum(subs) / len(subs)) * self.breakout_fdm)

    # --------------------------------------------------------------------------
    # LAYER 3: ACCELERATION ENGINE
    # --------------------------------------------------------------------------
    def layer3_acceleration_forecast(self, close, sigma_p):
        bases = [self._ewmac_base(close, sigma_p, f, s, sc) for f, s, sc in self.ewmac_pairs]
        AP, SC = (8, 16, 32, 64), (1.87, 1.90, 1.98, 2.05)
        
        subs = []
        for i in range(4):
            diff = bases[i] - bases[i].shift(AP[i])
            subs.append(self._clip_fc(diff * SC[i]))
            
        return self._clip_fc((sum(subs) / len(subs)) * self.accel_fdm)

    # --------------------------------------------------------------------------
    # LAYER 4: SKEWNESS
    # --------------------------------------------------------------------------
    def layer4_skew_forecast(self, close):
        ret = close.pct_change()
        subs = []
        for w, st, sm in self.skew_params:
            g1 = ret.rolling(w, min_periods=w // 2).skew() # bias-corrected G1
            # Negate: we want to fade negative skew
            neg_skew = -g1
            subs.append(self._clip_fc(neg_skew.ewm(span=sm, adjust=False).mean().mul(st)))
            
        return self._clip_fc((sum(subs) / len(subs)) * self.skew_fdm)

    # --------------------------------------------------------------------------
    # LAYER 5: VOLATILITY ATTENUATION
    # --------------------------------------------------------------------------
    def layer5_vol_attenuation(self, vol, window=1260, smooth=10):
        p_vol = vol.rolling(window, min_periods=self.ann_days).rank(pct=True)
        p_vol = p_vol.ewm(span=smooth, adjust=False).mean()
        return (1.5 - p_vol).clip(0.5, 1.5).fillna(1.0).rename("vol_att")

    # --------------------------------------------------------------------------
    # LAYER 6: BETA ROTATIONS (MACRO REGIME FILTER)
    # --------------------------------------------------------------------------
    def causal_savgol(self, series, window=21, polyorder=2):
        """Zero look-ahead Savitzky-Golay smoother."""
        v = series.values.astype(float)
        out = np.full(v.shape, np.nan)
        c = savgol_coeffs(window, polyorder, pos=window - 1, use='dot')
        for t in range(window - 1, len(v)):
            seg = v[t - window + 1: t + 1]
            if not np.isnan(seg).any():
                out[t] = np.dot(c, seg)
        return pd.Series(out, index=series.index)

    def macro_regime_filter(self, btc_close, spy_close):
        """
        Calculates a Risk-On / Risk-Off regime based on BTC/SPY 200MA Cross.
        If BTC/SPY ratio drops below its 200-day moving average, force Risk-Off (total exit).
        Regime 1: Risk-On (Ratio >= 200MA)
        Regime 2: Risk-Off (Ratio < 200MA)
        """
        # Strip timezones and reindex to avoid 100% NaN silent failure
        safe_btc = btc_close.copy()
        safe_spy = spy_close.copy()
        
        if safe_btc.index.tz is not None:
            safe_btc.index = safe_btc.index.tz_localize(None)
        if safe_spy.index.tz is not None:
            safe_spy.index = safe_spy.index.tz_localize(None)
            
        safe_spy = safe_spy.reindex(safe_btc.index).ffill()
        
        safe_btc = safe_btc.replace(0, np.nan).ffill()
        safe_spy = safe_spy.replace(0, np.nan).ffill()
        
        ratio = safe_btc / safe_spy
        ma_200 = ratio.rolling(200, min_periods=100).mean()
        
        # 1 = Risk On, 2 = Risk Off
        regime = pd.Series(np.where(ratio < ma_200, 2, 1), index=ratio.index)
        return regime

    # --------------------------------------------------------------------------
    # LAYER 7: CROSS-SECTIONAL MOMENTUM
    # --------------------------------------------------------------------------
    def _normalised_price(self, ret, vol_ann):
        safe_vol = vol_ann.replace(0.0, np.nan).ffill().fillna(1.0)
        r_norm = ret / (safe_vol / np.sqrt(self.ann_days))
        r_norm = r_norm.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return 100.0 + r_norm.cumsum()

    def layer7_cs_momentum_forecast(self, panel_prices, target_name):
        panel_ret = panel_prices.pct_change(fill_method=None).fillna(0.0)
        vol_pnl = pd.DataFrame()
        for col in panel_ret.columns:
            # We use vol_stack to get annualized vol
            vol_col, _ = self.vol_stack(panel_prices[col])
            vol_pnl[col] = self._normalised_price(panel_ret[col], vol_col)
        
        p_target = vol_pnl[target_name]
        p_class = vol_pnl.mean(axis=1)
        r = p_target - p_class # relative price
        
        fcs = []
        for h in [40, 80]:
            outperf = r.diff(1)
            f_raw = outperf.ewm(span=h, adjust=False).mean()
            # Expanding scaling constant to prevent lookahead bias
            scalar = self.fc_target / f_raw.abs().expanding(min_periods=self.ann_days).mean()
            scalar = scalar.replace([np.inf, -np.inf], 1.0)
            f_scaled = self._clip_fc(f_raw * scalar)
            fcs.append(f_scaled)
            
        # 1.1 is the within-family FDM for 2 low-corr assets
        fc_strat = (fcs[0] + fcs[1]) / 2.0 * 1.1
        return self._clip_fc(fc_strat)

    # --------------------------------------------------------------------------
    # COMBINATION ENGINE (DYNAMIC FDM)
    # --------------------------------------------------------------------------
    def fdm_from_corr(self, w, corr_matrix):
        w_arr = np.array(w)
        var = w_arr.T @ corr_matrix.values @ w_arr
        return float(1.0 / np.sqrt(var)) if var > 0 else 1.0

    def combine_forecasts(self, forecasts, weights):
        names = list(forecasts.keys())
        # Force all forecasts to 1D arrays/Series before concatenating to avoid multi-column issues
        clean_forecasts = [forecasts[n].iloc[:, 0] if isinstance(forecasts[n], pd.DataFrame) else forecasts[n] for n in names]
        f_df = pd.concat(clean_forecasts, axis=1)
        f_df.columns = names
        
        w_arr = np.array([weights[n] for n in names])
        w_arr = w_arr / w_arr.sum()
        
        # Table 52 Fixed Correlation Matrix
        c_fixed = pd.DataFrame(1.0, index=names, columns=names)
        fixed_corrs = {
            ("ewmac", "breakout"): 0.55,
            ("ewmac", "accel"): 0.25,
            ("ewmac", "skew"): 0.00,
            ("ewmac", "cs_mom"): 0.10,
            ("breakout", "accel"): 0.15,
            ("breakout", "skew"): 0.00,
            ("breakout", "cs_mom"): 0.10,
            ("accel", "skew"): 0.00,
            ("accel", "cs_mom"): 0.10,
            ("skew", "cs_mom"): 0.00,
        }
        
        for (f1, f2), val in fixed_corrs.items():
            if f1 in c_fixed.columns and f2 in c_fixed.columns:
                c_fixed.loc[f1, f2] = val
                c_fixed.loc[f2, f1] = val
                
        fdm = self.fdm_from_corr(w_arr, c_fixed)
        
        # sum(axis=1) will ignore NA by default, but we should just multiply and sum
        combined = (f_df * w_arr).sum(axis=1, skipna=False) * fdm
        return self._clip_fc(combined), fdm, c_fixed

    # --------------------------------------------------------------------------
    # FINAL: DEMOCRATIC VOTING & POSITION SIZING
    # --------------------------------------------------------------------------
    def position_from_forecast(self, forecast, vol, long_only=True, current_drawdown=None):
        """weight = (forecast / 10) * (vol_target / vol), with optional Soft CPPI drawdown scaling."""
        target_v = self.target_vol
        if current_drawdown is not None:
            # Soft CPPI: target_vol = target_vol * max(0.3, 1 - 1.2 * (DD / 0.15))
            dd_factor = np.maximum(0.3, 1.0 - 1.2 * (np.abs(current_drawdown) / 0.15))
            target_v = target_v * dd_factor

        w = (forecast / self.fc_target) * (target_v / vol)
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w.clip(0.0, 1.0) if long_only else w.clip(-1.0, 1.0)

    def apply_position_buffer(self, target_weights, buffer_threshold=0.10):
        """Carver position inertia: trade only if target weight changes by > buffer_threshold of max size."""
        buffered = target_weights.copy()
        curr_w = 0.0
        for i in range(len(target_weights)):
            tw = target_weights.iloc[i]
            if np.isnan(tw):
                buffered.iloc[i] = 0.0
                curr_w = 0.0
            else:
                if abs(tw - curr_w) >= buffer_threshold:
                    curr_w = tw
                buffered.iloc[i] = curr_w
        return buffered

    def generate_target_weights(self, close_prices, panel_prices=None, target_name=None, btc_prices=None, spy_prices=None, weights=None):
        """
        Combines all layers and applies Beta Rotation to output final positions.
        If panel_prices and target_name are provided, it computes CS Momentum as well.
        """
        if weights is None:
            # Default ensemble weights
            weights = {'ewmac': 0.40, 'breakout': 0.30, 'accel': 0.20, 'skew': 0.10}

        # 1. Volatility Infrastructure
        vol, sigma_p = self.vol_stack(close_prices)
        
        # 2. Layered Forecasts
        forecasts = {
            'ewmac': self.layer1_ewmac_forecast(close_prices, sigma_p),
            'breakout': self.layer2_breakout_forecast(close_prices),
            'accel': self.layer3_acceleration_forecast(close_prices, sigma_p),
            'skew': self.layer4_skew_forecast(close_prices)
        }
        
        if panel_prices is not None and target_name is not None:
            forecasts['cs_mom'] = self.layer7_cs_momentum_forecast(panel_prices, target_name)
            # Adjust weights down to accommodate 20% to cs_mom
            weights = {k: v * 0.8 for k, v in weights.items()}
            weights['cs_mom'] = 0.20

        # 3. Democratic Voting with Dynamic FDM
        combined_fc, self.last_fdm, self.last_corr = self.combine_forecasts(forecasts, weights)
        
        # 4. Volatility Attenuation (De-risking in localized vol spikes)
        vol_att = self.layer5_vol_attenuation(vol)
        attenuated_fc = self._clip_fc(combined_fc * vol_att)
        
        # 5. Position Sizing
        base_weight = self.position_from_forecast(attenuated_fc, vol, long_only=True)
        
        # 6. Macro Regime Override / Throttle
        if btc_prices is not None and spy_prices is not None:
            # Check if continuous macro throttle multiplier is passed
            regime = self.macro_regime_filter(btc_prices, spy_prices)
            aligned_base, aligned_regime = base_weight.align(regime, join='left')
            aligned_regime = aligned_regime.ffill().fillna(1)
            
            final_weight = pd.Series(np.where(aligned_regime == 2, 0.0, aligned_base), index=aligned_base.index)
        else:
            final_weight = base_weight
            
        # 7. Carver Position Buffering
        final_weight = self.apply_position_buffer(final_weight, buffer_threshold=0.05)
        return final_weight

    # --------------------------------------------------------------------------
    # BACKTESTING
    # --------------------------------------------------------------------------
    def backtest(self, close, weight, cost_bps=3.25, exec_lag=1):
        """Standard uncompounded backtest at the close, execute next bar."""
        r = close.pct_change(fill_method=None).fillna(0.0)
        
        # weights executed at close[t], earn returns at close[t+exec_lag]
        held = weight.shift(exec_lag).fillna(0.0)
        ret = held * r
        
        trn = held.diff().abs().fillna(0.0)
        cost = trn * (cost_bps / 10000.0)
        
        net = ret - cost
        equity = (1.0 + net).cumprod()
        
        return pd.DataFrame({"ret": r, "weight": weight, "held": held, "net": net, "equity": equity})
