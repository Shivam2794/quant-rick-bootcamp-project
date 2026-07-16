import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import vectorbt as vbt
import itertools
import warnings
import time

warnings.filterwarnings('ignore')

FEES = 0.002          # 20 bps per trade
MIN_TRADES_PER_YEAR = 5
MAX_ULCER_INDEX = 15

class UltraIndicatorFactory:
    @staticmethod
    def _safe_assign(df, name, condition):
        """Helper to assign boolean signals safely."""
        try:
            # Handle potential NaNs in condition safely
            df[name] = condition.fillna(False).astype(bool)
        except Exception:
            df[name] = False

    @classmethod
    def generate_all(cls, data):
        close = data['Close']
        high  = data['High']
        low   = data['Low']
        volume= data['Volume']
        open_ = data['Open']

        d = pd.DataFrame(index=data.index)
        d['Open']   = open_
        d['Close']  = close
        d['High']   = high
        d['Low']    = low
        d['Volume'] = volume

        c = d['Close']
        h = d['High']
        l = d['Low']
        v = d['Volume']

        # -------------------------------------------------------------
        # 1. Stochastic Oscillator
        # -------------------------------------------------------------
        # Standard: 14, 3, 3
        st_std = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
        if st_std is not None:
            k_std = st_std.iloc[:, 0]
            cls._safe_assign(d, 'Sig_Stoch_Std_OB', k_std > 80)
            cls._safe_assign(d, 'Sig_Stoch_Std_OS', k_std < 20)

        # Fast: 5, 3, 3
        st_fast = ta.stoch(h, l, c, k=5, d=3, smooth_k=3)
        if st_fast is not None:
            k_fast = st_fast.iloc[:, 0]
            cls._safe_assign(d, 'Sig_Stoch_Fast_OB', k_fast > 80)
            cls._safe_assign(d, 'Sig_Stoch_Fast_OS', k_fast < 20)

        # Macro: 21, 5, 5
        st_mac = ta.stoch(h, l, c, k=21, d=5, smooth_k=5)
        if st_mac is not None:
            k_mac = st_mac.iloc[:, 0]
            cls._safe_assign(d, 'Sig_Stoch_Macro_OB', k_mac > 80)
            cls._safe_assign(d, 'Sig_Stoch_Macro_OS', k_mac < 20)

        # -------------------------------------------------------------
        # 2. CCI (Commodity Channel Index)
        # -------------------------------------------------------------
        cci_std = ta.cci(h, l, c, length=20)
        cls._safe_assign(d, 'Sig_CCI_Std_Bull', cci_std > 100)
        
        cci_fast = ta.cci(h, l, c, length=14)
        cls._safe_assign(d, 'Sig_CCI_Fast_Bull', cci_fast > 100)
        
        cci_mac = ta.cci(h, l, c, length=50)
        cls._safe_assign(d, 'Sig_CCI_Macro_Bull', cci_mac > 100)

        # -------------------------------------------------------------
        # 3. Williams %R
        # -------------------------------------------------------------
        wr_std = ta.willr(h, l, c, length=14)
        cls._safe_assign(d, 'Sig_WillR_Std_OS', wr_std < -80)

        wr_fast = ta.willr(h, l, c, length=10)
        cls._safe_assign(d, 'Sig_WillR_Fast_OS', wr_fast < -80)

        wr_mac = ta.willr(h, l, c, length=50)
        cls._safe_assign(d, 'Sig_WillR_Macro_OS', wr_mac < -80)

        # -------------------------------------------------------------
        # 4. OBV (On-Balance Volume)
        # -------------------------------------------------------------
        obv = ta.obv(c, v)
        if obv is not None:
            ema20_obv = ta.ema(obv, length=20)
            ema50_obv = ta.ema(obv, length=50)
            cls._safe_assign(d, 'Sig_OBV_Fast', obv > ema20_obv)
            cls._safe_assign(d, 'Sig_OBV_Macro', obv > ema50_obv)

        # -------------------------------------------------------------
        # 5. Chaikin Money Flow (CMF)
        # -------------------------------------------------------------
        cmf_inst = ta.cmf(h, l, c, v, length=21)
        cls._safe_assign(d, 'Sig_CMF_Inst', cmf_inst > 0.05)

        cmf_fast = ta.cmf(h, l, c, v, length=14)
        cls._safe_assign(d, 'Sig_CMF_Fast', cmf_fast > 0.05)

        # -------------------------------------------------------------
        # 6. Volume Profile (PVT proxy for volume accumulation)
        # -------------------------------------------------------------
        pvt = ta.pvt(c, v)
        if pvt is not None:
            pvt_ema = ta.ema(pvt, length=21)
            cls._safe_assign(d, 'Sig_PVT_Acc', pvt > pvt_ema)

        # -------------------------------------------------------------
        # 7. ADX (Average Directional Index)
        # -------------------------------------------------------------
        adx_df = ta.adx(h, l, c, length=14)
        if adx_df is not None:
            # columns usually ADX_14, DMP_14, DMN_14
            adx_col = [col for col in adx_df.columns if col.startswith('ADX_')]
            dmp_col = [col for col in adx_df.columns if col.startswith('DMP_')]
            dmn_col = [col for col in adx_df.columns if col.startswith('DMN_')]
            if adx_col and dmp_col and dmn_col:
                adx = adx_df[adx_col[0]]
                dmp = adx_df[dmp_col[0]]
                dmn = adx_df[dmn_col[0]]
                
                # ADX > 25 and +DI > -DI
                cls._safe_assign(d, 'Sig_ADX_Std_Trend', (adx > 25) & (dmp > dmn))
                # ADX > 40 and +DI > -DI
                cls._safe_assign(d, 'Sig_ADX_Ext_Trend', (adx > 40) & (dmp > dmn))

        # -------------------------------------------------------------
        # 8. Parabolic SAR
        # -------------------------------------------------------------
        # Standard: 0.02, 0.20
        psar_std = ta.psar(h, l, c, af0=0.02, af=0.02, max_af=0.20)
        if psar_std is not None:
            l_col = [col for col in psar_std.columns if col.startswith('PSARl_')]
            if l_col:
                # If PSARl (long) exists, we are above SAR
                cls._safe_assign(d, 'Sig_PSAR_Std', pd.notna(psar_std[l_col[0]]))

        # Aggressive: 0.03, 0.30
        psar_agg = ta.psar(h, l, c, af0=0.03, af=0.03, max_af=0.30)
        if psar_agg is not None:
            l_col = [col for col in psar_agg.columns if col.startswith('PSARl_')]
            if l_col:
                cls._safe_assign(d, 'Sig_PSAR_Agg', pd.notna(psar_agg[l_col[0]]))

        # Macro: 0.01, 0.10
        psar_mac = ta.psar(h, l, c, af0=0.01, af=0.01, max_af=0.10)
        if psar_mac is not None:
            l_col = [col for col in psar_mac.columns if col.startswith('PSARl_')]
            if l_col:
                cls._safe_assign(d, 'Sig_PSAR_Mac', pd.notna(psar_mac[l_col[0]]))

        # -------------------------------------------------------------
        # 9. Ichimoku Cloud
        # -------------------------------------------------------------
        def process_ichi(res_tuple, prefix):
            if res_tuple is not None and isinstance(res_tuple, tuple) and len(res_tuple) > 0:
                idf = res_tuple[0]
                isa = [col for col in idf.columns if col.startswith('ISA_')]
                isb = [col for col in idf.columns if col.startswith('ISB_')]
                if isa and isb:
                    # Cloud breakout
                    cls._safe_assign(d, f'Sig_Ichi_{prefix}_Cloud', c > idf[isa[0]])
                    cls._safe_assign(d, f'Sig_Ichi_{prefix}_Span', idf[isa[0]] > idf[isb[0]])

        # Std: 9, 26, 52
        process_ichi(ta.ichimoku(h, l, c, tenkan=9, kijun=26, senkou=52), 'Std')
        # Crypto: 20, 60, 120
        process_ichi(ta.ichimoku(h, l, c, tenkan=20, kijun=60, senkou=120), 'Crypto')
        # Double: 18, 52, 104
        process_ichi(ta.ichimoku(h, l, c, tenkan=18, kijun=52, senkou=104), 'Double')

        # -------------------------------------------------------------
        # 10. Fibonacci Retracement (Rolling 200 Max)
        # -------------------------------------------------------------
        # Standard Key Levels: 23.6%, 38.2%, 50%, 61.8%, 78.6%
        # The Golden Pocket: 61.8% and 65%
        # Deep Value Retracement: 78.6% and 88.6%
        roll_max = h.rolling(200).max()
        roll_min = l.rolling(200).min()
        fib_range = roll_max - roll_min
        
        # Calculate support thresholds (price dropping X% from the high)
        fib_236 = roll_max - (fib_range * 0.236)
        fib_382 = roll_max - (fib_range * 0.382)
        fib_500 = roll_max - (fib_range * 0.500)
        fib_618 = roll_max - (fib_range * 0.618)
        fib_650 = roll_max - (fib_range * 0.650)
        fib_786 = roll_max - (fib_range * 0.786)
        fib_886 = roll_max - (fib_range * 0.886)
        
        # Signal: price is above the support level (support held)
        cls._safe_assign(d, 'Sig_FIB_236_Hold', c > fib_236)
        cls._safe_assign(d, 'Sig_FIB_382_Hold', c > fib_382)
        cls._safe_assign(d, 'Sig_FIB_500_Hold', c > fib_500)
        cls._safe_assign(d, 'Sig_FIB_618_Golden', c > fib_618)
        cls._safe_assign(d, 'Sig_FIB_650_Golden', c > fib_650)
        cls._safe_assign(d, 'Sig_FIB_786_Deep', c > fib_786)
        cls._safe_assign(d, 'Sig_FIB_886_Deep', c > fib_886)

        # -------------------------------------------------------------
        # 11. Linear Regression Slope
        # -------------------------------------------------------------
        slope_9 = ta.slope(c, length=9)
        cls._safe_assign(d, 'Sig_LinReg_Fast', slope_9 > 0)

        slope_14 = ta.slope(c, length=14)
        cls._safe_assign(d, 'Sig_LinReg_Swing', slope_14 > 0)

        slope_50 = ta.slope(c, length=50)
        cls._safe_assign(d, 'Sig_LinReg_Macro', slope_50 > 0)

        # -------------------------------------------------------------
        # Some essential standard trend baselines for combinatorics
        # -------------------------------------------------------------
        sma50 = ta.sma(c, length=50)
        sma200 = ta.sma(c, length=200)
        cls._safe_assign(d, 'Sig_SMA_50_200', sma50 > sma200)
        cls._safe_assign(d, 'Sig_Price_SMA200', c > sma200)

        # Drop any leftover NaNs from shifting
        d = d.dropna()
        return d


def calculate_metrics(portfolio):
    try:
        sr = portfolio.sharpe_ratio()
        if pd.isna(sr):
            return np.nan, np.nan, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        sortino = portfolio.sortino_ratio()
        trades  = portfolio.trades.count()
        wr      = portfolio.trades.win_rate() * 100
        exp     = portfolio.trades.expectancy()
        
        # BRUTAL FIX: Manual Profit Factor
        try:
            pnls = portfolio.trades.records_readable['PnL']
            gross_profits = np.sum(pnls[pnls > 0])
            gross_losses = np.abs(np.sum(pnls[pnls < 0]))
            profit_factor = float(gross_profits / gross_losses) if gross_losses != 0 else float('inf')
        except Exception:
            profit_factor = 0.0
            
        returns = portfolio.returns()
        skew    = returns.skew() if len(returns) > 30 else 0.0
        
        log_rets = np.log1p(returns)
        active = log_rets[log_rets != 0]
        if len(active) > 30:
            mu_log  = active.mean()
            var_log = active.var(ddof=1)
            kelly   = float(mu_log / var_log) if (pd.notna(var_log) and var_log > 1e-12) else 0.0
        else:
            kelly = 0.0

        drawdown_pct = portfolio.drawdown() * 100
        ulcer = float(np.sqrt(np.mean(drawdown_pct ** 2))) if len(drawdown_pct) > 0 else 0.0

        return sortino, sr, trades, wr, exp, skew, kelly, ulcer, profit_factor
    except Exception:
        return np.nan, np.nan, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


class UltraGrinder:
    def __init__(self, asset_name, raw_data):
        self.asset_name = asset_name
        self.raw_data = raw_data.dropna()
        self.full_df = UltraIndicatorFactory.generate_all(self.raw_data)
        split_idx = int(len(self.full_df) * 0.60)
        self.train_df = self.full_df.iloc[:split_idx]
        self.test_df = self.full_df.iloc[split_idx:]

    def _calendar_years(self, df):
        return (df.index[-1] - df.index[0]).days / 365.25

    def grind(self):
        print(f"\n============================================================")
        print(f"=== ULTRA GRINDING  {self.asset_name}  ===")
        print(f"============================================================")
        signals = [c for c in self.train_df.columns if c.startswith('Sig_')]
        print(f"  Signals available: {len(signals)}")
        print(f"  Signals list: {signals[:5]} ...")

        sig_cache  = {s: self.train_df[s].values for s in signals}
        open_vals = self.train_df['Open'].values
        years      = self._calendar_years(self.train_df)

        best_score = -np.inf
        self.best_combo = None
        self.best_metrics = None
        valid_found = 0

        # Exhaustive Combos k=1, 2, 3
        # CAUTION: If signals > 35, combinations explode. Let's strictly cap k=2 if combinations > 50,000 to avoid week-long runs.
        all_combos = []
        for k in (1, 2):
            all_combos.extend(list(itertools.combinations(signals, k)))
        
        # Only add k=3 if it won't take literally days.
        k3_len = len(signals) * (len(signals)-1) * (len(signals)-2) // 6
        if k3_len < 30000:
            all_combos.extend(list(itertools.combinations(signals, 3)))
        else:
            print(f"  [INFO] Skipping k=3 exhaustive search to stay within compute bounds (k=3 space is {k3_len} configs).")

        total_evals = len(all_combos) * 2

        print(f"  Exhaustive Grid Search: {total_evals} configurations, {years:.1f} training years")
        t0 = time.time()

        for i, combo in enumerate(all_combos):
            logics = ('OR', 'AND') if len(combo) > 1 else ('OR',)

            for logic in logics:
                entry = np.ones(len(open_vals), dtype=bool) if logic == 'AND' \
                        else np.zeros(len(open_vals), dtype=bool)
                for sig in combo:
                    if logic == 'AND':
                        entry &= sig_cache[sig]
                    else:
                        entry |= sig_cache[sig]

                if entry.all() or not entry.any():
                    continue

                # BRUTAL FIX: Shifting to execute on Open!
                shifted_entries = pd.Series(entry).shift(1).fillna(False).values
                shifted_exits = pd.Series(~entry).shift(1).fillna(False).values

                pf = vbt.Portfolio.from_signals(
                    open_vals, entries=shifted_entries, exits=shifted_exits,
                    freq='1d', fees=FEES
                )

                sortino, sharpe, trades, wr, exp, skew, kelly, ulcer, pf_factor = calculate_metrics(pf)
                tpy = trades / years

                if (tpy >= MIN_TRADES_PER_YEAR
                        and kelly > 0
                        and ulcer < MAX_ULCER_INDEX
                        and pf_factor >= 2.0
                        and pd.notna(sharpe)):

                    valid_found += 1
                    fitness = sharpe

                    if fitness > best_score:
                        best_score = fitness
                        self.best_combo = combo
                        self.best_logic = logic
                        self.best_metrics = {
                            'Sortino': sortino, 'Sharpe': sharpe, 'Trades': trades,
                            'Kelly': kelly, 'Skew': skew, 'Ulcer': ulcer, 'Logic': logic, 'ProfitFactor': pf_factor
                        }

            if i % 2500 == 0 and i > 0:
                print(f"  [combo {i:,}/{len(all_combos):,}] best_sharpe={best_score:.3f} | valid={valid_found}")

        elapsed = time.time() - t0
        print(f"\nGrind done in {elapsed:.1f}s -- {valid_found} ensembles survived all constraints.")
        
        if self.best_combo:
            print(f"  >> WINNER: {self.best_combo} [{self.best_logic}]")
            print(f"     Sharpe: {self.best_metrics['Sharpe']:.2f} | Kelly: {self.best_metrics['Kelly']:.2f} | Trades: {self.best_metrics['Trades']}")
        else:
            print("  >> WINNER: NONE (No strategy met institutional constraints)")
            
        return self.best_combo

    def apply_out_of_sample(self):
        print(f"[OOS] Validating WINNER out-of-sample...")
        
        open_vals = self.test_df['Open'].values
        entry = np.ones(len(open_vals), dtype=bool) if self.best_logic == 'AND' \
                else np.zeros(len(open_vals), dtype=bool)
                
        for sig in self.best_combo:
            if self.best_logic == 'AND':
                entry &= self.test_df[sig].values
            else:
                entry |= self.test_df[sig].values
                
        shifted_entries = pd.Series(entry).shift(1).fillna(False).values
        shifted_exits = pd.Series(~entry).shift(1).fillna(False).values
                
        pf = vbt.Portfolio.from_signals(
            open_vals, entries=shifted_entries, exits=shifted_exits,
            freq='1d', fees=FEES
        )
        sortino, sharpe, trades, wr, exp, skew, kelly, ulcer, pf_factor = calculate_metrics(pf)
        
        print(f"  [OOS RESULTS]")
        print(f"  Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | Kelly: {kelly:.2f} | PF: {pf_factor:.2f}")
        print(f"  Trades: {trades} | Win Rate: {wr:.1f}% | Ulcer: {ulcer:.2f}")
        
        self.oos_metrics = {
            'Sortino': sortino, 'Sharpe': sharpe, 'Trades': trades,
            'Kelly': kelly, 'Skew': skew, 'Ulcer': ulcer, 'WinRate': wr, 'ProfitFactor': pf_factor
        }


def main():
    assets = ['BTC-USD', 'SOL-USD', 'QQQ', 'DIA', 'GLD', 'SPY', 'SMH', 'XLV', 'XLU', 'SLV', 'ETH-USD']
    master_data = {}
    for a in assets:
        try:
            df = yf.download(a, start='2010-01-01', progress=False, auto_adjust=True)
            if not df.empty:
                # Handle multi-index yfinance output
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                master_data[a] = df
        except Exception as e:
            print(f"Failed to fetch {a}: {e}")

    results = []
    
    for asset in assets:
        if asset in master_data:
            # We must pass the raw data, length is checked after instantiation
            engine = UltraGrinder(asset, master_data[asset])
            if len(engine.train_df) < 500:
                print(f"Skipping {asset}: insufficient training data.")
                continue

            best = engine.grind()
            if best:
                engine.apply_out_of_sample()
                results.append({
                    'Asset': asset,
                    'Combo': engine.best_combo,
                    'Logic': engine.best_logic,
                    'IS_Sharpe': engine.best_metrics['Sharpe'],
                    'IS_Kelly': engine.best_metrics['Kelly'],
                    'OOS_Sharpe': engine.oos_metrics['Sharpe'],
                    'OOS_Kelly': engine.oos_metrics['Kelly'],
                    'OOS_Trades': engine.oos_metrics['Trades']
                })
    
    print("\n\n" + "="*80)
    print("=== FINAL ULTRA EXHAUSTIVE PORTFOLIO RESULTS ===")
    print("="*80)
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print(res_df.to_string())
    else:
        print("No assets found winning strategies.")

if __name__ == '__main__':
    main()
