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

class OmniIndicatorFactory:
    @staticmethod
    def _safe_assign(df, name, condition):
        """Helper to assign boolean signals safely."""
        try:
            df[name] = condition.fillna(False).astype(bool)
        except Exception:
            df[name] = False

    @classmethod
    def _as_series(cls, series_or_df):
        if isinstance(series_or_df, pd.DataFrame):
            return series_or_df.iloc[:, 0]
        return series_or_df

    @classmethod
    def generate_all(cls, df):
        d = df.copy()
        c = d['Close']
        h = d['High']
        l = d['Low']
        v = d['Volume']

        # =============================================================
        # LEGACY INDICATORS (from relentless_grinder.py)
        # =============================================================
        try:
            # 1. Moving Averages
            sma50 = ta.sma(c, length=50)
            sma200 = ta.sma(c, length=200)
            cls._safe_assign(d, 'Sig_SMA_Cross', sma50 > sma200)

            ema20 = ta.ema(c, length=20)
            ema50 = ta.ema(c, length=50)
            cls._safe_assign(d, 'Sig_EMA_Cross', ema20 > ema50)

            hma50 = ta.hma(c, length=50)
            cls._safe_assign(d, 'Sig_HMA', c > hma50)

            alma = ta.alma(c)
            if alma is not None and not alma.empty:
                cls._safe_assign(d, 'Sig_ALMA', c > cls._as_series(alma))

            zlema = ta.zlma(c)
            if zlema is not None and not zlema.empty:
                cls._safe_assign(d, 'Sig_ZLEMA', c > cls._as_series(zlema))

            kama = ta.kama(c)
            if kama is not None and not kama.empty:
                cls._safe_assign(d, 'Sig_KAMA', c > cls._as_series(kama))

            tema_fast = ta.tema(c, length=14)
            tema_slow = ta.tema(c, length=50)
            if tema_fast is not None and tema_slow is not None:
                cls._safe_assign(d, 'Sig_TEMA', tema_fast > tema_slow)

            # 2. Trend & Volatility
            supertrend = ta.supertrend(h, l, c)
            if supertrend is not None and not supertrend.empty:
                dir_cols = [col for col in supertrend.columns if col.startswith('SUPERTd')]
                if dir_cols:
                    cls._safe_assign(d, 'Sig_SuperTrend', supertrend[dir_cols[0]] > 0)

            psar = ta.psar(h, l, c)
            if psar is not None and not psar.empty:
                long_cols = [col for col in psar.columns if 'PSARl' in col]
                if long_cols:
                    cls._safe_assign(d, 'Sig_PSAR_Legacy', psar[long_cols[0]].notna())

            # 3. Oscillators
            macd = ta.macd(c)
            if macd is not None and not macd.empty:
                m_cols = [col for col in macd.columns if col.startswith('MACD_')]
                s_cols = [col for col in macd.columns if col.startswith('MACDs')]
                if m_cols and s_cols:
                    cls._safe_assign(d, 'Sig_MACD', macd[m_cols[0]] > macd[s_cols[0]])

            cls._safe_assign(d, 'Sig_RSI', ta.rsi(c, length=14) > 50)

            stc = ta.stc(c)
            if stc is not None and not stc.empty:
                stc_val = cls._as_series(stc.iloc[:, 0])
                cls._safe_assign(d, 'Sig_STC', stc_val > stc_val.shift(1))

            kst = ta.kst(c)
            if kst is not None and not kst.empty:
                cls._safe_assign(d, 'Sig_KST', kst.iloc[:, 0] > kst.iloc[:, 1])

            tsi = ta.tsi(c)
            if tsi is not None and not tsi.empty:
                cls._safe_assign(d, 'Sig_TSI', tsi.iloc[:, 0] > tsi.iloc[:, 1])

            cg = ta.cg(c)
            if cg is not None and not cg.empty:
                cg_val = cg.iloc[:, 0] if isinstance(cg, pd.DataFrame) else cg
                cls._safe_assign(d, 'Sig_CG', cg_val > cg_val.shift(1))

            # 4. Channels
            bbands = ta.bbands(c, length=20)
            if bbands is not None and not bbands.empty:
                cls._safe_assign(d, 'Sig_BB_Break', c > bbands.iloc[:, 2])

            kc = ta.kc(h, l, c)
            if kc is not None and not kc.empty:
                cls._safe_assign(d, 'Sig_KC_Break', c > kc.iloc[:, 2])

            donchian = ta.donchian(h, l)
            if donchian is not None and not donchian.empty:
                cls._safe_assign(d, 'Sig_DC_Break', c > donchian.iloc[:, 2].shift(1))

            # 5. Volume/Chop
            chop = ta.chop(h, l, c)
            if chop is not None and not chop.empty:
                cls._safe_assign(d, 'Sig_NotChoppy', cls._as_series(chop) < 50)

            obv = ta.obv(c, v)
            if obv is not None and not obv.empty:
                obv_s = cls._as_series(obv)
                obv_ma = ta.sma(obv_s, length=20)
                cls._safe_assign(d, 'Sig_OBV_Legacy', obv_s > obv_ma)

            cmf = ta.cmf(h, l, c, v)
            if cmf is not None and not cmf.empty:
                cls._safe_assign(d, 'Sig_CMF_Legacy', cls._as_series(cmf) > 0)

            # 6. Legacy Ichimoku (Confluence)
            ichi = ta.ichimoku(h, l, c)
            if isinstance(ichi, tuple): ichi = ichi[0]
            if ichi is not None and not ichi.empty and ichi.shape[1] >= 4:
                isa = [col for col in ichi.columns if col.startswith('ISA')]
                isb = [col for col in ichi.columns if col.startswith('ISB')]
                its = [col for col in ichi.columns if col.startswith('ITS')]
                iks = [col for col in ichi.columns if col.startswith('IKS')]
                if isa and isb and its and iks:
                    cls._safe_assign(d, 'Sig_Ichimoku_Legacy', 
                        (c > ichi[isa[0]]) & (c > ichi[isb[0]]) & (ichi[its[0]] > ichi[iks[0]])
                    )
                    
            # 7. Legacy Fib 618
            roll_high = h.rolling(200).max()
            roll_low  = l.rolling(200).min()
            cls._safe_assign(d, 'Sig_FIB_Legacy', c > (roll_high - (roll_high - roll_low) * 0.618))
            
        except Exception as e:
            print(f"[WARNING] Legacy generator error: {e}")

        # =============================================================
        # NEW INDICATORS (from ultra_grinder.py)
        # =============================================================
        # 1. Stochastic
        st_std = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
        if st_std is not None:
            cls._safe_assign(d, 'Sig_Stoch_Std_OB', st_std.iloc[:, 0] > 80)
            cls._safe_assign(d, 'Sig_Stoch_Std_OS', st_std.iloc[:, 0] < 20)
        st_fast = ta.stoch(h, l, c, k=5, d=3, smooth_k=3)
        if st_fast is not None:
            cls._safe_assign(d, 'Sig_Stoch_Fast_OB', st_fast.iloc[:, 0] > 80)
            cls._safe_assign(d, 'Sig_Stoch_Fast_OS', st_fast.iloc[:, 0] < 20)
        st_mac = ta.stoch(h, l, c, k=21, d=5, smooth_k=5)
        if st_mac is not None:
            cls._safe_assign(d, 'Sig_Stoch_Macro_OB', st_mac.iloc[:, 0] > 80)
            cls._safe_assign(d, 'Sig_Stoch_Macro_OS', st_mac.iloc[:, 0] < 20)

        # 2. CCI
        cls._safe_assign(d, 'Sig_CCI_Std_Bull', ta.cci(h, l, c, length=20) > 100)
        cls._safe_assign(d, 'Sig_CCI_Fast_Bull', ta.cci(h, l, c, length=14) > 100)
        cls._safe_assign(d, 'Sig_CCI_Macro_Bull', ta.cci(h, l, c, length=50) > 100)

        # 3. Williams %R
        cls._safe_assign(d, 'Sig_WillR_Std_OS', ta.willr(h, l, c, length=14) < -80)
        cls._safe_assign(d, 'Sig_WillR_Fast_OS', ta.willr(h, l, c, length=10) < -80)
        cls._safe_assign(d, 'Sig_WillR_Macro_OS', ta.willr(h, l, c, length=50) < -80)

        # 4. OBV
        if obv is not None:
            cls._safe_assign(d, 'Sig_OBV_Fast', obv > ta.ema(obv, length=20))
            cls._safe_assign(d, 'Sig_OBV_Macro', obv > ta.ema(obv, length=50))

        # 5. CMF
        cls._safe_assign(d, 'Sig_CMF_Inst', ta.cmf(h, l, c, v, length=21) > 0.05)
        cls._safe_assign(d, 'Sig_CMF_Fast', ta.cmf(h, l, c, v, length=14) > 0.05)

        # 6. PVT
        pvt = ta.pvt(c, v)
        if pvt is not None:
            cls._safe_assign(d, 'Sig_PVT_Acc', pvt > ta.ema(pvt, length=21))

        # 7. ADX
        adx_df = ta.adx(h, l, c, length=14)
        if adx_df is not None:
            adx_col = [col for col in adx_df.columns if col.startswith('ADX_')]
            dmp_col = [col for col in adx_df.columns if col.startswith('DMP_')]
            dmn_col = [col for col in adx_df.columns if col.startswith('DMN_')]
            if adx_col and dmp_col and dmn_col:
                adx = adx_df[adx_col[0]]
                dmp = adx_df[dmp_col[0]]
                dmn = adx_df[dmn_col[0]]
                cls._safe_assign(d, 'Sig_ADX_Std_Trend', (adx > 25) & (dmp > dmn))
                cls._safe_assign(d, 'Sig_ADX_Ext_Trend', (adx > 40) & (dmp > dmn))

        # 8. PSAR
        def add_psar(af0, af, max_af, name):
            ps = ta.psar(h, l, c, af0=af0, af=af, max_af=max_af)
            if ps is not None:
                lc = [col for col in ps.columns if col.startswith('PSARl_')]
                if lc: cls._safe_assign(d, name, pd.notna(ps[lc[0]]))

        add_psar(0.02, 0.02, 0.20, 'Sig_PSAR_Std')
        add_psar(0.03, 0.03, 0.30, 'Sig_PSAR_Agg')
        add_psar(0.01, 0.01, 0.10, 'Sig_PSAR_Mac')

        # 9. Ichimoku Clouds
        def process_ichi(res_tuple, prefix):
            if res_tuple and isinstance(res_tuple, tuple) and len(res_tuple) > 0:
                idf = res_tuple[0]
                isa = [col for col in idf.columns if col.startswith('ISA_')]
                isb = [col for col in idf.columns if col.startswith('ISB_')]
                if isa and isb:
                    cls._safe_assign(d, f'Sig_Ichi_{prefix}_Cloud', c > idf[isa[0]])
                    cls._safe_assign(d, f'Sig_Ichi_{prefix}_Span', idf[isa[0]] > idf[isb[0]])
        
        process_ichi(ta.ichimoku(h, l, c, tenkan=9, kijun=26, senkou=52), 'Std')
        process_ichi(ta.ichimoku(h, l, c, tenkan=20, kijun=60, senkou=120), 'Crypto')
        process_ichi(ta.ichimoku(h, l, c, tenkan=18, kijun=52, senkou=104), 'Double')

        # 10. Fibs
        roll_max = h.rolling(200).max()
        roll_min = l.rolling(200).min()
        fib_r = roll_max - roll_min
        
        cls._safe_assign(d, 'Sig_FIB_236_Hold', c > (roll_max - fib_r * 0.236))
        cls._safe_assign(d, 'Sig_FIB_382_Hold', c > (roll_max - fib_r * 0.382))
        cls._safe_assign(d, 'Sig_FIB_500_Hold', c > (roll_max - fib_r * 0.500))
        cls._safe_assign(d, 'Sig_FIB_618_Golden', c > (roll_max - fib_r * 0.618))
        cls._safe_assign(d, 'Sig_FIB_650_Golden', c > (roll_max - fib_r * 0.650))
        cls._safe_assign(d, 'Sig_FIB_786_Deep', c > (roll_max - fib_r * 0.786))
        cls._safe_assign(d, 'Sig_FIB_886_Deep', c > (roll_max - fib_r * 0.886))

        # 11. LinReg
        cls._safe_assign(d, 'Sig_LinReg_Fast', ta.slope(c, length=9) > 0)
        cls._safe_assign(d, 'Sig_LinReg_Swing', ta.slope(c, length=14) > 0)
        cls._safe_assign(d, 'Sig_LinReg_Macro', ta.slope(c, length=50) > 0)

        # Baseline
        sma50 = ta.sma(c, length=50)
        sma200 = ta.sma(c, length=200)
        cls._safe_assign(d, 'Sig_SMA_50_200', sma50 > sma200)
        cls._safe_assign(d, 'Sig_Price_SMA200', c > sma200)

        # Drop NaNs
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
        
        # BRUTAL FIX: Profit Factor Manual Calculation (Avoid VBT bug)
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
        if len(log_rets) > 30:
            mu_log  = log_rets.mean()
            var_log = log_rets.var(ddof=1)
            kelly   = float(mu_log / var_log) if (pd.notna(var_log) and var_log > 1e-12) else 0.0
        else:
            kelly = 0.0

        drawdown_pct = portfolio.drawdown() * 100
        ulcer = float(np.sqrt(np.mean(drawdown_pct ** 2))) if len(drawdown_pct) > 0 else 0.0

        return sortino, sr, trades, wr, exp, skew, kelly, ulcer, profit_factor
    except Exception:
        return np.nan, np.nan, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


class OmniGrinder:
    def __init__(self, asset_name, raw_data):
        self.asset_name = asset_name
        self.raw_data = raw_data.dropna()
        
        # BRUTAL FIX 1: Generate all features before split to prevent OOS NaN burn
        print(f"[GRINDER] Building expanded feature matrix for {self.asset_name}...")
        self.full_df = OmniIndicatorFactory.generate_all(self.raw_data)
        
        # BRUTAL FIX: Dynamic 60/40 Split instead of hardcoded date
        total_len = len(self.full_df)
        split_idx = int(total_len * 0.60)
        
        self.train_df = self.full_df.iloc[:split_idx]
        self.test_df = self.full_df.iloc[split_idx:]

    def _calendar_years(self, df):
        return (df.index[-1] - df.index[0]).days / 365.25

    def grind(self):
        print(f"\n============================================================")
        print(f"=== OMNI GRINDING  {self.asset_name}  ===")
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
        # We explicitly run k=3 without a cap because of the /goal directive
        all_combos = []
        for k in (1, 2, 3):
            all_combos.extend(list(itertools.combinations(signals, k)))
        
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

                # BRUTAL FIX 2: Shift entries to execute on next open
                shifted_entries = pd.Series(entry).shift(1).fillna(False).values
                shifted_exits = pd.Series(~entry).shift(1).fillna(False).values

                pf = vbt.Portfolio.from_signals(
                    open_vals, entries=shifted_entries, exits=shifted_exits,
                    freq='1d', fees=FEES
                )

                sortino, sharpe, trades, wr, exp, skew, kelly, ulcer, pf_factor = calculate_metrics(pf)
                tpy = trades / years

                # BRUTAL FIX: Enforce minimum 2.0 Profit Factor
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

            if i % 10000 == 0 and i > 0:
                print(f"  [combo {i:,}/{len(all_combos):,}] best_sharpe={best_score:.3f} | valid={valid_found}")

        elapsed = time.time() - t0
        print(f"\nGrind done in {elapsed:.1f}s -- {valid_found} ensembles survived all constraints.")
        
        if self.best_combo:
            print(f"  >> WINNER: {self.best_combo} [{self.best_logic}]")
            print(f"     Sharpe: {self.best_metrics['Sharpe']:.2f} | Kelly: {self.best_metrics['Kelly']:.2f} | Trades: {self.best_metrics['Trades']} | PF: {self.best_metrics['ProfitFactor']:.2f}")
        else:
            print("  >> WINNER: NONE (No strategy met institutional constraints)")
            
        return self.best_combo

    def apply_out_of_sample(self):
        print(f"[OOS] Validating WINNER out-of-sample...")
        
        open_vals = self.test_df['Open'].values
        entry = np.ones(len(open_vals), dtype=bool) if self.best_logic == 'AND' \
                else np.zeros(len(open_vals), dtype=bool)
                
        for sig in self.best_combo:
            if sig not in self.test_df.columns:
                continue
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
            'Kelly': kelly, 'Skew': skew, 'Ulcer': ulcer, 'WinRate': wr
        }


def main():
    assets = ['BTC-USD', 'SOL-USD', 'QQQ', 'DIA', 'GLD', 'SPY', 'SMH', 'XLV', 'XLU', 'SLV', 'ETH-USD']
    master_data = {}
    for a in assets:
        try:
            df = yf.download(a, start='2010-01-01', progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                master_data[a] = df
        except Exception as e:
            print(f"Failed to fetch {a}: {e}")

    results = []
    
    for asset in assets:
        if asset in master_data:
            engine = OmniGrinder(asset, master_data[asset])
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
    print("=== FINAL OMNI EXHAUSTIVE PORTFOLIO RESULTS ===")
    print("="*80)
    res_df = pd.DataFrame(results)
    if not res_df.empty:
        print(res_df.to_string())
    else:
        print("No assets found winning strategies.")

if __name__ == '__main__':
    main()
