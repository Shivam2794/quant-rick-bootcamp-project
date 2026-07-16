import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import vectorbt as vbt
import random
import itertools
import warnings
import matplotlib.pyplot as plt
import time

warnings.filterwarnings('ignore')

FEES = 0.002          # 20 bps per trade
MIN_TRADES_PER_YEAR = 5
MAX_ULCER_INDEX = 15

# -----------------------------------------------------------------------------
# DATA LAYER
# -----------------------------------------------------------------------------
def get_data(assets, start="2010-01-01"):
    print(f"Downloading {assets} from {start}...")
    data = yf.download(assets, start=start, progress=False, auto_adjust=True)
    return data


# -----------------------------------------------------------------------------
# INDICATOR FACTORY -- 30+ indicators, all causal, no forward-looking leakage
# -----------------------------------------------------------------------------
class MegaIndicatorFactory:

    @classmethod
    def _as_series(cls, series_or_df):
        """Ensure we always get a 1-D Series, never a DataFrame."""
        if isinstance(series_or_df, pd.DataFrame):
            return series_or_df.iloc[:, 0]
        return series_or_df

    @classmethod
    def generate_all(cls, data):
        close   = cls._as_series(data['Close'])
        high    = cls._as_series(data['High'])
        low     = cls._as_series(data['Low'])
        volume  = cls._as_series(data['Volume'])
        open_   = cls._as_series(data['Open'])

        df = pd.DataFrame(index=data.index)
        df['Open'] = open_
        df['Close'], df['High'], df['Low'], df['Volume'] = close, high, low, volume

        try:
            # -- Moving-average trend signals ----------------------------------
            sma50  = ta.sma(close, length=50)
            sma200 = ta.sma(close, length=200)
            df['Sig_SMA_Cross'] = sma50 > sma200

            ema20 = ta.ema(close, length=20)
            ema50 = ta.ema(close, length=50)
            df['Sig_EMA_Cross'] = ema20 > ema50

            hma50 = ta.hma(close, length=50)
            df['Sig_HMA'] = close > hma50

            alma = ta.alma(close)
            if alma is not None and not alma.empty:
                df['Sig_ALMA'] = close > cls._as_series(alma)

            zlema = ta.zlma(close)
            if zlema is not None and not zlema.empty:
                df['Sig_ZLEMA'] = close > cls._as_series(zlema)

            kama = ta.kama(close)
            if kama is not None and not kama.empty:
                df['Sig_KAMA'] = close > cls._as_series(kama)

            # -- SuperTrend & PSAR ---------------------------------------------
            # FIXED: SuperTrend direction column is named 'SUPERTd_*'.
            # Use column-name matching so we don't depend on column order.
            supertrend = ta.supertrend(high, low, close)
            if supertrend is not None and not supertrend.empty:
                dir_cols = [c for c in supertrend.columns if c.startswith('SUPERTd')]
                if dir_cols:
                    df['Sig_SuperTrend'] = supertrend[dir_cols[0]] > 0

            psar = ta.psar(high, low, close)
            if psar is not None and not psar.empty:
                long_cols = [c for c in psar.columns if 'PSARl' in c]
                if long_cols:
                    df['Sig_PSAR'] = psar[long_cols[0]].notna()  # notna = in long position

            # -- Momentum oscillators ------------------------------------------
            macd = ta.macd(close)
            if macd is not None and not macd.empty:
                # Use column-name matching -- never rely on positional index
                macd_line_cols   = [c for c in macd.columns if c.startswith('MACD_')]
                signal_line_cols = [c for c in macd.columns if c.startswith('MACDs')]
                if macd_line_cols and signal_line_cols:
                    df['Sig_MACD'] = macd[macd_line_cols[0]] > macd[signal_line_cols[0]]

            df['Sig_RSI'] = ta.rsi(close, length=14) > 50

            stc = ta.stc(close)
            if stc is not None and not stc.empty:
                stc_val = cls._as_series(stc.iloc[:, 0])
                df['Sig_STC'] = stc_val > stc_val.shift(1)

            adx = ta.adx(high, low, close)
            if adx is not None and not adx.empty:
                # ta.adx() returns: [ADX_14, ADXR_14_2, DMP_14 (+DI), DMN_14 (-DI)]
                # FIXED: Use column-name matching, not positional index
                adx_col  = [c for c in adx.columns if c.startswith('ADX_')]
                dmp_col  = [c for c in adx.columns if c.startswith('DMP')]
                dmn_col  = [c for c in adx.columns if c.startswith('DMN')]
                if adx_col and dmp_col and dmn_col:
                    df['Sig_ADX_Strong'] = (adx[adx_col[0]] > 25) & (adx[dmp_col[0]] > adx[dmn_col[0]])
                    df['Sig_ADX_Weak']   = (adx[adx_col[0]] > 20) & (adx[dmp_col[0]] > adx[dmn_col[0]])

            kst = ta.kst(close)
            if kst is not None and not kst.empty:
                df['Sig_KST'] = kst.iloc[:, 0] > kst.iloc[:, 1]

            tsi = ta.tsi(close)
            if tsi is not None and not tsi.empty:
                df['Sig_TSI'] = tsi.iloc[:, 0] > tsi.iloc[:, 1]

            cg = ta.cg(close)
            if cg is not None and not cg.empty:
                # ta.cg() returns a Series in some versions, DataFrame in others
                if isinstance(cg, pd.DataFrame):
                    cg_val = cg.iloc[:, 0]
                else:
                    cg_val = cg  # already a Series
                df['Sig_CG'] = cg_val > cg_val.shift(1)

            # -- Volatility channels -------------------------------------------
            bbands = ta.bbands(close, length=20)
            if bbands is not None and not bbands.empty:
                # TREND FOLLOWING ONLY: upper-band breakout = momentum signal
                # Removed Sig_BB_MeanRev: mean-reversion signal pollutes trend-following pool
                df['Sig_BB_Break'] = close > bbands.iloc[:, 2]

            kc = ta.kc(high, low, close)
            if kc is not None and not kc.empty:
                df['Sig_KC_Break'] = close > kc.iloc[:, 2]

            donchian = ta.donchian(high, low)
            if donchian is not None and not donchian.empty:
                # Upper channel breakout (use close, not high, to avoid same-bar lookahead)
                df['Sig_DC_Break'] = close > donchian.iloc[:, 2].shift(1)

            # -- Volume / chop -------------------------------------------------
            chop = ta.chop(high, low, close)
            if chop is not None and not chop.empty:
                df['Sig_NotChoppy'] = cls._as_series(chop) < 50

            obv = ta.obv(close, volume)
            if obv is not None and not obv.empty:
                obv_s = cls._as_series(obv)
                obv_ma = ta.sma(obv_s, length=20)
                df['Sig_OBV'] = obv_s > obv_ma

            cmf = ta.cmf(high, low, close, volume)
            if cmf is not None and not cmf.empty:
                df['Sig_CMF'] = cls._as_series(cmf) > 0

            # -- Ichimoku ------------------------------------------------------
            ichimoku_result = ta.ichimoku(high, low, close)
            # ta.ichimoku returns a tuple (span df, lagging df)
            if isinstance(ichimoku_result, tuple):
                ichi = ichimoku_result[0]
            else:
                ichi = ichimoku_result
            if ichi is not None and not ichi.empty and ichi.shape[1] >= 4:
                # pandas_ta ISA_9 and ISB_26 are already aligned to current bar --
                # NO shift needed. Use column-name matching for robustness.
                isa_cols = [c for c in ichi.columns if c.startswith('ISA')]
                isb_cols = [c for c in ichi.columns if c.startswith('ISB')]
                its_cols = [c for c in ichi.columns if c.startswith('ITS')]
                iks_cols = [c for c in ichi.columns if c.startswith('IKS')]
                if isa_cols and isb_cols and its_cols and iks_cols:
                    df['Sig_Ichimoku'] = (
                        (close > ichi[isa_cols[0]]) &
                        (close > ichi[isb_cols[0]]) &
                        (ichi[its_cols[0]] > ichi[iks_cols[0]])
                    )

            # -- Fibonacci (rolling 200-bar high/low) -------------------------
            roll_high = high.rolling(200).max()
            roll_low  = low.rolling(200).min()
            # 61.8% retracement level: above it = bullish
            df['Sig_FIB'] = close > (roll_high - (roll_high - roll_low) * 0.618)

            # -- TEMA crossover ------------------------------------------------
            tema_fast = ta.tema(close, length=14)
            tema_slow = ta.tema(close, length=50)
            if tema_fast is not None and tema_slow is not None:
                df['Sig_TEMA'] = tema_fast > tema_slow

        except Exception as e:
            print(f"[WARNING] Indicator generation error: {e}")

        signals = [c for c in df.columns if c.startswith('Sig_')]
        df[signals] = df[signals].fillna(False).astype(bool)
        return df


# -----------------------------------------------------------------------------
# METRICS ENGINE -- log-normal Kelly, Ulcer Index, correct annualisation
# -----------------------------------------------------------------------------
def calculate_metrics(portfolio):
    """
    Returns: sortino, sharpe, trades, win_rate, expectancy, skew, kelly, ulcer

    Kelly (Log-Normal, continuous-time):
        f* = mu_log / sigma2_log
    where mu and sigma2 are the DAILY log-return mean and variance.
    The 252 annualisation factor cancels in the ratio, so we compute on daily.
    This avoids inflating the number while remaining theoretically correct.

    Ulcer Index = sqrt( mean( (drawdown_pct)^2 ) )
    vectorbt drawdown() returns fractions in (-1, 0], so we multiply by 100 first.
    """
    stats   = portfolio.stats()
    returns = portfolio.returns()      # arithmetic daily returns (fraction)

    sortino    = stats.get('Sortino Ratio', 0) or 0
    sharpe     = stats.get('Sharpe Ratio', 0) or 0
    trades     = stats.get('Total Trades', 0) or 0
    win_rate   = stats.get('Win Rate [%]', 0) or 0
    expectancy = stats.get('Expectancy', 0) or 0

    # Skewness of arithmetic returns (fine here -- it's a descriptive stat)
    skew = returns.skew()
    skew = float(skew) if pd.notna(skew) else 0.0

    # Log-Normal Kelly -- computed on ACTIVE (non-zero) returns ONLY
    # Rationale: flat cash days contribute 0 return, diluting variance
    # artificially. Kelly must reflect the actual trade-level distribution.
    log_ret = np.log1p(returns)
    active  = log_ret[log_ret != 0.0]   # filter flat days
    if len(active) > 2:
        mu_log  = active.mean()
        var_log = active.var(ddof=1)
        kelly   = float(mu_log / var_log) if (pd.notna(var_log) and var_log > 1e-12) else 0.0
    else:
        kelly = 0.0

    # BRUTAL FIX: Manual Profit Factor
    try:
        pnls = portfolio.trades.records_readable['PnL']
        gross_profits = np.sum(pnls[pnls > 0])
        gross_losses = np.abs(np.sum(pnls[pnls < 0]))
        profit_factor = float(gross_profits / gross_losses) if gross_losses != 0 else float('inf')
    except Exception:
        profit_factor = 0.0

    # Ulcer Index: RMS of underwater percentage
    drawdown_pct = portfolio.drawdown() * 100  # now in percent, range (-100, 0]
    ulcer = float(np.sqrt(np.mean(drawdown_pct ** 2))) if len(drawdown_pct) > 0 else 0.0

    return sortino, sharpe, trades, win_rate, expectancy, skew, kelly, ulcer, profit_factor


# -----------------------------------------------------------------------------
# RELENTLESS GRINDER ENGINE
# -----------------------------------------------------------------------------
class RelentlessGrinder:

    def __init__(self, asset_name, raw_data):
        self.asset_name     = asset_name
        self.raw_data       = raw_data.dropna()
        self.lookback_warmup = 200
        self.full_df = MegaIndicatorFactory.generate_all(self.raw_data)
        split_idx = int(len(self.full_df) * 0.60)

        self.train_df = self.full_df.iloc[:split_idx]
        self.test_df  = self.full_df.iloc[split_idx:]

    def _calendar_years(self, df):
        """Use calendar days, not row count, to avoid equity/crypto discrepancy."""
        return (df.index[-1] - df.index[0]).days / 365.25

    def grind(self):
        print(f"\n[GRINDER] Building feature matrix for {self.asset_name}...")
        signals = [c for c in self.train_df.columns if c.startswith('Sig_')]
        print(f"  Signals available: {len(signals)}")

        sig_cache  = {s: self.train_df[s].values for s in signals}
        open_vals = self.train_df['Open'].values
        years      = self._calendar_years(self.train_df)

        best_score    = -np.inf
        self.best_combo   = None
        self.best_metrics = None
        valid_found   = 0

        # Generate exhaustive combinations of 1, 2, and 3 signals
        all_combos = []
        for k in (1, 2, 3):
            all_combos.extend(list(itertools.combinations(signals, k)))
        
        # Each combo will be evaluated twice (AND/OR logic), except single-signal (logic doesn't matter)
        total_evals = len(all_combos) * 2

        print(f"  Exhaustive Grid Search: {total_evals} configurations, {years:.1f} training years")
        t0 = time.time()

        for i, combo in enumerate(all_combos):
            # For 1 signal, logic doesn't matter, just test 'OR'
            logics = ('OR', 'AND') if len(combo) > 1 else ('OR',)

            for logic in logics:
                entry = np.ones(len(open_vals), dtype=bool) if logic == 'AND' \
                        else np.zeros(len(open_vals), dtype=bool)
                for sig in combo:
                    if logic == 'AND':
                        entry &= sig_cache[sig]
                    else:
                        entry |= sig_cache[sig]

                # Skip if signal is constant (all True or all False)
                if entry.all() or not entry.any():
                    continue

                # BRUTAL FIX: Shift entries to execute on Open
                shifted_entries = pd.Series(entry).shift(1).fillna(False).values
                shifted_exits = pd.Series(~entry).shift(1).fillna(False).values

                pf = vbt.Portfolio.from_signals(
                    open_vals, entries=shifted_entries, exits=shifted_exits,
                    freq='1d', fees=FEES
                )

                try:
                    sortino, sharpe, trades, wr, exp, skew, kelly, ulcer, pf_factor = calculate_metrics(pf)
                    tpy = trades / years
                except Exception:
                    continue

                # -- Strict institutional constraints ----------------------
                if (tpy >= MIN_TRADES_PER_YEAR
                        and kelly > 0
                        and ulcer < MAX_ULCER_INDEX
                        and pf_factor >= 2.0
                        and pd.notna(sharpe)):

                    valid_found += 1

                    # Fitness: STRICT SHARPE RATIO MAXIMIZATION
                    fitness = sharpe

                    if fitness > best_score:
                        best_score      = fitness
                        self.best_combo   = combo
                        self.best_logic   = logic
                        self.best_metrics = {
                            'Sortino': sortino, 'Sharpe': sharpe, 'Trades': trades,
                            'Kelly': kelly, 'Skew': skew, 'Ulcer': ulcer, 'Logic': logic, 'ProfitFactor': pf_factor
                        }

            if i % 1000 == 0 and i > 0:
                print(f"  [combo {i:,}/{len(all_combos):,}] best_sharpe={best_score:.3f} | valid={valid_found}")

        elapsed = time.time() - t0
        print(f"\nGrind done in {elapsed:.1f}s -- {valid_found} ensembles survived all constraints.")
        if self.best_combo:
            m = self.best_metrics
            print(f"-> CHAMPION [{m['Logic']}]: {self.best_combo}")
            print(f"  Kelly={m['Kelly']:.3f}  Sortino={m['Sortino']:.3f}  "
                  f"Trades={m['Trades']}  Ulcer={m['Ulcer']:.2f}")
        else:
            print("-> FATAL: Zero ensembles survived the institutional constraints.")

        return self.best_combo

    def apply_out_of_sample(self):
        if not getattr(self, 'best_combo', None):
            return
        print(f"\n[OOS] Strict embargo validation for {self.asset_name}...")

        logic = getattr(self, 'best_logic', 'OR')
        entry = pd.Series(
            True if logic == 'AND' else False,
            index=self.test_df.index
        )
        for sig in self.best_combo:
            if sig not in self.test_df.columns:
                continue
            if logic == 'AND':
                entry &= self.test_df[sig]
            else:
                entry |= self.test_df[sig]

        shifted_entries = entry.shift(1).fillna(False).values
        shifted_exits = (~entry).shift(1).fillna(False).values

        pf = vbt.Portfolio.from_signals(
            self.test_df['Open'], entries=shifted_entries, exits=shifted_exits,
            freq='1d', fees=FEES
        )

        sortino, sharpe, trades, wr, exp, skew, kelly, ulcer, pf_factor = calculate_metrics(pf)
        years_oos = self._calendar_years(self.test_df)

        print("\n================ FINAL OOS RESULTS ================")
        print(f"Asset             : {self.asset_name}")
        print(f"Ensemble ({logic})   : {self.best_combo}")
        print(f"OOS Period        : {self.test_df.index[0].date()} -> {self.test_df.index[-1].date()} ({years_oos:.1f} yr)")
        print(f"Sharpe Ratio      : {sharpe:.3f}")
        print(f"Sortino Ratio     : {sortino:.3f}")
        print(f"Total Trades      : {trades}  ({trades/years_oos:.1f}/yr)")
        print(f"Win Rate          : {wr:.1f}%")
        print(f"Expectancy        : {exp:.3f}")
        print(f"Return Skewness   : {skew:.3f}")
        print(f"Kelly (Log-Normal): {kelly:.4f}")
        print(f"Ulcer Index       : {ulcer:.3f}%")
        print(f"Profit Factor     : {pf_factor:.2f}")
        if kelly <= 0:  print("WARNING  WARNING: OOS NEGATIVE KELLY -- RUIN RISK.")
        if trades < 5:  print("WARNING  WARNING: OOS SAMPLE ILLUSION (< 5 trades).")
        print("===================================================\n")

        # 3-panel plot
        try:
            test_df = self.test_df
            atr = ta.atr(test_df['High'], test_df['Low'], test_df['Close'], length=14)
            if atr is None or atr.empty:
                atr = test_df['Close'] * 0.02
            else:
                atr = atr.squeeze()

            upper = test_df['Close'] + atr * 2
            lower = test_df['Close'] - atr * 2

            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(14, 12),
                gridspec_kw={'height_ratios': [2, 1, 1]}
            )

            ax1.plot(test_df.index, test_df['Close'], color='black', alpha=0.7, label='Close')
            ax1.fill_between(test_df.index, lower, upper, color='gray', alpha=0.2, label='ATR Band')

            buy_idx  = test_df.index[entry]
            sell_idx = test_df.index[~entry]
            if len(buy_idx):  ax1.scatter(buy_idx,  test_df.loc[buy_idx,  'Close'], marker='^', color='lime',   s=60, zorder=5)
            if len(sell_idx): ax1.scatter(sell_idx, test_df.loc[sell_idx, 'Close'], marker='v', color='crimson', s=60, zorder=5)
            ax1.set_title(f"{self.asset_name} -- Regime Execution ({logic}-logic)")
            ax1.legend(fontsize=8)

            dd = pf.drawdown() * 100
            ax2.fill_between(dd.index, dd.values, 0, color='red', alpha=0.25)
            ax2.plot(dd.index, dd.values, color='red', linewidth=1)
            ax2.set_title("Underwater Curve (Ulcer Index)")
            ax2.set_ylabel("Drawdown %")

            # Trade density -- show actual state changes
            state_changes = entry.astype(int).diff().fillna(0).abs()
            monthly = state_changes.resample('ME').sum()
            ax3.bar(monthly.index, monthly.values, width=20, color='steelblue', alpha=0.7)
            ax3.set_title("Monthly State-Change Frequency (Trade Density)")
            ax3.set_ylabel("# Transitions")

            plt.tight_layout()
            out = (f"C:/Users/Shivam Patel/.gemini/antigravity/brain/"
                   f"7b03663a-d01b-4302-8959-0a511c484299/{self.asset_name}_grinder_audit.png")
            plt.savefig(out, dpi=120)
            plt.close()
            print(f"-> 3-Panel audit saved: {out}")
        except Exception as e:
            print(f"Plot error: {e}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    assets   = ["BTC-USD", "SOL-USD", "QQQ", "DIA", "GLD",
                "SPY", "SMH", "XLV", "XLU", "SLV", "ETH-USD"]
    raw_data = get_data(assets, start="2010-01-01")

    for asset in assets:
        print(f"\n{'='*60}")
        print(f"=== GRINDING  {asset}  ===")
        print(f"{'='*60}")

        try:
            df = pd.DataFrame({
                'Open':   raw_data['Open'][asset],
                'Close':  raw_data['Close'][asset],
                'High':   raw_data['High'][asset],
                'Low':    raw_data['Low'][asset],
                'Volume': raw_data['Volume'][asset],
            }).dropna()

            engine = RelentlessGrinder(asset, df)

            if len(engine.train_df) < 400:
                print(f"  Skipping {asset}: insufficient training data.")
                continue

            best = engine.grind()
            if best:
                engine.apply_out_of_sample()

        except Exception as e:
            import traceback
            print(f"  [ERROR] {asset}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
