import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import vectorbt as vbt
from itertools import combinations, product
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

FEES = 0.002 # 20 bps

def get_data(assets):
    print(f"Downloading data for {assets} starting from 2018-01-01...")
    data = yf.download(assets, start="2018-01-01")
    return data

class IndicatorFactory:
    """
    SOLID Principle: Open/Closed. Handles pure feature generation.
    """
    @staticmethod
    def calc_vectorized_poc(high, low, close, volume, window=30, bins=10):
        n = len(close)
        if n < window:
            return pd.Series(np.nan, index=close.index)
            
        poc = np.full(n, np.nan)
        h_vals, l_vals, c_vals, v_vals = high.values, low.values, close.values, volume.values
        
        for i in range(window, n):
            h_win, l_win, c_win, v_win = h_vals[i-window:i], l_vals[i-window:i], c_vals[i-window:i], v_vals[i-window:i]
            min_p, max_p = np.min(l_win), np.max(h_win)
            
            if max_p == min_p:
                poc[i] = c_win[-1]
                continue
                
            step = (max_p - min_p) / bins
            typ_price = (h_win + l_win + c_win) / 3
            bin_indices = np.clip(np.floor((typ_price - min_p) / step).astype(int), 0, bins - 1)
            
            vol_profile = np.bincount(bin_indices, weights=v_win, minlength=bins)
            poc[i] = min_p + (np.argmax(vol_profile) + 0.5) * step
            
        return pd.Series(poc, index=close.index)

    @classmethod
    def generate_all(cls, data, tema_params):
        open_, close, high, low, volume = data['Open'], data['Close'], data['High'], data['Low'], data['Volume']
        df = pd.DataFrame(index=data.index)
        df['Open'], df['Close'], df['High'], df['Low'], df['Volume'] = open_, close, high, low, volume
        
        f, m, s = tema_params
        df['TEMA_Fast'], df['TEMA_Med'], df['TEMA_Slow'] = ta.tema(close, length=f), ta.tema(close, length=m), ta.tema(close, length=s)
        df['TEMA_Signal'] = (df['TEMA_Fast'] > df['TEMA_Med']) & (df['TEMA_Med'] > df['TEMA_Slow'])
        
        macd = ta.macd(close)
        df['MACD_Signal'] = (macd.iloc[:, 0] > macd.iloc[:, 2]) if (macd is not None and not macd.empty) else False
        
        df['RSI_Signal'] = ta.rsi(close, length=14) > 50
        
        aroon = ta.aroon(high=high, low=low)
        df['Aroon_Signal'] = (aroon.iloc[:, 1] > aroon.iloc[:, 0]) if (aroon is not None and not aroon.empty) else False
        
        stc = ta.stc(close)
        df['STC_Signal'] = (stc.iloc[:, 0] > stc.iloc[:, 0].shift(1)) if (stc is not None and not stc.empty) else False
        
        bbands = ta.bbands(close)
        df['BB_Signal'] = ((close > bbands.iloc[:, 2]) | (close < bbands.iloc[:, 0])) if (bbands is not None and not bbands.empty) else False
        
        donchian = ta.donchian(high, low)
        df['DC_Signal'] = (high >= donchian.iloc[:, 2]) if (donchian is not None and not donchian.empty) else False
        
        adx = ta.adx(high, low, close)
        df['ADX_Signal'] = ((adx.iloc[:, 0] > 25) & (adx.iloc[:, 1] > adx.iloc[:, 2])) if (adx is not None and not adx.empty) else False
        
        ichimoku, _ = ta.ichimoku(high, low, close)
        df['Ichimoku_Signal'] = ((close > ichimoku.iloc[:, 0]) & (close > ichimoku.iloc[:, 1]) & (ichimoku.iloc[:, 2] > ichimoku.iloc[:, 3])) if (ichimoku is not None and not ichimoku.empty) else False
        
        df['OBV'] = ta.obv(close, volume)
        df['OBV_Signal'] = df['OBV'] > ta.sma(df['OBV'], length=20)
        df['CMF_Signal'] = ta.cmf(high, low, close, volume) > 0
        df['CCI_Signal'] = ta.cci(high, low, close) > 100
        df['ROC_Signal'] = ta.roc(close, length=10) > 0
        
        stoch = ta.stoch(high, low, close)
        df['Stoch_Signal'] = (stoch.iloc[:, 0] > stoch.iloc[:, 1]) if (stoch is not None and not stoch.empty) else False
            
        vwap = ta.vwap(high, low, close, volume)
        df['VWAP_Signal'] = (close > vwap) if (vwap is not None and not vwap.empty) else False
            
        df['POC_Signal'] = close > cls.calc_vectorized_poc(high, low, close, volume, window=30, bins=10)
        
        roll_high, roll_low = high.rolling(200).max(), low.rolling(200).min()
        df['FIB_Signal'] = close > (roll_high - (roll_high - roll_low) * 0.618)
        
        signals = [col for col in df.columns if col.endswith('_Signal')]
        df[signals] = df[signals].fillna(False)
        return df

class GenesisEngine:
    def __init__(self, asset_name, raw_data, lookback_warmup=200):
        self.asset_name = asset_name
        self.raw_data = raw_data.dropna()
        self.lookback_warmup = lookback_warmup
        split_idx = int(len(self.raw_data) * 0.60)
        self.train_raw = self.raw_data.iloc[:split_idx]
        self.test_raw = self.raw_data.iloc[split_idx:]

    def _calculate_metrics(self, portfolio):
        stats = portfolio.stats()
        returns = portfolio.returns()
        
        sortino = stats.get('Sortino Ratio', 0)
        sharpe = stats.get('Sharpe Ratio', 0)
        
        win_rate = stats.get('Win Rate [%]', 0)
        avg_win = stats.get('Avg Winning Trade [%]', 0)
        avg_loss = stats.get('Avg Losing Trade [%]', 0)
        expectancy = stats.get('Expectancy', 0)
        
        skew = returns.skew()
        skew = skew if pd.notna(skew) else 0
        
        # CONTINUOUS KELLY FRACTION (Log-Normal model — correct)
        # f* = μ_log / σ²_log (daily units; 252 cancels in numerator/denominator)
        log_ret = np.log1p(returns)
        active = log_ret[log_ret != 0.0]
        if len(active) > 2:
            mu_log  = active.mean()
            var_log = active.var(ddof=1)
            kelly = float(mu_log / var_log) if (pd.notna(var_log) and var_log > 1e-12) else 0.0
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
            
        # Ulcer Index
        drawdown = portfolio.drawdown()
        ulcer_index = np.sqrt(np.mean(drawdown ** 2)) * 100 if len(drawdown) > 0 else 0
        
        return sortino, sharpe, win_rate, expectancy, skew, kelly, ulcer_index, profit_factor

    def run_sensitivity_search(self):
        print(f"\n[SENSITIVITY] Grid Searching TEMA for {self.asset_name}...")
        close = self.train_raw['Close']
        results = {}
        
        for f, m, s in product(range(4, 41, 10), range(40, 91, 10), range(100, 201, 20)):
            if not (f < m < s): continue
            t_f, t_m, t_s = ta.tema(close, length=f), ta.tema(close, length=m), ta.tema(close, length=s)
            if t_f is None or t_m is None or t_s is None: continue
            
            entries = (t_f > t_m) & (t_m > t_s)
            shifted_entries = entries.shift(1).fillna(False)
            pf = vbt.Portfolio.from_signals(self.train_raw['Open'], entries=shifted_entries, exits=~shifted_entries, freq='1d', fees=FEES)
            results[(f, m, s)] = pf.stats().get('Sortino Ratio', 0)
            
        best_score, best_params = -999, (10, 64, 126)
        
        for (f, m, s), sortino in results.items():
            if pd.isna(sortino): continue
            neighbors = [results.get((nf, nm, ns)) for nf, nm, ns in [(f-10,m,s), (f+10,m,s), (f,m-10,s), (f,m+10,s), (f,m,s-20), (f,m,s+20)] if results.get((nf, nm, ns)) is not None]
            
            robust_score = sortino
            if neighbors:
                robust_score = (sortino * 0.4) + (np.nanmean(neighbors) * 0.6) - (np.nanstd(neighbors) * 1.5)
                
            if robust_score > best_score:
                best_score, best_params = robust_score, (f, m, s)
                
        self.tema_params = best_params
        return best_params

    def run_boruta(self):
        print("\n[BORUTA] Embargo-safe indicator generation and combination search...")
        
        # Embargo Strict: Generate features ONLY on the In-Sample set
        self.train_df = IndicatorFactory.generate_all(self.train_raw, self.tema_params)
        signals = [col for col in self.train_df.columns if col.endswith('_Signal')]
        
        best_score = -999
        self.best_combo = None
        self.best_metrics = None
        
        sig_cache = {sig: self.train_df[sig].values for sig in signals}
        open_vals = self.train_df['Open'].values
        
        for r in range(1, 4):
            for combo in combinations(signals, r):
                combined_entry = np.zeros(len(open_vals), dtype=bool)
                for sig in combo:
                    combined_entry |= sig_cache[sig]
                    
                shifted_entries = pd.Series(combined_entry).shift(1).fillna(False).values
                shifted_exits = pd.Series(~combined_entry).shift(1).fillna(False).values
                    
                pf = vbt.Portfolio.from_signals(open_vals, entries=shifted_entries, exits=shifted_exits, freq='1d', fees=FEES)
                
                try:
                    sortino, sharpe, win_rate, expectancy, skew, kelly, ulcer, pf_factor = self._calculate_metrics(pf)
                    trades = pf.stats().get('Total Trades', 0)
                    years  = (self.train_df.index[-1] - self.train_df.index[0]).days / 365.25
                    tpy    = trades / years if years > 0 else 0
                except: continue
                
                if tpy >= 2 and pd.notna(sortino) and kelly > 0 and pf_factor >= 2.0 and sortino > best_score:
                    best_score = sortino
                    self.best_combo = combo
                    self.best_metrics = {"Sortino": sortino, "Sharpe": sharpe, "Kelly": kelly, "Skew": skew, "Ulcer Index": ulcer, "Profit Factor": pf_factor}

        if self.best_combo:
            print(f"-> BEST ENSEMBLE: {self.best_combo} | KELLY: {self.best_metrics['Kelly']:.2f}")
        else:
            print("-> FATAL: NO COMBINATION SURVIVED CONTINUOUS KELLY CONSTRAINT.")
            
        return self.best_combo

    def apply_out_of_sample(self):
        if not getattr(self, 'best_combo', None): return
        print("\n[OUT-OF-SAMPLE] Strict Embargo Calculation & 3-Panel Audit...")
        
        # EMBARGO STRICT: No pre-slicing by date, use dynamic slicing
        test_df = self.test_raw.copy()
        test_df_full = IndicatorFactory.generate_all(self.raw_data, self.tema_params)
        test_df = test_df_full.loc[self.test_raw.index[0]:]
        
        combined_entry = pd.Series(False, index=test_df.index)
        for sig in self.best_combo:
            combined_entry |= test_df[sig]
            
        shifted_entries = combined_entry.shift(1).fillna(False).values
        shifted_exits = (~combined_entry).shift(1).fillna(False).values
            
        pf = vbt.Portfolio.from_signals(test_df['Open'], entries=shifted_entries, exits=shifted_exits, freq='1d', fees=FEES)
        
        sortino, sharpe, win_rate, expectancy, skew, kelly, ulcer, pf_factor = self._calculate_metrics(pf)
        
        print("\n================ FINAL OOS RESULTS ================")
        print(f"Strategy Ensemble : {self.best_combo}")
        print(f"Sharpe Ratio      : {sharpe:.2f}")
        print(f"Sortino Ratio     : {sortino:.2f}")
        print(f"Total Trades      : {pf.stats().get('Total Trades', 0)}")
        print(f"--- INSTITUTIONAL CASINO METRICS ---")
        print(f"Win Rate          : {win_rate:.2f}%")
        print(f"Expectancy        : {expectancy:.2f}")
        print(f"Return Skewness   : {skew:.2f}")
        print(f"Continuous Kelly  : {kelly:.2f}")
        print(f"Ulcer Index       : {ulcer:.2f}")
        print(f"Profit Factor     : {pf_factor:.2f}")
        if kelly <= 0: print("WARNING: NEGATIVE KELLY OBSERVED. GUARANTEED EVENTUAL RUIN.")
        print("===================================================\n")
        
        try:
            atr = ta.atr(test_df['High'], test_df['Low'], test_df['Close'], length=14)
            atr = atr if atr is not None and not atr.empty else test_df['Close'] * 0.02
            upper_band, lower_band = test_df['Close'] + (atr * 2), test_df['Close'] - (atr * 2)
            
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
            
            ax1.plot(test_df.index, test_df['Close'], color='black', alpha=0.7, label='Close')
            ax1.fill_between(test_df.index, lower_band, upper_band, color='gray', alpha=0.2, label='ATR Band')
            
            buy_signals = test_df.index[combined_entry]
            sell_signals = test_df.index[~combined_entry]
            
            if len(buy_signals) > 0: ax1.scatter(buy_signals, test_df.loc[buy_signals, 'Close'], marker='^', color='green', s=100)
            if len(sell_signals) > 0: ax1.scatter(sell_signals, test_df.loc[sell_signals, 'Close'], marker='v', color='red', s=100)
            
            ax1.set_title(f"{self.asset_name} - Regime Execution")
            
            drawdown = pf.drawdown() * 100
            ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
            ax2.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
            ax2.set_title("Visual Ulcer Index (Drawdown)")
            
            trade_mask = combined_entry.astype(int) + (~combined_entry).astype(int)
            monthly_trades = trade_mask.resample('ME').sum()
            ax3.bar(monthly_trades.index, monthly_trades.values, width=20, color='blue', alpha=0.6)
            ax3.set_title("Concurrency Audit")
            
            plt.tight_layout()
            filepath = f"C:/Users/Shivam Patel/.gemini/antigravity/brain/7b03663a-d01b-4302-8959-0a511c484299/{self.asset_name}_institutional_audit.png"
            plt.savefig(filepath)
            plt.close()
        except Exception as e:
            print(f"Plot failed: {e}")

def main():
    assets = ["BTC-USD", "SOL-USD", "QQQ", "DIA", "GLD", "SPY", "SMH", "XLV", "XLU", "SLV", "ETH-USD"]
    raw_data = get_data(assets)
    
    for asset in assets:
        print(f"\n{'='*60}")
        print(f"=== {asset} ===")
        print(f"{'='*60}")
        
        try:
            asset_data = pd.DataFrame({
                'Open': raw_data['Open'][asset],
                'Close': raw_data['Close'][asset],
                'High': raw_data['High'][asset],
                'Low': raw_data['Low'][asset],
                'Volume': raw_data['Volume'][asset]
            })
            
            engine = GenesisEngine(asset, asset_data)
            if len(engine.train_raw) < 200: continue
            
            engine.run_sensitivity_search()
            if engine.run_boruta():
                engine.apply_out_of_sample()
        except Exception as e:
            print(f"Error on {asset}: {e}")

if __name__ == "__main__":
    main()
