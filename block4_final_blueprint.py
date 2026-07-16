import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import vectorbt as vbt
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

FEES = 0.002
SAFE_HAVEN = 'GLD'

# These are the optimized structural 'Champions' derived from the Omni-Grinder
CHAMPION_ENSEMBLES = {
    'BTC-USD': ['Sig_EMA_Cross', 'Sig_FIB_236_Hold'],
    'SOL-USD': ['Sig_SMA_Cross', 'Sig_RSI'],
    'QQQ': ['Sig_TEMA', 'Sig_CMF_Inst'],
    'DIA': ['Sig_FIB_618_Golden'],
    'GLD': ['Sig_SuperTrend', 'Sig_TSI'],
    'SPY': ['Sig_FIB_382_Hold', 'Sig_WillR_Std_OS'],
    'SMH': ['Sig_TEMA', 'Sig_RSI'],
    'XLV': ['Sig_MACD', 'Sig_FIB_500_Hold'],
    'XLU': ['Sig_HMA', 'Sig_Stoch_Std_OB'],
    'SLV': ['Sig_KC_Break', 'Sig_WillR_Fast_OS'],
    'ETH-USD': ['Sig_EMA_Cross', 'Sig_Stoch_Fast_OB']
}

class MegaIndicatorFactory:
    """The fully sanitized Block 4 feature engineering factory."""
    
    @classmethod
    def _as_series(cls, series_or_df):
        if isinstance(series_or_df, pd.DataFrame):
            return series_or_df.iloc[:, 0]
        return series_or_df

    @classmethod
    def _safe_assign(cls, df, name, series):
        if series is not None and not series.empty:
            df[name] = series.fillna(False).astype(bool)
        else:
            df[name] = False

    @classmethod
    def generate_all(cls, data):
        c = data['Close']
        h = data['High']
        l = data['Low']
        v = data['Volume']

        df = pd.DataFrame(index=data.index)
        
        # 1. Moving Averages
        sma50 = ta.sma(c, length=50)
        sma200 = ta.sma(c, length=200)
        cls._safe_assign(df, 'Sig_SMA_Cross', sma50 > sma200)

        ema20 = ta.ema(c, length=20)
        ema50 = ta.ema(c, length=50)
        cls._safe_assign(df, 'Sig_EMA_Cross', ema20 > ema50)

        hma50 = ta.hma(c, length=50)
        cls._safe_assign(df, 'Sig_HMA', c > hma50)

        tema_fast = ta.tema(c, length=14)
        tema_slow = ta.tema(c, length=50)
        if tema_fast is not None and tema_slow is not None:
            cls._safe_assign(df, 'Sig_TEMA', tema_fast > tema_slow)

        # 2. Trend
        supertrend = ta.supertrend(h, l, c)
        if supertrend is not None and not supertrend.empty:
            dir_cols = [col for col in supertrend.columns if col.startswith('SUPERTd')]
            if dir_cols:
                cls._safe_assign(df, 'Sig_SuperTrend', supertrend[dir_cols[0]] > 0)

        # 3. Oscillators
        macd = ta.macd(c)
        if macd is not None and not macd.empty:
            m_cols = [col for col in macd.columns if col.startswith('MACD_')]
            s_cols = [col for col in macd.columns if col.startswith('MACDs')]
            if m_cols and s_cols:
                cls._safe_assign(df, 'Sig_MACD', macd[m_cols[0]] > macd[s_cols[0]])

        cls._safe_assign(df, 'Sig_RSI', ta.rsi(c, length=14) > 50)
        
        tsi = ta.tsi(c)
        if tsi is not None and not tsi.empty:
            cls._safe_assign(df, 'Sig_TSI', tsi.iloc[:, 0] > tsi.iloc[:, 1])
            
        st_std = ta.stoch(h, l, c, k=14, d=3, smooth_k=3)
        if st_std is not None:
            cls._safe_assign(df, 'Sig_Stoch_Std_OB', st_std.iloc[:, 0] > 80)

        cls._safe_assign(df, 'Sig_WillR_Std_OS', ta.willr(h, l, c, length=14) < -80)
        cls._safe_assign(df, 'Sig_WillR_Fast_OS', ta.willr(h, l, c, length=10) < -80)

        # 4. Channels
        kc = ta.kc(h, l, c)
        if kc is not None and not kc.empty:
            cls._safe_assign(df, 'Sig_KC_Break', c > kc.iloc[:, 2])

        # 5. Volume/Flow
        cls._safe_assign(df, 'Sig_CMF_Inst', ta.cmf(h, l, c, v, length=21) > 0.05)

        # 6. Fibs
        roll_max = h.rolling(200).max()
        roll_min = l.rolling(200).min()
        fib_r = roll_max - roll_min
        
        cls._safe_assign(df, 'Sig_FIB_236_Hold', c > (roll_max - fib_r * 0.236))
        cls._safe_assign(df, 'Sig_FIB_382_Hold', c > (roll_max - fib_r * 0.382))
        cls._safe_assign(df, 'Sig_FIB_500_Hold', c > (roll_max - fib_r * 0.500))
        cls._safe_assign(df, 'Sig_FIB_618_Golden', c > (roll_max - fib_r * 0.618))

        return df

def fetch_data(assets):
    raw_data = {}
    print(f"Downloading {assets} from 2018-01-01...")
    for a in assets:
        try:
            df = yf.download(a, start='2018-01-01', progress=False, auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                raw_data[a] = df
        except Exception as e:
            print(f"Failed to fetch {a}: {e}")
    return raw_data

def calculate_metrics(pf):
    trades = pf.trades.records_readable
    total_trades = len(trades)
    
    if total_trades == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    sortino = pf.sortino_ratio() if callable(pf.sortino_ratio) else pf.sortino_ratio
    sharpe = pf.sharpe_ratio() if callable(pf.sharpe_ratio) else pf.sharpe_ratio
    
    win_rate = (trades['Return'] > 0).mean() * 100
    avg_win = trades[trades['Return'] > 0]['Return'].mean()
    avg_loss = abs(trades[trades['Return'] < 0]['Return'].mean())
    avg_loss = avg_loss if not pd.isna(avg_loss) and avg_loss != 0 else 1e-5
    avg_win = avg_win if not pd.isna(avg_win) else 0

    exp = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)
    skew = trades['Return'].skew()
    
    kelly = (win_rate / 100) - ((1 - win_rate / 100) / (avg_win / avg_loss)) if avg_loss > 0 else 0
    kelly = kelly * 100

    ulcer = pf.drawdown().mean() * 100 if pf.drawdown() is not None else 0
    
    pos_returns = trades[trades['Return'] > 0]['Return'].sum()
    neg_returns = abs(trades[trades['Return'] < 0]['Return'].sum())
    pf_factor = pos_returns / neg_returns if neg_returns > 0 else float('inf')

    return sortino, sharpe, total_trades, win_rate, exp, skew, kelly, ulcer, pf_factor

def main():
    assets = list(CHAMPION_ENSEMBLES.keys())
    raw_data = fetch_data(assets)
    
    # 1. Align Dates (Intersection of all assets)
    common_index = None
    for a in assets:
        if a in raw_data:
            if common_index is None:
                common_index = raw_data[a].index
            else:
                common_index = common_index.intersection(raw_data[a].index)
                
    # 2. Build Structural Feature Matrices
    print("\n[BLUEPRINT] Building Omni-Grinder feature matrices and 28-RSI rankings...")
    rsis = pd.DataFrame(index=common_index)
    regime_signals = pd.DataFrame(index=common_index)
    close_prices = pd.DataFrame(index=common_index)
    open_prices = pd.DataFrame(index=common_index)
    
    for a in assets:
        if a not in raw_data:
            continue
            
        df = raw_data[a].loc[common_index]
        close_prices[a] = df['Close']
        open_prices[a] = df['Open']
        
        # 28-day Absolute Momentum (RSI cross-sectional ranking)
        rsis[a] = ta.rsi(df['Close'], length=28)
        
        # Build MegaIndicator Matrix
        feature_df = MegaIndicatorFactory.generate_all(df)
        
        # Retrieve Champion Ensemble
        champs = CHAMPION_ENSEMBLES.get(a, [])
        asset_regime = pd.Series(True, index=common_index)
        for sig in champs:
            if sig in feature_df.columns:
                asset_regime &= feature_df[sig]
                
        regime_signals[a] = asset_regime
        
    rsis = rsis.fillna(50)
    
    # 3. Portfolio Amalgamation (Execution Engine)
    print("\n[BLUEPRINT] Executing Cross-Sectional Ranking & GLD Safe Haven Override...")
    
    allocations = pd.DataFrame(0.0, index=common_index, columns=assets)
    N_SLOTS = 3
    
    # Pre-allocate rows
    current_allocs = {a: 0.0 for a in assets}
    
    for i in range(len(common_index)):
        if i % 5 == 0:  # Rebalance every 5 days (Weekly)
            row_rsi = rsis.iloc[i]
            row_sig = regime_signals.iloc[i]
            
            ranked = row_rsi.sort_values(ascending=False)
            top_assets = ranked.head(N_SLOTS).index.tolist()
            
            new_allocs = {a: 0.0 for a in assets}
            
            for asset in top_assets:
                # Core Rule: Asset must be Top 3, RSI > 50, AND Omni-Regime must be True
                if row_rsi[asset] > 50 and row_sig[asset]:
                    new_allocs[asset] += (1.0 / N_SLOTS)
                else:
                    new_allocs[SAFE_HAVEN] += (1.0 / N_SLOTS)
                    
            current_allocs = new_allocs
            
        for a in assets:
            allocations.at[common_index[i], a] = current_allocs[a]
            
    # CRITICAL: Next-Open Execution Constraint to eliminate Lookahead Bias
    shifted_allocs = allocations.shift(1).fillna(0.0)
    
    print("\n[BLUEPRINT] Running VectorBT Institutional Audit...")
    
    pf = vbt.Portfolio.from_orders(
        close_prices, # used for valuation
        size=shifted_allocs, # using weights as size is a simplification, let's use from_weights instead
        size_type='targetpercent',
        price=open_prices, # Executed at Open!
        freq='1d', 
        fees=FEES,
        group_by=True
    )
    
    # The output 
    sortino, sharpe, trades, wr, exp, skew, kelly, ulcer, pf_factor = calculate_metrics(pf)
    
    print(f"\n=======================================================")
    print(f"=== BLOCK 4: OMNI-GRINDER / AMALGAMATOR SYNTHESIS ===")
    print(f"=======================================================")
    print(f"Sharpe Ratio      : {sharpe:.2f}")
    print(f"Sortino Ratio     : {sortino:.2f}")
    print(f"Total Trades      : {trades}")
    print(f"--- INSTITUTIONAL CASINO METRICS ---")
    print(f"Win Rate          : {wr:.2f}%")
    print(f"Expectancy        : {exp:.2f}")
    print(f"Return Skewness   : {skew:.2f}")
    print(f"Continuous Kelly  : {kelly:.2f}")
    print(f"Ulcer Index       : {ulcer:.2f}")
    print(f"Profit Factor     : {pf_factor:.2f}")
    print(f"=======================================================")
    
    if kelly < 0:
        print("WARNING: NEGATIVE KELLY OBSERVED. GUARANTEED EVENTUAL RUIN.")
        
    try:
        # Generate the Final Blueprint Visual Audit
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1, 1]})
        
        pf.cumulative_returns().plot(ax=ax1, color='gold', lw=2)
        ax1.set_title("Block 4 Synthesis: Combined Equity Curve (Next-Open Execution, 20bps Fees)", color='white')
        ax1.set_facecolor('black')
        ax1.grid(color='gray', linestyle='--', alpha=0.3)
        ax1.tick_params(colors='white')
        
        # Allocations
        shifted_allocs.plot.area(ax=ax2, colormap='tab20', alpha=0.8, legend=False)
        ax2.set_title("Portfolio Allocations (Safe Haven: GLD)", color='white')
        ax2.set_facecolor('black')
        ax2.grid(color='gray', linestyle='--', alpha=0.3)
        ax2.tick_params(colors='white')
        
        # Drawdowns
        pf.drawdown() * 100
        dd = pf.drawdown() * 100
        ax3.fill_between(dd.index, dd, 0, color='red', alpha=0.5)
        ax3.set_title("Strategy Drawdown (%)", color='white')
        ax3.set_facecolor('black')
        ax3.grid(color='gray', linestyle='--', alpha=0.3)
        ax3.tick_params(colors='white')
        
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        plt.savefig('block4_final_blueprint_audit.png', facecolor=fig.get_facecolor(), edgecolor='none')
        print(f"\n[SUCCESS] Final structural audit saved to block4_final_blueprint_audit.png")
        
    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == '__main__':
    main()
