import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import vectorbt as vbt
import matplotlib.pyplot as plt

FEES = 0.002
RISK_ASSETS = ['SPY', 'QQQ', 'DIA', 'SMH', 'SLV']
SAFE_HAVEN = 'GLD'

def safe_assign(df, name, condition):
    try:
        df[name] = condition.fillna(False).astype(bool)
    except Exception:
        df[name] = False

def get_signal_spy_qqq(df):
    d = df.copy()
    c, h, l = d['Close'], d['High'], d['Low']
    
    st_mac = ta.stoch(h, l, c, k=21, d=5, smooth_k=5)
    safe_assign(d, 'Stoch', st_mac.iloc[:, 0] < 20 if st_mac is not None else pd.Series(False, index=c.index))
    
    ps = ta.psar(h, l, c, af0=0.01, af=0.01, max_af=0.10)
    lc = [col for col in ps.columns if col.startswith('PSARl_')] if ps is not None else []
    safe_assign(d, 'PSAR', pd.notna(ps[lc[0]]) if lc else pd.Series(False, index=c.index))
    
    r_max = h.rolling(200).max()
    r_min = l.rolling(200).min()
    fib_r = r_max - r_min
    safe_assign(d, 'FIB', c > (r_max - fib_r * 0.236))
    
    return d['Stoch'] | d['PSAR'] | d['FIB']

def get_signal_smh(df):
    d = df.copy()
    c, h, l = d['Close'], d['High'], d['Low']
    
    alma = ta.alma(c)
    safe_assign(d, 'ALMA', c > alma.iloc[:, 0] if isinstance(alma, pd.DataFrame) else c > alma)
    
    chop = ta.chop(h, l, c)
    safe_assign(d, 'NotChoppy', chop < 50 if chop is not None else pd.Series(False, index=c.index))
    
    ichi = ta.ichimoku(h, l, c, tenkan=18, kijun=52, senkou=104)
    if isinstance(ichi, tuple): ichi = ichi[0]
    isa = [col for col in ichi.columns if col.startswith('ISA_')] if ichi is not None else []
    isb = [col for col in ichi.columns if col.startswith('ISB_')] if ichi is not None else []
    safe_assign(d, 'Ichi', ichi[isa[0]] > ichi[isb[0]] if isa and isb else pd.Series(False, index=c.index))
    
    return d['ALMA'] | d['NotChoppy'] | d['Ichi']

def get_signal_dia(df):
    d = df.copy()
    c, h, l = d['Close'], d['High'], d['Low']
    
    tsi = ta.tsi(c)
    safe_assign(d, 'TSI', tsi.iloc[:, 0] > tsi.iloc[:, 1] if tsi is not None else pd.Series(False, index=c.index))
    
    willr = ta.willr(h, l, c, length=50)
    safe_assign(d, 'WillR', willr < -80 if willr is not None else pd.Series(False, index=c.index))
    
    ichi = ta.ichimoku(h, l, c, tenkan=18, kijun=52, senkou=104)
    if isinstance(ichi, tuple): ichi = ichi[0]
    isa = [col for col in ichi.columns if col.startswith('ISA_')] if ichi is not None else []
    safe_assign(d, 'Ichi', c > ichi[isa[0]] if isa else pd.Series(False, index=c.index))
    
    return d['TSI'] | d['WillR'] | d['Ichi']

def get_signal_slv(df):
    d = df.copy()
    c, h, l, v = d['Close'], d['High'], d['Low'], d['Volume']
    
    cmf = ta.cmf(h, l, c, v)
    safe_assign(d, 'CMF', cmf > 0 if cmf is not None else pd.Series(False, index=c.index))
    
    pvt = ta.pvt(c, v)
    safe_assign(d, 'PVT', pvt > ta.ema(pvt, length=21) if pvt is not None else pd.Series(False, index=c.index))
    
    ichi = ta.ichimoku(h, l, c, tenkan=9, kijun=26, senkou=52)
    if isinstance(ichi, tuple): ichi = ichi[0]
    isa = [col for col in ichi.columns if col.startswith('ISA_')] if ichi is not None else []
    safe_assign(d, 'Ichi', c > ichi[isa[0]] if isa else pd.Series(False, index=c.index))
    
    return d['CMF'] & d['PVT'] & d['Ichi']

def main():
    assets = RISK_ASSETS + [SAFE_HAVEN]
    print("Fetching master portfolio data...")
    df_raw = yf.download(assets, start='2010-01-01', progress=False, auto_adjust=True)
    
    prices = df_raw['Close'].dropna()
    open_prices = df_raw['Open'].reindex(prices.index).fillna(method='ffill')
    high_prices = df_raw['High'].reindex(prices.index).fillna(method='ffill')
    low_prices = df_raw['Low'].reindex(prices.index).fillna(method='ffill')
    vol_prices = df_raw['Volume'].reindex(prices.index).fillna(0)

    # Precalculate RSI 28 and Signals for all Risk Assets
    print("Calculating signals and cross-sectional momentum...")
    rsis = pd.DataFrame(index=prices.index)
    signals = pd.DataFrame(index=prices.index)
    
    for asset in RISK_ASSETS:
        c, h, l, v = prices[asset], high_prices[asset], low_prices[asset], vol_prices[asset]
        single_df = pd.DataFrame({'Open': open_prices[asset], 'High': h, 'Low': l, 'Close': c, 'Volume': v})
        
        # Calculate 28-period RSI
        rsis[asset] = ta.rsi(c, length=28)
        
        # Calculate Signals
        if asset in ['SPY', 'QQQ']: sig = get_signal_spy_qqq(single_df)
        elif asset == 'SMH': sig = get_signal_smh(single_df)
        elif asset == 'DIA': sig = get_signal_dia(single_df)
        elif asset == 'SLV': sig = get_signal_slv(single_df)
        
        signals[asset] = sig.fillna(False).values
        
    rsis = rsis.dropna()
    signals = signals.reindex(rsis.index)
    prices = prices.reindex(rsis.index)
    open_prices = open_prices.reindex(rsis.index)

    # Initialize allocation weights DataFrame
    allocations = pd.DataFrame(0.0, index=rsis.index, columns=assets)
    
    # Run the Cross-Sectional Ranking Engine (Rebalance weekly - every 5 days for simplicity)
    current_allocs = {a: 0.0 for a in assets}
    N_SLOTS = 3
    
    for i in range(len(rsis)):
        # Rebalance every 5 days
        if i % 5 == 0:
            row_rsi = rsis.iloc[i]
            row_sig = signals.iloc[i]
            
            # Rank descending
            ranked = row_rsi.sort_values(ascending=False)
            top_assets = ranked.head(N_SLOTS).index.tolist()
            
            new_allocs = {a: 0.0 for a in assets}
            
            for asset in top_assets:
                # The rule: if RSI > 50 AND Omni-Signal == True -> hold asset
                if row_rsi[asset] > 50 and row_sig[asset]:
                    new_allocs[asset] += (1.0 / N_SLOTS)
                else:
                    # Park in safe haven
                    new_allocs[SAFE_HAVEN] += (1.0 / N_SLOTS)
                    
            current_allocs = new_allocs
            
        # Write daily allocs (shifted by 1 to execute on Next Open!)
        # Actually vectorbt handles from_orders or we can just shift the target weights.
        for a in assets:
            allocations.at[rsis.index[i], a] = current_allocs[a]
            
    # Shift allocations by 1 day to ensure next-open execution and zero look-ahead bias
    allocations = allocations.shift(1).fillna(0.0)

    print("Simulating final portfolio via vectorbt...")
    
    pf = vbt.Portfolio.from_orders(
        close=open_prices,
        size=allocations.values,
        size_type='targetpercent',
        fees=FEES,
        freq='1d',
        group_by=True  # Group all columns into a single portfolio
    )
    
    # Compare against Buy & Hold SPY
    bh = vbt.Portfolio.from_holding(open_prices['SPY'], freq='1d')
    
    def calc_stats(port):
        rets = port.returns()
        val = port.value()
        
        # Ensure rets and val are Series, not DataFrames
        if isinstance(rets, pd.DataFrame):
            rets = rets.iloc[:, 0]
        if isinstance(val, pd.DataFrame):
            val = val.iloc[:, 0]
            
        total_ret = (val.iloc[-1] / val.iloc[0] - 1) * 100
        
        # Annualized volatility
        ann_vol = float(rets.std() * np.sqrt(252))
        # Annualized return
        days = (rets.index[-1] - rets.index[0]).days
        ann_ret = float((val.iloc[-1] / val.iloc[0]) ** (365.25 / days) - 1)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # Max drawdown
        cum_max = val.cummax()
        drawdown = (val - cum_max) / cum_max
        max_dd = float(drawdown.min() * 100)
        return total_ret, sharpe, max_dd

    pf_ret, pf_sharpe, pf_dd = calc_stats(pf)
    bh_ret, bh_sharpe, bh_dd = calc_stats(bh)

    print("\n--- OMNI-AMALGAMATED PORTFOLIO STATS ---")
    print(f"Total Return: {pf_ret:.2f}% (B&H SPY: {bh_ret:.2f}%)")
    print(f"Sharpe Ratio: {pf_sharpe:.2f} (B&H SPY: {bh_sharpe:.2f})")
    print(f"Max Drawdown: {pf_dd:.2f}% (B&H SPY: {bh_dd:.2f}%)")
    
    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    axes[0].plot(pf.value(), label='Omni Amalgamated Portfolio', color='blue')
    axes[0].plot(bh.value(), label='Buy & Hold SPY', color='gray', alpha=0.6)
    axes[0].set_yscale('log')
    axes[0].set_title('Master Equity Curve (Log Scale)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot asset allocations over time
    allocations.plot(ax=axes[1], kind='area', stacked=True, colormap='tab10', alpha=0.8)
    axes[1].set_title('Dynamic Asset Allocation (Safe Haven Shifts)')
    axes[1].set_ylabel('Capital %')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('portfolio_amalgamation_v3.png', dpi=300, bbox_inches='tight')
    print("Saved 'portfolio_amalgamation_v3.png'")

if __name__ == '__main__':
    main()
