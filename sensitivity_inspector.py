import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import vectorbt as vbt
import matplotlib.pyplot as plt

FEES = 0.002
SPLIT_DATE = '2021-12-31'

def safe_assign(df, name, condition):
    try:
        df[name] = condition.fillna(False).astype(bool)
    except Exception:
        df[name] = False

def calculate_sharpe(portfolio):
    try:
        sr = portfolio.sharpe_ratio()
        return sr if pd.notna(sr) else 0.0
    except Exception:
        return 0.0

def generate_signals(df, stoch_k, psar_af, fib_win):
    d = df.copy()
    c = d['Close']
    h = d['High']
    l = d['Low']

    # 1. Stoch Macro
    st_mac = ta.stoch(h, l, c, k=stoch_k, d=5, smooth_k=5)
    if st_mac is not None and not st_mac.empty:
        safe_assign(d, 'Sig_Stoch_Macro_OS', st_mac.iloc[:, 0] < 20)
    else:
        d['Sig_Stoch_Macro_OS'] = False

    # 2. PSAR Mac
    ps = ta.psar(h, l, c, af0=psar_af, af=psar_af, max_af=psar_af*10)
    if ps is not None and not ps.empty:
        lc = [col for col in ps.columns if col.startswith('PSARl_')]
        if lc: 
            safe_assign(d, 'Sig_PSAR_Mac', pd.notna(ps[lc[0]]))
        else:
            d['Sig_PSAR_Mac'] = False
    else:
        d['Sig_PSAR_Mac'] = False

    # 3. FIB 236
    roll_max = h.rolling(fib_win).max()
    roll_min = l.rolling(fib_win).min()
    fib_r = roll_max - roll_min
    safe_assign(d, 'Sig_FIB_236_Hold', c > (roll_max - fib_r * 0.236))

    d = d.dropna()
    return d

def evaluate_combo(df):
    entry = df['Sig_Stoch_Macro_OS'] | df['Sig_PSAR_Mac'] | df['Sig_FIB_236_Hold']
    shifted_entries = entry.shift(1).fillna(False).values
    shifted_exits = (~entry).shift(1).fillna(False).values
    open_vals = df['Open'].values
    
    pf = vbt.Portfolio.from_signals(
        open_vals, entries=shifted_entries, exits=shifted_exits,
        freq='1d', fees=FEES
    )
    return calculate_sharpe(pf)

def main():
    print("Fetching SPY data...")
    df = yf.download("SPY", start='2010-01-01', progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Defaults
    def_stoch_k = 21
    def_psar_af = 0.01
    def_fib_win = 200

    print("Running Stochastic 'K' Sensitivity Analysis...")
    stoch_k_range = list(range(10, 41, 2))
    stoch_res_is = []
    stoch_res_oos = []
    
    for k in stoch_k_range:
        d = generate_signals(df, k, def_psar_af, def_fib_win)
        train_df = d.loc[:SPLIT_DATE]
        test_df = d.loc[pd.to_datetime(SPLIT_DATE) + pd.Timedelta(days=1):]
        
        stoch_res_is.append(evaluate_combo(train_df))
        stoch_res_oos.append(evaluate_combo(test_df))

    print("Running PSAR 'AF0' Sensitivity Analysis...")
    psar_af_range = [round(x, 3) for x in np.arange(0.005, 0.031, 0.002)]
    psar_res_is = []
    psar_res_oos = []
    
    for af in psar_af_range:
        d = generate_signals(df, def_stoch_k, af, def_fib_win)
        train_df = d.loc[:SPLIT_DATE]
        test_df = d.loc[pd.to_datetime(SPLIT_DATE) + pd.Timedelta(days=1):]
        
        psar_res_is.append(evaluate_combo(train_df))
        psar_res_oos.append(evaluate_combo(test_df))

    print("Running Fibonacci Rolling Window Sensitivity Analysis...")
    fib_win_range = list(range(100, 310, 20))
    fib_res_is = []
    fib_res_oos = []
    
    for w in fib_win_range:
        d = generate_signals(df, def_stoch_k, def_psar_af, w)
        train_df = d.loc[:SPLIT_DATE]
        test_df = d.loc[pd.to_datetime(SPLIT_DATE) + pd.Timedelta(days=1):]
        
        fib_res_is.append(evaluate_combo(train_df))
        fib_res_oos.append(evaluate_combo(test_df))

    # Plotting
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    axes[0].plot(stoch_k_range, stoch_res_is, label='IS Sharpe', marker='o')
    axes[0].plot(stoch_k_range, stoch_res_oos, label='OOS Sharpe', marker='s')
    axes[0].axvline(def_stoch_k, color='r', linestyle='--', label='Default (21)')
    axes[0].set_title('Stochastic K Parameter Sensitivity (SPY)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(psar_af_range, psar_res_is, label='IS Sharpe', marker='o')
    axes[1].plot(psar_af_range, psar_res_oos, label='OOS Sharpe', marker='s')
    axes[1].axvline(def_psar_af, color='r', linestyle='--', label='Default (0.01)')
    axes[1].set_title('PSAR AF0 Parameter Sensitivity (SPY)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(fib_win_range, fib_res_is, label='IS Sharpe', marker='o')
    axes[2].plot(fib_win_range, fib_res_oos, label='OOS Sharpe', marker='s')
    axes[2].axvline(def_fib_win, color='r', linestyle='--', label='Default (200)')
    axes[2].set_title('Fibonacci Lookback Window Sensitivity (SPY)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parameter_sensitivity_SPY.png', dpi=300, bbox_inches='tight')
    print("Saved 'parameter_sensitivity_SPY.png'. Analysis Complete.")
    
if __name__ == '__main__':
    main()
