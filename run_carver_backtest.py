import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import sys
import warnings

warnings.filterwarnings("ignore")

from carver_engine import build_carver_engine, ANN_DAYS

COST_BPS = 3.25

def backtest(close, weight, cost_bps=COST_BPS, exec_lag=1):
    """
    Simulates portfolio execution with execution lag and transaction costs.
    """
    ret = close.pct_change().fillna(0.0)
    
    # exec_lag=1 ensures no lookahead bias.
    # The weight decided at time T is executed at the close of time T
    # and held during the return from T to T+1.
    held = weight.shift(exec_lag).fillna(0.0)
    
    turnover = held.diff().abs().fillna(held.abs())
    net = held * ret - turnover * (cost_bps / 1e4)
    
    return pd.DataFrame({
        "ret": ret, 
        "weight": weight, 
        "held": held,
        "net": net, 
        "equity": (1.0 + net).cumprod()
    })

def sharpe(net, ann_days=ANN_DAYS, rf=0.0):
    ex = net - rf / ann_days
    sd = ex.std(ddof=1)
    if not sd:
        return float("nan")
    return float(ex.mean() / sd * np.sqrt(ann_days))

def main():
    print("=========================================================")
    print("   BOOTCAMP 2 REPLICATION: FULL 8-LAYER CARVER ENGINE")
    print("=========================================================\n")
    
    # 1. The Exact Universe
    symbols = ("BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "LTC-USD")
    target_asset = "BTC-USD"
    start_date = "2018-01-01"
    
    print(f"Downloading Crypto Panel: {symbols}")
    print(f"Start Date: {start_date}")
    
    panel = yf.download(list(symbols), start=start_date, auto_adjust=True, progress=False)["Close"]
    panel = panel.ffill().dropna()
    
    print("\nRunning Layers 1-7 (No Cross-Sectional Momentum)...")
    w_no_cs = build_carver_engine(panel, target=target_asset, use_cs=False)
    res_no_cs = backtest(panel[target_asset], w_no_cs)
    
    print("Running Full 8-Layer Engine (WITH Cross-Sectional Momentum)...")
    w_with_cs = build_carver_engine(panel, target=target_asset, use_cs=True)
    res_with_cs = backtest(panel[target_asset], w_with_cs)
    
    bh_res = backtest(panel[target_asset], pd.Series(1.0, index=panel.index))
    
    print("\n---------------------------------------------------------")
    print("RESULTS:")
    print(f"Buy & Hold {target_asset:<15} Sharpe: {sharpe(bh_res['net']):.2f}")
    print(f"Carver Engine (Layers 1-7)   Sharpe: {sharpe(res_no_cs['net']):.2f}")
    print(f"Full 8-Layer Carver Engine   Sharpe: {sharpe(res_with_cs['net']):.2f}")
    print("---------------------------------------------------------\n")
    
    # 2. Plotting
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    ax[0].plot(res_with_cs.index, res_with_cs["equity"], color="k", lw=2, label="Full 8-Layer Carver Engine")
    ax[0].plot(res_no_cs.index, res_no_cs["equity"], color="tab:blue", lw=1, alpha=0.7, label="Carver Engine (Layers 1-7)")
    ax[0].plot(bh_res.index, bh_res["equity"], color="gray", ls="--", lw=1, label=f"Buy & Hold {target_asset}")
    ax[0].set_yscale("log")
    ax[0].set_title(f"Bootcamp Replication: {target_asset} Strategy Equity Curve (Log Scale)")
    ax[0].set_ylabel("Cumulative Return")
    ax[0].legend(loc="upper left")
    ax[0].grid(True, alpha=0.3)
    
    ax[1].fill_between(w_with_cs.index, 0, w_with_cs, color="k", alpha=0.3, label="Executed Weight")
    ax[1].set_ylim(0, 1.05)
    ax[1].set_ylabel("Portfolio Weight")
    ax[1].set_title("Layer 7 Position Sizing (Long-Only, Target 20% Volatility)")
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = "final_145_sharpe_equity.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Equity curve plot saved to {plot_path}")

if __name__ == "__main__":
    main()
