import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np

# Download dummy data
df = yf.download("SPY", start="2023-01-01", end="2023-12-31", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Calculate Ichimoku
ichi_tuple = ta.ichimoku(df['High'], df['Low'], df['Close'])
ichi = ichi_tuple[0]

# Check if Ichimoku adds future rows or has lookahead
print("Original DF shape:", df.shape)
print("Ichi shape:", ichi.shape)
print("Ichi tails:\n", ichi.tail())

# Test Kelly Fraction logic
returns = pd.Series([0.01, -0.01, 0.0, 0.02, 0.0, -0.02, 0.05])
log_rets = np.log1p(returns)

# Flawed method
active = log_rets[log_rets != 0]
if len(active) > 0:
    mu_active = active.mean()
    var_active = active.var(ddof=1)
    kelly_active = mu_active / var_active
else:
    kelly_active = 0.0

# Correct method
mu_full = log_rets.mean()
var_full = log_rets.var(ddof=1)
kelly_full = mu_full / var_full

print(f"Flawed Kelly (active only): {kelly_active:.4f}")
print(f"Correct Kelly (full series): {kelly_full:.4f}")
