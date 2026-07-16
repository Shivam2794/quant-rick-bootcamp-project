import vectorbt as vbt
import pandas as pd
import numpy as np

prices = pd.Series([100, 105, 110, 108, 105, 110, 115])
entries = pd.Series([False, True, True, False, False, True, False])
exits = pd.Series([False, False, False, True, True, False, True])

# Run vbt with fees
pf = vbt.Portfolio.from_signals(prices, entries=entries, exits=exits, fees=0.01)

print("Orders:")
print(pf.orders.records_readable)
print(f"Total Trades: {pf.trades.count()}")
