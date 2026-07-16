import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load data
results_path = r"backtest_results.parquet"
df = pd.read_parquet(results_path)

# Ensure index is datetime
df.index = pd.to_datetime(df.index)

# Plot equity curve
plt.style.use('dark_background') # Looks more premium
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot Equity and HWM
ax1.plot(df.index, df['equity'], label='Omni-Engine Equity', color='#00ffcc', linewidth=1.5)
ax1.plot(df.index, df['hwm'], label='High Water Mark', color='#888888', linestyle='--', alpha=0.5)
ax1.fill_between(df.index, df['equity'], df['hwm'], color='red', alpha=0.3, label='Drawdown Area')

# Formatting
ax1.set_title('FTMO Omni-Engine: Equity Curve & Drawdown Profile', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Date', fontsize=12)
ax1.set_ylabel('Account Equity ($)', fontsize=12)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Grid and Legend
ax1.grid(True, linestyle='-', alpha=0.2, color='white')
ax1.legend(loc='upper left', fontsize=12, framealpha=0.2)

# Format X-axis for dates
ax1.xaxis.set_major_locator(mdates.YearLocator())
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()

plt.tight_layout()

# Save plot to artifacts directory
output_path = r"C:\Users\Shivam Patel\.gemini\antigravity\brain\7b03663a-d01b-4302-8959-0a511c484299\equity_curve.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")
