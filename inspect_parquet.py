import pandas as pd
import sys

try:
    df = pd.read_parquet("universe_data.parquet")
    print(f"Shape: {df.shape}")
    print(f"Columns type: {type(df.columns)}")
    if isinstance(df.columns, pd.MultiIndex):
        l0 = df.columns.get_level_values(0).unique().tolist()
        l1 = df.columns.get_level_values(1).unique()[:8].tolist()
        print(f"Level 0 (price fields): {l0}")
        print(f"Level 1 (assets, first 8): {l1}")
    else:
        print(f"Flat columns (first 8): {df.columns[:8].tolist()}")
    print(f"Index type: {type(df.index)}")
    print(f"Index[0]: {df.index[0]}")
    print(f"Index[-1]: {df.index[-1]}")
    print(f"Sample dtypes:\n{df.dtypes.head(5)}")
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
