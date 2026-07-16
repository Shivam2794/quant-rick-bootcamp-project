import pandas as pd
import numpy as np

s = pd.Series([1.0, 2.0, np.nan, np.nan, 5.0])
print("Original:")
print(s)
print("\nEWM Mean (adjust=False):")
print(s.ewm(span=3, adjust=False).mean())
