# scripts/check_leakage.py
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_parquet(Path("data/features/train_fe.parquet"))

target = df["Sales"]
suspects = []

for col in df.columns.drop(["Sales", "Id"], errors="ignore"):
    # 1️⃣ identical values row-wise (works for any dtype)
    if target.equals(df[col]):
        print(f"🚩 identical column: {col}")
        suspects.append(col)
        continue

    # 2️⃣ numeric Pearson R == 1  (skip non-numeric)
    if pd.api.types.is_numeric_dtype(df[col]):
        x = df[col]
        # constant columns give std=0 → NaN corr; skip them
        if x.nunique(dropna=False) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = target.corr(x)
            if r == 1:
                print(f"🚩 perfectly correlated numeric column: {col}")
                suspects.append(col)

print("\nPotential leakage features:", suspects or "None ✅")
