#!/usr/bin/env python
"""
check_feature_pipeline.py

Post-run validation for feature_engineering.py outputs.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
FEATURE_DIR   = Path("data/features")
TRAIN_FILE    = FEATURE_DIR / "train_fe.parquet"
TEST_FILE     = FEATURE_DIR / "test_fe.parquet"
HOL_SENTINEL  = 32_767           # keep in sync with pipeline constant
# --------------------------------------------------------------------------


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("🔍 Loading engineered Parquet files …")
    return (
        pd.read_parquet(TRAIN_FILE),
        pd.read_parquet(TEST_FILE),
    )


def _column_parity(train: pd.DataFrame, test: pd.DataFrame) -> None:
    missing = (set(train.columns) - {"Sales"}) - set(test.columns)
    extra   = (set(test.columns)  - {"Id"})    - set(train.columns)
    assert not missing and not extra, f"Column mismatch – missing={missing}, extra={extra}"


def _nan_check(train: pd.DataFrame, test: pd.DataFrame) -> None:
    assert not train.drop(columns=["Sales"], errors="ignore").isna().any().any(), \
        "NaNs found in train features"
    assert not test.drop(columns=["Id"], errors="ignore").isna().any().any(), \
        "NaNs found in test features"


def _date_integrity(df: pd.DataFrame, name: str) -> None:
    assert "Date" in df.columns, f"'Date' missing in {name}"
    assert pd.api.types.is_datetime64_any_dtype(df["Date"]), \
        f"'Date' dtype not datetime64 in {name}"
    assert df["Date"].notna().all(), f"NaT detected in {name}.Date"


def _holiday_sentinel_ok(train: pd.DataFrame, test: pd.DataFrame) -> None:
    for col in ["DaysToNextHoliday", "DaysSincePrevHoliday"]:
        for df, tag in [(train, "train"), (test, "test")]:
            sentinel_rows = (df[col] == HOL_SENTINEL).sum()
            print(f"   {tag}: {sentinel_rows:,} rows have {col} = {HOL_SENTINEL}")
            # sentinel is *expected* at data boundaries; just ensure dtype/int16
            assert pd.api.types.is_integer_dtype(df[col]), f"{col} not integer dtype"


def _promo_flag(train: pd.DataFrame) -> None:
    spot = (
        train.sort_values(["Store", "Date"])
             .groupby("Store", sort=False)
             .head(10)
             .copy()
    )
    spot["Promo_shift_fwd1"] = (
        spot.groupby("Store", sort=False)["Promo"]
            .shift(-1)
            .fillna(0)
            .astype(int)
    )
    assert (spot["Promo_shift_fwd1"] == spot["Promo_in_1"]).all(), \
        "Promo_in_1 mismatch vs Promo(t+1) on sample rows"


def _at_least_one_engineered(train: pd.DataFrame) -> None:
    engineered = [c for c in train.columns if c.startswith((
        "Sales_lag", "Sales_roll", "Promo_in_", "Trend_", "Roll7_by_Roll30"
    ))]
    assert engineered, "No engineered feature columns detected!"
    return engineered


def main() -> None:
    train, test = _load()

    _column_parity(train, test)
    _nan_check(train, test)
    _date_integrity(train, "train")
    _date_integrity(test,  "test")
    _holiday_sentinel_ok(train, test)
    _promo_flag(train)
    engineered = _at_least_one_engineered(train)

    print("✅ All checks passed!")
    print(f"   train_fe shape: {train.shape}")
    print(f"   test_fe  shape: {test.shape}")
    print("   Engineered feature count:", len(engineered))


if __name__ == "__main__":
    main()
