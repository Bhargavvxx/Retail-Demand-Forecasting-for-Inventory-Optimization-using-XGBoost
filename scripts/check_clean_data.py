#!/usr/bin/env python
# --------------------------------------------------------------
#  Sanity checks for Rossmann train_clean / test_clean datasets
# --------------------------------------------------------------
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np

LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(format=LOG_FMT, level=logging.INFO)
log = logging.getLogger(__name__)


# -----------------------------------------------------------------
# helper: make sure a column is binary {0,1} or nullable boolean
# -----------------------------------------------------------------
def _assert_binary(col: pd.Series, name: str) -> None:
    bad = col.dropna().unique()
    if not set(bad).issubset({0, 1, False, True}):
        raise ValueError(f"{name} contains non-binary values: {bad}")


def sanity_check(clean_dir: str | Path) -> None:
    clean_dir = Path(clean_dir)

    # ------------------------ load --------------------------------
    train = pd.read_parquet(clean_dir / "train_clean.parquet")
    test  = pd.read_parquet(clean_dir / "test_clean.parquet")

    log.info("train: %s  | nulls=%d", train.shape, train.isna().sum().sum())
    log.info("test : %s  | nulls=%d", test.shape,  test.isna().sum().sum())

    # 1️⃣  Basic NA / dtype assertions -----------------------------
    assert train.isna().sum().sum() == 0, "train contains NaNs"
    assert test.isna().sum().sum()  == 0, "test contains NaNs"

    # expected dtypes for critical columns
    must_be_int8  = ["Open", "Promo", "SchoolHoliday", "IsPromoMonth"]
    must_be_int16 = ["CompetitionOpenMonths", "Promo2OpenMonths"]
    must_be_float = ["Sales", "Customers"]

    for col in must_be_int8:
        if col in train.columns:
            assert train[col].dtype == np.int8,  f"{col} not int8"
            assert test[col].dtype  == np.int8,  f"{col} not int8 (test)"
    for col in must_be_int16:
        assert train[col].dtype == np.int16, f"{col} not int16"
        assert test[col].dtype  == np.int16, f"{col} not int16 (test)"
    for col in must_be_float:
        if col in train.columns:          # not present in test
            assert train[col].dtype == np.float32, f"{col} not float32"

    # 2️⃣  Range / value checks ------------------------------------
    if "Sales" in train.columns:
        assert (train["Sales"] >= 0).all(), "negative sales found"

    _assert_binary(train["Open"],          "Open")
    _assert_binary(test["Open"],           "Open (test)")
    _assert_binary(train["IsPromoMonth"],  "IsPromoMonth")
    _assert_binary(test["IsPromoMonth"],   "IsPromoMonth (test)")

    # dates in sensible bounds (should be 2013-2015)
    for df, name in [(train, "train"), (test, "test")]:
        min_d, max_d = df["Date"].min(), df["Date"].max()
        assert min_d.year >= 2012 and max_d.year <= 2015, (
            f"{name} date range off: {min_d} → {max_d}"
        )

    # 3️⃣  Store counts / duplicates -------------------------------
    n_unique = train["Store"].nunique()
    assert n_unique == 1115, f"unexpected number of stores: {n_unique}"

    # each Store-Date should be unique
    dupes = train.duplicated(subset=["Store", "Date"]).sum()
    assert dupes == 0, f"{dupes} duplicate Store-Date rows in train"

    # 4️⃣  Quick descriptive stats ---------------------------------
    log.info("Sales  (train):\n%s", train["Sales"].describe())
    if "Customers" in train.columns:
        log.info("Customers(train):\n%s", train["Customers"].describe())

    log.info("✅ All sanity checks passed!")


# -----------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check clean Rossmann data")
    parser.add_argument("--clean-dir", required=True,
                        help="folder containing train_clean.parquet / test_clean.parquet")
    args = parser.parse_args()

    sanity_check(args.clean_dir)
