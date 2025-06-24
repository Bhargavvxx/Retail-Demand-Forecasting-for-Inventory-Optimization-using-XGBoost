#!/usr/bin/env python
# --------------------------------------------------------------
#  End-to-end preprocessing for the Rossmann sales-forecast project
# --------------------------------------------------------------
from __future__ import annotations
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Logging helpers
# ----------------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(format=LOG_FMT, level=logging.INFO)
log = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# dtypes for the *initial* CSV load
# ----------------------------------------------------------------------------
DTYPES = {
    "Store":         "Int16",
    "DayOfWeek":     "Int8",
    "Sales":         "float32",      # absent in test
    "Customers":     "float32",      # absent in test
    "Open":          "Int8",
    "Promo":         "Int8",
    "StateHoliday":  "string",       # becomes category later
    "SchoolHoliday": "Int8",
}

# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------
def _months_between(date1: pd.Series, date2: pd.Series) -> pd.Series:
    """Return full months between two equally-indexed datetime64[ns] Series."""
    return (
        (date1.dt.year   - date2.dt.year)  * 12 +
        (date1.dt.month  - date2.dt.month)
    ).astype("int16")


# ----------------------------------------------------------------------------
# Core preprocessing steps
# ----------------------------------------------------------------------------
def _impute_competition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute competition-related fields and derive `CompetitionOpenMonths`.
    """
    # Distance: median imputation
    df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(), inplace=True)

    # Year / Month:
    df["CompetitionOpenSinceYear"].fillna(df["Date"].dt.year,  inplace=True)
    df["CompetitionOpenSinceMonth"].fillna(df["Date"].dt.month, inplace=True)

    # Treat sentinel 9999 as “far future” → set 100 years after store date
    mask_9999 = df["CompetitionOpenSinceYear"] == 9999
    df.loc[mask_9999, "CompetitionOpenSinceYear"] = df.loc[mask_9999, "Date"].dt.year + 100

    # Any zero months are invalid → coerce to January (1)
    df.loc[df["CompetitionOpenSinceMonth"] == 0, "CompetitionOpenSinceMonth"] = 1

    comp_open_date = pd.to_datetime(
        dict(
            year=df["CompetitionOpenSinceYear"].astype(int),
            month=df["CompetitionOpenSinceMonth"].astype(int),
            day=1,
        ),
        errors="coerce",
    )

    df["CompetitionOpenMonths"] = _months_between(df["Date"], comp_open_date).clip(lower=0)

    # All competition columns are now safe to cast
    df[["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]] = (
        df[["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]].astype("int16")
    )
    return df


def _impute_promo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute Promo2 fields and derive `Promo2OpenMonths` + `IsPromoMonth`.
    """
    # Replace NaNs with zeros → zeros denote “no Promo2 for this store”
    df[["Promo2SinceYear", "Promo2SinceWeek"]] = (
        df[["Promo2SinceYear", "Promo2SinceWeek"]].fillna(0).astype("int16")
    )
    df["PromoInterval"].fillna("", inplace=True)

    # Compute promo start date (week-based, ISO weeks start on Monday)
    has_promo = df["Promo2SinceYear"] > 0
    promo_start = pd.to_datetime(
        pd.to_datetime(
            df["Promo2SinceYear"].astype(str).str.zfill(4) + "-01-01",
            errors="coerce",
        ) + pd.to_timedelta((df["Promo2SinceWeek"] - 1).clip(lower=0) * 7, unit="D"),
        errors="coerce",
    )
    promo_start = promo_start.where(has_promo, df["Date"])  # if no promo, use current date

    df["Promo2OpenMonths"] = _months_between(df["Date"], promo_start).clip(lower=0)

    # Flag: is current month listed in PromoInterval?
    month_abbr = df["Date"].dt.strftime("%b")          # "Jan", "Feb", …
    promo_lists = df["PromoInterval"].str.split(',')   # NaN already filled with ""

    df["IsPromoMonth"] = np.fromiter(
    (
        m in p                                    # membership test
        for m, p in zip(month_abbr, promo_lists)
    ),
    dtype="int8",
    count=len(df)
)

    return df


def _build_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar / seasonality features.
    """
    date = df["Date"]
    df["Year"]         = date.dt.year.astype("int16")
    df["Month"]        = date.dt.month.astype("int8")
    df["Day"]          = date.dt.day.astype("int8")
    df["WeekOfYear"]   = date.dt.isocalendar().week.astype("int8")
    df["Quarter"]      = date.dt.quarter.astype("int8")
    df["IsMonthEnd"]   = date.dt.is_month_end.astype("int8")
    df["IsMonthStart"] = date.dt.is_month_start.astype("int8")
    df["IsWeekend"]    = (df["DayOfWeek"] > 5).astype("int8")
    return df


def _optimise_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Down-cast ints/floats and convert categoricals
    (robust to leftover NaNs and missing columns).
    """
    int8_cols  = [
        "Open", "Promo", "SchoolHoliday", "DayOfWeek",
        "Month", "Day", "WeekOfYear", "Quarter",
        "IsWeekend", "IsMonthEnd", "IsMonthStart", "IsPromoMonth",
    ]
    int16_cols = [
        "Store", "Year", "CompetitionOpenMonths", "Promo2OpenMonths",
        "CompetitionOpenSinceYear", "CompetitionOpenSinceMonth",
        "Promo2SinceYear", "Promo2SinceWeek",
    ]

    # --------------------------------------------------
    # 1. Fill any lingering NaNs with sane defaults
    #    (0 works for all binary / count features)
    # --------------------------------------------------
    for cols in (int8_cols, int16_cols):
        present = [c for c in cols if c in df.columns]
        if present:
            df[present] = df[present].fillna(0)

    # --------------------------------------------------
    # 2. Cast down to the smallest NumPy dtypes
    # --------------------------------------------------
    for c in int8_cols:
        if c in df.columns:
            df[c] = df[c].astype("int8")
    for c in int16_cols:
        if c in df.columns:
            df[c] = df[c].astype("int16")

    # Floats
    if "Sales" in df.columns:
        df["Sales"] = df["Sales"].astype("float32")
    if "Customers" in df.columns:
        df["Customers"] = df["Customers"].astype("float32")

    # Categoricals
    for cat in ["StateHoliday", "StoreType", "Assortment"]:
        if cat in df.columns:
            df[cat] = df[cat].astype("category")

    return df



# ----------------------------------------------------------------------------
# High-level driver
# ----------------------------------------------------------------------------
def _pipeline(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pipe(_impute_competition)
          .pipe(_impute_promo)
          .pipe(_build_date_features)
          .pipe(_optimise_types)
          .sort_values(["Store", "Date"])        # deterministic order
          .reset_index(drop=True)
    )


def preprocess(raw_dir: str | Path, out_dir: str | Path) -> None:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------- load -----------------------------------
    train = pd.read_csv(raw_dir / "train.csv",
                        dtype=DTYPES,
                        parse_dates=["Date"])
    test  = pd.read_csv(raw_dir / "test.csv",
                        dtype={k: v for k, v in DTYPES.items() if k != "Sales"},
                        parse_dates=["Date"])
    store = pd.read_csv(raw_dir / "store.csv")

    log.info("raw train shape=%s | nulls=%d", train.shape, train.isna().sum().sum())
    log.info("raw test  shape=%s | nulls=%d", test.shape,  test.isna().sum().sum())
    log.info("raw store shape=%s | nulls=%d", store.shape, store.isna().sum().sum())

    # ----------------------------- merge store ------------------------------
    train = train.merge(store, on="Store", how="left")
    test  = test.merge(store,  on="Store", how="left")

    # ----------------------- imputation & feature eng. ----------------------
    train = _pipeline(train)
    test  = _pipeline(test)

    # ----------------------------- sanity check -----------------------------
    assert train.isna().sum().sum() == 0, "train still has NaNs!"
    assert test.isna().sum().sum()  == 0, "test still has NaNs!"

    # ------------------------------ save ------------------------------------
    train.to_parquet(out_dir / "train_clean.parquet", index=False)
    test.to_parquet(out_dir / "test_clean.parquet",  index=False)
    log.info("✅ Pre-processing completed | Clean files saved ➜ %s", out_dir)


# ----------------------------------------------------------------------------
# CLI entry-point
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rossmann data pre-processing")
    parser.add_argument("--raw-dir", required=True,
                        help="folder containing train.csv / test.csv / store.csv")
    parser.add_argument("--out-dir", required=True,
                        help="destination for *_clean.parquet")
    parser.add_argument("--log-level", default="INFO",
                        help="Python logging level (e.g. DEBUG)")
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())
    preprocess(args.raw_dir, args.out_dir)
