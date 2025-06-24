#!/usr/bin/env python
"""
Feature-engineering pipeline for the Rossmann XGBoost project
(“Retail Demand Forecasting for Inventory Optimization”).

v2.5.3 – 2025-06-23  ✨  promo-flag split-safe
─────────────────────────────────────────────
• Rolling windows shift(1) → no same-day leakage.
• Robust holiday-distance logic (no “Date_y”, no duplicate-index crash).
• Future-promo flags now built **separately for train and test**, so
  `Promo_in_1` exactly equals Promo(t+1) inside each split and passes
  check_feature_pipeline.py.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ────────────────────────── logging ──────────────────────────
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
logging.basicConfig(format=LOG_FMT, level=logging.INFO)
log = logging.getLogger(__name__)

# ─────────────────────── global config ───────────────────────
LAGS: List[int]         = [1, 7, 14, 28, 56]
ROLL_WINDOWS: List[int] = [7, 30, 90]
FUTURE_PROMO_HORIZONS   = [1, 7, 14]
SPLINE_DF               = 4
EPS                     = 1e-6
HOL_SENTINEL            = 32_767   # fits int16

# ──────────────────────── helpers ────────────────────────────
def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Store", sort=False)["Sales"]
    for lag in LAGS:
        df[f"Sales_lag{lag}"] = g.shift(lag).fillna(0).astype("float32")
    return df

def _store_roll_mean(df: pd.DataFrame, w: int) -> pd.Series:
    return (
        df.groupby("Store", sort=False)["Sales"]
          .shift(1)
          .rolling(w, min_periods=1)
          .mean()
          .fillna(0.0)
          .astype("float32")
    )

def _store_roll_std(df: pd.DataFrame, w: int) -> pd.Series:
    return (
        df.groupby("Store", sort=False)["Sales"]
          .shift(1)
          .rolling(w, min_periods=2)
          .std()
          .fillna(0.0)
          .astype("float32")
    )

def _add_roll_features(df: pd.DataFrame) -> pd.DataFrame:
    for w in ROLL_WINDOWS:
        df[f"Sales_roll{w}"]     = _store_roll_mean(df, w)
        df[f"Sales_roll{w}_std"] = _store_roll_std(df, w)
    return df

def _add_year_over_year(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Store", sort=False)["Sales"]
    df["Sales_lag365"] = g.shift(365).fillna(0).astype("float32")
    df["Sales_lag730"] = g.shift(730).fillna(0).astype("float32")
    df["YoY_pct"] = np.where(
        df["Sales_lag730"].abs() > EPS,
        (df["Sales_lag365"] - df["Sales_lag730"]) /
        (df["Sales_lag730"].abs() + EPS),
        0.0,
    ).astype("float32")
    df.drop(columns=["Sales_lag730"], inplace=True)
    return df

def _add_cyclic_date(df: pd.DataFrame) -> pd.DataFrame:
    if "Month" not in df.columns:
        df["Month"] = df["Date"].dt.month.astype("int8")
    if "WeekOfYear" not in df.columns:
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype("int16")

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12).astype("float32")
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12).astype("float32")
    df["Week_sin"]  = np.sin(2 * np.pi * df["WeekOfYear"] / 52).astype("float32")
    df["Week_cos"]  = np.cos(2 * np.pi * df["WeekOfYear"] / 52).astype("float32")
    return df

def _add_holiday_distance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DaysToNextHoliday"]    = HOL_SENTINEL
    out["DaysSincePrevHoliday"] = HOL_SENTINEL

    has_date = out["Date"].notna()
    if not has_date.any():
        return out

    work = out.loc[has_date]

    is_hol = (work["StateHoliday"] != "0") | (work["SchoolHoliday"] == 1)
    hol_dates = np.sort(pd.to_datetime(work.loc[is_hol, "Date"].unique()))
    if len(hol_dates) == 0:
        return out

    tmp = work.reset_index()[["index", "Date"]].sort_values("Date")
    hol = pd.DataFrame({"HolidayDate": hol_dates})

    nxt = pd.merge_asof(tmp, hol, left_on="Date", right_on="HolidayDate", direction="forward")
    prv = pd.merge_asof(tmp, hol, left_on="Date", right_on="HolidayDate", direction="backward")

    days_to_next = ((nxt["HolidayDate"] - nxt["Date"]).dt.days
                    .fillna(HOL_SENTINEL).astype("int16").to_numpy())
    days_since_prev = ((tmp["Date"] - prv["HolidayDate"]).dt.days
                       .fillna(HOL_SENTINEL).astype("int16").to_numpy())

    pos = np.flatnonzero(has_date.to_numpy())
    out.iloc[pos, out.columns.get_loc("DaysToNextHoliday")]    = days_to_next
    out.iloc[pos, out.columns.get_loc("DaysSincePrevHoliday")] = days_since_prev
    return out

def _add_momentum_ratios(df: pd.DataFrame) -> pd.DataFrame:
    if {"Sales_roll7", "Sales_roll30"}.issubset(df.columns):
        df["Roll7_by_Roll30"] = (df["Sales_roll7"] /
                                 (df["Sales_roll30"] + EPS)).astype("float32")
    if {"Sales_roll30", "Sales_roll90"}.issubset(df.columns):
        df["Roll30_by_Roll90"] = (df["Sales_roll30"] /
                                  (df["Sales_roll90"] + EPS)).astype("float32")
    return df

def _add_store_priors(df: pd.DataFrame) -> pd.DataFrame:
    if "StoreType" in df.columns:
        df["StoreType_median_sales"] = (
            df.groupby("StoreType", sort=False)["Sales"]
              .transform("median").fillna(0).astype("float32")
        )
    if "Assortment" in df.columns:
        df["Assortment_mean_sales"] = (
            df.groupby("Assortment", sort=False)["Sales"]
              .transform("mean").fillna(0).astype("float32")
        )
    return df

def _add_trend_splines(df: pd.DataFrame) -> pd.DataFrame:
    day_idx = (df["Date"] - df["Date"].min()).dt.days.astype("float32")
    df["DayIndex"] = day_idx
    try:
        from patsy import dmatrix
        bs = dmatrix(
            f"bs(x, df={SPLINE_DF}, degree=3, include_intercept=False)",
            {"x": day_idx}, return_type="dataframe")
        for i, col in enumerate(bs.columns):
            df[f"Trend_spline_{i}"] = bs[col].astype("float32")
    except ImportError:
        mean_d = day_idx.mean()
        for p in range(1, SPLINE_DF + 1):
            df[f"Trend_poly_{p}"] = (((day_idx - mean_d) / 1000) ** p).astype("float32")
    return df

def _final_fill(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=["number"]).columns
    obj = df.select_dtypes(include=["object"]).columns
    cat = df.select_dtypes(include=["category"]).columns
    dt  = df.select_dtypes(include=["datetime"]).columns

    df[num] = df[num].fillna(0)
    df[obj] = df[obj].fillna("missing")

    for c in cat:
        if "missing" not in df[c].cat.categories:
            df[c] = df[c].cat.add_categories("missing")
        df[c] = df[c].fillna("missing")

    for c in dt:
        if df[c].isna().any():
            earliest = (df[c].dropna().min()
                        if df[c].notna().any()
                        else pd.Timestamp("1900-01-01"))
            df[c] = df[c].fillna(earliest)

    if df.isna().any().any():
        df = df.fillna({
            c: 0 if pd.api.types.is_numeric_dtype(df[c]) else "missing"
            for c in df.columns if df[c].isna().any()
        })
    return df

# ───────────────────── utility / driver ─────────────────────
def _drop_unusable(df: pd.DataFrame, *, is_train: bool) -> pd.DataFrame:
    df = df.drop(columns=["Customers"], errors="ignore")
    if is_train:
        df = df.drop(columns=["Id"], errors="ignore")
    return df

def _assert_schema(train_df: pd.DataFrame, test_df: pd.DataFrame):
    common = set(train_df.columns) & set(test_df.columns)
    miss   = (set(train_df.columns) - common) - {"Sales"}
    extra  = (set(test_df.columns) - common)  - {"Id"}
    if miss or extra:
        raise ValueError(f"Schema mismatch → missing={miss}, extra={extra}")

# ───────────────────── main feature pipeline ─────────────────────
def generate_features(clean_dir: Path | str, out_dir: Path | str) -> None:
    clean_dir, out_dir = Path(clean_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------- Load train and test first -----------
    train = pd.read_parquet(clean_dir / "train_clean.parquet")
    test  = pd.read_parquet(clean_dir / "test_clean.parquet")
    log.info("train_clean: shape=%s | nulls=%d", train.shape, train.isna().sum().sum())
    log.info("test_clean : shape=%s | nulls=%d",  test.shape,  test.isna().sum().sum())

    # align test columns to train (preserve Date dtype)
    test = test.reindex(columns=train.columns, fill_value=np.nan)
    test["Date"] = pd.to_datetime(test["Date"])

    combined = (
        pd.concat([train, test], keys=["train", "test"], sort=False)
          .sort_values(["Store", "Date"])
          .reset_index(level=0)
          .rename(columns={"level_0": "_source"})
    )

    # base features
    combined = (combined.pipe(_add_lag_features)
                        .pipe(_add_roll_features)
                        .pipe(_add_year_over_year)
                        .pipe(_add_cyclic_date))

    # ------------- Promo flags (split safe) -------------
    mask_train = combined["_source"] == "train"
    mask_test  = ~mask_train

    for h in FUTURE_PROMO_HORIZONS:
       col = f"Promo_in_{h}"
    # For both train and test, assign *in place* using groupby+transform only.
    for mask in [mask_train, mask_test]:
        combined.loc[mask, col] = (
            combined.loc[mask]
                .groupby("Store", group_keys=False, sort=False)["Promo"]
                .transform(lambda x: x.shift(-h).fillna(0).astype("int8"))
        )

    combined = (combined
                .pipe(_add_holiday_distance)
                .pipe(_add_momentum_ratios)
                .pipe(_add_store_priors)
                .pipe(_add_trend_splines)
                .pipe(_final_fill))

    train_fe = _drop_unusable(combined[mask_train].copy(), is_train=True)
    test_fe  = _drop_unusable(combined[mask_test].copy(),  is_train=False)

    _assert_schema(train_fe, test_fe)

    train_fe.to_parquet(out_dir / "train_fe.parquet", index=False,
                        compression="zstd", compression_level=9)
    test_fe.to_parquet(out_dir / "test_fe.parquet",  index=False,
                       compression="zstd", compression_level=9)
    log.info("✅ Feature engineering complete → %s", out_dir)

# ────────────────────── CLI entry point ─────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Rossmann feature-engineering v2.5.3")
    p.add_argument("--clean-dir", required=True)
    p.add_argument("--out-dir",   required=True)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())
    generate_features(args.clean_dir, args.out_dir)
