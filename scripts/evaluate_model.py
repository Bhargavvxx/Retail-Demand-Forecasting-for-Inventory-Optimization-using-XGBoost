#!/usr/bin/env python
"""
evaluate_model.py  –  Hold-out evaluation for the Rossmann XGBoost forecaster
──────────────────────────────────────────────────────────────────────────────
▸ Uses the **last N days** of the engineered training set as an unseen test
  window (default 42 days ≈ competition horizon).

▸ Loads the trained XGBoost wrapper (*.pkl) and predicts that window.

▸ Computes industry-standard regression metrics *and* a naïve baseline:
      – RMSE, MAE
      – RMSPE, MAPE
      – Improvement vs “last-week same-day” baseline
▸ Writes a timestamped CSV / YAML with metrics, plus per-store residuals CSV.

▸ Optional PNG plots for visual QA (disabled on headless servers).

Requires: pandas, numpy, scikit-learn≥1.4, matplotlib (optional), PyYAML (optional)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ───────────── optional deps ─────────────
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore

# ───────────── logging ─────────────
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=DATE_FMT)
log = logging.getLogger("evaluate_model")


# ─────────── utilities ───────────
def _rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    pct_err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(np.square(pct_err))))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    pct_err = np.abs(y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.mean(pct_err))


def _encode_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """Match dtypes expected by the trained model."""
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            X[col] = (X[col] - pd.Timestamp("1970-01-01")).dt.days.astype("int32")
        elif X[col].dtype == "object":
            X[col] = X[col].astype("category")
    return X


def _latest_pkl(model_dir: Path) -> Path:
    picks = sorted(model_dir.glob("*.pkl"))
    if not picks:
        log.error("No .pkl models found in %s", model_dir)
        sys.exit(1)
    return picks[-1]


def _load_model(path: Path) -> xgb.XGBRegressor:
    with path.open("rb") as f:
        return pickle.load(f)


# ─────────── evaluation routine ───────────
def evaluate(
    feature_dir: Path,
    model_dir: Path,
    reports_dir: Path,
    horizon: int = 42,
    plot_png: bool = False,
) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(feature_dir / "train_fe.parquet")
    df.sort_values(["Store", "Date"], inplace=True)

    cutoff = df["Date"].max() - pd.Timedelta(days=horizon)
    holdout = df[df["Date"] > cutoff].copy()
    train   = df[df["Date"] <= cutoff].copy()

    if holdout.empty:
        log.error("Hold-out window yielded 0 rows – adjust horizon.")
        sys.exit(1)

    log.info("Hold-out window: %s → %s | rows=%d",
             holdout["Date"].min().date(), holdout["Date"].max().date(), len(holdout))

    model_path = _latest_pkl(model_dir)
    model = _load_model(model_path)
    log.info("Loaded model: %s", model_path.name)

    # ---------- predictions ----------
    X_hold = holdout.drop(columns=["Sales", "Id"], errors="ignore")
    X_hold = _encode_dataframe(X_hold)
    y_true = holdout["Sales"].values.astype("float32")
    y_pred = model.predict(X_hold)

    # ---------- baseline (lag-7) ----------
    if "Sales_lag7" in holdout.columns:
        baseline = holdout["Sales_lag7"].values.astype("float32")
    else:
        log.warning("Sales_lag7 not present – baseline set to zero.")
        baseline = np.zeros_like(y_true)

    # ---------- metrics ----------
    rmse   = mean_squared_error(y_true, y_pred, squared=False)
    mae    = mean_absolute_error(y_true, y_pred)
    rmspe  = _rmspe(y_true, y_pred)
    mape   = _mape(y_true, y_pred)

    rmse_b = mean_squared_error(y_true, baseline, squared=False)
    imp_rmse = (rmse_b - rmse) / rmse_b * 100 if rmse_b else np.nan

    metrics: Dict[str, Any] = {
        "timestamp": stamp,
        "rows_holdout": int(len(holdout)),
        "rmse": float(rmse),
        "mae": float(mae),
        "rmspe": float(rmspe),
        "mape": float(mape),
        "baseline_rmse": float(rmse_b),
        "rmse_improvement_%": float(imp_rmse),
        "model_path": str(model_path),
        "feature_file": str(feature_dir / "train_fe.parquet"),
        "horizon_days": horizon,
    }

    # save YAML / CSV
    yaml_path = reports_dir / f"eval_{stamp}.yml"
    if yaml is not None:
        with yaml_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(metrics, f, sort_keys=False)
        log.info("Saved metrics → %s", yaml_path)
    else:
        log.warning("pyyaml not installed – metrics YAML skipped (%s)", yaml_path.name)

    # per-store residuals
    holdout["y_pred"]   = y_pred
    holdout["residual"] = y_true - y_pred
    resid_path = reports_dir / f"residuals_{stamp}.csv"
    holdout[["Store", "Date", "Sales", "y_pred", "residual"]].to_csv(resid_path, index=False)
    log.info("Saved per-row residuals → %s", resid_path)

    # optional plots
    if plot_png and plt is not None:
        stores = holdout["Store"].unique()[:4]  # first four stores
        for sid in stores:
            sub = holdout[holdout["Store"] == sid]
            plt.figure(figsize=(10, 4))
            plt.plot(sub["Date"], sub["Sales"], label="Actual")
            plt.plot(sub["Date"], sub["y_pred"], label="Predicted")
            plt.title(f"Store {sid} – hold-out horizon")
            plt.legend()
            plt.tight_layout()
            png_path = reports_dir / f"plot_store{sid}_{stamp}.png"
            plt.savefig(png_path, dpi=120)
            plt.close()
            log.info("Saved plot → %s", png_path)

    # console summary
    log.info(
        "RMSE %.2f | MAE %.2f | RMSPE %.4f | MAPE %.4f  | ΔRMSE vs baseline = %.2f%%",
        rmse, mae, rmspe, mape, imp_rmse,
    )


# ─────────── CLI ───────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Evaluate Rossmann XGBoost hold-out performance")
    ap.add_argument("--feature-dir", required=True, help="Folder with train_fe.parquet")
    ap.add_argument("--model-dir", required=True, help="Folder where *.pkl model lives")
    ap.add_argument("--reports-dir", required=True, help="Destination for eval outputs")
    ap.add_argument("--horizon", type=int, default=42, help="Hold-out window in days")
    ap.add_argument("--plot", action="store_true", help="Save PNG plots for a few stores")
    args = ap.parse_args()

    evaluate(
        feature_dir=Path(args.feature_dir),
        model_dir=Path(args.model_dir),
        reports_dir=Path(args.reports_dir),
        horizon=args.horizon,
        plot_png=args.plot,
    )
