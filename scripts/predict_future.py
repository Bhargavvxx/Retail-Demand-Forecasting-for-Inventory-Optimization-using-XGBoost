#!/usr/bin/env python
"""
Batch-forecast Rossmann Store Sales with a trained XGBoost model.

▸ Picks the most recent *.pkl in --model-dir unless --model-path is given.
▸ Reads **test_fe.parquet** produced by the feature-engineering pipeline.
▸ Encodes datetimes → “days since epoch”, object → category (as during training).
▸ Writes forecast CSV + run log (+ optional diagnostic PNG) to --reports-dir.

Usage
-----
python scripts/predict_future.py \
       --feature-dir data/features \
       --model-dir   models \
       --reports-dir reports \
       --plot
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import xgboost as xgb

# ────────────────────────── logging ──────────────────────────
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=DATE_FMT)
log = logging.getLogger("predict_future")

# ───────────────────────── plotting (optional) ───────────────
try:
    import matplotlib

    matplotlib.use("Agg")  # headless-friendly
    import matplotlib.pyplot as plt  # noqa
except ImportError:  # pragma: no cover
    plt = None  # type: ignore

# ───────────────────────── utilities ─────────────────────────
def _encode_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the minimal encoding expected by the trained booster."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = (df[col] - pd.Timestamp("1970-01-01")).dt.days.astype("int32")
        elif df[col].dtype == "object":
            df[col] = df[col].astype("category")
    return df


def _latest_model(model_dir: Path) -> Path:
    """Return the most recent *.pkl model inside *model_dir*."""
    pkl_files = sorted(model_dir.glob("*.pkl"))
    if not pkl_files:
        log.error("No *.pkl model found in %s", model_dir)
        sys.exit(1)
    return pkl_files[-1]


def _load_model(path: Path) -> xgb.XGBRegressor:
    """Deserialize a pickled :class:`~xgboost.XGBRegressor`."""
    with path.open("rb") as f:
        model = pickle.load(f)
    if not isinstance(model, xgb.XGBRegressor):
        log.error("Loaded object is not an XGBRegressor (%s)", type(model))
        sys.exit(1)
    return model


def _load_features(feature_dir: Path) -> pd.DataFrame:
    """Read **test_fe.parquet** written by the feature pipeline."""
    test_path = feature_dir / "test_fe.parquet"
    if not test_path.exists():
        log.error("test_fe.parquet missing in %s", feature_dir)
        sys.exit(1)
    df = pd.read_parquet(test_path)
    log.info("Loaded test features: shape=%s | nulls=%d", df.shape, df.isna().any().sum())
    return df


def _prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Drop target/ID columns and apply encoding."""
    X = df.drop(columns=["Sales", "Id"], errors="ignore")
    return _encode_dataframe(X)


def _save_forecast(df_out: pd.DataFrame, reports_dir: Path, stamp: str) -> Path:
    out_path = reports_dir / f"forecast_{stamp}.csv"
    df_out.to_csv(out_path, index=False)
    log.info("Saved forecast → %s | rows=%d", out_path, len(df_out))
    return out_path


def _quick_plot(df: pd.DataFrame, stamp: str, reports_dir: Path) -> None:
    if plt is None:
        log.warning("--plot flag ignored (matplotlib not installed)")
        return

    sample = df.groupby("Store", sort=False).head(50)
    fig, ax = plt.subplots(figsize=(8, 4))
    for store_id, g in sample.groupby("Store"):
        ax.plot(g["Date"], g["Prediction"], label=f"Store {store_id}", alpha=0.7)

    ax.set_title("Sample forecast (50 days × 5 stores)")
    ax.set_ylabel("Predicted Sales")
    ax.set_xlabel("Date")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()

    png = reports_dir / f"forecast_{stamp}.png"
    fig.savefig(png, dpi=120)
    log.info("Diagnostic plot saved → %s", png)


# ─────────────────────────── main ────────────────────────────
def main(args: argparse.Namespace) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # file-level logger
    fh = logging.FileHandler(reports_dir / f"predict_{stamp}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FMT, datefmt=DATE_FMT))
    logging.getLogger().addHandler(fh)

    # pick model
    model_path = args.model_path or _latest_model(args.model_dir)
    model = _load_model(model_path)
    log.info("Loaded model: %s", model_path.name)

    # load & encode features
    df_test = _load_features(args.feature_dir)
    X_test = _prepare_X(df_test)

    # predict
    preds = model.predict(X_test).astype("float32")

    # build output DF – keep identifying columns if present
    out_cols: Dict[str, Any] = {"Prediction": preds}
    ids = [c for c in ("Id", "Store", "Date") if c in df_test.columns]
    for c in ids:
        out_cols[c] = df_test[c].values

    df_out = pd.DataFrame(out_cols)
    df_out = df_out[["Prediction", *[c for c in df_out.columns if c != "Prediction"]]]

    _save_forecast(df_out, reports_dir, stamp)

    if args.plot:
        # ensure Date is available for the plot
        if "Date" not in df_out.columns and "Date" in df_test.columns:
            df_plot = df_out.join(df_test["Date"])
        else:
            df_plot = df_out
        _quick_plot(df_plot, stamp, reports_dir)

    log.info("✅ Forecasting complete. Output folder: %s", reports_dir.resolve())


# ────────────────────────── CLI entry ─────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="predict_future.py",
        description="Generate Rossmann sales forecasts using a trained XGBoost model.",
    )
    parser.add_argument(
        "--feature-dir",
        required=True,
        type=Path,
        help="Folder containing test_fe.parquet",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory with *.pkl models (most recent will be used by default)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Explicit path to a specific *.pkl model (overrides --model-dir)",
    )
    parser.add_argument(
        "--reports-dir",
        required=True,
        type=Path,
        help="Destination for forecast CSV / logs / figures",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a quick diagnostic PNG (requires matplotlib)",
    )
    main(parser.parse_args())
