#!/usr/bin/env python
"""
Train an XGBoost regressor for the **Rossmann Store Sales** forecasting project.

Key Features
------------
* Time‑series‑aware CV via `TimeSeriesSplit` (no look‑ahead leakage).
* Randomised hyper‑parameter search (override via YAML if desired).
* Robust logging → console **and** file with ISO timestamps.
* Automatic dtype handling:
    • `datetime64[ns]` → integer days since epoch.
    • `object` → pandas `category`.
    • `enable_categorical=True` so XGBoost uses native cat boost.
* Artifacts saved: booster JSON, pickled wrapper, CV results, feature importances,
  run‑metadata YAML.
* Reproducible – fixed `random_state`; captures git hash if repository available.

Usage
-----
```bash
python scripts/train_model.py \
       --feature-dir data/features \
       --model-dir   models \
       --reports-dir reports \
       --cv 5 \
       --n-iter 40
```

Requires: pandas, numpy, scikit-learn>=1.4, xgboost>=1.6, (optional) PyYAML, GitPython
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

# ────────────────── optional deps ──────────────────
try:
    import yaml  # hyper‑parameter YAML override
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

try:
    import git  # git metadata
except ImportError:  # pragma: no cover
    git = None  # type: ignore

# ───────────────────── logging ─────────────────────
LOG_FMT  = "%(asctime)s | %(levelname)-8s | %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt=DATE_FMT)
log = logging.getLogger("train_model")


# ───────────── hyper‑parameter grid ───────────────
DEFAULT_PARAM_DISTRIBUTION: Dict[str, List[Any]] = {
    "n_estimators": [300, 400, 500, 600, 800],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.2],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [0, 1, 5, 10],
}

# ──────────────────── utilities ────────────────────
def _load_features(feature_dir: Path) -> pd.DataFrame:
    fpath = feature_dir / "train_fe.parquet"
    if not fpath.exists():
        log.error("Feature file not found: %s", fpath)
        sys.exit(1)
    df = pd.read_parquet(fpath)
    log.info("Loaded train features: shape=%s | nulls=%d", df.shape, df.isna().sum().sum())
    return df


def _encode_dataframe(X: pd.DataFrame) -> pd.DataFrame:
    """Convert dtypes so XGBoost accepts the DataFrame."""
    for col in X.columns:
        if pd.api.types.is_datetime64_any_dtype(X[col]):
            # days since epoch as int32
            X[col] = (X[col] - pd.Timestamp("1970-01-01")).dt.days.astype("int32")
        elif X[col].dtype == "object":
            X[col] = X[col].astype("category")
    return X


def _prepare_X_y(df: pd.DataFrame):
    if "Sales" not in df.columns:
        log.error("Target column 'Sales' not found in feature set!")
        sys.exit(1)
    X = df.drop(columns=["Sales", "Id"], errors="ignore")
    X = _encode_dataframe(X)
    y = df["Sales"].astype("float32")
    return X, y


def _git_hash() -> Optional[str]:
    if git is None:
        return None
    try:
        repo = git.Repo(Path(__file__).resolve().parent, search_parent_directories=True)
        return repo.head.commit.hexsha[:7] + ("*" if repo.is_dirty() else "")
    except Exception:  # pragma: no cover
        return None


def _dump_yaml(obj: Dict[str, Any], path: Path):
    if yaml is None:
        log.warning("pyyaml not installed – skipping YAML dump: %s", path)
        return
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    log.info("Saved: %s", path)


# ─────────────────── main routine ──────────────────
def main(args: argparse.Namespace) -> None:
    feature_dir = Path(args.feature_dir)
    model_dir = Path(args.model_dir)
    reports_dir = Path(args.reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # file logger
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = logging.FileHandler(reports_dir / f"train_{stamp}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FMT))
    logging.getLogger().addHandler(fh)

    # data
    df = _load_features(feature_dir)
    X, y = _prepare_X_y(df)

    # parameter grid
    if args.param_yaml and yaml is not None and Path(args.param_yaml).exists():
        with open(args.param_yaml, "r", encoding="utf-8") as stream:
            param_dist = yaml.safe_load(stream)
        log.info("Loaded param grid from %s", args.param_yaml)
    else:
        param_dist = DEFAULT_PARAM_DISTRIBUTION

    # CV splitter
    tss = TimeSeriesSplit(n_splits=args.cv_splits)

    # base model
    base_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        tree_method="hist",
        n_jobs=os.cpu_count() or 4,
        enable_categorical=True,  # native categorical handling
    )

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=args.n_iter,
        scoring="neg_root_mean_squared_error",
        cv=tss,
        verbose=1,
        n_jobs=os.cpu_count() or 4,
        random_state=42,
        refit=True,
        error_score="raise",
    )

    log.info("Starting RandomizedSearchCV (n_iter=%d, cv=%d)…", args.n_iter, args.cv_splits)
    search.fit(X, y)
    best_rmse = -search.best_score_
    log.info("Best CV RMSE: %.4f", best_rmse)
    log.info("Best params: %s", search.best_params_)

    best_model = search.best_estimator_

    # save artifacts
    booster_path = model_dir / f"xgb_rossmann_{stamp}.json"
    best_model.get_booster().save_model(booster_path)
    wrapper_path = model_dir / f"xgb_rossmann_{stamp}.pkl"
    with wrapper_path.open("wb") as f:
        pickle.dump(best_model, f)
    log.info("Saved booster → %s", booster_path)
    log.info("Saved wrapper → %s", wrapper_path)

    # CV results
    cv_df = pd.DataFrame(search.cv_results_)
    cv_path = reports_dir / f"cv_results_{stamp}.csv"
    cv_df.to_csv(cv_path, index=False)
    log.info("Saved CV results → %s", cv_path)

    # feature importance
    fi = (
        pd.Series(best_model.feature_importances_, index=X.columns, name="importance")
        .sort_values(ascending=False)
    )
    fi_path = reports_dir / f"feature_importance_{stamp}.csv"
    fi.to_csv(fi_path, header=True)
    log.info("Saved feature importance → %s", fi_path)

    # metadata
    meta: Dict[str, Any] = {
        "timestamp": stamp,
        "git": _git_hash(),
        "best_rmse": float(best_rmse),
        "best_params": search.best_params_,
        "cv_splits": args.cv_splits,
        "n_iter": args.n_iter,
        "feature_file": str(feature_dir / "train_fe.parquet"),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "xgboost": xgb.__version__,
    }
    meta_path = reports_dir / f"run_{stamp}.yml"
    _dump_yaml(meta, meta_path)

    log.info("✅ Training complete.")


# ───────────────────────── CLI ──────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser("Train XGBoost for Rossmann Store Sales")
    p.add_argument("--feature-dir", required=True, help="Directory containing train_fe.parquet")
    p.add_argument("--model-dir", required=True, help="Where to write model artifacts (.json, .pkl)")
    p.add_argument("--reports-dir", required=True, help="Where to write CV results, logs, YAML")
    p.add_argument(
        "--cv",
        type=int,
        default=5,
        dest="cv_splits",
        help="Number of TimeSeriesSplit folds (default 5)",
    )
    p.add_argument("--n-iter", type=int, default=40, help="RandomizedSearchCV iterations (default 40)")
    p.add_argument("--param-yaml", help="YAML file with param_distributions override")
    main(p.parse_args())
