# 🛒 Rossmann XGBoost Forecasting

> **Accurate six‑week demand forecasts for Rossmann stores using a fully leak‑free feature‑engineering pipeline and a tuned XGBoost regressor.**

---

## 🚦 Pipeline Overview

| Stage                  | Script                              | Main Artifact(s)                                                                            |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------- |
| 1. Pre‑processing      | `scripts/data_preprocessing.py`     | `data/processed/train_clean.parquet`  <br> `data/processed/test_clean.parquet`              |
| 2. Feature engineering | `scripts/feature_engineering.py`    | `data/features/train_fe.parquet`  <br> `data/features/test_fe.parquet`                      |
| 3. Leakage checks      | `scripts/check_feature_pipeline.py` | Pass / fail console output                                                                  |
| 4. Model training      | `scripts/train_model.py`            | `models/xgb_rossmann_<timestamp>.json`  <br> booster + wrapper \*.pkl, CV & importance CSVs |
| 5. Hold‑out evaluation | `scripts/evaluate_model.py`         | metrics YAML/CSV + residual plot PNG                                                        |
| 6. Future forecasting  | `scripts/predict_future.py`         | `reports/forecast_<timestamp>.csv`  <br> optional forecast PNG                              |

---

## 📂 Repository Layout

```text
.
├── data/
│   ├── raw/              # original Kaggle CSVs
│   ├── processed/        # clean parquet after stage 1
│   └── features/         # engineered parquet after stage 2
│
├── models/               # saved boosters / wrappers
│
├── reports/              # CV results, feature importance, residuals, forecasts
│   ├── *.csv  *.yml  *.png
│   └── *.log             # timestamped run logs
│
└── scripts/              # all pipeline code
```

---

## 🛠️ Getting Started

### 1 · Clone & create environment

```bash
git clone https://github.com/your‑handle/rossmann_xgboost_forecasting.git
cd rossmann_xgboost_forecasting

# Conda (recommended)
conda env create -f environment.yml
conda activate rossmann-xgb

# — or —
# pip
pip install -r requirements.txt
```

### 2 · Add the Kaggle CSVs

Download **train.csv**, **test.csv**, **store.csv** from the [Kaggle competition page](https://www.kaggle.com/c/rossmann-store-sales/data) and place them in `data/raw/`:

```text
data/raw/
├── train.csv
├── test.csv
└── store.csv
```

### 3 · Run the pipeline end‑to‑end

```bash
# ➊ Clean & enrich raw data
python scripts/data_preprocessing.py \
       --raw-dir   data/raw \
       --out-dir   data/processed

# ➋ Engineer leakage‑free features
python scripts/feature_engineering.py \
       --clean-dir data/processed \
       --out-dir   data/features

# ➌ Sanity‑check feature parity & leakage
python scripts/check_feature_pipeline.py  # must finish without assertion errors

# ➍ Train XGBoost with 40‑run random search & 5‑fold TimeSeriesSplit
python scripts/train_model.py \
       --feature-dir data/features \
       --model-dir   models \
       --reports-dir reports \
       --cv 5 \
       --n-iter 40
```

Sample training output:

```yaml
Best CV RMSE : 751.83
Best params  : {max_depth: 10, n_estimators: 600, learning_rate: 0.05, …}
Artifacts    : models/xgb_rossmann_20250624‑134512.{json,pkl}
```

```bash
# ➎ Evaluate on a hold‑out window (last 42 days of train)
python scripts/evaluate_model.py \
       --feature-dir data/features \
       --model-dir   models \
       --reports-dir reports \
       --horizon 42 \
       --plot            # include to save PNG residual plot

# ➏ Forecast the Kaggle test set (6 weeks)
python scripts/predict_future.py \
       --feature-dir data/features \
       --model-dir   models \
       --reports-dir reports \
       --plot
```

Generated forecast files:

```text
reports/
├── forecast_YYYYMMDD-hhmmss.csv   # Store, Date, Sales_pred
└── forecast_YYYYMMDD-hhmmss.png   # optional visual sanity check
```

Merge `Sales_pred` with the `Id` column from **test.csv** to prepare the Kaggle submission.

---

## 🧩 Feature Overview

| Category             | Examples                                                                |
| -------------------- | ----------------------------------------------------------------------- |
| **Lags**             | `Sales_lag1`, `Sales_lag7`, `Sales_lag14`, `Sales_lag28`, `…_lag365`    |
| **Moving windows**   | `Sales_roll7`, `Sales_roll30` + std‑dev counterparts                    |
| **Promo look‑ahead** | `Promo_in_1`, `Promo_in_7`, `Promo_in_14` – generated leak‑free         |
| **Calendar**         | `Day`, `Month`, `Quarter`, `IsWeekend`, cyclic sin/cos month & ISO week |
| **Holiday distance** | `DaysSincePrevHoliday`, `DaysToNextHoliday` (32767 = none)              |
| **Store priors**     | median sales by `StoreType`, mean by `Assortment`                       |
| **Trend splines**    | cubic B‑spline basis `Trend_spline_0…4` on a global day‑index           |

See `reports/feature_importance_<timestamp>.csv` for the ranked list.

---

## 🏷️ Key Design Decisions

* **Leak‑proofing** – every feature uses only past information.  Future‑promo flags are generated *per split* after concatenating train/test to avoid overlap.
* **TimeSeriesSplit CV** – preserves temporal order inside folds.
* **Native categorical** in XGBoost ≥ 1.6 – faster and lower memory than one‑hot.
* **Parquet + Zstandard** – speedy I/O during iterative modelling.
* **Reproducibility** – fixed random seeds, timestamped artifacts, optional Git hash captured in YAML.

---

## 📈 Results (hold‑out 42 days)

| Metric    | Value |
| --------- | ----- |
| **RMSE**  | ≈ 450 |
| **MAE**   | ≈ 317 |
| **RMSPE** | 7.9 % |
| **MAPE**  | 5.8 % |

### Next Ideas

* Hierarchical modelling – per‑`StoreType` models blended at inference.
* External regressors – German public holidays, regional weather, CPI.
* Model ensemble – LightGBM + CatBoost + Prophet residuals.

---

## 👤 Author

Bhargav Patil

## 🤝 Contributing

1. Fork → create feature branch → commit → open PR.
2. Run `pre-commit run --all-files` (black, isort, flake8) before pushing.
3. Ensure `pytest` passes.

---

## 📜 License

Released under the MIT License.  See `LICENSE` for details.
