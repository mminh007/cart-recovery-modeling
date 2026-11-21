# ☁️ Google Colab Training Guide for Tabular ML Models

> **Applies to:** Sprint 2 — Training the 5 Tabular ML models (`LightGBM`, `XGBoost`, `CausalML`, `Lifelines`, `Channel Selector`).

---

## 1. Local ↔ Google Drive ↔ Google Colab Workflow

Because the raw REES46 dataset (~13.7 GB CSV) is large, **do not parse raw CSVs repeatedly inside Colab sessions**. Follow this streamlined workflow instead:

1. **Run Feature Engineering Once (Local or High-RAM Colab):**
   Execute `rees46_processor.py` and `hillstrom_processor.py` (via `python -m src.data_prep.feature_engineering`) to produce compact, schema-validated `.parquet` tables (`intent_train.parquet`, `churn_train.parquet`, `uplift_train.parquet`, `timing_train.parquet`, `channel_train.parquet`).
2. **Sync Compact Artifacts to Google Drive:**
   Organize your Google Drive workspace as follows:
   ```text
   MyDrive/cart-abandonment/
   ├── data/processed/ml/       # Compact .parquet feature tables
   ├── configs/                 # Copied from fine-tuning/configs/
   ├── mlruns/                  # Persistent MLflow tracking store across Colab sessions
   └── exported_models/         # ONNX + serialized model artifacts for Repo 2
   ```
3. **Train via Template Notebook:**
   Open `notebooks/colab_ml_training_template.ipynb` in Google Colab, mount Google Drive, and run training + hyperparameter tuning.

---

## 2. Recommended Colab Runtimes per Model

| Model | Recommended Colab Runtime | Notes |
|---|---|---|
| **Intent Classifier (`LightGBM`)** | Standard / High-RAM CPU or T4 GPU | Histogram-based GBDT runs fast on multi-core CPUs; 50 Optuna trials complete in ~3–5 minutes. |
| **Churn Predictor (`XGBoost`)** | T4 GPU (`tree_method="hist"`, `device="cuda"`) | 4–6× speedup during Optuna cross-validation sweeps. |
| **Uplift Model (`CausalML` X-Learner)** | Standard CPU (Hillstrom 64K) / High-RAM (Criteo 14M) | Train baseline on Hillstrom first (<1 min), then benchmark scalability on Criteo samples. |
| **Timing Optimizer (`Lifelines` CoxPH)** | Standard CPU | Fits in under 30 seconds. |
| **Channel Selector (`LightGBM` Multiclass)** | Standard CPU | Lightweight 4-class tabular classifier. |

---

## 3. Syncing Trained Models Back to Local MLOps Stack

Once models pass their blocking Quality Gates (`AUC-ROC >= 0.80` for Intent, `AUC-ROC >= 0.75` for Churn, `AUUC >= 0.03` for Uplift):
1. Download `mlruns/` or `exported_models/` from Google Drive into `fine-tuning/`.
2. Inspect runs locally via `mlflow ui --backend-store-uri ./mlruns -p 5000` or verify via Airflow's `cart_abandon_ml_training` DAG.
