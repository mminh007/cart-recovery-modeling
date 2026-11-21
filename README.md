# 🧠 Cart Abandonment AI — Repo 1: Modeling, Fine-tuning & MLOps Pipeline

> **Repository 1 of 3** in the **Cart Abandonment Recovery AI System**.  
> Owns the end-to-end training lifecycle: **Data Ingestion → Leakage-Free Feature Engineering → Tabular ML Training (Google Colab) → Vietnamese LLM Fine-tuning (vast.ai) → Model Registry & Export (MLflow / HuggingFace Hub)**.

---

## 📁 Repository Structure

```text
fine-tuning/
├── README.md                             # Project overview & quickstart guide
├── pyproject.toml                        # Dependencies, packaging & pytest/ruff config
├── Makefile                              # Common developer & MLOps commands
├── docker-compose.yml                    # Local MLOps stack (Postgres + MLflow + Airflow + Jupyter)
├── dvc.yaml                              # Reproducible DVC data pipeline stages
├── .env.example                          # Environment variables template (Local & Cloud S3/GCS)
│
├── docs/                                 # 📚 Technical documentation & infrastructure guides
│   ├── README.md                         # Documentation index
│   ├── data-storage-and-pipeline.md      # Data lake architecture, leakage prevention & cloud scale-up
│   ├── colab-ml-guide.md                 # Guide for training Tabular ML models on Google Colab
│   └── vast-ai-gpu-guide.md              # GPU sizing, pricing & runbook on vast.ai for LLM/vLLM
│
├── bugs/                                 # 🐞 Bug reports & post-mortem Root Cause Analysis
│   ├── README.md                         # Bug index & resolution workflow
│   └── BUG_TEMPLATE.md                   # Standardized Root Cause Analysis template
│
├── configs/                              # ⚙️ Model hyperparameters & shared feature contract
│   ├── feature_schema.yaml               # Shared feature schema contract between Repo 1 & Repo 2
│   ├── ml/                               # Hyperparameter configs for all 5 Tabular ML models
│   └── llm/                              # SFT (QLoRA), DPO & vLLM serving configurations
│
├── dags/                                 # 🔄 Apache Airflow orchestration DAGs
│   ├── cart_abandonment_data_dag.py      # Ingestion, validation & feature engineering DAG
│   └── cart_abandonment_training_dag.py  # Model training verification & MLflow Quality Gates DAG
│
├── data/                                 # 💾 Datasets (tracked by DVC, ignored by Git)
│   ├── raw/                              # Immutable raw CSVs (REES46, Hillstrom, Criteo, Tiki)
│   ├── processed/                        # Validated datasets (.parquet for ML, .jsonl for LLM)
│   └── synthetic/                        # Synthetic SFT/DPO data generation scripts & outputs
│
├── notebooks/                            # 📓 Exploratory Data Analysis (EDA) & Colab templates
│   ├── 01_rees46_eda.ipynb               # Deep-dive EDA on 109.9M REES46 clickstream events
│   ├── 02_hillstrom_criteo_eda.ipynb     # Uplift RCT analysis (Hillstrom/Criteo) & Tiki catalog EDA
│   └── colab_ml_training_template.ipynb  # Standardized Google Colab training notebook template
│
├── src/                                  # 🧩 Core Python package
│   ├── data_prep/                        # Data validators & leakage-free feature processors
│   ├── ml_models/                        # Base trainer contract & Tabular ML modules
│   ├── llm_finetune/                     # QLoRA SFT & DPO alignment modules
│   ├── evaluation/                       # Metrics & blocking Quality Gate evaluators
│   ├── serving/                          # vLLM inference wrappers
│   └── utils/                            # Cloud-ready storage abstraction, schema validator & MLflow
│
├── scripts/                              # 🚀 CLI helper scripts (Kaggle / synthetic ingestion)
├── docker/                               # 🐳 Dockerfiles for Airflow, Training & vLLM Serving
└── tests/                                # ✅ Automated unit & integration test suite
```

---

## ⚡ Quickstart (Sprint 1)

All commands below should be executed from the `fine-tuning/` directory (`D:\Card-abadon-analysis\fine-tuning`):

### 1. Run End-to-End Smoke Pipeline with Synthetic Demo Data (No 14GB Download Required)

```bash
# Step 1: Generate schema-accurate synthetic raw CSVs into data/raw/
python scripts/download_datasets.py --synthetic-demo

# Step 2: Validate raw schemas, null rates, and domain constraints
python -m src.data_prep.quality_filter --data-root data/raw --report-path data/processed/quality_report.json

# Step 3: Run leakage-free feature engineering & schema contract validation
python -m src.data_prep.feature_engineering --raw-dir data/raw --output-dir data/processed/ml --schema-path configs/feature_schema.yaml

# Step 4: Run automated test suite
python -m pytest tests/
```

### 2. Download Full Datasets from Kaggle

```bash
# Configure KAGGLE_USERNAME and KAGGLE_KEY in .env or ~/.kaggle/kaggle.json
python scripts/download_datasets.py --dataset all
```

### 3. Launch Local MLOps Stack (Airflow + MLflow + Postgres + Jupyter)

```bash
cp .env.example .env
docker compose up -d --build
```

- **MLflow Tracking & Registry UI:** http://localhost:5000
- **Apache Airflow UI:** http://localhost:8080
- **Jupyter Lab:** http://localhost:8888 (Token: `cart-abandon`)

---

## 📚 Documentation & Guides

- [Data Storage Architecture & Cloud Scale-Up](./docs/data-storage-and-pipeline.md)
- [Google Colab Tabular ML Training Guide](./docs/colab-ml-guide.md)
- [vast.ai GPU Rental & Sizing Guide for LLM / vLLM](./docs/vast-ai-gpu-guide.md)
- [Bug Reports & Root Cause Analysis](./bugs/README.md)
