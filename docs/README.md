# 📚 Documentation Index (`/docs`)

Welcome to the technical documentation hub for **Repo 1 (`fine-tuning`) — Cart Abandonment Modeling, Fine-tuning & MLOps Pipeline**.

---

## 🗺️ Contents

### 1. Engineering & Infrastructure Guides (`./`)
- [data-storage-and-pipeline.md](./data-storage-and-pipeline.md) — Medallion-lite storage layers (`Raw → Validated → Parquet Gold`), temporal data leakage prevention, DVC + Airflow + MLflow lineage, and zero-code-change cloud migration (`s3://` / `gs://`).
- [colab-ml-guide.md](./colab-ml-guide.md) — Workflow for training the 5 Tabular ML models on Google Colab synced with Google Drive and MLflow.
- [vast-ai-gpu-guide.md](./vast-ai-gpu-guide.md) — GPU VRAM sizing, hourly cost comparison (`RTX 4090`, `RTX A6000`, `L40S`, `A100`), instance selection filters, and step-by-step operational runbook on **vast.ai** for QLoRA SFT, DPO, and vLLM serving.

### 2. Workspace System Specifications (`../../docs/`)
- High-level multi-repo system design documents live in the parent workspace directory (`../docs/` relative to the monorepo root).

### 3. Bug Tracking & Post-Mortems (`../bugs/`)
- See [../bugs/README.md](../bugs/README.md) for Root Cause Analysis reports and regression test tracking.
