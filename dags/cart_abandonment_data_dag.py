"""Airflow DAG 1: Cart Abandonment Data Ingestion, Validation & Feature Engineering.

Designed for Local Docker Compose (Phase 1) and zero-code-change migration to
Cloud Composer / AWS MWAA (Phase 2) via environment variables.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def task_ensure_raw_data(**context: Any) -> dict[str, str]:
    """Ensure raw data exists; generate synthetic demo data if USE_SYNTHETIC_DEMO=true."""
    data_root = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "data")))
    raw_dir = data_root / "raw"
    use_demo = os.getenv("USE_SYNTHETIC_DEMO", "false").lower() == "true"

    rees_has_csv = (raw_dir / "rees46").exists() and any((raw_dir / "rees46").glob("*.csv"))
    if not rees_has_csv and use_demo:
        from scripts.download_datasets import generate_synthetic_demo_data

        generated = generate_synthetic_demo_data(raw_dir)
        return {k: str(v) for k, v in generated.items()}

    if not rees_has_csv:
        raise FileNotFoundError(
            f"No raw CSVs found in {raw_dir / 'rees46'}. Populate data/raw/ via Kaggle CLI "
            "or set USE_SYNTHETIC_DEMO=true for smoke testing."
        )
    return {"status": "raw_files_present", "raw_dir": str(raw_dir)}


def task_validate_raw_data(**context: Any) -> dict[str, Any]:
    """Run Stage 2 schema & null quality checks on raw CSVs."""
    import json

    from src.data_prep.quality_filter import validate_raw_directory

    data_root = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "data")))
    report = validate_raw_directory(data_root / "raw")
    report_path = data_root / "processed" / "quality_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if not report["overall_passed"]:
        raise ValueError(f"Raw data quality gate failed! See {report_path}")
    return {"overall_passed": True, "report_path": str(report_path)}


def task_run_feature_engineering(**context: Any) -> dict[str, Any]:
    """Run Stage 3 leakage-free feature engineering & schema contract validation."""
    from src.data_prep.feature_engineering import run_feature_pipeline

    data_root = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "data")))
    max_rows_env = os.getenv("MAX_REES46_ROWS", "").strip()
    max_rows = int(max_rows_env) if max_rows_env else None

    return run_feature_pipeline(
        raw_dir=data_root / "raw",
        ml_output_dir=data_root / "processed" / "ml",
        llm_output_dir=data_root / "processed" / "llm",
        schema_path=REPO_ROOT / "configs" / "feature_schema.yaml",
        max_rees46_rows=max_rows,
    )


def task_log_dataset_lineage_mlflow(**context: Any) -> dict[str, str]:
    """Initialize MLflow experiments and record processed dataset summary."""
    from src.utils.mlflow_config import get_tracking_uri, initialize_experiments

    exps = initialize_experiments()
    return {"tracking_uri": get_tracking_uri(), "experiments_count": str(len(exps))}


try:
    from airflow import DAG  # type: ignore[import-not-found]
    from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]

    DEFAULT_ARGS = {
        "owner": "cart-abandon-mlops",
        "depends_on_past": False,
        "retries": 2,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="cart_abandon_data_pipeline",
        default_args=DEFAULT_ARGS,
        description="Ingest, validate, and build leakage-free Parquet features for ML & LLM",
        schedule=None,
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["cart-abandonment", "data-pipeline", "sprint-1"],
    ) as dag:
        t_ingest = PythonOperator(
            task_id="ensure_raw_data",
            python_callable=task_ensure_raw_data,
        )
        t_validate = PythonOperator(
            task_id="validate_raw_data",
            python_callable=task_validate_raw_data,
        )
        t_features = PythonOperator(
            task_id="build_and_validate_features",
            python_callable=task_run_feature_engineering,
        )
        t_lineage = PythonOperator(
            task_id="log_mlflow_lineage",
            python_callable=task_log_dataset_lineage_mlflow,
        )

        t_ingest >> t_validate >> t_features >> t_lineage
except ImportError:
    dag = None
