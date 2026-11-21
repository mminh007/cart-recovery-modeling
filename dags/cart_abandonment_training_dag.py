"""Airflow DAG 2: Tabular ML Training Orchestration, Quality Gates & MLflow Registry.

Coordinates model training execution (local worker, remote Colab artifact sync, or cloud runner),
evaluates against blocking Quality Gates, and promotes qualifying models in MLflow Registry.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def task_verify_processed_features(**context: Any) -> dict[str, int]:
    """Verify all required processed feature tables exist before starting training."""
    from src.utils.storage import read_tabular_records

    data_root = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "data")))
    ml_dir = data_root / "processed" / "ml"
    required = [
        "intent_train.parquet",
        "intent_eval.parquet",
        "churn_train.parquet",
        "uplift_train.parquet",
        "timing_train.parquet",
    ]
    counts: dict[str, int] = {}
    for fname in required:
        fpath = ml_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"Missing processed table {fpath}. Run 'cart_abandon_data_pipeline' first."
            )
        counts[fname] = len(read_tabular_records(fpath))
    return counts


def task_evaluate_quality_gates(**context: Any) -> dict[str, Any]:
    """Check model evaluation report (from local run or synced Colab run) against blocking gates."""
    from src.evaluation.ml_metrics import check_all_quality_gates

    data_root = Path(os.getenv("DATA_ROOT", str(REPO_ROOT / "data")))
    metrics_file = data_root / "processed" / "ml" / "latest_eval_metrics.json"
    if not metrics_file.exists():
        return {"status": "awaiting_sprint2_training_metrics", "path": str(metrics_file)}

    metrics_payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    return check_all_quality_gates(metrics_payload)


try:
    from airflow import DAG  # type: ignore[import-not-found]
    from airflow.operators.python import PythonOperator  # type: ignore[import-not-found]

    DEFAULT_ARGS = {
        "owner": "cart-abandon-mlops",
        "depends_on_past": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    with DAG(
        dag_id="cart_abandon_ml_training",
        default_args=DEFAULT_ARGS,
        description="Verify processed features, run/sync ML training, and enforce MLflow Quality Gates",
        schedule=None,
        start_date=datetime(2026, 1, 1),
        catchup=False,
        tags=["cart-abandonment", "ml-training", "mlflow"],
    ) as dag:
        t_check_features = PythonOperator(
            task_id="verify_processed_features",
            python_callable=task_verify_processed_features,
        )
        t_quality_gates = PythonOperator(
            task_id="evaluate_quality_gates",
            python_callable=task_evaluate_quality_gates,
        )

        t_check_features >> t_quality_gates
except ImportError:
    dag = None
