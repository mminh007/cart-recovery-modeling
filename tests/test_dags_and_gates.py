"""Tests for Airflow DAG task callables, Quality Gate enforcement, and DVC/Docker configs."""

from __future__ import annotations

from pathlib import Path

import pytest

from dags.cart_abandonment_data_dag import (
    task_ensure_raw_data,
    task_log_dataset_lineage_mlflow,
    task_run_feature_engineering,
    task_validate_raw_data,
)
from dags.cart_abandonment_training_dag import (
    task_evaluate_quality_gates,
    task_verify_processed_features,
)
from src.evaluation.ml_metrics import check_all_quality_gates, compute_binary_auc_roc


def test_airflow_dags_execute_end_to_end_with_demo_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("MLFLOW_LOCAL_DIR", str(tmp_path / "mlruns"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("USE_SYNTHETIC_DEMO", "true")
    monkeypatch.chdir(Path(__file__).resolve().parents[1])

    ingest_res = task_ensure_raw_data()
    assert "rees46" in ingest_res

    val_res = task_validate_raw_data()
    assert val_res["overall_passed"] is True

    feat_res = task_run_feature_engineering()
    assert len(feat_res["tables"]) == 10

    lineage_res = task_log_dataset_lineage_mlflow()
    assert int(lineage_res["experiments_count"]) == 7

    counts = task_verify_processed_features()
    assert counts["intent_train.parquet"] > 0

    gate_status = task_evaluate_quality_gates()
    assert gate_status["status"] == "awaiting_sprint2_training_metrics"


def test_quality_gates_blocking_vs_recommended() -> None:
    auc = compute_binary_auc_roc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert auc == 1.0

    passing_payload = {
        "intent_classifier": {"auc_roc": 0.84},
        "churn_predictor": {"auc_roc": 0.78},
        "uplift_model": {"auuc": 0.045},
        "timing_optimizer": {"c_index": 0.60},  # Below recommended, NOT blocking
    }
    res = check_all_quality_gates(passing_payload)
    assert res["all_blocking_passed"] is True
    assert res["models"]["timing_optimizer"]["passed"] is False

    failing_payload = {
        "intent_classifier": {"auc_roc": 0.72},  # Blocking failure (< 0.80)
    }
    with pytest.raises(RuntimeError, match="Blocking Quality Gate"):
        check_all_quality_gates(failing_payload, raise_on_blocking_failure=True)
