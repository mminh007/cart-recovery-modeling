"""Unit tests for YAML config loading, FeatureSchemaRegistry, and cloud-ready StorageConfig."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config_loader import load_yaml_config
from src.utils.feature_schema import FeatureSchemaRegistry
from src.utils.mlflow_config import initialize_experiments
from src.utils.storage import StorageConfig, read_tabular_records, write_tabular_records


def test_load_all_yaml_configs() -> None:
    """Verify every YAML config in configs/ loads cleanly."""
    config_root = Path("configs")
    yaml_files = list(config_root.rglob("*.yaml"))
    assert len(yaml_files) >= 10

    for cfg_file in yaml_files:
        parsed = load_yaml_config(cfg_file)
        assert isinstance(parsed, dict)
        assert len(parsed) > 0, f"Empty parsed dict for {cfg_file}"


def test_feature_schema_registry_loads_all_five_models() -> None:
    registry = FeatureSchemaRegistry.from_yaml("configs/feature_schema.yaml")
    assert registry.schema_version == "1.0.0"
    expected_models = {
        "intent_classifier",
        "churn_predictor",
        "uplift_model",
        "timing_optimizer",
        "channel_selector",
    }
    assert set(registry.models.keys()) == expected_models

    intent_spec = registry.get("intent_classifier")
    assert intent_spec.target.name == "purchased_within_7d"
    assert len(intent_spec.features) == 17


def test_feature_schema_validation_catches_invalid_rows() -> None:
    registry = FeatureSchemaRegistry.from_yaml("configs/feature_schema.yaml")
    intent_spec = registry.get("intent_classifier")

    valid_row = {
        "session_duration_sec": 120.5,
        "session_views_count": 5,
        "session_categories_count": 2,
        "session_brands_count": 2,
        "cart_items_count": 1,
        "cart_total_value": 299.99,
        "cart_avg_price": 299.99,
        "cart_max_price": 299.99,
        "has_removed_from_cart": 0,
        "prior_sessions_count": 3,
        "prior_views_count": 12,
        "prior_purchases_count": 1,
        "prior_conversion_rate": 0.33,
        "days_since_first_seen": 10.0,
        "event_hour": 21,
        "day_of_week": 4,
        "is_weekend": 0,
        "purchased_within_7d": 1,
    }
    assert intent_spec.validate_records([valid_row]) == []

    # Out-of-range hour and invalid binary target
    bad_row = dict(valid_row)
    bad_row["event_hour"] = 28
    bad_row["purchased_within_7d"] = 2
    errors = intent_spec.validate_records([bad_row])
    assert len(errors) == 2
    assert any("event_hour" in e for e in errors)
    assert any("purchased_within_7d" in e for e in errors)


def test_storage_config_cloud_uri_resolution(tmp_path: Path) -> None:
    s3_cfg = StorageConfig(backend="s3", s3_bucket_uri="s3://my-ml-bucket")
    assert (
        s3_cfg.resolve_uri("processed/ml/intent_train.parquet")
        == "s3://my-ml-bucket/processed/ml/intent_train.parquet"
    )

    gcs_cfg = StorageConfig(backend="gcs", gcs_bucket_uri="gs://my-gcs-lake/")
    assert (
        gcs_cfg.resolve_uri("/raw/rees46/2019-Oct.csv")
        == "gs://my-gcs-lake/raw/rees46/2019-Oct.csv"
    )

    sample = [{"user_id": 101, "score": 0.85}, {"user_id": 102, "score": 0.12}]
    out_file = tmp_path / "sample.parquet"
    write_tabular_records(sample, out_file)
    roundtrip = read_tabular_records(out_file)
    assert roundtrip == sample


def test_mlflow_experiment_initialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    exps = initialize_experiments()
    assert len(exps) == 7
    assert "intent_classifier" in exps
    assert "llm_dpo" in exps
