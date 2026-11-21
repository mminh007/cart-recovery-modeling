"""Unit & integration tests for dataset ingestion, quality validation, and feature engineering."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts.download_datasets import generate_synthetic_demo_data
from src.data_prep.feature_engineering import run_feature_pipeline
from src.data_prep.quality_filter import validate_raw_directory
from src.data_prep.rees46_processor import process_rees46_directory
from src.utils.storage import read_tabular_records


def test_end_to_end_synthetic_ingestion_validation_and_features(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    ml_out = tmp_path / "processed" / "ml"
    llm_out = tmp_path / "processed" / "llm"

    generated = generate_synthetic_demo_data(raw_dir, seed=7, n_users=80)
    assert set(generated.keys()) == {"rees46", "hillstrom", "criteo", "tiki"}

    quality = validate_raw_directory(raw_dir)
    assert quality["overall_passed"] is True

    summary = run_feature_pipeline(
        raw_dir=raw_dir,
        ml_output_dir=ml_out,
        llm_output_dir=llm_out,
        schema_path="configs/feature_schema.yaml",
    )
    assert summary["vietnamese_catalog_items"] == 8
    assert len(summary["tables"]) == 10

    intent_rows = read_tabular_records(ml_out / "intent_train.parquet")
    assert len(intent_rows) > 0
    assert {"purchased_within_7d", "cart_total_value", "prior_sessions_count"}.issubset(
        intent_rows[0].keys()
    )

    uplift_rows = read_tabular_records(ml_out / "uplift_train.parquet")
    assert len(uplift_rows) == 480  # 80% of 600 synthetic Hillstrom rows


def test_rees46_processor_prevents_temporal_leakage(tmp_path: Path) -> None:
    """Verify future purchases within 7d set target=1 WITHOUT leaking into prior_purchases_count."""
    rees_dir = tmp_path / "rees46"
    rees_dir.mkdir(parents=True)
    csv_file = rees_dir / "leakage_check.csv"

    with csv_file.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "event_time",
            "event_type",
            "product_id",
            "category_id",
            "category_code",
            "brand",
            "price",
            "user_id",
            "user_session",
        ])
        # Session 1: User views & abandons cart on Oct 5 (0 prior purchases!)
        writer.writerow([
            "2019-10-05 10:00:00 UTC", "view", 1001, 2053, "electronics.smartphone", "apple", "999.00", 777, "sess-1"
        ])
        writer.writerow([
            "2019-10-05 10:02:00 UTC", "cart", 1001, 2053, "electronics.smartphone", "apple", "999.00", 777, "sess-1"
        ])
        # Session 2: User comes back 2 days later (Oct 7) and purchases!
        writer.writerow([
            "2019-10-07 14:00:00 UTC", "purchase", 1001, 2053, "electronics.smartphone", "apple", "999.00", 777, "sess-2"
        ])
        # Session 3: User abandons another cart on Oct 10 (now SHOULD have 1 prior purchase, 0 future purchase!)
        writer.writerow([
            "2019-10-10 09:00:00 UTC", "cart", 1002, 2053, "electronics.smartphone", "samsung", "499.00", 777, "sess-3"
        ])

    tables = process_rees46_directory(rees_dir)
    all_intent = tables["intent_train"] + (
        tables["intent_eval"] if tables["intent_eval"] != tables["intent_train"] else []
    )
    by_sess = {r["user_session"]: r for r in all_intent}

    assert by_sess["sess-1"]["prior_purchases_count"] == 0
    assert by_sess["sess-1"]["purchased_within_7d"] == 1

    assert by_sess["sess-3"]["prior_purchases_count"] == 1
    assert by_sess["sess-3"]["purchased_within_7d"] == 0
