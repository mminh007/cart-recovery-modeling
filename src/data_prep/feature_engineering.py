"""End-to-end Feature Engineering pipeline with strict FeatureSchemaRegistry validation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_prep.catalog_loader import export_catalog_jsonl
from src.data_prep.hillstrom_processor import process_hillstrom_directory
from src.data_prep.rees46_processor import process_rees46_directory
from src.utils.feature_schema import FeatureSchemaRegistry
from src.utils.storage import write_tabular_records

DATASET_TO_SCHEMA_MODEL: dict[str, str] = {
    "intent_train": "intent_classifier",
    "intent_eval": "intent_classifier",
    "churn_train": "churn_predictor",
    "churn_eval": "churn_predictor",
    "uplift_train": "uplift_model",
    "uplift_eval": "uplift_model",
    "timing_train": "timing_optimizer",
    "timing_eval": "timing_optimizer",
    "channel_train": "channel_selector",
    "channel_eval": "channel_selector",
}


def run_feature_pipeline(
    raw_dir: str | Path = "data/raw",
    ml_output_dir: str | Path = "data/processed/ml",
    llm_output_dir: str | Path = "data/processed/llm",
    schema_path: str | Path = "configs/feature_schema.yaml",
    max_rees46_rows: int | None = None,
) -> dict[str, Any]:
    """Process all raw datasets, enforce feature_schema.yaml contract, and export Parquet tables."""
    raw_root = Path(raw_dir)
    ml_out = Path(ml_output_dir)
    llm_out = Path(llm_output_dir)
    ml_out.mkdir(parents=True, exist_ok=True)
    llm_out.mkdir(parents=True, exist_ok=True)

    registry = FeatureSchemaRegistry.from_yaml(schema_path)

    all_tables: dict[str, list[dict[str, Any]]] = {}
    all_tables.update(
        process_rees46_directory(raw_root / "rees46", max_rows_per_file=max_rees46_rows)
    )
    all_tables.update(process_hillstrom_directory(raw_root / "hillstrom"))

    exported_files: dict[str, dict[str, Any]] = {}
    for table_name, records in all_tables.items():
        schema_key = DATASET_TO_SCHEMA_MODEL[table_name]
        spec = registry.get(schema_key)
        errors = spec.validate_records(records, require_target=True)
        if errors:
            joined = "\n  - ".join(errors[:10])
            raise ValueError(
                f"Schema validation failed for '{table_name}' ({schema_key}):\n  - {joined}"
            )

        out_file = ml_out / f"{table_name}.parquet"
        write_tabular_records(records, out_file)
        exported_files[table_name] = {
            "path": str(out_file),
            "rows": len(records),
            "schema_model": schema_key,
        }

    catalog_count = 0
    tiki_dir = raw_root / "tiki_shopee_catalog"
    if tiki_dir.exists() and any(tiki_dir.glob("*.csv")):
        catalog_jsonl = llm_out / "vietnamese_catalog.jsonl"
        catalog_count = export_catalog_jsonl(tiki_dir, catalog_jsonl)

    summary = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": registry.schema_version,
        "tables": exported_files,
        "vietnamese_catalog_items": catalog_count,
    }
    summary_path = ml_out / "feature_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Cart Abandonment Feature Engineering Pipeline")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ml"))
    parser.add_argument("--llm-output-dir", type=Path, default=Path("data/processed/llm"))
    parser.add_argument("--schema-path", type=Path, default=Path("configs/feature_schema.yaml"))
    parser.add_argument("--max-rees46-rows", type=int, default=None)
    args = parser.parse_args()

    summary = run_feature_pipeline(
        raw_dir=args.raw_dir,
        ml_output_dir=args.output_dir,
        llm_output_dir=args.llm_output_dir,
        schema_path=args.schema_path,
        max_rees46_rows=args.max_rees46_rows,
    )
    print(f"Feature engineering completed ({len(summary['tables'])} tables):")
    for t_name, meta in summary["tables"].items():
        print(f"  - {t_name}: {meta['rows']} rows -> {meta['path']}")
    print(f"  - vietnamese_catalog: {summary['vietnamese_catalog_items']} products")


if __name__ == "__main__":
    main()
