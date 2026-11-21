"""Raw dataset schema & quality validation gate (Stage 2 of Data Pipeline)."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_RAW_COLUMNS: dict[str, list[str]] = {
    "rees46": [
        "event_time",
        "event_type",
        "product_id",
        "category_id",
        "category_code",
        "brand",
        "price",
        "user_id",
        "user_session",
    ],
    "hillstrom": [
        "recency",
        "history_segment",
        "history",
        "mens",
        "womens",
        "zip_code",
        "newbie",
        "channel",
        "segment",
        "visit",
        "conversion",
        "spend",
    ],
    "criteo_uplift": [
        *(f"f{i}" for i in range(12)),
        "treatment",
        "conversion",
        "visit",
        "exposure",
    ],
    "tiki_shopee_catalog": [
        "id",
        "name",
        "description",
        "original_price",
        "price",
        "fulfillment_type",
        "brand",
        "review_counts",
        "rating_average",
        "favorite_count",
        "pay_later",
        "current_seller",
        "date_created",
        "number_of_images",
        "vnd_cashback",
        "has_video",
        "quantity_sold",
    ],
}

VALID_REES46_EVENTS = {"view", "cart", "remove_from_cart", "purchase"}


def validate_csv_file(
    csv_path: Path,
    expected_columns: list[str],
    dataset_type: str,
    max_scan_rows: int = 50_000,
) -> dict[str, Any]:
    """Validate CSV headers, null rates, and domain constraints on up to max_scan_rows."""
    issues: list[str] = []
    null_counts: dict[str, int] = {col: 0 for col in expected_columns}
    rows_checked = 0
    negative_prices = 0
    invalid_events = 0

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        actual_cols = reader.fieldnames or []
        missing_cols = [c for c in expected_columns if c not in actual_cols]
        if missing_cols:
            return {
                "file": str(csv_path),
                "passed": False,
                "rows_checked": 0,
                "issues": [f"Missing required columns: {missing_cols}"],
            }

        for row in reader:
            rows_checked += 1
            for col in expected_columns:
                val = row.get(col)
                if val is None or val.strip() == "":
                    null_counts[col] += 1

            if dataset_type in ("rees46", "tiki_shopee_catalog"):
                try:
                    if float(row.get("price") or 0.0) < 0:
                        negative_prices += 1
                except ValueError:
                    issues.append(f"Unparseable price at row {rows_checked}")

            if dataset_type == "rees46":
                ev = (row.get("event_type") or "").strip()
                if ev not in VALID_REES46_EVENTS:
                    invalid_events += 1

            if rows_checked >= max_scan_rows:
                break

    if rows_checked == 0:
        issues.append("CSV file contains 0 data rows")
    if negative_prices > 0:
        issues.append(f"Found {negative_prices} rows with negative price")
    if invalid_events > 0:
        issues.append(f"Found {invalid_events} rows with invalid REES46 event_type")

    # Non-nullable key columns check for REES46
    if dataset_type == "rees46":
        for mandatory_col in ("event_time", "event_type", "product_id", "user_id", "user_session"):
            if null_counts[mandatory_col] > 0:
                issues.append(
                    f"Mandatory column '{mandatory_col}' has {null_counts[mandatory_col]} nulls"
                )

    return {
        "file": str(csv_path),
        "passed": len(issues) == 0,
        "rows_checked": rows_checked,
        "null_rates": {
            col: round(null_counts[col] / max(rows_checked, 1), 4) for col in expected_columns
        },
        "issues": issues,
    }


def validate_raw_directory(data_root: str | Path = "data/raw") -> dict[str, Any]:
    """Validate all CSV datasets found under data_root."""
    root = Path(data_root)
    dataset_reports: dict[str, Any] = {}
    overall_passed = True

    for subdir_name, expected_cols in EXPECTED_RAW_COLUMNS.items():
        subdir = root / subdir_name
        csv_files = sorted(subdir.glob("*.csv")) if subdir.exists() else []
        if not csv_files:
            dataset_reports[subdir_name] = {
                "status": "missing_files",
                "passed": False,
                "files": [],
            }
            overall_passed = False
            continue

        file_results = [validate_csv_file(f, expected_cols, subdir_name) for f in csv_files]
        ds_passed = all(r["passed"] for r in file_results)
        if not ds_passed:
            overall_passed = False

        dataset_reports[subdir_name] = {
            "status": "validated",
            "passed": ds_passed,
            "files": file_results,
        }

    return {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "overall_passed": overall_passed,
        "datasets": dataset_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw datasets against Kaggle schemas")
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/processed/quality_report.json"),
    )
    args = parser.parse_args()

    report = validate_raw_directory(args.data_root)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Quality validation overall_passed={report['overall_passed']} -> {args.report_path}")
    if not report["overall_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
