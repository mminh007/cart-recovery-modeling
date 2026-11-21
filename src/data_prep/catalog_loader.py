"""Vietnamese Tiki/Shopee product catalog loader & cleaner for LLM Grounding (Sprint 3)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_clean_catalog(catalog_dir: str | Path) -> list[dict[str, Any]]:
    """Load Vietnamese product CSVs, filter invalid entries, and normalize VND prices."""
    folder = Path(catalog_dir)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Vietnamese catalog CSV files found in {folder}")

    seen_ids: set[int] = set()
    items: list[dict[str, Any]] = []

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                name = (row.get("name") or "").strip()
                if not name:
                    continue
                try:
                    pid = int(float(row.get("id") or 0))
                    price_vnd = int(round(float(row.get("price") or 0)))
                    orig_vnd = int(round(float(row.get("original_price") or price_vnd)))
                except ValueError:
                    continue

                if price_vnd <= 0 or pid in seen_ids:
                    continue
                seen_ids.add(pid)

                orig_vnd = max(orig_vnd, price_vnd)
                discount_pct = (
                    round((orig_vnd - price_vnd) / orig_vnd * 100.0, 1) if orig_vnd > 0 else 0.0
                )

                items.append(
                    {
                        "product_id": pid,
                        "product_name": name,
                        "brand": (row.get("brand") or "OEM").strip() or "OEM",
                        "price_vnd": price_vnd,
                        "original_price_vnd": orig_vnd,
                        "listed_discount_pct": discount_pct,
                        "rating_average": round(float(row.get("rating_average") or 0.0), 2),
                        "review_counts": int(float(row.get("review_counts") or 0)),
                        "quantity_sold": int(float(row.get("quantity_sold") or 0)),
                        "source_category_file": csv_path.stem,
                    }
                )

    return items


def export_catalog_jsonl(catalog_dir: str | Path, output_jsonl: str | Path) -> int:
    """Clean Vietnamese catalog and write to JSONL for LLM SFT/DPO generation."""
    items = load_clean_catalog(catalog_dir)
    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)
