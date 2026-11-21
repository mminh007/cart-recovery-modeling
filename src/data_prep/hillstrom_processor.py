"""Hillstrom Email Marketing processor for Uplift Modeling (CausalML) and Channel Selection."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def _derive_optimal_channel(
    channel_raw: str, newbie: int, recency: int, history: float, mobile_score: float
) -> str:
    """Deterministic multi-channel assignment combining Hillstrom channel with VN channels."""
    if mobile_score >= 0.72 and recency <= 4:
        return "push"
    if channel_raw == "Multichannel" or history >= 350.0:
        return "zalo"
    if channel_raw == "Phone" and newbie == 0:
        return "sms"
    return "email"


def process_hillstrom_directory(hillstrom_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Parse Hillstrom CSV(s) into Uplift and Channel Selector train/eval splits."""
    folder = Path(hillstrom_dir)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No Hillstrom CSV files found in {folder}")

    uplift_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    customer_seq = 0

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                customer_seq += 1
                recency = max(0, int(float(row["recency"])))
                history = round(max(0.0, float(row["history"])), 2)
                mens = 1 if int(float(row["mens"])) > 0 else 0
                womens = 1 if int(float(row["womens"])) > 0 else 0
                newbie = 1 if int(float(row["newbie"])) > 0 else 0

                zip_raw = (row.get("zip_code") or "").strip()
                channel_raw = (row.get("channel") or "").strip()
                segment_raw = (row.get("segment") or "").strip()

                visit = 1 if int(float(row.get("visit") or 0)) > 0 else 0
                conversion = 1 if int(float(row.get("conversion") or 0)) > 0 else 0
                spend = round(max(0.0, float(row.get("spend") or 0.0)), 2)

                treatment = 0 if segment_raw.lower() == "no e-mail" else 1
                seg_norm = segment_raw.lower().replace(" ", "_").replace("-", "")

                zip_urban = 1 if zip_raw == "Urban" else 0
                zip_surburban = 1 if zip_raw == "Surburban" else 0
                zip_rural = 1 if zip_raw == "Rural" else 0

                ch_web = 1 if channel_raw == "Web" else 0
                ch_phone = 1 if channel_raw == "Phone" else 0
                ch_multi = 1 if channel_raw == "Multichannel" else 0

                uplift_rows.append(
                    {
                        "customer_id": customer_seq,
                        "treatment": int(treatment),
                        "treatment_segment": seg_norm,
                        "recency": int(recency),
                        "history": float(history),
                        "mens": int(mens),
                        "womens": int(womens),
                        "newbie": int(newbie),
                        "zip_urban": int(zip_urban),
                        "zip_surburban": int(zip_surburban),
                        "zip_rural": int(zip_rural),
                        "channel_web": int(ch_web),
                        "channel_phone": int(ch_phone),
                        "channel_multichannel": int(ch_multi),
                        "visit": int(visit),
                        "spend": float(spend),
                        "conversion": int(conversion),
                    }
                )

                # Mobile affinity heuristic bounded in [0.0, 1.0]
                raw_aff = (
                    0.45 * ch_web
                    + 0.35 * ch_multi
                    + 0.15 * newbie
                    + max(0.0, (12 - recency) / 40.0)
                )
                mobile_score = round(min(1.0, max(0.0, raw_aff)), 4)
                opt_channel = _derive_optimal_channel(
                    channel_raw, newbie, recency, history, mobile_score
                )

                channel_rows.append(
                    {
                        "customer_id": customer_seq,
                        "recency": int(recency),
                        "history": float(history),
                        "mens": int(mens),
                        "womens": int(womens),
                        "newbie": int(newbie),
                        "channel_web": int(ch_web),
                        "channel_phone": int(ch_phone),
                        "channel_multichannel": int(ch_multi),
                        "mobile_affinity_score": float(mobile_score),
                        "optimal_channel": opt_channel,
                    }
                )

    split_u = max(1, int(len(uplift_rows) * 0.8))
    split_c = max(1, int(len(channel_rows) * 0.8))

    return {
        "uplift_train": uplift_rows[:split_u],
        "uplift_eval": uplift_rows[split_u:] or uplift_rows[:1],
        "channel_train": channel_rows[:split_c],
        "channel_eval": channel_rows[split_c:] or channel_rows[:1],
    }
