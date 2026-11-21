"""Leakage-free REES46 clickstream processor for Intent, Churn, and Survival Timing models."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def parse_rees46_timestamp(raw: str) -> datetime:
    """Parse 'YYYY-MM-DD HH:MM:SS UTC' into a timezone-aware UTC datetime."""
    cleaned = raw.strip()
    if cleaned.endswith(" UTC"):
        cleaned = cleaned[:-4]
    return datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


@dataclass
class SessionSummary:
    """Aggregated statistics for a single user_session."""

    user_id: int
    user_session: str
    start_time: datetime
    end_time: datetime
    views_count: int = 0
    carts_count: int = 0
    removes_count: int = 0
    purchases_count: int = 0
    total_events: int = 0
    categories: set[str] = field(default_factory=set)
    brands: set[str] = field(default_factory=set)
    cart_prices: list[float] = field(default_factory=list)
    purchase_spend: float = 0.0


def _split_train_eval(
    records: list[dict[str, Any]], eval_ratio: float = 0.2
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic temporal/index split (80/20) preserving chronological order."""
    if len(records) <= 1:
        return records, records
    split_idx = max(1, int(len(records) * (1.0 - eval_ratio)))
    return records[:split_idx], records[split_idx:]


def process_rees46_directory(
    rees46_dir: str | Path,
    max_rows_per_file: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Stream REES46 CSV files and build Intent, Churn, and Timing feature tables."""
    folder = Path(rees46_dir)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No REES46 CSV files found in {folder}")

    sessions_by_key: dict[tuple[int, str], SessionSummary] = {}
    user_purchases: dict[int, list[datetime]] = defaultdict(list)
    user_hour_counts: dict[int, Counter[int]] = defaultdict(Counter)
    min_ts: datetime | None = None
    max_ts: datetime | None = None

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row_idx, row in enumerate(reader, start=1):
                if max_rows_per_file is not None and row_idx > max_rows_per_file:
                    break

                ts = parse_rees46_timestamp(row["event_time"])
                user_id = int(row["user_id"])
                session_id = (row["user_session"] or f"anon-{user_id}").strip()
                ev_type = (row["event_type"] or "").strip()
                price = max(0.0, float(row.get("price") or 0.0))
                cat_code = (row.get("category_code") or "").strip()
                brand = (row.get("brand") or "").strip()

                if min_ts is None or ts < min_ts:
                    min_ts = ts
                if max_ts is None or ts > max_ts:
                    max_ts = ts

                key = (user_id, session_id)
                sess = sessions_by_key.get(key)
                if sess is None:
                    sess = SessionSummary(
                        user_id=user_id,
                        user_session=session_id,
                        start_time=ts,
                        end_time=ts,
                    )
                    sessions_by_key[key] = sess
                else:
                    if ts < sess.start_time:
                        sess.start_time = ts
                    if ts > sess.end_time:
                        sess.end_time = ts

                sess.total_events += 1
                user_hour_counts[user_id][ts.hour] += 1
                if cat_code:
                    sess.categories.add(cat_code.split(".")[0])
                if brand:
                    sess.brands.add(brand)

                if ev_type == "view":
                    sess.views_count += 1
                elif ev_type == "cart":
                    sess.carts_count += 1
                    sess.cart_prices.append(price)
                elif ev_type == "remove_from_cart":
                    sess.removes_count += 1
                elif ev_type == "purchase":
                    sess.purchases_count += 1
                    sess.purchase_spend += price
                    user_purchases[user_id].append(ts)

    if min_ts is None or max_ts is None:
        raise ValueError("REES46 CSV files contained zero valid event rows")

    # Group sessions per user sorted chronologically
    user_sessions: dict[int, list[SessionSummary]] = defaultdict(list)
    for sess in sessions_by_key.values():
        user_sessions[sess.user_id].append(sess)
    for uid in user_sessions:
        user_sessions[uid].sort(key=lambda s: s.start_time)
        if uid in user_purchases:
            user_purchases[uid].sort()

    intent_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for uid, sess_list in user_sessions.items():
        first_seen = sess_list[0].start_time
        prior_sessions = 0
        prior_views = 0
        prior_purchases = 0
        prior_hour_counter: Counter[int] = Counter()

        for idx, sess in enumerate(sess_list):
            # Abandoned cart session: added to cart, no purchase inside same session
            if sess.carts_count >= 1 and sess.purchases_count == 0:
                abandon_ts = sess.end_time
                window_7d = abandon_ts + timedelta(days=7)

                # Target: any purchase strictly after abandon_ts within 7 days
                bought_in_7d = any(
                    abandon_ts < p_ts <= window_7d for p_ts in user_purchases.get(uid, [])
                )

                cart_total = round(sum(sess.cart_prices), 2)
                cart_count = len(sess.cart_prices)
                cart_avg = round(cart_total / max(cart_count, 1), 2)
                cart_max = round(max(sess.cart_prices) if sess.cart_prices else 0.0, 2)
                prior_conv = (
                    round(min(1.0, prior_purchases / prior_sessions), 4)
                    if prior_sessions > 0
                    else 0.0
                )
                days_since_first = round(
                    max(0.0, (abandon_ts - first_seen).total_seconds() / 86400.0), 3
                )
                duration_sec = round(
                    max(0.0, (sess.end_time - sess.start_time).total_seconds()), 1
                )
                dow = abandon_ts.weekday()
                is_wknd = 1 if dow >= 5 else 0

                intent_rows.append(
                    {
                        "user_id": uid,
                        "user_session": sess.user_session,
                        "cart_abandon_time": abandon_ts.isoformat(),
                        "session_duration_sec": float(duration_sec),
                        "session_views_count": int(sess.views_count),
                        "session_categories_count": int(len(sess.categories)),
                        "session_brands_count": int(len(sess.brands)),
                        "cart_items_count": int(sess.carts_count),
                        "cart_total_value": float(cart_total),
                        "cart_avg_price": float(cart_avg),
                        "cart_max_price": float(cart_max),
                        "has_removed_from_cart": int(1 if sess.removes_count > 0 else 0),
                        "prior_sessions_count": int(prior_sessions),
                        "prior_views_count": int(prior_views),
                        "prior_purchases_count": int(prior_purchases),
                        "prior_conversion_rate": float(prior_conv),
                        "days_since_first_seen": float(days_since_first),
                        "event_hour": int(abandon_ts.hour),
                        "day_of_week": int(dow),
                        "is_weekend": int(is_wknd),
                        "purchased_within_7d": int(1 if bought_in_7d else 0),
                    }
                )

                # Survival Timing features (horizon = 168h / 7 days)
                next_sess = sess_list[idx + 1] if (idx + 1) < len(sess_list) else None
                if next_sess is not None:
                    delta_hours = max(
                        0.1, (next_sess.start_time - abandon_ts).total_seconds() / 3600.0
                    )
                    if delta_hours <= 168.0:
                        hours_until_return = round(delta_hours, 2)
                        event_returned = 1
                    else:
                        hours_until_return = 168.0
                        event_returned = 0
                else:
                    remaining_h = max(0.5, (max_ts - abandon_ts).total_seconds() / 3600.0)
                    hours_until_return = round(min(168.0, remaining_h), 2)
                    event_returned = 0

                pref_hour = (
                    prior_hour_counter.most_common(1)[0][0]
                    if prior_hour_counter
                    else abandon_ts.hour
                )

                timing_rows.append(
                    {
                        "user_id": uid,
                        "user_session": sess.user_session,
                        "hours_until_return": float(hours_until_return),
                        "abandon_hour": int(abandon_ts.hour),
                        "abandon_day_of_week": int(dow),
                        "is_weekend": int(is_wknd),
                        "cart_total_value": float(cart_total),
                        "prior_sessions_count": int(prior_sessions),
                        "preferred_active_hour": int(pref_hour),
                        "event_returned": int(event_returned),
                    }
                )

            # Advance historical accumulators AFTER extracting features (zero leakage)
            prior_sessions += 1
            prior_views += sess.views_count
            prior_purchases += 1 if sess.purchases_count > 0 else 0
            prior_hour_counter[sess.start_time.hour] += 1

    # Churn dataset using midpoint observation cutoff
    total_span_sec = max(3600.0, (max_ts - min_ts).total_seconds())
    cutoff_ts = min_ts + timedelta(seconds=total_span_sec * 0.55)
    midpoint_prior_ts = min_ts + timedelta(seconds=total_span_sec * 0.275)
    churn_horizon_ts = cutoff_ts + timedelta(days=30)

    churn_rows: list[dict[str, Any]] = []
    for uid, sess_list in user_sessions.items():
        hist_sessions = [s for s in sess_list if s.end_time <= cutoff_ts]
        if not hist_sessions:
            continue

        future_sessions = [s for s in sess_list if cutoff_ts < s.start_time <= churn_horizon_ts]
        churned_30d = 1 if len(future_sessions) == 0 else 0

        last_visit = max(s.end_time for s in hist_sessions)
        recency_days = round(max(0.0, (cutoff_ts - last_visit).total_seconds() / 86400.0), 3)
        sessions_count = len(hist_sessions)
        events_count = sum(s.total_events for s in hist_sessions)
        total_spend = round(sum(s.purchase_spend for s in hist_sessions), 2)
        order_sessions = sum(1 for s in hist_sessions if s.purchases_count > 0)
        avg_order_val = round(total_spend / order_sessions, 2) if order_sessions > 0 else 0.0
        pages_per_sess = round(events_count / max(sessions_count, 1), 2)

        cart_sess_n = sum(1 for s in hist_sessions if s.carts_count > 0)
        abandoned_sess_n = sum(
            1 for s in hist_sessions if s.carts_count > 0 and s.purchases_count == 0
        )
        abandon_rate = (
            round(abandoned_sess_n / cart_sess_n, 4) if cart_sess_n > 0 else 0.0
        )

        early_events = sum(s.total_events for s in hist_sessions if s.end_time <= midpoint_prior_ts)
        recent_events = sum(s.total_events for s in hist_sessions if s.end_time > midpoint_prior_ts)
        trend_ratio = round((recent_events + 1.0) / (early_events + 1.0), 4)

        churn_rows.append(
            {
                "user_id": uid,
                "cutoff_time": cutoff_ts.isoformat(),
                "recency_days": float(recency_days),
                "sessions_count_30d": int(sessions_count),
                "events_count_30d": int(events_count),
                "total_spend_30d": float(total_spend),
                "avg_order_value": float(avg_order_val),
                "pages_per_session": float(pages_per_sess),
                "cart_abandon_rate": float(abandon_rate),
                "activity_trend_ratio": float(trend_ratio),
                "churned_30d": int(churned_30d),
            }
        )

    intent_rows.sort(key=lambda r: str(r["cart_abandon_time"]))
    intent_train, intent_eval = _split_train_eval(intent_rows)
    churn_train, churn_eval = _split_train_eval(churn_rows)
    timing_train, timing_eval = _split_train_eval(timing_rows)

    return {
        "intent_train": intent_train,
        "intent_eval": intent_eval,
        "churn_train": churn_train,
        "churn_eval": churn_eval,
        "timing_train": timing_train,
        "timing_eval": timing_eval,
    }
