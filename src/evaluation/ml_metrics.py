"""Evaluation metrics & blocking Quality Gate checks for Tabular ML and LLM."""

from __future__ import annotations

from typing import Any

QUALITY_GATES: dict[str, tuple[str, float, bool]] = {
    # model_name: (metric_name, min_threshold, is_blocking)
    "intent_classifier": ("auc_roc", 0.80, True),
    "churn_predictor": ("auc_roc", 0.75, True),
    "uplift_model": ("auuc", 0.03, True),
    "timing_optimizer": ("c_index", 0.65, False),
    "channel_selector": ("accuracy", 0.70, False),
}


def compute_binary_auc_roc(y_true: list[int], y_score: list[float]) -> float:
    """Compute exact ROC-AUC via Mann-Whitney U statistic (zero external C-deps needed)."""
    if len(y_true) != len(y_score) or not y_true:
        raise ValueError("y_true and y_score must be non-empty and equal length")

    n_pos = sum(1 for y in y_true if y == 1)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    paired = sorted(zip(y_score, y_true), key=lambda x: x[0])
    rank_sum_pos = 0.0
    idx = 0
    n = len(paired)
    while idx < n:
        j = idx
        while j < n and paired[j][0] == paired[idx][0]:
            j += 1
        avg_rank = (idx + 1 + j) / 2.0
        for k in range(idx, j):
            if paired[k][1] == 1:
                rank_sum_pos += avg_rank
        idx = j

    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(float(auc), 6)


def check_all_quality_gates(
    evaluated_models: dict[str, dict[str, float]],
    raise_on_blocking_failure: bool = True,
) -> dict[str, Any]:
    """Check model metrics against project blocking & recommended Quality Gates."""
    results: dict[str, Any] = {}
    blocking_failures: list[str] = []

    for model_name, metrics in evaluated_models.items():
        if model_name not in QUALITY_GATES:
            continue
        metric_key, threshold, is_blocking = QUALITY_GATES[model_name]
        actual = float(metrics.get(metric_key, 0.0))
        passed = actual >= threshold
        results[model_name] = {
            "metric": metric_key,
            "actual": actual,
            "threshold": threshold,
            "blocking": is_blocking,
            "passed": passed,
        }
        if not passed and is_blocking:
            blocking_failures.append(
                f"{model_name}: {metric_key}={actual:.4f} < blocking threshold {threshold:.4f}"
            )

    summary = {
        "all_blocking_passed": len(blocking_failures) == 0,
        "blocking_failures": blocking_failures,
        "models": results,
    }
    if blocking_failures and raise_on_blocking_failure:
        raise RuntimeError("Blocking Quality Gate(s) failed: " + "; ".join(blocking_failures))
    return summary
