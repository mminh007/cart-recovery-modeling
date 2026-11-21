"""Unified MLflow tracking & Model Registry configuration (Local, Docker, Colab, Cloud)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_EXPERIMENTS: dict[str, str] = {
    "intent_classifier": "cart-abandon/ml/intent-classifier",
    "churn_predictor": "cart-abandon/ml/churn-predictor",
    "uplift_model": "cart-abandon/ml/uplift-model",
    "timing_optimizer": "cart-abandon/ml/timing-optimizer",
    "channel_selector": "cart-abandon/ml/channel-selector",
    "llm_sft": "cart-abandon/llm/sft-finetune",
    "llm_dpo": "cart-abandon/llm/dpo-alignment",
}


def get_tracking_uri() -> str:
    """Return configured MLflow tracking URI, defaulting to local ./mlruns directory."""
    env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if env_uri:
        return env_uri
    local_dir = Path("mlruns").resolve()
    return local_dir.as_uri()


def initialize_experiments(tracking_uri: str | None = None) -> dict[str, str]:
    """Ensure all 7 project experiments exist in MLflow (or local manifest fallback)."""
    resolved_uri = tracking_uri or get_tracking_uri()
    created: dict[str, str] = {}

    try:
        import mlflow  # type: ignore[import-untyped]

        mlflow.set_tracking_uri(resolved_uri)
        for key, exp_name in DEFAULT_EXPERIMENTS.items():
            exp = mlflow.get_experiment_by_name(exp_name)
            exp_id = exp.experiment_id if exp else mlflow.create_experiment(exp_name)
            created[key] = str(exp_id)
        return created
    except ImportError:
        if resolved_uri.startswith("file:///"):
            manifest_dir = Path(resolved_uri.replace("file:///", ""))
        else:
            manifest_dir = Path(os.getenv("MLFLOW_LOCAL_DIR", "mlruns"))
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "experiments_manifest.json"
        for idx, (key, exp_name) in enumerate(DEFAULT_EXPERIMENTS.items(), start=1):
            created[key] = f"local-exp-{idx:02d}"
        manifest_payload: dict[str, Any] = {
            "tracking_uri": resolved_uri,
            "initialized_at": datetime.now(timezone.utc).isoformat(),
            "mode": "local_manifest_fallback",
            "experiments": {
                key: {"name": DEFAULT_EXPERIMENTS[key], "id": created[key]} for key in created
            },
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
        return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize MLflow experiments for Cart Abandonment AI")
    parser.add_argument("--init", action="store_true", help="Create all standard MLflow experiments")
    parser.add_argument("--tracking-uri", type=str, default=None, help="Override MLFLOW_TRACKING_URI")
    args = parser.parse_args()

    if args.init:
        exps = initialize_experiments(args.tracking_uri)
        print(f"Initialized {len(exps)} MLflow experiments (URI: {args.tracking_uri or get_tracking_uri()}):")
        for key, exp_id in exps.items():
            print(f"  - {DEFAULT_EXPERIMENTS[key]} (id={exp_id})")


if __name__ == "__main__":
    main()
