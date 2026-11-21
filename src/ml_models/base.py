"""Base interface for all Tabular ML trainers enforcing FeatureSchemaRegistry contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.utils.feature_schema import FeatureSchemaRegistry, ModelFeatureSpec


class BaseTabularTrainer(ABC):
    """Contract-enforcing base class for LightGBM, XGBoost, CausalML, and Lifelines trainers."""

    def __init__(
        self,
        model_name: str,
        schema_path: str | Path = "configs/feature_schema.yaml",
    ) -> None:
        self.model_name = model_name
        self.registry = FeatureSchemaRegistry.from_yaml(schema_path)
        self.spec: ModelFeatureSpec = self.registry.get(model_name)

    def validate_training_data(self, records: list[dict[str, Any]]) -> None:
        errors = self.spec.validate_records(records, require_target=True)
        if errors:
            raise ValueError(
                f"[{self.model_name}] Training data failed schema contract: {errors[:5]}"
            )

    @abstractmethod
    def fit(self, train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, float]:
        """Fit model and return evaluation metrics dictionary."""

    @abstractmethod
    def save_artifact(self, output_dir: str | Path) -> Path:
        """Persist trained model artifact for MLflow / Repo 2 consumption."""
