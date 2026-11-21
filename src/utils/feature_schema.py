"""Shared Feature Schema validation contract between Repo 1 (Training) and Repo 2 (Serving).

Uses Pydantic v2 when installed (Docker / Colab / Repo 2) and seamless dataclass validation
in lightweight environments so schema checks never break.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.config_loader import load_yaml_config


@dataclass(frozen=True)
class FeatureFieldSpec:
    """Specification for a single feature or target column."""

    name: str
    dtype: str
    group: str = "default"
    min_value: float | None = None
    max_value: float | None = None

    def validate_value(self, value: Any, row_idx: int = 0) -> list[str]:
        errors: list[str] = []
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return [f"Row {row_idx}: '{self.name}' is null/NaN"]

        if self.dtype == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                errors.append(
                    f"Row {row_idx}: '{self.name}' expected int, got {type(value).__name__} ({value!r})"
                )
                return errors
            numeric_val = float(value)
        elif self.dtype == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(
                    f"Row {row_idx}: '{self.name}' expected float, got {type(value).__name__} ({value!r})"
                )
                return errors
            numeric_val = float(value)
        elif self.dtype == "str":
            if not isinstance(value, str) or not value.strip():
                errors.append(f"Row {row_idx}: '{self.name}' expected non-empty str, got {value!r}")
            return errors
        else:
            return errors

        if self.min_value is not None and numeric_val < self.min_value:
            errors.append(
                f"Row {row_idx}: '{self.name}'={numeric_val} below min_value={self.min_value}"
            )
        if self.max_value is not None and numeric_val > self.max_value:
            errors.append(
                f"Row {row_idx}: '{self.name}'={numeric_val} above max_value={self.max_value}"
            )
        return errors


@dataclass(frozen=True)
class ModelFeatureSpec:
    """Feature and target schema for one ML model."""

    model_name: str
    model_type: str
    entity_keys: list[str]
    target: FeatureFieldSpec
    features: list[FeatureFieldSpec] = field(default_factory=list)
    timestamp_col: str | None = None
    treatment_col: str | None = None
    duration_col: str | None = None

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    @property
    def required_training_columns(self) -> list[str]:
        cols = list(self.entity_keys)
        if self.treatment_col:
            cols.append(self.treatment_col)
        if self.duration_col:
            cols.append(self.duration_col)
        cols.extend(self.feature_names)
        cols.append(self.target.name)
        return cols

    def validate_records(
        self,
        records: list[dict[str, Any]],
        require_target: bool = True,
        max_errors: int = 20,
    ) -> list[str]:
        """Validate a list of feature dictionaries (training rows or online inference payloads)."""
        if not records:
            return [f"[{self.model_name}] Empty dataset provided for validation"]

        errors: list[str] = []
        for idx, row in enumerate(records):
            for feat in self.features:
                if feat.name not in row:
                    errors.append(f"Row {idx}: missing feature '{feat.name}'")
                else:
                    errors.extend(feat.validate_value(row[feat.name], row_idx=idx))

            if require_target:
                if self.target.name not in row:
                    errors.append(f"Row {idx}: missing target '{self.target.name}'")
                else:
                    errors.extend(self.target.validate_value(row[self.target.name], row_idx=idx))

            if len(errors) >= max_errors:
                return errors[:max_errors]
        return errors


class FeatureSchemaRegistry:
    """Registry loaded from configs/feature_schema.yaml."""

    def __init__(self, schema_version: str, models: dict[str, ModelFeatureSpec]) -> None:
        self.schema_version = schema_version
        self.models = models

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/feature_schema.yaml") -> FeatureSchemaRegistry:
        raw = load_yaml_config(path)
        version = str(raw.get("schema_version", "1.0.0"))
        models_raw = raw.get("models", {})
        parsed_models: dict[str, ModelFeatureSpec] = {}

        for model_name, spec in models_raw.items():
            target_raw = spec["target"]
            target_spec = FeatureFieldSpec(
                name=target_raw["name"],
                dtype=target_raw["dtype"],
                group="target",
                min_value=target_raw.get("min_value"),
                max_value=target_raw.get("max_value"),
            )
            features = [
                FeatureFieldSpec(
                    name=f["name"],
                    dtype=f["dtype"],
                    group=f.get("group", "default"),
                    min_value=f.get("min_value"),
                    max_value=f.get("max_value"),
                )
                for f in spec.get("features", [])
            ]
            parsed_models[model_name] = ModelFeatureSpec(
                model_name=model_name,
                model_type=spec.get("model_type", "unknown"),
                entity_keys=list(spec.get("entity_keys", [])),
                target=target_spec,
                features=features,
                timestamp_col=spec.get("timestamp_col"),
                treatment_col=spec.get("treatment_col"),
                duration_col=spec.get("duration_col"),
            )

        return cls(schema_version=version, models=parsed_models)

    def get(self, model_name: str) -> ModelFeatureSpec:
        if model_name not in self.models:
            available = ", ".join(sorted(self.models))
            raise KeyError(f"Unknown model '{model_name}'. Available: {available}")
        return self.models[model_name]
