from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .types import CostBucket, CostPrediction


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float, bool))


@dataclass(frozen=True)
class TabularEncoder:
    numeric_keys: List[str]
    categorical_values: Dict[str, List[str]]
    encoder_name: str = "tabular_v1"

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, Any]]) -> "TabularEncoder":
        keys = sorted({key for row in rows for key in row})
        numeric_keys: List[str] = []
        categorical_values: Dict[str, List[str]] = {}
        for key in keys:
            values = [row.get(key) for row in rows]
            if all(value is None or _is_numeric(value) for value in values):
                numeric_keys.append(key)
            else:
                categorical_values[key] = sorted({str(value) for value in values})
        return cls(numeric_keys=numeric_keys, categorical_values=categorical_values)

    def transform(self, row: Mapping[str, Any]) -> Dict[str, float]:
        encoded: Dict[str, float] = {}
        for key in self.numeric_keys:
            encoded[f"num:{key}"] = float(row.get(key, 0.0) or 0.0)
        for key, values in self.categorical_values.items():
            raw_value = str(row.get(key, "<missing>"))
            for value in values:
                encoded[f"cat:{key}={value}"] = 1.0 if raw_value == value else 0.0
        return encoded

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder_name": self.encoder_name,
            "numeric_keys": list(self.numeric_keys),
            "categorical_values": {key: list(values) for key, values in self.categorical_values.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TabularEncoder":
        return cls(
            numeric_keys=[str(value) for value in payload["numeric_keys"]],
            categorical_values={str(key): [str(item) for item in values] for key, values in payload["categorical_values"].items()},
            encoder_name=str(payload.get("encoder_name", "tabular_v1")),
        )


@dataclass(frozen=True)
class LinearScoreModel:
    encoder: TabularEncoder
    weights: Dict[str, float]
    intercept: float
    model_name: str
    seed: int = 0

    def predict_score(self, features: Mapping[str, Any]) -> float:
        encoded = self.encoder.transform(features)
        return float(self.intercept + sum(self.weights.get(name, 0.0) * value for name, value in encoded.items()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder.to_dict(),
            "weights": dict(self.weights),
            "intercept": float(self.intercept),
            "model_name": self.model_name,
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinearScoreModel":
        return cls(
            encoder=TabularEncoder.from_dict(payload["encoder"]),
            weights={str(key): float(value) for key, value in payload["weights"].items()},
            intercept=float(payload["intercept"]),
            model_name=str(payload["model_name"]),
            seed=int(payload.get("seed", 0)),
        )


@dataclass(frozen=True)
class CostPolicyModel:
    bucket_models: Dict[str, LinearScoreModel]
    timeout_model: LinearScoreModel
    seed: int = 0

    def predict(self, features: Mapping[str, Any]) -> CostPrediction:
        ordered_buckets = [bucket.value for bucket in CostBucket]
        bucket = max(
            ordered_buckets,
            key=lambda name: (self.bucket_models[name].predict_score(features), -ordered_buckets.index(name)),
        )
        raw_timeout = self.timeout_model.predict_score(features)
        timeout_risk = 1.0 / (1.0 + math.exp(-raw_timeout))
        timeout_risk = round(timeout_risk, 6)
        confidence = round(max(self.bucket_models[name].predict_score(features) for name in ordered_buckets), 6)
        return CostPrediction(
            bucket=CostBucket(bucket),
            timeout_risk=timeout_risk,
            confidence=confidence,
            details={"seed": self.seed},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bucket_models": {name: model.to_dict() for name, model in self.bucket_models.items()},
            "timeout_model": self.timeout_model.to_dict(),
            "seed": int(self.seed),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CostPolicyModel":
        return cls(
            bucket_models={str(name): LinearScoreModel.from_dict(model) for name, model in payload["bucket_models"].items()},
            timeout_model=LinearScoreModel.from_dict(payload["timeout_model"]),
            seed=int(payload.get("seed", 0)),
        )


def _train_linear_model(
    rows: Sequence[Mapping[str, Any]],
    target_values: Sequence[float],
    model_name: str,
    seed: int,
) -> LinearScoreModel:
    encoder = TabularEncoder.fit(rows)
    encoded_rows = [encoder.transform(row) for row in rows]
    feature_names = sorted({name for row in encoded_rows for name in row})
    mean_target = sum(target_values) / float(len(target_values) or 1)
    weights: Dict[str, float] = {}
    mean_features: Dict[str, float] = {}
    for name in feature_names:
        values = [row.get(name, 0.0) for row in encoded_rows]
        mean_value = sum(values) / float(len(values) or 1)
        mean_features[name] = mean_value
        variance = sum((value - mean_value) ** 2 for value in values)
        if variance == 0.0:
            weights[name] = 0.0
            continue
        covariance = sum((value - mean_value) * (target - mean_target) for value, target in zip(values, target_values))
        weights[name] = covariance / variance
    intercept = mean_target - sum(weights[name] * mean_features[name] for name in feature_names)
    return LinearScoreModel(
        encoder=encoder,
        weights=weights,
        intercept=intercept,
        model_name=model_name,
        seed=seed,
    )


def train_region_policy(rows: Sequence[Mapping[str, Any]], seed: int = 0) -> LinearScoreModel:
    feature_rows = [row["region_features"] for row in rows]
    targets = [float(row["region_target"]) for row in rows]
    return _train_linear_model(feature_rows, targets, model_name="region_policy", seed=seed)


def train_instance_policy(rows: Sequence[Mapping[str, Any]], seed: int = 0) -> LinearScoreModel:
    feature_rows = [row["instance_action_features"] for row in rows]
    targets = [float(row["instance_target"]) for row in rows]
    return _train_linear_model(feature_rows, targets, model_name="instance_policy", seed=seed)


def train_cost_policy(rows: Sequence[Mapping[str, Any]], seed: int = 0) -> CostPolicyModel:
    feature_rows = [row["instance_action_features"] for row in rows]
    bucket_models: Dict[str, LinearScoreModel] = {}
    for bucket in CostBucket:
        targets = [1.0 if row["cost_bucket"] == bucket.value else 0.0 for row in rows]
        bucket_models[bucket.value] = _train_linear_model(feature_rows, targets, model_name=f"cost_policy:{bucket.value}", seed=seed)
    timeout_targets = [float(row["timeout_target"]) for row in rows]
    timeout_model = _train_linear_model(feature_rows, timeout_targets, model_name="cost_policy:timeout", seed=seed)
    return CostPolicyModel(bucket_models=bucket_models, timeout_model=timeout_model, seed=seed)
