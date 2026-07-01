"""Canonical registry for reproducible model builders used by CLI and scripts.

Keeping this mapping in one place avoids drift between the Typer commands and
the experiment runner, which is essential for educational reliability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.ai import CosineSimilarityClassifier, NaiveBayesClassifier
from src.ml_pipeline import TFIDFModel, WordFrequencyModel

ModelBuilder = Callable[[], Any]


@dataclass(frozen=True)
class ModelMetadata:
    """Human-readable properties that explain model behavior and tradeoffs."""

    complexity: str
    supports_probability: bool
    interpretability: str
    preferred_data_scale: str
    failure_mode: str
    use_when: tuple[str, ...]


@dataclass(frozen=True)
class ModelSpec:
    """Registered model specification.

    Fields:
        key: CLI / script identifier.
        label: Human-friendly, stable display string.
        builder: Callable that creates a fresh model instance.
        metadata: Human-readable engineering profile for learning and selection.
    """

    key: str
    label: str
    builder: ModelBuilder
    metadata: ModelMetadata


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        "word_frequency",
        "Word Frequency",
        lambda: WordFrequencyModel(),
        ModelMetadata(
            complexity="low",
            supports_probability=False,
            interpretability="high",
            preferred_data_scale="small-to-medium",
            failure_mode="lexical overlap can overfit repeated tokens",
            use_when=(
                "baseline comparisons",
                "explainability-first pilots",
                "small sparse vocabularies",
            ),
        ),
    ),
    ModelSpec(
        "tfidf",
        "TF-IDF",
        lambda: TFIDFModel(),
        ModelMetadata(
            complexity="low-to-medium",
            supports_probability=False,
            interpretability="medium",
            preferred_data_scale="small-to-large",
            failure_mode="unstable vectors on very short inputs",
            use_when=(
                "feature contrast needs",
                "noisy long-tail text",
                "simple retrieval-like routing",
            ),
        ),
    ),
    ModelSpec(
        "naive_bayes",
        "Naive Bayes",
        lambda: NaiveBayesClassifier(),
        ModelMetadata(
            complexity="low-to-medium",
            supports_probability=True,
            interpretability="high",
            preferred_data_scale="small-to-large",
            failure_mode="class imbalance can inflate prior confidence",
            use_when=(
                "triage flows",
                "short text classification",
                "decision points needing probabilities",
            ),
        ),
    ),
    ModelSpec(
        "cosine",
        "Cosine Similarity",
        lambda: CosineSimilarityClassifier(),
        ModelMetadata(
            complexity="medium",
            supports_probability=False,
            interpretability="medium",
            preferred_data_scale="small-to-medium",
            failure_mode="noisy short inputs can produce weak vectors",
            use_when=(
                "topic routing",
                "semantic nearest-neighbor style prompts",
                "quick prototyping of classes",
            ),
        ),
    ),
)


def model_builders() -> dict[str, ModelBuilder]:
    """Return a deterministic mapping of model keys to constructors."""
    return {spec.key: spec.builder for spec in MODEL_REGISTRY}


def model_labels() -> dict[str, str]:
    """Return a deterministic mapping for human-readable model names."""
    return {spec.key: spec.label for spec in MODEL_REGISTRY}


def sorted_model_keys() -> list[str]:
    """Return registry keys in declaration order."""
    return [spec.key for spec in MODEL_REGISTRY]


def model_metadata() -> dict[str, dict[str, Any]]:
    """Return a deterministic mapping of metadata per model key."""
    return {
        spec.key: {
            "complexity": spec.metadata.complexity,
            "supports_probability": spec.metadata.supports_probability,
            "interpretability": spec.metadata.interpretability,
            "preferred_data_scale": spec.metadata.preferred_data_scale,
            "failure_mode": spec.metadata.failure_mode,
            "use_when": list(spec.metadata.use_when),
        }
        for spec in MODEL_REGISTRY
    }
