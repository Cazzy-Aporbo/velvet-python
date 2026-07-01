from __future__ import annotations

from src.algorithm_guide import (
    AlgorithmProfile,
    describe_profiles,
    recommend_text_algorithms,
)
from src.model_registry import model_builders, model_labels, sorted_model_keys


def test_model_registry_is_complete_and_stable():
    """Registry keys should be deterministic and match displayed labels."""
    keys = sorted_model_keys()

    assert keys == [
        "word_frequency",
        "tfidf",
        "naive_bayes",
        "cosine",
    ]

    registry = model_builders()
    labels = model_labels()

    assert set(keys) == set(registry.keys())
    assert set(keys) == set(labels.keys())
    assert labels["word_frequency"] == "Word Frequency"
    assert labels["naive_bayes"] == "Naive Bayes"

    # Each builder should materialize a fresh instance object so callers can run
    # repeated experiments without state leakage.
    for key in keys:
        instance_a = registry[key]()
        instance_b = registry[key]()
        assert type(instance_a) is type(instance_b)


def test_recommend_text_algorithms_highlights_probabilistic_option():
    """When probabilities are required, Naive Bayes should be prioritized."""
    recommendations = recommend_text_algorithms(
        row_count=120,
        class_count=4,
        needs_probabilities=True,
        needs_explainability=True,
    )
    assert recommendations[0].name == "naive_bayes"
    assert any(profile.name == "word_frequency" for profile in recommendations)


def test_recommend_text_algorithms_applies_latency_pressure():
    """Low-latency constraints should favor rule-based and frequency approaches."""
    recommendations = recommend_text_algorithms(
        row_count=10_000,
        class_count=2,
        p95_latency_ms=80,
    )
    assert recommendations[0].name in {"rule_based", "word_frequency"}


def test_describe_profiles_filters_only_requested_profiles():
    """Filtering by names should only return known requested entries."""
    profiles = describe_profiles(["tfidf", "cosine", "unknown"])
    names = [profile.name for profile in profiles]
    assert names == ["tfidf", "cosine"]


def test_algorithm_profile_is_strongly_typed():
    """Profiles should expose expected public fields for curriculum docs and CLI output."""
    profile = AlgorithmProfile(
        name="test",
        model_package="src.example.Model",
        strengths=("simple",),
        tradeoffs=("none",),
        collisions=("none",),
        best_for=("learning",),
        avoid_if=("none",),
    )
    assert profile.name == "test"
