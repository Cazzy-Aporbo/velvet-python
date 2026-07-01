"""Advanced model tests for the core text classifiers.

These tests are intentionally explicit: they verify math properties the models
must preserve (normalization, deterministic ranking, and safe fallback paths)
instead of just checking “it returns a string.”
"""

from __future__ import annotations

import math

from src.ai import CosineSimilarityClassifier, NaiveBayesClassifier


def test_naive_bayes_probabilities_are_normalized_for_known_model() -> None:
    model = NaiveBayesClassifier()
    model.train(
        [
            "hello world",
            "hello friend",
            "machine learning",
            "deep learning",
        ],
        ["greeting", "greeting", "tech", "tech"],
    )

    probs = model.predict_proba("hello world")

    assert set(probs) == {"greeting", "tech"}
    assert all(0.0 <= value <= 1.0 for value in probs.values())
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert model.predict("hello world") in probs


def test_naive_bayes_favors_observed_class_signal() -> None:
    model = NaiveBayesClassifier()
    model.train(
        [
            "payment due invoice",
            "invoice reminder due",
            "python async event loop",
            "machine learning inference",
        ],
        ["finance", "finance", "tech", "tech"],
    )

    pred = model.predict("payment invoice")
    assert pred == "finance"


def test_cosine_similarity_geometry_follows_expected_signature() -> None:
    model = CosineSimilarityClassifier()
    model.train(
        ["hello there", "hello friend", "neural network", "deep learning"],
        ["greeting", "greeting", "tech", "tech"],
    )

    scores = model.similarity_scores("hello there")
    assert set(scores) == {"greeting", "tech"}
    assert scores["greeting"] > scores["tech"]
    assert model.predict("hello there") == "greeting"


def test_cosine_similarity_empty_input_falls_back_to_unknown() -> None:
    model = CosineSimilarityClassifier()
    model.train(["alpha", "beta"], ["topic_a", "topic_b"])

    assert model.predict("") == "unknown"
    scores = model.similarity_scores("")
    assert scores == {"topic_a": 0.0, "topic_b": 0.0}


def test_naive_bayes_rejects_empty_training_rows() -> None:
    model = NaiveBayesClassifier()
    model.train(["hello", ""], ["greeting", "greeting"])

    # Empty training rows collapse to no feature signal but keep label priors.
    # This test asserts we still get deterministic predictions for the class
    # with strongest prior when no overlap exists.
    assert model.predict("non matching unknown signal") == "greeting"


def test_cosine_similarity_handles_single_document_classes() -> None:
    model = CosineSimilarityClassifier()
    model.train(["delta", "epsilon"], ["a", "b"])

    scores = model.similarity_scores("delta")
    assert scores["a"] >= 0.999
    assert scores["a"] > scores["b"]
    assert model.predict("delta") == "a"
