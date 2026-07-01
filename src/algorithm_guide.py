"""Algorithm selection guidance used by the Velvet Python learning track.

The goal is not to prescribe a single "best" model for every project.
The goal is to make tradeoffs explicit so engineers can make the right
first decision in a few seconds, then harden it later with experiments.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class AlgorithmProfile:
    """Portable profile for a beginner-to-production text classification model."""

    name: str
    model_package: str
    strengths: tuple[str, ...]
    tradeoffs: tuple[str, ...]
    collisions: tuple[str, ...]
    best_for: tuple[str, ...]
    avoid_if: tuple[str, ...]


PROFILES: tuple[AlgorithmProfile, ...] = (
    AlgorithmProfile(
        name="rule_based",
        model_package="src.ai.classify_text",
        strengths=(
            "Fast, deterministic, easiest to explain to non-technical readers",
            "No dependencies and no training state",
        ),
        tradeoffs=(
            "Limited precision, not meant to generalize beyond explicit patterns",
            "No uncertainty score by default",
        ),
        collisions=(
            "Keyword drift: rules become stale as data language changes",
            "Can overfit to legacy vocabulary and miss edge cases",
        ),
        best_for=(
            "quick smoke check",
            "baseline guard rail",
            "first-pass triage demos",
        ),
        avoid_if=(
            "need probabilistic output",
            "vocabulary is changing weekly",
            "classes are implicit and unlisted",
        ),
    ),
    AlgorithmProfile(
        name="word_frequency",
        model_package="src.ml_pipeline.WordFrequencyModel",
        strengths=(
            "Simple supervised baseline",
            "Highly interpretable token-overlap decisions",
            "No external ML dependencies",
        ),
        tradeoffs=(
            "Sensitive to repeated tokens",
            "Weak with long-form nuance and phrase-level context",
        ),
        collisions=(
            "When labels have uneven data volumes, the majority class dominates token counts",
            "Rare edge cases may be silently underrepresented",
        ),
        best_for=(
            "new teams learning train/infer loops",
            "small-medium datasets",
            "first version of an internal governance model",
        ),
        avoid_if=(
            "strictly low-latency and zero false positive budget",
            "heavy synonym/phrase ambiguity",
        ),
    ),
    AlgorithmProfile(
        name="tfidf",
        model_package="src.ml_pipeline.TFIDFModel",
        strengths=(
            "Down-weights common words automatically",
            "Performs better than raw frequency on noisy text",
            "Still transparent: sparse vector math is inspectable",
        ),
        tradeoffs=(
            "More code and failure modes than word-frequency",
            "Harder to hand-explain than simple word counts",
        ),
        collisions=(
            "Duplicate punctuation can shift token boundaries",
            "Short inputs can produce unstable cosine scores",
        ),
        best_for=(
            "feature-based comparisons",
            "short-form text signals",
            "first upgrade from word_frequency",
        ),
        avoid_if=(
            "extremely tiny datasets with many tied vectors",
            "strictly fixed latency at p95<50ms",
        ),
    ),
    AlgorithmProfile(
        name="naive_bayes",
        model_package="src.ai.NaiveBayesClassifier",
        strengths=(
            "Fast baseline with probability outputs",
            "Good default for short text and discrete classes",
            "Numerically stable with explicit smoothing",
        ),
        tradeoffs=(
            "Strong conditional-independence assumption",
            "Can be overconfident on unseen jargon",
        ),
        collisions=(
            "Highly imbalanced labels inflate priors",
            "Very short labels with long documents weaken interpretability",
        ),
        best_for=(
            "spam-like routing",
            "triage workflows",
            "early production pilots",
        ),
        avoid_if=(
            "feature dependence is known to be high",
            "you need calibrated confidence intervals",
        ),
    ),
    AlgorithmProfile(
        name="cosine",
        model_package="src.ai.CosineSimilarityClassifier",
        strengths=(
            "Stable geometric intuition",
            "Good when semantic neighborhoods are meaningful",
            "Fast enough for small to medium class sets",
        ),
        tradeoffs=(
            "Needs good preprocessing for token consistency",
            "No calibrated probabilities in the default path",
        ),
        collisions=(
            "Very short inputs can map to near-zero vectors",
            "Class centroids are sensitive to data drift",
        ),
        best_for=(
            "topic-routing experiments",
            "pilot recommendation classifiers",
            "teaching vector similarity",
        ),
        avoid_if=(
            "strict explainability is required for every prediction",
            "tokenization changes weekly",
        ),
    ),
)


def recommend_text_algorithms(
    row_count: int,
    class_count: int,
    *,
    needs_probabilities: bool = False,
    needs_explainability: bool = True,
    p95_latency_ms: int | None = None,
) -> list[AlgorithmProfile]:
    """Return an ordered list of profiles for text-classification use cases.

    This scoring model is intentionally explicit and simple:
    - small datasets should start with simpler methods,
    - probability requirements add Naive Bayes first,
    - low-latency defaults to rule-based and word-frequency,
    - explainability needs favor frequency/naive bayes over geometric similarity.
    """
    scores: dict[str, int] = {profile.name: 0 for profile in PROFILES}

    for profile in PROFILES:
        if row_count < 300 and profile.name in {"rule_based", "word_frequency", "tfidf", "naive_bayes"}:
            scores[profile.name] += 1
        if class_count > 6 and profile.name in {"naive_bayes", "cosine", "tfidf"}:
            scores[profile.name] += 1
        if needs_probabilities and profile.name == "naive_bayes":
            scores[profile.name] += 3
        if needs_explainability and profile.name in {"rule_based", "word_frequency", "naive_bayes"}:
            scores[profile.name] += 2
        if p95_latency_ms is not None and p95_latency_ms <= 100 and profile.name in {"rule_based", "word_frequency"}:
            scores[profile.name] += 2
        if p95_latency_ms is not None and p95_latency_ms > 200 and profile.name in {"tfidf", "cosine", "naive_bayes"}:
            scores[profile.name] += 1

    ordered = sorted(
        PROFILES,
        key=lambda profile: (-scores[profile.name], profile.name),
    )
    return ordered


def describe_profiles(names: Iterable[str] | None = None) -> list[AlgorithmProfile]:
    """Filter known profiles by name.

    If names is empty, all profiles are returned.
    """
    if names is None:
        return list(PROFILES)

    requested = {name.lower() for name in names}
    return [profile for profile in PROFILES if profile.name.lower() in requested]
