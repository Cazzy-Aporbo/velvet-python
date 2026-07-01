"""
Text classification — three strategies, one interface.

Strategy 1: Rule-based (pattern matching on known tokens).
Strategy 2: Naive Bayes (Bayes' theorem applied to word frequencies).
Strategy 3: Cosine similarity against learned centroids.

All three are deterministic given the same training data, which makes
them testable without randomness. The mathematical foundations are
documented inline so every step can be followed with pen and paper.

    P(class | text) = P(text | class) * P(class) / P(text)   [Bayes]
    sim(a, b) = (a · b) / (||a|| * ||b||)                    [Cosine]
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict


def classify_text(text: str) -> str:
    """Classify text by scanning for known token patterns.

    This is the simplest possible classifier: no training, no state,
    just ordered rules evaluated top-to-bottom. Fast, transparent,
    and useful as a baseline to measure learned classifiers against.

    Returns one of: 'empty', 'greeting', 'number', 'other'.
    """
    if not text:
        return "empty"
    if "hello" in text.lower():
        return "greeting"
    if text.isdigit():
        return "number"
    return "other"



class NaiveBayesClassifier:
    """Multinomial Naive Bayes built on word-frequency counts.

    Training phase:
        For each class c, count how often each word w appears.
        Store log-probabilities to avoid floating-point underflow
        on long documents:

            log P(w | c) = log( count(w, c) + alpha )
                         - log( total_words_in_c + alpha * |V| )

        where alpha is Laplace smoothing (default 1.0) and |V| is
        the vocabulary size.

    Prediction phase:
        For a new document, sum log P(w | c) for every word w,
        add log P(c), and pick the class with the highest score.

    This is the same math behind spam filters, sentiment analysis,
    and document routing — implemented in ~40 lines of Python.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.class_word_counts: dict[str, Counter] = defaultdict(Counter)
        self.class_totals: dict[str, int] = defaultdict(int)
        self.class_doc_counts: dict[str, int] = defaultdict(int)
        self.vocab: set = set()
        self.n_docs: int = 0

    def train(self, texts: list[str], labels: list[str]) -> None:
        """Learn word distributions per class from labeled examples."""
        for text, label in zip(texts, labels, strict=False):
            words = text.lower().split()
            self.class_word_counts[label].update(words)
            self.class_totals[label] += len(words)
            self.class_doc_counts[label] += 1
            self.vocab.update(words)
            self.n_docs += 1

    def predict(self, text: str) -> str:
        """Return the most probable class for a document.

        Computes log P(c | text) ∝ log P(c) + Σ log P(w | c)
        and returns argmax over all known classes.
        """
        words = text.lower().split()
        best_class, best_score = "", float("-inf")
        v = len(self.vocab)

        for cls in self.class_doc_counts:
            # Prior: log P(c) = log(docs_in_c / total_docs)
            score = math.log(self.class_doc_counts[cls] / self.n_docs)

            # Likelihood: log P(w | c) with Laplace smoothing
            total = self.class_totals[cls]
            counts = self.class_word_counts[cls]
            for w in words:
                score += math.log((counts[w] + self.alpha) /
                                  (total + self.alpha * v))

            if score > best_score:
                best_score = score
                best_class = cls

        return best_class

    def predict_proba(self, text: str) -> dict[str, float]:
        """Return normalized probabilities for all classes.

        Converts log-scores to probabilities via the log-sum-exp trick
        to maintain numerical stability:

            P(c) = exp(score_c - max_score) / Σ exp(score_i - max_score)
        """
        words = text.lower().split()
        scores: dict[str, float] = {}
        v = len(self.vocab)

        for cls in self.class_doc_counts:
            score = math.log(self.class_doc_counts[cls] / self.n_docs)
            total = self.class_totals[cls]
            counts = self.class_word_counts[cls]
            for w in words:
                score += math.log((counts[w] + self.alpha) /
                                  (total + self.alpha * v))
            scores[cls] = score

        # Log-sum-exp normalization
        max_score = max(scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in scores.items()}
        total_exp = sum(exp_scores.values())
        return {c: v / total_exp for c, v in exp_scores.items()}



class CosineSimilarityClassifier:
    """Classify by measuring angular distance to class centroids.

    Training: build a centroid vector for each class by averaging
    the term-frequency vectors of all documents in that class.

    Prediction: compute cosine similarity between the input document
    and every centroid, return the closest class.

        cos(θ) = (A · B) / (||A|| · ||B||)

    When cos(θ) = 1, the vectors point in the same direction.
    When cos(θ) = 0, they are orthogonal (completely unrelated).
    """

    def __init__(self) -> None:
        self.centroids: dict[str, dict[str, float]] = {}

    def _vectorize(self, text: str) -> dict[str, float]:
        """Convert text to a normalized term-frequency vector."""
        words = text.lower().split()
        counts = Counter(words)
        total = len(words) or 1
        return {w: c / total for w, c in counts.items()}

    @staticmethod
    def _dot(a: dict[str, float], b: dict[str, float]) -> float:
        """Sparse dot product — only iterate over shared keys."""
        return sum(a[k] * b[k] for k in a if k in b)

    @staticmethod
    def _norm(v: dict[str, float]) -> float:
        """L2 norm of a sparse vector: sqrt(Σ v_i²)."""
        return math.sqrt(sum(x * x for x in v.values()))

    def train(self, texts: list[str], labels: list[str]) -> None:
        """Build centroid vectors by averaging per-class documents."""
        groups: dict[str, list[dict[str, float]]] = defaultdict(list)
        for text, label in zip(texts, labels, strict=False):
            groups[label].append(self._vectorize(text))

        for label, vectors in groups.items():
            merged: dict[str, float] = defaultdict(float)
            for v in vectors:
                for k, val in v.items():
                    merged[k] += val
            n = len(vectors)
            self.centroids[label] = {k: v / n for k, v in merged.items()}

    def predict(self, text: str) -> str:
        """Return the class whose centroid is most similar to the input."""
        vec = self._vectorize(text)
        vec_norm = self._norm(vec)
        if vec_norm == 0:
            return "unknown"

        best_class, best_sim = "unknown", -1.0
        for label, centroid in self.centroids.items():
            c_norm = self._norm(centroid)
            if c_norm == 0:
                continue
            sim = self._dot(vec, centroid) / (vec_norm * c_norm)
            if sim > best_sim:
                best_sim = sim
                best_class = label
        return best_class

    def similarity_scores(self, text: str) -> dict[str, float]:
        """Return cosine similarity to every class centroid."""
        vec = self._vectorize(text)
        vec_norm = self._norm(vec)
        if vec_norm == 0:
            return dict.fromkeys(self.centroids, 0.0)

        return {
            label: self._dot(vec, cent) / (vec_norm * self._norm(cent))
            for label, cent in self.centroids.items()
            if self._norm(cent) > 0
        }



def generate_number(seed: int | None = None) -> int:
    """Return a pseudo-random digit 0-9.

    Seeding makes this fully deterministic — useful for verifying
    that stochastic pipelines produce stable outputs when the
    random state is controlled.
    """
    rng = random.Random(seed)
    return rng.randint(0, 9)
