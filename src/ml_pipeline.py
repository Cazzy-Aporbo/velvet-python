"""
ML pipeline — train, predict, evaluate, and compare.

This module implements a complete machine learning pipeline using only
the Python standard library. No NumPy, no scikit-learn — just the
algorithms themselves, so every step is visible and auditable.

The pipeline supports two model types:
    1. WordFrequencyModel  — classify by most-frequent token overlap
    2. TFIDFModel          — classify using TF-IDF weighted vectors

Both follow the same interface: train(texts, labels) → predict(text).

Key concepts demonstrated:
    - Term Frequency:   TF(t, d) = count(t in d) / |d|
    - Inverse Doc Freq: IDF(t) = log(N / df(t))
    - TF-IDF:           weight(t, d) = TF(t, d) * IDF(t)
    - Accuracy:         correct_predictions / total_predictions
    - Confusion matrix construction from raw predictions
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Tuple



class WordFrequencyModel:
    """Classify text by matching tokens against learned frequency tables.

    Training builds a frequency table per class. Prediction scores each
    class by summing the frequencies of overlapping tokens, then returns
    the class with the highest cumulative score.

    This is intentionally simple — it serves as the baseline that more
    sophisticated models (TF-IDF, Naive Bayes) are measured against.
    """

    def __init__(self) -> None:
        self.class_frequencies: Dict[str, Counter] = {}
        self.is_trained: bool = False

    def train(self, texts: List[str], labels: List[str]) -> None:
        """Build per-class word frequency tables from labeled data."""
        groups: Dict[str, List[str]] = defaultdict(list)
        for text, label in zip(texts, labels):
            groups[label].append(text)

        for label, group_texts in groups.items():
            counts = Counter()
            for t in group_texts:
                counts.update(t.lower().split())
            self.class_frequencies[label] = counts

        self.is_trained = True

    def predict(self, text: str) -> str:
        """Return the class with the highest token-overlap score."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")

        tokens = text.lower().split()
        best_class, best_score = "unknown", 0

        for label, freq in self.class_frequencies.items():
            score = sum(freq.get(t, 0) for t in tokens)
            if score > best_score:
                best_score = score
                best_class = label

        return best_class



class TFIDFModel:
    """Classify text using TF-IDF weighted cosine similarity.

    TF-IDF captures two ideas at once:
        - Words that appear often in a document are important to it (TF).
        - Words that appear in every document are not discriminative (IDF).

    The product TF * IDF gives high weight to words that are frequent
    in a specific document but rare across the corpus — exactly the
    tokens that distinguish one class from another.

    Training computes a centroid TF-IDF vector per class.
    Prediction finds the centroid closest to the input (cosine similarity).
    """

    def __init__(self) -> None:
        self.centroids: Dict[str, Dict[str, float]] = {}
        self.idf: Dict[str, float] = {}
        self.is_trained: bool = False

    def _compute_tf(self, text: str) -> Dict[str, float]:
        """Term frequency: count(word) / total_words_in_document."""
        words = text.lower().split()
        counts = Counter(words)
        n = len(words) or 1
        return {w: c / n for w, c in counts.items()}

    def _compute_idf(self, documents: List[str]) -> Dict[str, float]:
        """Inverse document frequency: log(N / docs_containing_term).

        Measures how rare a term is across the entire corpus.
        Rare terms get high IDF; ubiquitous terms get low IDF.
        """
        n = len(documents)
        df: Counter = Counter()
        for doc in documents:
            unique_words = set(doc.lower().split())
            df.update(unique_words)
        return {w: math.log(n / count) for w, count in df.items() if count > 0}

    def train(self, texts: List[str], labels: List[str]) -> None:
        """Compute IDF over the corpus, then build class centroids."""
        self.idf = self._compute_idf(texts)

        # Group documents by class and compute TF-IDF vectors
        groups: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        for text, label in zip(texts, labels):
            tf = self._compute_tf(text)
            tfidf = {w: tf_val * self.idf.get(w, 0) for w, tf_val in tf.items()}
            groups[label].append(tfidf)

        # Average vectors per class to form centroids
        for label, vectors in groups.items():
            merged: Dict[str, float] = defaultdict(float)
            for v in vectors:
                for k, val in v.items():
                    merged[k] += val
            n = len(vectors)
            self.centroids[label] = {k: val / n for k, val in merged.items()}

        self.is_trained = True

    def predict(self, text: str) -> str:
        """Return the class whose TF-IDF centroid is closest (cosine)."""
        if not self.is_trained:
            raise RuntimeError("Model has not been trained.")

        tf = self._compute_tf(text)
        vec = {w: tf_val * self.idf.get(w, 0) for w, tf_val in tf.items()}
        vec_norm = math.sqrt(sum(v * v for v in vec.values()))
        if vec_norm == 0:
            return "unknown"

        best_class, best_sim = "unknown", -1.0
        for label, centroid in self.centroids.items():
            dot = sum(vec.get(k, 0) * v for k, v in centroid.items())
            c_norm = math.sqrt(sum(v * v for v in centroid.values()))
            if c_norm == 0:
                continue
            sim = dot / (vec_norm * c_norm)
            if sim > best_sim:
                best_sim = sim
                best_class = label

        return best_class



def evaluate(model, data: List[Tuple[str, str]]) -> float:
    """Compute classification accuracy: correct / total.

    Works with any model that implements predict(text) -> str.
    Returns 0.0 on empty data to avoid division by zero.
    """
    if not data:
        return 0.0
    correct = sum(1 for text, label in data if model.predict(text) == label)
    return correct / len(data)


def confusion_matrix(
    model, data: List[Tuple[str, str]]
) -> Dict[str, Dict[str, int]]:
    """Build a confusion matrix: actual → predicted → count.

    Reading the matrix:
        matrix["spam"]["ham"] = 3  means 3 spam documents were
        misclassified as ham (false negatives for spam).

    The diagonal (matrix[c][c]) contains correct predictions.
    Off-diagonal entries reveal systematic misclassification patterns.
    """
    matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for text, actual in data:
        predicted = model.predict(text)
        matrix[actual][predicted] += 1
    return {k: dict(v) for k, v in matrix.items()}