"""
Mock ML pipeline for demonstration and testing.

This simulates training, prediction, and evaluation without
any heavy ML libraries. Everything is deterministic.
"""

from __future__ import annotations
from collections import Counter
from typing import List


class MockModel:
    """
    A fake ML model that "learns" by counting words.
    """

    def __init__(self):
        self.word_counts: dict[str, int] = {}
        self.is_trained: bool = False

    def train(self, texts: List[str]) -> None:
        """
        Train the model by counting words in training texts.
        """
        counts = Counter()
        for text in texts:
            counts.update(text.lower().split())
        self.word_counts = dict(counts)
        self.is_trained = True

    def predict(self, text: str) -> str:
        """
        Fake prediction: return the most common word seen during training
        if text contains it, otherwise return 'unknown'.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet.")

        tokens = text.lower().split()
        for token in tokens:
            if token in self.word_counts:
                return token
        return "unknown"


def evaluate(model: MockModel, data: List[tuple[str, str]]) -> float:
    """
    Evaluate the model on labeled data.
    Returns accuracy (correct / total).
    """
    correct = 0
    for text, label in data:
        if model.predict(text) == label:
            correct += 1
    return correct / len(data) if data else 0.0