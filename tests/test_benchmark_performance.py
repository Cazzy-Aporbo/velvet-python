import time

import pytest

from src.data_utils import load_dataset
from src.ml_pipeline import TFIDFModel, WordFrequencyModel


@pytest.mark.slow
def test_freq_training_speed():
    data = load_dataset()
    texts, labels = [t for t, _ in data], [label for _, label in data]
    model = WordFrequencyModel()
    start = time.perf_counter()
    model.train(texts, labels)
    assert time.perf_counter() - start < 0.1


@pytest.mark.slow
def test_freq_prediction_throughput():
    data = load_dataset()
    texts, labels = [t for t, _ in data], [label for _, label in data]
    model = WordFrequencyModel()
    model.train(texts, labels)
    start = time.perf_counter()
    for text in texts * 1000:
        model.predict(text)
    assert time.perf_counter() - start < 0.5


@pytest.mark.slow
def test_tfidf_training_speed():
    data = load_dataset()
    texts, labels = [t for t, _ in data], [label for _, label in data]
    model = TFIDFModel()
    start = time.perf_counter()
    model.train(texts, labels)
    assert time.perf_counter() - start < 0.1
