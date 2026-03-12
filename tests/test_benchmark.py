import time
from src.data_utils import load_dataset
from src.ml_pipeline import WordFrequencyModel


def test_training_speed():
    data = load_dataset()
    texts = [t for t, _ in data]
    labels = [l for _, l in data]
    model = WordFrequencyModel()
    start = time.perf_counter()
    model.train(texts, labels)
    assert time.perf_counter() - start < 0.1


def test_prediction_speed():
    data = load_dataset()
    texts = [t for t, _ in data]
    labels = [l for _, l in data]
    model = WordFrequencyModel()
    model.train(texts, labels)
    start = time.perf_counter()
    for text in texts * 1000:
        model.predict(text)
    assert time.perf_counter() - start < 0.5