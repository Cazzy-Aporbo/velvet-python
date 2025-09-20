"""
Benchmark-style tests for the mock ML pipeline.

These ensure training and prediction run within expected time limits.
"""

import time
from your_package.data_utils import load_dataset
from your_package.ml_pipeline import MockModel


def test_training_speed():
    dataset = load_dataset()
    texts = [text for text, _ in dataset]

    model = MockModel()

    start = time.perf_counter()
    model.train(texts)
    end = time.perf_counter()

    duration = end - start
    # Training should be nearly instant (< 0.1s on GitHub runners)
    assert duration < 0.1, f"Training took too long: {duration:.4f}s"


def test_prediction_speed():
    dataset = load_dataset()
    texts = [text for text, _ in dataset]

    model = MockModel()
    model.train(texts)

    start = time.perf_counter()
    for text in texts * 1000:  # repeat to simulate workload
        _ = model.predict(text)
    end = time.perf_counter()

    duration = end - start
    # Prediction loop should be very fast (< 0.2s total)
    assert duration < 0.2, f"Prediction took too long: {duration:.4f}s"