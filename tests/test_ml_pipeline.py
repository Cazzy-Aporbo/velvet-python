"""
Tests for the mock ML pipeline.
"""

import pytest
from your_package.ml_pipeline import MockModel, evaluate


@pytest.fixture
def trained_model():
    """
    Fixture that trains a MockModel on sample data.
    """
    model = MockModel()
    training_texts = ["hello world", "hello ai", "machine learning"]
    model.train(training_texts)
    return model


def test_model_training(trained_model):
    assert trained_model.is_trained
    assert "hello" in trained_model.word_counts
    assert trained_model.word_counts["hello"] >= 2


def test_prediction_known(trained_model):
    # The word "hello" was in training data
    assert trained_model.predict("say hello") == "hello"


def test_prediction_unknown(trained_model):
    # The word "goodbye" was not in training data
    assert trained_model.predict("goodbye now") == "unknown"


def test_evaluation_accuracy(trained_model):
    test_data = [
        ("hello ai", "hello"),       # should be correct
        ("world says hello", "hello"),
        ("totally unseen words", "unknown"),
    ]
    acc = evaluate(trained_model, test_data)
    # Accuracy must be between 0 and 1
    assert 0.0 <= acc <= 1.0
    # This dataset ensures accuracy > 0
    assert acc > 0


def test_model_not_trained():
    model = MockModel()
    with pytest.raises(RuntimeError):
        model.predict("anything")