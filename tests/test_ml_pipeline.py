import pytest

from src.ml_pipeline import TFIDFModel, WordFrequencyModel, confusion_matrix, evaluate


@pytest.fixture
def trained_freq_model():
    model = WordFrequencyModel()
    texts = ["hello world", "hello friend", "machine learning", "deep learning"]
    labels = ["greeting", "greeting", "tech", "tech"]
    model.train(texts, labels)
    return model


@pytest.fixture
def trained_tfidf_model():
    model = TFIDFModel()
    texts = ["hello world", "hello friend", "machine learning", "deep learning"]
    labels = ["greeting", "greeting", "tech", "tech"]
    model.train(texts, labels)
    return model


def test_freq_model_training(trained_freq_model):
    assert trained_freq_model.is_trained
    assert "greeting" in trained_freq_model.class_frequencies


def test_freq_prediction_known(trained_freq_model):
    assert trained_freq_model.predict("hello there") == "greeting"


def test_freq_prediction_tech(trained_freq_model):
    assert trained_freq_model.predict("learning algorithms") == "tech"


def test_tfidf_model_training(trained_tfidf_model):
    assert trained_tfidf_model.is_trained
    assert len(trained_tfidf_model.centroids) == 2


def test_tfidf_prediction(trained_tfidf_model):
    result = trained_tfidf_model.predict("hello there")
    assert isinstance(result, str)
    assert result in ("greeting", "tech", "unknown")


def test_evaluation_accuracy(trained_freq_model):
    test_data = [
        ("hello friend", "greeting"),
        ("learning models", "tech"),
    ]
    acc = evaluate(trained_freq_model, test_data)
    assert 0.0 <= acc <= 1.0


def test_evaluate_empty_data(trained_freq_model):
    assert evaluate(trained_freq_model, []) == 0.0


def test_confusion_matrix_structure(trained_freq_model):
    test_data = [("hello", "greeting"), ("learning", "tech")]
    cm = confusion_matrix(trained_freq_model, test_data)
    assert isinstance(cm, dict)
    for _actual, pred_counts in cm.items():
        assert isinstance(pred_counts, dict)


def test_model_not_trained():
    model = WordFrequencyModel()
    with pytest.raises(RuntimeError):
        model.predict("anything")


def test_tfidf_not_trained():
    model = TFIDFModel()
    with pytest.raises(RuntimeError):
        model.predict("anything")
