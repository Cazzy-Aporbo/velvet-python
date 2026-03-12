from src.data_utils import load_dataset, train_test_split
from src.ml_pipeline import TFIDFModel, WordFrequencyModel, confusion_matrix, evaluate


def test_full_pipeline_freq():
    dataset = load_dataset()
    assert len(dataset) > 0

    train_data, test_data = train_test_split(dataset, test_ratio=0.3, seed=42)
    texts = [t for t, _ in train_data]
    labels = [label for _, label in train_data]

    model = WordFrequencyModel()
    model.train(texts, labels)
    assert model.is_trained

    preds = [model.predict(text) for text, _ in test_data]
    assert all(isinstance(p, str) for p in preds)

    acc = evaluate(model, test_data)
    assert 0.0 <= acc <= 1.0


def test_full_pipeline_tfidf():
    dataset = load_dataset()
    train_data, test_data = train_test_split(dataset, test_ratio=0.3, seed=42)
    texts = [t for t, _ in train_data]
    labels = [label for _, label in train_data]

    model = TFIDFModel()
    model.train(texts, labels)
    assert model.is_trained

    acc = evaluate(model, test_data)
    assert 0.0 <= acc <= 1.0


def test_confusion_matrix_integration():
    dataset = load_dataset()
    texts = [t for t, _ in dataset]
    labels = [label for _, label in dataset]

    model = WordFrequencyModel()
    model.train(texts, labels)

    cm = confusion_matrix(model, dataset)
    assert isinstance(cm, dict)
    assert len(cm) > 0
