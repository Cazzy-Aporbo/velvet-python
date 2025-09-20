"""
High-level integration test for the mock ML pipeline.

This simulates a full ML workflow:
- Load config
- Load dataset
- Train model
- Predict
- Evaluate performance
"""

from pathlib import Path
from your_package.data_utils import load_dataset, load_config
from your_package.ml_pipeline import MockModel, evaluate


def test_full_pipeline(tmp_path: Path):
    # ---------------------------
    # 1. Load configuration
    # ---------------------------
    config_path = Path("src/your_package/config.yaml")
    config = load_config(config_path)

    assert config["model"]["type"] == "MockModel"
    assert config["training"]["epochs"] == 2

    # ---------------------------
    # 2. Load dataset
    # ---------------------------
    dataset = load_dataset()
    assert len(dataset) > 0

    # ---------------------------
    # 3. Train model
    # ---------------------------
    model = MockModel()
    texts = [text for text, _ in dataset]
    model.train(texts)

    assert model.is_trained
    assert "hello" in model.word_counts

    # ---------------------------
    # 4. Run predictions
    # ---------------------------
    preds = [model.predict(text) for text, _ in dataset]

    assert all(isinstance(p, str) for p in preds)

    # ---------------------------
    # 5. Evaluate
    # ---------------------------
    acc = evaluate(model, dataset)

    # Accuracy should be within [0,1] and non-zero
    assert 0.0 <= acc <= 1.0
    assert acc > 0.2  # pipeline should perform decently

    # ---------------------------
    # 6. Smoke check: config + model interaction
    # ---------------------------
    max_vocab = config["model"]["max_vocab_size"]
    assert len(model.word_counts) <= max_vocab