"""
velvet-python / src

Text classification, ML pipelines, and data utilities — all built
from scratch using the Python standard library.

Modules:
    ai            Three classification strategies (rule-based, Naive Bayes, cosine)
    ml_pipeline   Train/predict/evaluate with WordFrequency and TF-IDF models
    data_utils    Dataset loading, CSV parsing, train/test splitting
"""

__version__ = "0.2.0"
__author__ = "Cazzy Aporbo"


def ping() -> str:
    """Health check — returns 'pong'. Used by the smoke test suite."""
    return "pong"