"""
A toy "AI" module with deterministic behavior.
"""

from __future__ import annotations
import random


def classify_text(text: str) -> str:
    """
    Fake text classifier that labels text based on simple rules.
    Always deterministic, so tests are guaranteed to pass.
    """
    if not text:
        return "empty"
    if "hello" in text.lower():
        return "greeting"
    if text.isdigit():
        return "number"
    return "other"


def generate_number(seed: int | None = None) -> int:
    """
    Fake AI generator — returns a pseudo-random number,
    but seed ensures reproducibility for tests.
    """
    rng = random.Random(seed)
    return rng.randint(0, 9)