import pytest
from src.ai import classify_text, generate_number


def test_classify_empty():
    assert classify_text("") == "empty"


def test_classify_greeting():
    assert classify_text("Hello world!") == "greeting"


def test_classify_number():
    assert classify_text("12345") == "number"


@pytest.mark.parametrize("text, expected", [
    ("hello there", "greeting"),
    ("HELLO", "greeting"),
    ("42", "number"),
    ("anything else", "other"),
])
def test_classify_parametrized(text, expected):
    assert classify_text(text) == expected


def test_generate_number_seeded():
    assert generate_number(seed=1) == generate_number(seed=1)


def test_generate_number_range():
    for s in range(5):
        n = generate_number(seed=s)
        assert 0 <= n <= 9


def test_generate_number_different_seeds():
    nums = {generate_number(seed=s) for s in range(5)}
    assert len(nums) > 1