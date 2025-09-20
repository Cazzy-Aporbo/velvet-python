"""
Example pytest test suite.
All tests here are simple and guaranteed to pass.
"""

import pytest
from your_package import ping


def test_ping_returns_pong():
    """Check that ping() returns the expected string."""
    assert ping() == "pong"


def test_math_is_correct():
    """Basic math test (always true)."""
    assert 2 + 2 == 4


def test_string_contains():
    """Check substrings (always true)."""
    text = "hello world"
    assert "hello" in text
    assert text.startswith("h")


def test_list_membership():
    """Check that values exist in a list."""
    numbers = [1, 2, 3, 4, 5]
    assert 3 in numbers
    assert len(numbers) == 5


def test_dictionary_lookup():
    """Ensure dictionary access works as expected."""
    data = {"a": 1, "b": 2}
    assert data["a"] == 1
    assert "b" in data


def test_parametrized_examples():
    """Same test logic over multiple inputs using pytest.mark.parametrize."""
    @pytest.mark.parametrize("x, y, expected", [
        (1, 1, 2),
        (2, 3, 5),
        (10, 5, 15),
    ])
    def inner(x, y, expected):
        assert x + y == expected

    # Run the inner parametrized test function
    inner()


def test_raises_example():
    """Show how to test that code raises an exception."""
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0