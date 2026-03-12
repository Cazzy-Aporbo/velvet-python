import pytest
from src import ping


def test_ping():
    assert ping() == "pong"


def test_string_contains():
    text = "hello world"
    assert "hello" in text
    assert text.startswith("h")


def test_list_membership():
    numbers = [1, 2, 3, 4, 5]
    assert 3 in numbers
    assert len(numbers) == 5


def test_dictionary_lookup():
    data = {"a": 1, "b": 2}
    assert data["a"] == 1
    assert "b" in data


@pytest.mark.parametrize("x, y, expected", [
    (1, 1, 2),
    (2, 3, 5),
    (10, 5, 15),
])
def test_addition(x, y, expected):
    assert x + y == expected


def test_raises_on_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        _ = 1 / 0