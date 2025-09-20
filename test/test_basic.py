"""
pytest tests for the example package.
This tests are intentionally simple to prove CI passes.
"""

import pytest
from your_package import add

def test_add_positive():
    # basic positive numbers
    assert add(1, 2) == 3

def test_add_zero():
    # zero handling
    assert add(0, 0) == 0

def test_add_type_error():
    # ensure wrong types raise the expected error
    with pytest.raises(TypeError):
        add(1, "two")