"""
Trivial test to guarantee GitHub Actions passes.
"""

from your_package import ping


def test_ping():
    # This will always pass
    assert ping() == "pong"