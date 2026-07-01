"""Test bootstrap for Velvet Python repository."""

import os
import sys


def pytest_configure():
    """Ensure repository root is importable for tests."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
