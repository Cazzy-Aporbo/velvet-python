"""
Velvet Python - Test Suite
==========================

Author: Cazzy Aporbo, MS
Created: January 2025

This is the root test package for Velvet Python. 

I organize tests the same way I organize code - clearly and with purpose.
Each test file tests one specific module or functionality. No mysteries.

Test Organization:
- tests/           (root tests - CLI, main package)
- MODULE/tests/    (module-specific tests)

I learned to write tests BEFORE pushing to production after breaking
a client's site on a Friday afternoon. Now I test everything.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import velvet_python
# This ensures tests can run from any directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CONFIG = {
    'author': 'Cazzy Aporbo, MS',
    'project': 'Velvet Python',
    'min_coverage': 80,  # I aim for 80% coverage minimum
    'test_timeout': 30,   # Default timeout for async tests
    'verbose': True,      # I like seeing what's happening
}

# Shared test utilities
def get_test_data_path() -> Path:
    """
    Get the path to test data directory.
    
    I keep test data separate from test code for clarity.
    """
    return Path(__file__).parent / 'data'


def get_fixture_path(filename: str) -> Path:
    """
    Get the path to a test fixture file.
    
    Args:
        filename: Name of the fixture file
        
    Returns:
        Path to the fixture file
    """
    return get_test_data_path() / 'fixtures' / filename


# Test markers explanation
"""
I use pytest markers to categorize tests:

@pytest.mark.unit - Fast, isolated unit tests (run these constantly)
@pytest.mark.integration - Tests that integrate multiple components
@pytest.mark.slow - Tests that take > 1 second (run less frequently)
@pytest.mark.requires_net - Tests that need internet connection
@pytest.mark.benchmark - Performance tests

To run only unit tests: pytest -m unit
To skip slow tests: pytest -m "not slow"
"""

__all__ = [
    'TEST_CONFIG',
    'get_test_data_path', 
    'get_fixture_path',
]

__version__ = '0.1.0'
__author__ = 'Cazzy Aporbo, MS'
