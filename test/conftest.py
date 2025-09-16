"""
Pytest Configuration and Shared Fixtures

Author: Cazzy Aporbo, MS
Created: January 2025; Updated: September 16 2025

This file is automatically loaded by pytest. It contains fixtures and
configuration that all tests can use. Think of it as the test kitchen -
all the ingredients and tools are here.

I learned to use conftest.py properly after writing the same test setup
in 20 different files. Never again. DRY (Don't Repeat Yourself) applies
to tests too!
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, MagicMock
import json
import yaml

import pytest
from click.testing import CliRunner
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from velvet_python import VelvetPython
from velvet_python.cli import app


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers.
    
    I define all markers here so pytest doesn't complain about unknown markers.
    """
    config.addinivalue_line(
        "markers", "unit: Unit tests - fast and isolated"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests - test multiple components"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests - take more than 1 second"
    )
    config.addinivalue_line(
        "markers", "requires_net: Tests that require internet connection"
    )
    config.addinivalue_line(
        "markers", "benchmark: Performance benchmarking tests"
    )
    config.addinivalue_line(
        "markers", "cli: Command-line interface tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: Asynchronous tests"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically add markers based on test location/name.
    
    This saves me from having to manually mark every test.
    """
    for item in items:
        # Add markers based on test file location
        if "test_cli" in str(item.fspath):
            item.add_marker(pytest.mark.cli)
        
        if "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        
        # Mark async tests
        if "async" in item.name:
            item.add_marker(pytest.mark.asyncio)


# =============================================================================
# DIRECTORY FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for testing.
    
    I use this constantly for tests that need to create files.
    The directory is automatically cleaned up after the test.
    
    Yields:
        Path to temporary directory
    """
    temp_path = tempfile.mkdtemp(prefix="velvet_test_")
    temp_path = Path(temp_path)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def project_dir(temp_dir: Path) -> Path:
    """
    Create a mock project directory with standard structure.
    
    Sets up a minimal project structure for testing.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path to project directory
    """
    # Create standard project structure
    (temp_dir / "velvet_python").mkdir()
    (temp_dir / "tests").mkdir()
    (temp_dir / "docs").mkdir()
    
    # Create essential files
    (temp_dir / "README.md").write_text("# Test Project\nAuthor: Cazzy Aporbo, MS")
    (temp_dir / "requirements.txt").write_text("rich>=13.0.0\ntyper>=0.9.0")
    (temp_dir / "pyproject.toml").write_text("""
[project]
name = "test-project"
version = "0.1.0"
author = [{name = "Cazzy Aporbo, MS"}]
""")
    
    return temp_dir


# =============================================================================
# CLI FIXTURES
# =============================================================================

@pytest.fixture
def cli_runner() -> CliRunner:
    """
    Create a Click test runner for CLI testing.
    
    This isolates CLI tests from the actual filesystem.
    
    Returns:
        CliRunner instance
    """
    return CliRunner()


@pytest.fixture
def mock_console() -> Mock:
    """
    Create a mock Rich console for testing output.
    
    I use this to test console output without actual printing.
    
    Returns:
        Mock Console object
    """
    mock = Mock(spec=Console)
    mock.print = Mock()
    mock.input = Mock(return_value="yes")
    return mock


# =============================================================================
# DATA FIXTURES
# =============================================================================

@pytest.fixture
def sample_requirements() -> list:
    """
    Sample requirements for testing.
    
    Returns:
        List of requirement strings
    """
    return [
        "requests==2.31.0",
        "pandas==2.2.0",
        "numpy==1.26.0",
        "flask==3.0.0",
        "pytest==8.0.0",
        "black==24.1.0",
    ]


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """
    Sample configuration dictionary.
    
    Returns:
        Configuration dictionary
    """
    return {
        "project": {
            "name": "test-project",
            "version": "0.1.0",
            "author": "Cazzy Aporbo, MS",
            "description": "Test project for Velvet Python",
        },
        "python": {
            "version": "3.11",
            "virtual_env": ".venv",
        },
        "dependencies": {
            "production": ["requests", "pandas"],
            "development": ["pytest", "black", "mypy"],
        },
        "settings": {
            "debug": True,
            "testing": True,
        }
    }


@pytest.fixture
def config_files(temp_dir: Path, sample_config: Dict) -> Dict[str, Path]:
    """
    Create various configuration files for testing.
    
    Args:
        temp_dir: Temporary directory
        sample_config: Sample configuration
        
    Returns:
        Dictionary mapping file type to path
    """
    files = {}
    
    # Create JSON config
    json_file = temp_dir / "config.json"
    json_file.write_text(json.dumps(sample_config, indent=2))
    files["json"] = json_file
    
    # Create YAML config
    yaml_file = temp_dir / "config.yaml"
    yaml_file.write_text(yaml.dump(sample_config))
    files["yaml"] = yaml_file
    
    # Create TOML config
    toml_file = temp_dir / "pyproject.toml"
    toml_content = """
[project]
name = "test-project"
version = "0.1.0"

[tool.velvet]
author = "Cazzy Aporbo, MS"
"""
    toml_file.write_text(toml_content)
    files["toml"] = toml_file
    
    # Create .env file
    env_file = temp_dir / ".env"
    env_content = """
DEBUG=true
DATABASE_URL=sqlite:///test.db
SECRET_KEY=test-secret-key
AUTHOR=Cazzy Aporbo, MS
"""
    env_file.write_text(env_content)
    files["env"] = env_file
    
    return files


# =============================================================================
# MOCK OBJECTS
# =============================================================================

@pytest.fixture
def mock_velvet_python() -> Mock:
    """
    Create a mock VelvetPython instance.
    
    Returns:
        Mock VelvetPython object
    """
    mock = Mock(spec=VelvetPython)
    mock.version = "0.1.0"
    mock.author = "Cazzy Aporbo, MS"
    mock.modules = {
        "01-environments": {
            "name": "Environment Management",
            "difficulty": "Beginner",
            "status": "completed"
        },
        "09-concurrency": {
            "name": "Concurrency Patterns",
            "difficulty": "Advanced",
            "status": "in-progress"
        }
    }
    mock.get_module_info = Mock(return_value=mock.modules.get)
    mock.list_modules = Mock(return_value=list(mock.modules.values()))
    
    return mock


@pytest.fixture
def mock_subprocess() -> Mock:
    """
    Mock subprocess for testing commands without execution.
    
    Returns:
        Mock subprocess module
    """
    mock = MagicMock()
    mock.run = Mock(return_value=Mock(
        returncode=0,
        stdout="Success",
        stderr=""
    ))
    return mock


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================

@pytest.fixture
def clean_env() -> Generator[None, None, None]:
    """
    Provide a clean environment for testing.
    
    Saves and restores environment variables.
    """
    original_env = os.environ.copy()
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def test_env(clean_env) -> Dict[str, str]:
    """
    Set up test environment variables.
    
    Args:
        clean_env: Clean environment fixture
        
    Returns:
        Dictionary of test environment variables
    """
    test_vars = {
        "VELVET_ENV": "testing",
        "VELVET_DEBUG": "true",
        "VELVET_AUTHOR": "Cazzy Aporbo, MS",
        "DATABASE_URL": "sqlite:///:memory:",
        "SECRET_KEY": "test-secret-key-123",
    }
    
    os.environ.update(test_vars)
    return test_vars


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

@pytest.fixture
def assert_pastel_colors():
    """
    Helper to assert our pastel color theme is used.
    
    Returns:
        Assertion function
    """
    def _assert(text: str):
        """Check if text contains our pastel color codes"""
        pastel_colors = [
            "#FFE4E1",  # Misty Rose
            "#E6E6FA",  # Lavender  
            "#F0E6FF",  # Alice Blue variant
            "#FFF0F5",  # Lavender Blush
            "#FFEFD5",  # Papaya Whip
            "#F5DEB3",  # Wheat
            "#DDA0DD",  # Plum
            "#D8BFD8",  # Thistle
        ]
        
        has_color = any(color in text for color in pastel_colors)
        assert has_color, f"Text doesn't contain pastel colors: {text[:100]}"
    
    return _assert


# =============================================================================
# PERFORMANCE FIXTURES
# =============================================================================

@pytest.fixture
def benchmark_timer():
    """
    Simple timer for performance testing.
    
    I use this for quick performance checks in tests.
    
    Returns:
        Timer context manager
    """
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timer():
        start = time.perf_counter()
        result = {"elapsed": 0}
        
        yield result
        
        result["elapsed"] = time.perf_counter() - start
    
    return timer


# =============================================================================
# ASYNC FIXTURES
# =============================================================================

@pytest.fixture
async def async_client():
    """
    Async HTTP client for testing async endpoints.
    
    Returns:
        Async client
    """
    import httpx
    
    async with httpx.AsyncClient() as client:
        yield client


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture
def test_db(temp_dir: Path) -> Path:
    """
    Create a test SQLite database.
    
    Args:
        temp_dir: Temporary directory
        
    Returns:
        Path to database file
    """
    db_path = temp_dir / "test.db"
    
    # Create minimal schema
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS environments (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            author TEXT DEFAULT 'Cazzy Aporbo, MS'
        )
    """)
    
    conn.commit()
    conn.close()
    
    return db_path


# =============================================================================
# CLEANUP
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Automatic cleanup after each test.
    
    This runs after every test to ensure no test artifacts remain.
    I added this after tests started interfering with each other.
    """
    yield
    
    # Cleanup any temporary files created during tests
    temp_patterns = [
        "velvet_test_*",
        "test_env_*", 
        "benchmark_*",
    ]
    
    import glob
    import tempfile
    temp_dir = Path(tempfile.gettempdir())
    
    for pattern in temp_patterns:
        for path in temp_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)


# =============================================================================
# DOCSTRING
# =============================================================================

"""
Available Fixtures:
-------------------
- temp_dir: Temporary directory (auto-cleaned)
- project_dir: Mock project structure
- cli_runner: Click CLI test runner
- mock_console: Mock Rich console
- sample_requirements: List of requirements
- sample_config: Configuration dictionary
- config_files: Various config file formats
- mock_velvet_python: Mock VelvetPython instance
- mock_subprocess: Mock subprocess module
- clean_env: Clean environment variables
- test_env: Test environment variables
- assert_pastel_colors: Check for pastel theme
- benchmark_timer: Performance timing
- async_client: Async HTTP client
- test_db: SQLite test database

Usage in tests:
def test_something(temp_dir, mock_console):
    # Use fixtures directly by name
    pass

Author: Cazzy Aporbo, MS
"""
