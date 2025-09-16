"""
Velvet Python - Building Python Mastery Through Working Code
=============================================================

A comprehensive learning journey through Python's ecosystem, from fundamentals 
to production systems. Every pattern tested, every benchmark measured, every 
concept visualized.

Author: Cazzy Aporbo, MS
License: MIT (Code), CC-BY-4.0 (Documentation)
Repository: https://github.com/Cazzy-Aporbo/velvet-python
Started: January 2025
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Version information
__version__ = "0.1.0"
__author__ = "Cazzy Aporbo, MS"
__email__ = ""
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 Cazzy Aporbo, MS"
__url__ = "https://github.com/Cazzy-Aporbo/velvet-python"

# Project metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__copyright__",
    "__url__",
    "VelvetPython",
    "get_module_info",
    "list_modules",
    "check_environment",
    "MODULES",
    "PASTEL_THEME",
]

# Pastel color theme constants
PASTEL_THEME = {
    "primary": {
        "misty_rose": "#FFE4E1",
        "lavender": "#EDE5FF",
        "alice_blue": "#F0E6FF",
        "lavender_blush": "#FFF0F5",
        "light_goldenrod": "#FFEFD5",
        "wheat": "#F5DEB3",
    },
    "secondary": {
        "plum": "#DDA0DD",
        "thistle": "#D8BFD8",
        "light_pink": "#FFB6C1",
        "pale_green": "#98FB98",
        "sky_blue": "#87CEEB",
    },
    "text": {
        "primary": "#4A4A4A",
        "secondary": "#706B70",
        "accent": "#8B7D8B",
        "muted": "#858585",
    },
    "backgrounds": {
        "gradient_1": "linear-gradient(135deg, #FFE4E1 0%, #F0E6FF 100%)",
        "gradient_2": "linear-gradient(135deg, #EDE5FF 0%, #FFF0F5 100%)",
        "gradient_3": "linear-gradient(135deg, #F0E6FF 0%, #FFEFD5 100%)",
        "gradient_4": "linear-gradient(135deg, #FFF0F5 0%, #F5DEB3 100%)",
    }
}

# Module registry with metadata
MODULES = {
    "01-environments": {
        "name": "Environment Management",
        "difficulty": "Beginner",
        "topics": ["venv", "pip", "conda", "poetry", "pyenv"],
        "key_learning": "Setting up and managing Python development environments",
        "status": "planned",
    },
    "02-packaging": {
        "name": "Package Distribution",
        "difficulty": "Intermediate",
        "topics": ["setuptools", "wheels", "pyproject.toml", "twine", "PyPI"],
        "key_learning": "Creating and distributing Python packages",
        "status": "planned",
    },
    "03-cli-applications": {
        "name": "CLI Development",
        "difficulty": "Intermediate",
        "topics": ["argparse", "click", "typer", "rich", "configuration"],
        "key_learning": "Building robust command-line interfaces",
        "status": "planned",
    },
    "04-datetime-handling": {
        "name": "DateTime Mastery",
        "difficulty": "Intermediate",
        "topics": ["datetime", "pytz", "pendulum", "dateutil", "timezones"],
        "key_learning": "Handling dates, times, and timezones correctly",
        "status": "planned",
    },
    "05-text-processing": {
        "name": "Text Processing",
        "difficulty": "Intermediate",
        "topics": ["regex", "unicode", "encoding", "string methods", "parsing"],
        "key_learning": "Text manipulation and encoding best practices",
        "status": "planned",
    },
    "06-nlp-essentials": {
        "name": "Natural Language Processing",
        "difficulty": "Advanced",
        "topics": ["spacy", "nltk", "transformers", "tokenization", "embeddings"],
        "key_learning": "NLP fundamentals and modern techniques",
        "status": "planned",
    },
    "07-http-clients": {
        "name": "HTTP & APIs",
        "difficulty": "Intermediate",
        "topics": ["requests", "httpx", "aiohttp", "REST", "authentication"],
        "key_learning": "Consuming and building web APIs",
        "status": "planned",
    },
    "08-data-stores": {
        "name": "Database Systems",
        "difficulty": "Advanced",
        "topics": ["SQLAlchemy", "psycopg2", "pymongo", "redis", "migrations"],
        "key_learning": "Database patterns and ORM vs raw SQL",
        "status": "planned",
    },
    "09-concurrency": {
        "name": "Concurrency Patterns",
        "difficulty": "Advanced",
        "topics": ["threading", "multiprocessing", "asyncio", "GIL", "patterns"],
        "key_learning": "Choosing the right concurrency model",
        "status": "in-progress",
    },
    "10-media-processing": {
        "name": "Media Processing",
        "difficulty": "Advanced",
        "topics": ["Pillow", "OpenCV", "moviepy", "audio", "streaming"],
        "key_learning": "Image, audio, and video processing",
        "status": "planned",
    },
    "11-numerical-computing": {
        "name": "Numerical Computing",
        "difficulty": "Advanced",
        "topics": ["numpy", "scipy", "numba", "BLAS", "vectorization"],
        "key_learning": "High-performance numerical computation",
        "status": "planned",
    },
    "12-data-visualization": {
        "name": "Data Visualization",
        "difficulty": "Intermediate",
        "topics": ["matplotlib", "plotly", "altair", "bokeh", "dashboards"],
        "key_learning": "Creating effective data visualizations",
        "status": "planned",
    },
    "13-machine-learning": {
        "name": "Machine Learning",
        "difficulty": "Advanced",
        "topics": ["scikit-learn", "xgboost", "feature engineering", "deployment"],
        "key_learning": "ML workflows from prototype to production",
        "status": "planned",
    },
    "14-web-frameworks": {
        "name": "Web Frameworks",
        "difficulty": "Advanced",
        "topics": ["FastAPI", "Flask", "Django", "middleware", "deployment"],
        "key_learning": "Building and deploying web applications",
        "status": "planned",
    },
    "15-authentication": {
        "name": "Authentication & Security",
        "difficulty": "Advanced",
        "topics": ["JWT", "OAuth2", "sessions", "cryptography", "OWASP"],
        "key_learning": "Securing applications properly",
        "status": "planned",
    },
    "16-task-queues": {
        "name": "Task Queues",
        "difficulty": "Advanced",
        "topics": ["Celery", "RQ", "Huey", "Redis", "monitoring"],
        "key_learning": "Background job processing patterns",
        "status": "planned",
    },
    "17-data-validation": {
        "name": "Data Validation",
        "difficulty": "Intermediate",
        "topics": ["Pydantic", "marshmallow", "cerberus", "forms", "sanitization"],
        "key_learning": "Input validation and type safety",
        "status": "planned",
    },
    "18-testing-quality": {
        "name": "Testing Strategies",
        "difficulty": "Advanced",
        "topics": ["pytest", "hypothesis", "mocking", "coverage", "TDD"],
        "key_learning": "Writing tests that catch real bugs",
        "status": "planned",
    },
    "19-performance": {
        "name": "Performance Optimization",
        "difficulty": "Expert",
        "topics": ["profiling", "cython", "numba", "memory", "algorithms"],
        "key_learning": "Finding and fixing performance bottlenecks",
        "status": "planned",
    },
    "20-architecture": {
        "name": "Architecture Patterns",
        "difficulty": "Expert",
        "topics": ["DDD", "CQRS", "hexagonal", "microservices", "patterns"],
        "key_learning": "Designing maintainable systems",
        "status": "planned",
    },
    "21-desktop-applications": {
        "name": "Desktop Applications",
        "difficulty": "Advanced",
        "topics": ["PyQt6", "Tkinter", "Kivy", "packaging", "distribution"],
        "key_learning": "Building modern desktop GUIs",
        "status": "planned",
    },
    "22-algorithms": {
        "name": "Algorithms & Data Structures",
        "difficulty": "Expert",
        "topics": ["complexity", "optimization", "graphs", "dynamic programming"],
        "key_learning": "Algorithmic thinking in Python",
        "status": "planned",
    },
    "23-development-tools": {
        "name": "Development Tools",
        "difficulty": "Intermediate",
        "topics": ["IDE setup", "debugging", "linting", "formatting", "productivity"],
        "key_learning": "Optimizing development workflow",
        "status": "planned",
    },
}


class VelvetPython:
    """
    Main interface for the Velvet Python learning system.
    
    Author: Cazzy Aporbo, MS
    """
    
    def __init__(self) -> None:
        """Initialize the Velvet Python system."""
        self.version = __version__
        self.author = __author__
        self.modules = MODULES
        self.theme = PASTEL_THEME
        self._check_python_version()
    
    def _check_python_version(self) -> None:
        """Verify Python version meets minimum requirements."""
        min_version = (3, 10)
        current_version = sys.version_info[:2]
        
        if current_version < min_version:
            raise RuntimeError(
                f"Python {min_version[0]}.{min_version[1]}+ required, "
                f"but {current_version[0]}.{current_version[1]} found. "
                "Please upgrade your Python installation."
            )
    
    def get_module_info(self, module_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific module.
        
        Args:
            module_id: Module identifier (e.g., "09-concurrency")
            
        Returns:
            Module metadata dictionary or None if not found
        """
        return self.modules.get(module_id)
    
    def list_modules(
        self, 
        difficulty: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all modules with optional filtering.
        
        Args:
            difficulty: Filter by difficulty level
            status: Filter by development status
            
        Returns:
            List of module metadata dictionaries
        """
        modules = []
        
        for module_id, info in self.modules.items():
            if difficulty and info["difficulty"] != difficulty:
                continue
            if status and info["status"] != status:
                continue
            
            modules.append({
                "id": module_id,
                **info
            })
        
        return sorted(modules, key=lambda x: x["id"])
    
    def get_learning_path(self) -> List[str]:
        """
        Get the recommended learning path through modules.
        
        Returns:
            Ordered list of module IDs
        """
        return sorted(self.modules.keys())
    
    def check_environment(self) -> Dict[str, Any]:
        """
        Check the current environment setup.
        
        Returns:
            Environment information dictionary
        """
        import platform
        import os
        
        return {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "velvet_version": self.version,
            "author": self.author,
            "modules_available": len(self.modules),
            "modules_completed": sum(
                1 for m in self.modules.values() 
                if m["status"] == "completed"
            ),
            "modules_in_progress": sum(
                1 for m in self.modules.values() 
                if m["status"] == "in-progress"
            ),
            "virtual_env": os.environ.get("VIRTUAL_ENV") is not None,
            "project_root": Path(__file__).parent.parent,
        }
    
    def __repr__(self) -> str:
        """String representation of VelvetPython instance."""
        return (
            f"VelvetPython(version={self.version}, "
            f"modules={len(self.modules)}, "
            f"author='{self.author}')"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"Velvet Python v{self.version} by {self.author}\n"
            f"A comprehensive Python learning journey with {len(self.modules)} modules"
        )


# Convenience functions
def get_module_info(module_id: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a specific module.
    
    Author: Cazzy Aporbo, MS
    """
    vp = VelvetPython()
    return vp.get_module_info(module_id)


def list_modules(
    difficulty: Optional[str] = None,
    status: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List available modules with optional filtering.
    
    Author: Cazzy Aporbo, MS
    """
    vp = VelvetPython()
    return vp.list_modules(difficulty=difficulty, status=status)


def check_environment() -> Dict[str, Any]:
    """
    Check the current Python environment.
    
    Author: Cazzy Aporbo, MS
    """
    vp = VelvetPython()
    return vp.check_environment()


# Display project info when imported
if __name__ == "__main__":
    import json
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    
    console = Console()
    
    # Create header with gradient effect
    header = Panel(
        Text(
            f"Velvet Python v{__version__}",
            style="bold #8B7D8B",
            justify="center"
        ),
        subtitle=f"by {__author__}",
        style="#FFE4E1",
        padding=(1, 2),
    )
    console.print(header)
    
    # Environment check
    vp = VelvetPython()
    env = vp.check_environment()
    
    console.print("\n[#8B7D8B]Environment Information:[/#8B7D8B]")
    for key, value in env.items():
        if key != "project_root":
            console.print(f"  [#706B70]{key}:[/#706B70] {value}")
    
    # Module overview table
    console.print("\n[#8B7D8B]Module Overview:[/#8B7D8B]")
    
    table = Table(
        title="Learning Modules",
        style="#706B70",
        header_style="#8B7D8B bold",
        border_style="#DDA0DD",
        title_style="#8B7D8B bold",
    )
    
    table.add_column("Module", style="#FFE4E1")
    table.add_column("Name", style="#EDE5FF")
    table.add_column("Difficulty", style="#F0E6FF")
    table.add_column("Status", style="#FFF0F5")
    
    for module_id in sorted(MODULES.keys()):
        info = MODULES[module_id]
        
        # Color code status
        status_color = {
            "completed": "[green]",
            "in-progress": "[yellow]",
            "planned": "[#706B70]",
        }.get(info["status"], "")
        
        table.add_row(
            module_id,
            info["name"],
            info["difficulty"],
            f"{status_color}{info['status']}[/]"
        )
    
    console.print(table)
    
    # Footer
    console.print(
        f"\n[#706B70]Repository:[/#706B70] {__url__}"
    )
    console.print(
        f"[#706B70]License:[/#706B70] {__license__} (Code), CC-BY-4.0 (Documentation)"
    )
