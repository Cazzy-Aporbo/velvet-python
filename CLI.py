"""
Velvet Python CLI — project tooling and interactive demos.

Commands:
    velvet info       Show project structure and environment
    velvet test       Run the pytest suite
    velvet lint       Run ruff on src/ and tests/
    velvet classify   Interactive text classification demo
    velvet tree       Display project as a Rich tree
    velvet clean      Remove __pycache__, .pytest_cache, etc.
    velvet version    Print version and Python info
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from src import __version__, __author__

PROJECT_ROOT = Path(__file__).parent
console = Console()

app = typer.Typer(
    name="velvet",
    help="Velvet Python CLI",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)


def _gradient(text: str) -> Text:
    """Render text with a pastel gradient — lavender to rose."""
    colors = [
        "#EDE5FF", "#E8DFF9", "#E3D9F3", "#DED3ED",
        "#D9CDE7", "#DCC7E1", "#DFC1DB", "#E2BBD5",
        "#E5B5CF", "#FFE4E1",
    ]
    out = Text()
    for i, ch in enumerate(text):
        idx = min(int(i / max(len(text), 1) * len(colors)), len(colors) - 1)
        out.append(ch, style=colors[idx])
    return out


def _header() -> None:
    console.print(Panel(
        Text.assemble(
            _gradient("VELVET PYTHON"), "\n",
            Text(f"v{__version__}", style="#8B7D8B"), " · ",
            Text(__author__, style="#706B70"),
            justify="center",
        ),
        border_style="#DDA0DD",
        padding=(1, 2),
    ))


@app.command()
def info() -> None:
    """Show project structure and environment."""
    _header()

    tbl = Table(show_header=False, box=None, padding=(0, 2))
    tbl.add_column("Key", style="#8B7D8B")
    tbl.add_column("Value", style="#706B70")

    tbl.add_row("Python", sys.version.split()[0])
    tbl.add_row("Platform", sys.platform)
    tbl.add_row("Venv", "active" if sys.prefix != sys.base_prefix else "none")
    tbl.add_row("src/ modules", str(len(list((PROJECT_ROOT / "src").glob("*.py")))))
    tbl.add_row("tests/", str(len(list((PROJECT_ROOT / "tests").glob("test_*.py")))))
    tbl.add_row("scripts/", str(len(list((PROJECT_ROOT / "scripts").glob("*.py")))))
    tbl.add_row("pyfiles/", str(len(list((PROJECT_ROOT / "pyfiles").glob("*.py")))))

    console.print(Panel(tbl, title="[#8B7D8B]Environment[/#8B7D8B]", border_style="#E6E6FA", padding=(1, 2)))


@app.command()
def test(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    coverage: bool = typer.Option(False, "--cov", "-c"),
) -> None:
    """Run the pytest suite."""
    _header()
    cmd = [sys.executable, "-m", "pytest", "tests/", "--tb=short"]
    if verbose:
        cmd.append("-v")
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    raise typer.Exit(result.returncode)


@app.command()
def lint() -> None:
    """Lint src/ and tests/ with ruff."""
    _header()
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "src/", "tests/"],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        console.print("[green]No issues found.[/green]")
    raise typer.Exit(result.returncode)


@app.command()
def classify(
    text: str = typer.Argument(..., help="Text to classify"),
    strategy: str = typer.Option("all", "--strategy", "-s", help="rule | bayes | cosine | all"),
) -> None:
    """Classify text using strategies from src/ai.py."""
    from src.ai import classify_text, NaiveBayesClassifier, CosineSimilarityClassifier
    from src.data_utils import load_dataset

    _header()
    data = load_dataset()
    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    tbl = Table(title=f"[#8B7D8B]Classifying:[/#8B7D8B] {text}", border_style="#DDA0DD")
    tbl.add_column("Strategy", style="#FFE4E1")
    tbl.add_column("Prediction", style="#EDE5FF")

    if strategy in ("rule", "all"):
        tbl.add_row("Rule-based", classify_text(text))

    if strategy in ("bayes", "all"):
        nb = NaiveBayesClassifier()
        nb.train(texts, labels)
        pred = nb.predict(text)
        proba = nb.predict_proba(text)
        top_p = max(proba.values())
        tbl.add_row("Naive Bayes", f"{pred} ({top_p:.0%})")

    if strategy in ("cosine", "all"):
        cs = CosineSimilarityClassifier()
        cs.train(texts, labels)
        tbl.add_row("Cosine Similarity", cs.predict(text))

    console.print(tbl)


@app.command()
def tree(
    depth: int = typer.Option(3, "--depth", "-d"),
) -> None:
    """Display project structure as a Rich tree."""
    _header()
    root_tree = Tree("[#8B7D8B]velvet-python[/#8B7D8B]", style="#DDA0DD")

    skip = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

    def _walk(node, path: Path, level: int = 0) -> None:
        if level >= depth:
            return
        try:
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            return
        for item in items:
            if item.name in skip or item.name.startswith("."):
                continue
            if item.is_dir():
                branch = node.add(f"[#FFE4E1]{item.name}/[/#FFE4E1]")
                _walk(branch, item, level + 1)
            else:
                ext_style = {
                    ".py": "#EDE5FF", ".md": "#F0E6FF",
                    ".yml": "#FFF0F5", ".yaml": "#FFF0F5",
                    ".toml": "#FFF0F5",
                }.get(item.suffix, "#FFEFD5")
                node.add(f"[{ext_style}]{item.name}[/{ext_style}]")

    _walk(root_tree, PROJECT_ROOT)
    console.print(root_tree)


@app.command()
def clean(
    all_: bool = typer.Option(False, "--all", "-a", help="Remove build artifacts too"),
) -> None:
    """Remove __pycache__, .pytest_cache, and other transient files."""
    _header()
    patterns = [
        "**/__pycache__", "**/*.pyc", "**/.pytest_cache",
        "**/.mypy_cache", "**/.ruff_cache", "**/.coverage",
    ]
    if all_:
        patterns.extend(["build", "dist", "**/*.egg-info"])

    removed = 0
    for pattern in patterns:
        for p in PROJECT_ROOT.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            removed += 1

    console.print(f"[green]Cleaned {removed} items.[/green]")


@app.command()
def version() -> None:
    """Print version and Python info."""
    console.print(_gradient(f"Velvet Python v{__version__}"))
    console.print(f"[#706B70]{__author__}[/#706B70]")
    console.print(f"[#706B70]Python {sys.version.split()[0]}[/#706B70]")


if __name__ == "__main__":
    app()
