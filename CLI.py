"""
Velvet Python CLI - Command Line Interface
===========================================

Interactive command-line interface for navigating and managing the 
Velvet Python learning system.

Author: Cazzy Aporbo, MS
License: MIT
"""

from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import typer
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.tree import Tree
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.style import Style

from velvet_python import (
    __version__,
    __author__,
    VelvetPython,
    MODULES,
    PASTEL_THEME,
)

# Initialize Typer app with custom help
app = typer.Typer(
    name="velvet",
    help=f"Velvet Python CLI - Building Python Mastery Through Working Code\nAuthor: {__author__}",
    add_completion=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)

# Initialize Rich console with theme
console = Console()

# CLI state management
state = {
    "current_module": None,
    "project_root": Path(__file__).parent.parent,
    "vp": VelvetPython(),
}


def create_gradient_text(text: str, start_color: str = "#FFE4E1", end_color: str = "#EDE5FF") -> Text:
    """
    Create gradient colored text.
    
    Author: Cazzy Aporbo, MS
    """
    gradient_text = Text()
    colors = [
        "#FFE4E1", "#FCDEDD", "#F9D8D9", "#F6D2D5", "#F3CCD1",
        "#F0C6CD", "#EDC0C9", "#EABAC5", "#E7B4C1", "#EDE5FF"
    ]
    
    for i, char in enumerate(text):
        color_idx = int((i / len(text)) * len(colors))
        color_idx = min(color_idx, len(colors) - 1)
        gradient_text.append(char, style=colors[color_idx])
    
    return gradient_text


def print_header():
    """
    Print the Velvet Python header.
    
    Author: Cazzy Aporbo, MS
    """
    header_text = create_gradient_text("VELVET PYTHON")
    version_text = Text(f"v{__version__}", style="#8B7D8B")
    author_text = Text(f"by {__author__}", style="#706B70")
    
    header_panel = Panel(
        Text.assemble(
            header_text, "\n",
            version_text, " | ",
            author_text,
            justify="center"
        ),
        style="#FFE4E1",
        border_style="#DDA0DD",
        padding=(1, 2),
    )
    console.print(header_panel)


@app.callback()
def callback():
    """
    Velvet Python CLI - Your companion for Python mastery.
    
    Author: Cazzy Aporbo, MS
    """
    pass


@app.command()
def info():
    """
    Display project information and environment status.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    env = state["vp"].check_environment()
    
    # Environment info panel
    env_table = Table(show_header=False, box=None, padding=(0, 2))
    env_table.add_column("Key", style="#8B7D8B")
    env_table.add_column("Value", style="#706B70")
    
    env_table.add_row("Python Version", env["python_version"])
    env_table.add_row("Platform", env["platform"])
    env_table.add_row("Virtual Environment", "✓ Active" if env["virtual_env"] else "✗ Not Active")
    env_table.add_row("Modules Available", str(env["modules_available"]))
    env_table.add_row("Modules Completed", str(env["modules_completed"]))
    env_table.add_row("Modules In Progress", str(env["modules_in_progress"]))
    
    console.print(Panel(
        env_table,
        title="[#8B7D8B]Environment Information[/#8B7D8B]",
        border_style="#E6E6FA",
        padding=(1, 2),
    ))


@app.command()
def modules(
    difficulty: Optional[str] = typer.Option(None, "--difficulty", "-d", help="Filter by difficulty"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
    detailed: bool = typer.Option(False, "--detailed", "-v", help="Show detailed information"),
):
    """
    List all available learning modules.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    modules_list = state["vp"].list_modules(difficulty=difficulty, status=status)
    
    if not modules_list:
        console.print("[#CD919E]No modules found matching criteria.[/#CD919E]")
        return
    
    if detailed:
        # Detailed view with topics
        for module in modules_list:
            # Module header
            title = f"[#FFE4E1]{module['id']}[/#FFE4E1]: [#EDE5FF]{module['name']}[/#EDE5FF]"
            
            # Create content table
            content = Table(show_header=False, box=None)
            content.add_column("Property", style="#8B7D8B")
            content.add_column("Value", style="#706B70")
            
            content.add_row("Difficulty", module["difficulty"])
            content.add_row("Status", module["status"])
            content.add_row("Topics", ", ".join(module["topics"]))
            content.add_row("Key Learning", module["key_learning"])
            
            console.print(Panel(
                content,
                title=title,
                border_style="#DDA0DD",
                padding=(1, 2),
            ))
            console.print()
    else:
        # Simple table view
        table = Table(
            title="[#8B7D8B]Learning Modules[/#8B7D8B]",
            header_style="#8B7D8B bold",
            border_style="#DDA0DD",
        )
        
        table.add_column("Module", style="#FFE4E1")
        table.add_column("Name", style="#EDE5FF")
        table.add_column("Difficulty", style="#F0E6FF")
        table.add_column("Status", style="#FFF0F5")
        table.add_column("Topics", style="#FFEFD5", max_width=30)
        
        for module in modules_list:
            status_style = {
                "completed": "[green]",
                "in-progress": "[yellow]",
                "planned": "[#706B70]",
            }.get(module["status"], "")
            
            table.add_row(
                module["id"],
                module["name"],
                module["difficulty"],
                f"{status_style}{module['status']}[/]",
                ", ".join(module["topics"][:3]) + "..."
            )
        
        console.print(table)


@app.command()
def start(
    module_id: str = typer.Argument(..., help="Module ID to start (e.g., 09-concurrency)"),
):
    """
    Start working on a specific module.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    module_info = state["vp"].get_module_info(module_id)
    
    if not module_info:
        console.print(f"[#CD919E]Module '{module_id}' not found.[/#CD919E]")
        raise typer.Exit(1)
    
    module_path = state["project_root"] / module_id
    
    # Check if module directory exists
    if not module_path.exists():
        console.print(f"[#FFE4E1]Module directory not found. Creating structure...[/#FFE4E1]")
        
        with Progress(
            SpinnerColumn(style="#DDA0DD"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Creating module structure...", total=7)
            
            # Create directory structure
            directories = [
                module_path,
                module_path / "src",
                module_path / "tests",
                module_path / "benchmarks",
                module_path / "examples",
                module_path / "notebooks",
                module_path / "docs",
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                progress.advance(task)
        
        console.print("[green]✓[/green] Module structure created successfully!")
    
    # Display module information
    console.print(Panel(
        f"[#8B7D8B]Module:[/#8B7D8B] {module_id}\n"
        f"[#8B7D8B]Name:[/#8B7D8B] {module_info['name']}\n"
        f"[#8B7D8B]Difficulty:[/#8B7D8B] {module_info['difficulty']}\n"
        f"[#8B7D8B]Status:[/#8B7D8B] {module_info['status']}\n"
        f"[#8B7D8B]Path:[/#8B7D8B] {module_path}",
        title="[#FFE4E1]Starting Module[/#FFE4E1]",
        border_style="#DDA0DD",
        padding=(1, 2),
    ))
    
    # Set current module
    state["current_module"] = module_id
    
    # Offer to open in editor
    if Confirm.ask("[#8B7D8B]Open module in VS Code?[/#8B7D8B]"):
        subprocess.run(["code", str(module_path)])


@app.command()
def test(
    module_id: Optional[str] = typer.Argument(None, help="Module ID to test"),
    coverage: bool = typer.Option(True, "--coverage/--no-coverage", help="Run with coverage"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run tests for a module or entire project.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    if module_id:
        test_path = state["project_root"] / module_id / "tests"
        if not test_path.exists():
            console.print(f"[#CD919E]No tests found for module '{module_id}'[/#CD919E]")
            raise typer.Exit(1)
    else:
        test_path = state["project_root"]
    
    console.print(f"[#8B7D8B]Running tests...[/#8B7D8B]")
    
    cmd = ["pytest", str(test_path)]
    
    if coverage:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])
    
    if verbose:
        cmd.append("-vv")
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        console.print("[green]✓[/green] All tests passed!")
    else:
        console.print("[#CD919E]✗ Some tests failed[/#CD919E]")
        raise typer.Exit(result.returncode)


@app.command()
def benchmark(
    module_id: str = typer.Argument(..., help="Module ID to benchmark"),
    compare: bool = typer.Option(False, "--compare", "-c", help="Compare with baseline"),
):
    """
    Run performance benchmarks for a module.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    benchmark_path = state["project_root"] / module_id / "benchmarks"
    
    if not benchmark_path.exists():
        console.print(f"[#CD919E]No benchmarks found for module '{module_id}'[/#CD919E]")
        raise typer.Exit(1)
    
    console.print(f"[#8B7D8B]Running benchmarks for {module_id}...[/#8B7D8B]")
    
    # Find benchmark files
    benchmark_files = list(benchmark_path.glob("*.py"))
    
    if not benchmark_files:
        console.print("[#CD919E]No benchmark files found[/#CD919E]")
        raise typer.Exit(1)
    
    for bench_file in benchmark_files:
        console.print(f"\n[#EDE5FF]Running {bench_file.name}...[/#EDE5FF]")
        result = subprocess.run(
            [sys.executable, str(bench_file)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            console.print(result.stdout)
        else:
            console.print(f"[#CD919E]Error: {result.stderr}[/#CD919E]")


@app.command()
def run(
    module_id: str = typer.Argument(..., help="Module ID"),
    example: Optional[str] = typer.Option(None, "--example", "-e", help="Run specific example"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Run interactive app"),
):
    """
    Run module examples or interactive app.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    module_path = state["project_root"] / module_id
    
    if interactive:
        app_path = module_path / "app.py"
        if not app_path.exists():
            console.print(f"[#CD919E]No interactive app found for {module_id}[/#CD919E]")
            raise typer.Exit(1)
        
        console.print(f"[#8B7D8B]Launching interactive app...[/#8B7D8B]")
        subprocess.run(["streamlit", "run", str(app_path)])
    else:
        examples_path = module_path / "examples"
        
        if example:
            example_file = examples_path / f"{example}.py"
            if not example_file.exists():
                console.print(f"[#CD919E]Example '{example}' not found[/#CD919E]")
                raise typer.Exit(1)
            
            console.print(f"[#8B7D8B]Running example: {example}[/#8B7D8B]")
            subprocess.run([sys.executable, str(example_file)])
        else:
            # List available examples
            example_files = list(examples_path.glob("*.py"))
            
            if not example_files:
                console.print("[#CD919E]No examples found[/#CD919E]")
                raise typer.Exit(1)
            
            console.print("[#8B7D8B]Available examples:[/#8B7D8B]")
            for ex_file in example_files:
                console.print(f"  - {ex_file.stem}")
            
            console.print("\n[#706B70]Run with: velvet run {module_id} --example {name}[/#706B70]")


@app.command()
def format(
    check: bool = typer.Option(False, "--check", help="Check only, don't modify"),
):
    """
    Format code using black and isort.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    console.print("[#8B7D8B]Formatting code...[/#8B7D8B]")
    
    with Progress(
        SpinnerColumn(style="#DDA0DD"),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Black formatting
        task1 = progress.add_task("Running black...", total=1)
        black_cmd = ["black", str(state["project_root"])]
        if check:
            black_cmd.append("--check")
        
        black_result = subprocess.run(black_cmd, capture_output=True)
        progress.advance(task1)
        
        # isort formatting
        task2 = progress.add_task("Running isort...", total=1)
        isort_cmd = ["isort", str(state["project_root"])]
        if check:
            isort_cmd.append("--check-only")
        
        isort_result = subprocess.run(isort_cmd, capture_output=True)
        progress.advance(task2)
    
    if black_result.returncode == 0 and isort_result.returncode == 0:
        if check:
            console.print("[green]✓[/green] Code formatting is correct!")
        else:
            console.print("[green]✓[/green] Code formatted successfully!")
    else:
        console.print("[#CD919E]✗ Formatting issues found[/#CD919E]")
        if check:
            console.print("[#706B70]Run 'velvet format' to fix[/#706B70]")


@app.command()
def lint():
    """
    Run code quality checks with ruff and mypy.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    console.print("[#8B7D8B]Running linters...[/#8B7D8B]")
    
    # Run ruff
    console.print("\n[#EDE5FF]Running ruff...[/#EDE5FF]")
    ruff_result = subprocess.run(
        ["ruff", "check", str(state["project_root"])],
        capture_output=False
    )
    
    # Run mypy
    console.print("\n[#EDE5FF]Running mypy...[/#EDE5FF]")
    mypy_result = subprocess.run(
        ["mypy", str(state["project_root"])],
        capture_output=False
    )
    
    if ruff_result.returncode == 0 and mypy_result.returncode == 0:
        console.print("\n[green]✓[/green] No linting issues found!")
    else:
        console.print("\n[#CD919E]✗ Linting issues detected[/#CD919E]")


@app.command()
def docs(
    serve: bool = typer.Option(False, "--serve", "-s", help="Serve documentation locally"),
    build: bool = typer.Option(False, "--build", "-b", help="Build documentation"),
):
    """
    Manage project documentation.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    if build:
        console.print("[#8B7D8B]Building documentation...[/#8B7D8B]")
        result = subprocess.run(["mkdocs", "build"], capture_output=False)
        
        if result.returncode == 0:
            console.print("[green]✓[/green] Documentation built successfully!")
            console.print("[#706B70]Output: site/[/#706B70]")
    
    if serve:
        console.print("[#8B7D8B]Serving documentation at http://localhost:8000[/#8B7D8B]")
        console.print("[#706B70]Press Ctrl+C to stop[/#706B70]")
        subprocess.run(["mkdocs", "serve"])


@app.command()
def tree(
    module_id: Optional[str] = typer.Argument(None, help="Show tree for specific module"),
    depth: int = typer.Option(3, "--depth", "-d", help="Tree depth"),
):
    """
    Display project structure as a tree.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    if module_id:
        root = state["project_root"] / module_id
        tree_title = f"[#8B7D8B]{module_id}[/#8B7D8B]"
    else:
        root = state["project_root"]
        tree_title = "[#8B7D8B]Velvet Python[/#8B7D8B]"
    
    if not root.exists():
        console.print(f"[#CD919E]Path not found: {root}[/#CD919E]")
        raise typer.Exit(1)
    
    tree = Tree(tree_title, style="#DDA0DD")
    
    def add_directory(tree_node, path, current_depth=0):
        """Recursively add directories to tree."""
        if current_depth >= depth:
            return
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
            
            for item in items:
                # Skip hidden and cache directories
                if item.name.startswith('.') or item.name == '__pycache__':
                    continue
                
                if item.is_dir():
                    branch = tree_node.add(f"[#FFE4E1]{item.name}/[/#FFE4E1]")
                    add_directory(branch, item, current_depth + 1)
                else:
                    # Color code by file type
                    if item.suffix == '.py':
                        style = "#EDE5FF"
                    elif item.suffix in ['.md', '.txt']:
                        style = "#F0E6FF"
                    elif item.suffix in ['.yml', '.yaml', '.toml']:
                        style = "#FFF0F5"
                    else:
                        style = "#FFEFD5"
                    
                    tree_node.add(f"[{style}]{item.name}[/{style}]")
        except PermissionError:
            pass
    
    add_directory(tree, root)
    console.print(tree)


@app.command()
def progress():
    """
    Show learning progress overview.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    # Calculate progress statistics
    total_modules = len(MODULES)
    completed = sum(1 for m in MODULES.values() if m["status"] == "completed")
    in_progress = sum(1 for m in MODULES.values() if m["status"] == "in-progress")
    planned = sum(1 for m in MODULES.values() if m["status"] == "planned")
    
    # Progress bar
    progress_pct = (completed / total_modules) * 100 if total_modules > 0 else 0
    
    # Create progress visualization
    progress_bar = Progress(
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )
    
    with progress_bar:
        task = progress_bar.add_task(
            f"[#8B7D8B]Overall Progress: {completed}/{total_modules} modules[/#8B7D8B]",
            total=total_modules,
            completed=completed
        )
    
    # Statistics table
    stats_table = Table(show_header=False, box=None, padding=(0, 2))
    stats_table.add_column("Status", style="#8B7D8B")
    stats_table.add_column("Count", style="#706B70")
    stats_table.add_column("Percentage", style="#DDA0DD")
    
    stats_table.add_row(
        "Completed",
        str(completed),
        f"{(completed/total_modules)*100:.1f}%"
    )
    stats_table.add_row(
        "In Progress",
        str(in_progress),
        f"{(in_progress/total_modules)*100:.1f}%"
    )
    stats_table.add_row(
        "Planned",
        str(planned),
        f"{(planned/total_modules)*100:.1f}%"
    )
    
    console.print(Panel(
        stats_table,
        title="[#8B7D8B]Learning Statistics[/#8B7D8B]",
        border_style="#E6E6FA",
        padding=(1, 2),
    ))
    
    # Next recommended module
    next_module = None
    for module_id, info in MODULES.items():
        if info["status"] == "in-progress":
            next_module = module_id
            break
    
    if not next_module:
        for module_id, info in MODULES.items():
            if info["status"] == "planned":
                next_module = module_id
                break
    
    if next_module:
        console.print(
            f"\n[#8B7D8B]Next recommended module:[/#8B7D8B] "
            f"[#FFE4E1]{next_module}[/#FFE4E1] - "
            f"[#EDE5FF]{MODULES[next_module]['name']}[/#EDE5FF]"
        )


@app.command()
def clean(
    cache: bool = typer.Option(False, "--cache", "-c", help="Clean cache files"),
    build: bool = typer.Option(False, "--build", "-b", help="Clean build artifacts"),
    all: bool = typer.Option(False, "--all", "-a", help="Clean everything"),
):
    """
    Clean project artifacts and cache.
    
    Author: Cazzy Aporbo, MS
    """
    print_header()
    
    if all:
        cache = build = True
    
    if not (cache or build):
        console.print("[#706B70]Specify what to clean: --cache, --build, or --all[/#706B70]")
        raise typer.Exit(1)
    
    console.print("[#8B7D8B]Cleaning project...[/#8B7D8B]")
    
    patterns_to_clean = []
    
    if cache:
        patterns_to_clean.extend([
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/.ruff_cache",
            "**/htmlcov",
            "**/.coverage",
        ])
    
    if build:
        patterns_to_clean.extend([
            "build",
            "dist",
            "**/*.egg-info",
            "site",
        ])
    
    removed_count = 0
    
    for pattern in patterns_to_clean:
        for path in state["project_root"].glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed_count += 1
    
    console.print(f"[green]✓[/green] Cleaned {removed_count} items")


@app.command()
def version():
    """
    Show version information.
    
    Author: Cazzy Aporbo, MS
    """
    console.print(create_gradient_text(f"Velvet Python v{__version__}"))
    console.print(f"[#706B70]by {__author__}[/#706B70]")
    console.print(f"[#706B70]Python {sys.version}[/#706B70]")


def main():
    """
    Main CLI entry point.
    
    Author: Cazzy Aporbo, MS
    """
    app()


if __name__ == "__main__":
    main()
