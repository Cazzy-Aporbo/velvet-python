"""
Environment Tools Benchmarking

Author: Cazzy Aporbo, MS
Created: January 2025

I benchmark everything because "fast" is subjective. This module tests
the actual performance of different environment management tools so you
can make informed decisions based on data, not marketing.
"""

import time
import subprocess
import shutil
import tempfile
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
import plotly.graph_objects as go

console = Console()


@dataclass
class BenchmarkResult:
    """
    Stores the result of a benchmark run.
    
    Author: Cazzy Aporbo, MS
    """
    tool: str
    operation: str
    times: List[float]  # List of execution times
    mean_time: float
    median_time: float
    std_dev: float
    min_time: float
    max_time: float
    success_rate: float  # Percentage of successful runs
    
    def __str__(self):
        return (f"{self.tool} - {self.operation}: "
                f"{self.mean_time:.2f}s (±{self.std_dev:.2f}s)")


class EnvironmentBenchmark:
    """
    Benchmarks different Python environment management tools.
    
    I run these benchmarks periodically to track performance changes
    and validate my tool choices. Real numbers beat opinions.
    
    Author: Cazzy Aporbo, MS
    """
    
    TOOLS = ['venv', 'virtualenv', 'pipenv', 'poetry', 'conda', 'uv']
    
    # Test packages for installation benchmarks
    TEST_PACKAGES = [
        'requests',
        'numpy',
        'pandas', 
        'flask',
        'pytest',
        'black',
        'mypy',
        'sqlalchemy',
        'rich',
        'pydantic'
    ]
    
    def __init__(self, iterations: int = 3):
        """
        Initialize the benchmark suite.
        
        Args:
            iterations: Number of times to run each benchmark
        """
        self.iterations = iterations
        self.results: List[BenchmarkResult] = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix='env_bench_'))
        
    def cleanup(self):
        """Clean up temporary directories"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _measure_time(self, func: Callable, *args, **kwargs) -> Tuple[float, bool]:
        """
        Measure execution time of a function.
        
        Returns:
            Tuple of (execution_time, success)
        """
        start_time = time.perf_counter()
        try:
            func(*args, **kwargs)
            success = True
        except Exception as e:
            console.print(f"[yellow]Benchmark failed: {e}[/yellow]")
            success = False
        end_time = time.perf_counter()
        
        return end_time - start_time, success
    
    def _run_benchmark(self, 
                      tool: str, 
                      operation: str,
                      benchmark_func: Callable) -> BenchmarkResult:
        """
        Run a benchmark multiple times and collect statistics.
        
        Args:
            tool: Tool name
            operation: Operation being benchmarked  
            benchmark_func: Function to benchmark
            
        Returns:
            BenchmarkResult with statistics
        """
        times = []
        successes = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task(
                f"Benchmarking {tool} - {operation}",
                total=self.iterations
            )
            
            for i in range(self.iterations):
                exec_time, success = self._measure_time(benchmark_func)
                if success:
                    times.append(exec_time)
                    successes += 1
                progress.advance(task)
        
        if times:
            result = BenchmarkResult(
                tool=tool,
                operation=operation,
                times=times,
                mean_time=statistics.mean(times),
                median_time=statistics.median(times),
                std_dev=statistics.stdev(times) if len(times) > 1 else 0,
                min_time=min(times),
                max_time=max(times),
                success_rate=(successes / self.iterations) * 100
            )
        else:
            # All runs failed
            result = BenchmarkResult(
                tool=tool,
                operation=operation,
                times=[],
                mean_time=float('inf'),
                median_time=float('inf'),
                std_dev=0,
                min_time=float('inf'),
                max_time=float('inf'),
                success_rate=0
            )
        
        self.results.append(result)
        return result
    
    def benchmark_environment_creation(self) -> None:
        """
        Benchmark creating a new virtual environment.
        
        This tests the raw speed of environment creation.
        """
        console.print("\n[cyan]Benchmarking Environment Creation...[/cyan]")
        
        for tool in self.TOOLS:
            # Skip if tool not available
            if not self._check_tool_available(tool):
                console.print(f"[yellow]Skipping {tool} - not installed[/yellow]")
                continue
            
            def create_env():
                env_path = self.temp_dir / f"{tool}_env_{time.time()}"
                
                if tool == 'venv':
                    subprocess.run(
                        ['python', '-m', 'venv', str(env_path)],
                        capture_output=True,
                        check=True
                    )
                
                elif tool == 'virtualenv':
                    subprocess.run(
                        ['virtualenv', str(env_path)],
                        capture_output=True,
                        check=True
                    )
                
                elif tool == 'pipenv':
                    subprocess.run(
                        ['pipenv', 'install'],
                        cwd=str(env_path),
                        capture_output=True,
                        check=True,
                        env={**os.environ, 'PIPENV_VENV_IN_PROJECT': '1'}
                    )
                
                elif tool == 'poetry':
                    # Create pyproject.toml first
                    pyproject = env_path / 'pyproject.toml'
                    env_path.mkdir(exist_ok=True)
                    pyproject.write_text("""
[tool.poetry]
name = "test"
version = "0.1.0"
description = ""
authors = ["Test"]

[tool.poetry.dependencies]
python = "^3.10"
""")
                    subprocess.run(
                        ['poetry', 'install'],
                        cwd=str(env_path),
                        capture_output=True,
                        check=True
                    )
                
                elif tool == 'conda':
                    subprocess.run(
                        ['conda', 'create', '-p', str(env_path), 'python=3.10', '-y'],
                        capture_output=True,
                        check=True
                    )
                
                elif tool == 'uv':
                    subprocess.run(
                        ['uv', 'venv', str(env_path)],
                        capture_output=True,
                        check=True
                    )
                
                # Clean up
                if env_path.exists():
                    shutil.rmtree(env_path, ignore_errors=True)
            
            self._run_benchmark(tool, "Environment Creation", create_env)
    
    def benchmark_package_installation(self) -> None:
        """
        Benchmark installing packages.
        
        This tests how fast each tool can install a standard set of packages.
        """
        console.print("\n[cyan]Benchmarking Package Installation...[/cyan]")
        
        # Create requirements file
        req_file = self.temp_dir / 'requirements.txt'
        req_file.write_text('\n'.join(self.TEST_PACKAGES))
        
        for tool in ['venv', 'uv']:  # Focus on main tools for package installation
            if not self._check_tool_available(tool):
                continue
            
            def install_packages():
                env_path = self.temp_dir / f"{tool}_pkg_env"
                
                # Create environment first
                if tool == 'venv':
                    subprocess.run(
                        ['python', '-m', 'venv', str(env_path)],
                        capture_output=True,
                        check=True
                    )
                    
                    # Install packages
                    pip_path = env_path / ('Scripts' if os.name == 'nt' else 'bin') / 'pip'
                    subprocess.run(
                        [str(pip_path), 'install', '-r', str(req_file)],
                        capture_output=True,
                        check=True
                    )
                
                elif tool == 'uv':
                    subprocess.run(
                        ['uv', 'venv', str(env_path)],
                        capture_output=True,
                        check=True
                    )
                    subprocess.run(
                        ['uv', 'pip', 'install', '-r', str(req_file)],
                        capture_output=True,
                        check=True,
                        env={**os.environ, 'VIRTUAL_ENV': str(env_path)}
                    )
                
                # Clean up
                if env_path.exists():
                    shutil.rmtree(env_path, ignore_errors=True)
            
            self._run_benchmark(tool, "Package Installation", install_packages)
    
    def benchmark_dependency_resolution(self) -> None:
        """
        Benchmark dependency resolution speed.
        
        This tests how fast tools can resolve complex dependency trees.
        """
        console.print("\n[cyan]Benchmarking Dependency Resolution...[/cyan]")
        
        # Create a complex requirements file with potential conflicts
        complex_req = self.temp_dir / 'complex_requirements.txt'
        complex_req.write_text("""
django>=3.0,<4.0
flask>=2.0
requests>=2.25.0
sqlalchemy>=1.4.0
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
tensorflow>=2.6.0
pytest>=6.0.0
""")
        
        for tool in ['pip', 'pipenv', 'poetry']:
            if not self._check_tool_available(tool):
                continue
            
            def resolve_deps():
                work_dir = self.temp_dir / f"{tool}_resolve_{time.time()}"
                work_dir.mkdir(exist_ok=True)
                
                if tool == 'pip':
                    # Use pip-compile if available
                    subprocess.run(
                        ['pip-compile', str(complex_req), '-o', str(work_dir / 'resolved.txt')],
                        capture_output=True,
                        check=True
                    )
                
                elif tool == 'pipenv':
                    # Copy requirements to Pipfile format
                    pipfile = work_dir / 'Pipfile'
                    pipfile.write_text("""
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
django = ">=3.0,<4.0"
flask = ">=2.0"
requests = ">=2.25.0"
""")
                    subprocess.run(
                        ['pipenv', 'lock'],
                        cwd=str(work_dir),
                        capture_output=True,
                        check=True
                    )
                
                elif tool == 'poetry':
                    pyproject = work_dir / 'pyproject.toml'
                    pyproject.write_text("""
[tool.poetry]
name = "test"
version = "0.1.0"
description = ""
authors = ["Test"]

[tool.poetry.dependencies]
python = "^3.10"
django = "^3.0"
flask = "^2.0"
requests = "^2.25.0"
""")
                    subprocess.run(
                        ['poetry', 'lock'],
                        cwd=str(work_dir),
                        capture_output=True,
                        check=True
                    )
                
                # Clean up
                if work_dir.exists():
                    shutil.rmtree(work_dir, ignore_errors=True)
            
            self._run_benchmark(tool, "Dependency Resolution", resolve_deps)
    
    def _check_tool_available(self, tool: str) -> bool:
        """Check if a tool is installed and available"""
        try:
            if tool == 'venv':
                # venv is a module, not a command
                import venv
                return True
            elif tool == 'pip':
                return shutil.which('pip') is not None
            else:
                return shutil.which(tool) is not None
        except ImportError:
            return False
    
    def display_results(self) -> None:
        """Display benchmark results in a beautiful table"""
        if not self.results:
            console.print("[yellow]No benchmark results to display[/yellow]")
            return
        
        # Group results by operation
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = []
            operations[result.operation].append(result)
        
        # Create table for each operation
        for operation, results in operations.items():
            table = Table(title=f"[bold cyan]{operation}[/bold cyan]")
            table.add_column("Tool", style="cyan")
            table.add_column("Mean Time", style="yellow")
            table.add_column("Median Time", style="green")
            table.add_column("Std Dev", style="magenta")
            table.add_column("Min/Max", style="blue")
            table.add_column("Success Rate", style="white")
            
            # Sort by mean time
            results.sort(key=lambda x: x.mean_time)
            
            for result in results:
                if result.success_rate > 0:
                    table.add_row(
                        result.tool,
                        f"{result.mean_time:.2f}s",
                        f"{result.median_time:.2f}s",
                        f"±{result.std_dev:.2f}s",
                        f"{result.min_time:.2f}s/{result.max_time:.2f}s",
                        f"{result.success_rate:.0f}%"
                    )
                else:
                    table.add_row(
                        result.tool,
                        "Failed",
                        "Failed",
                        "-",
                        "-",
                        "0%"
                    )
            
            console.print(table)
            console.print()
    
    def export_results(self, output_path: Path, format: str = 'json') -> bool:
        """
        Export benchmark results for later analysis.
        
        Args:
            output_path: Where to save results
            format: Export format ('json', 'csv')
            
        Returns:
            Success boolean
        """
        try:
            if format == 'json':
                import json
                data = []
                for result in self.results:
                    data.append({
                        'tool': result.tool,
                        'operation': result.operation,
                        'mean_time': result.mean_time,
                        'median_time': result.median_time,
                        'std_dev': result.std_dev,
                        'min_time': result.min_time,
                        'max_time': result.max_time,
                        'success_rate': result.success_rate,
                        'raw_times': result.times
                    })
                
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            elif format == 'csv':
                import csv
                with open(output_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        'Tool', 'Operation', 'Mean Time', 'Median Time',
                        'Std Dev', 'Min Time', 'Max Time', 'Success Rate'
                    ])
                    
                    for result in self.results:
                        writer.writerow([
                            result.tool, result.operation, result.mean_time,
                            result.median_time, result.std_dev, result.min_time,
                            result.max_time, result.success_rate
                        ])
            
            console.print(f"[green]✓ Results exported to {output_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error exporting results: {e}[/red]")
            return False
    
    def create_visualization(self) -> go.Figure:
        """
        Create an interactive visualization of benchmark results.
        
        Returns:
            Plotly figure object
        """
        if not self.results:
            return None
        
        # Group by operation
        operations = {}
        for result in self.results:
            if result.operation not in operations:
                operations[result.operation] = {'tools': [], 'times': []}
            operations[result.operation]['tools'].append(result.tool)
            operations[result.operation]['times'].append(result.mean_time)
        
        # Create subplots
        fig = go.Figure()
        
        colors = ['#FFE4E1', '#E6E6FA', '#F0E6FF', '#FFF0F5', '#FFEFD5', '#DDA0DD']
        
        for i, (operation, data) in enumerate(operations.items()):
            fig.add_trace(go.Bar(
                name=operation,
                x=data['tools'],
                y=data['times'],
                marker_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            title="Environment Tool Benchmarks",
            xaxis_title="Tool",
            yaxis_title="Time (seconds)",
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        return fig


def benchmark_tools(iterations: int = 3) -> None:
    """
    Main benchmark function to run all benchmarks.
    
    Author: Cazzy Aporbo, MS
    """
    console.print(Panel(
        "[bold cyan]Python Environment Tools Benchmark Suite[/bold cyan]\n"
        "by Cazzy Aporbo, MS\n\n"
        "This will test the performance of various Python environment tools.\n"
        "Each benchmark will be run multiple times for accuracy.",
        border_style="cyan"
    ))
    
    benchmark = EnvironmentBenchmark(iterations=iterations)
    
    try:
        # Run benchmarks
        benchmark.benchmark_environment_creation()
        benchmark.benchmark_package_installation()
        benchmark.benchmark_dependency_resolution()
        
        # Display results
        console.print("\n[bold green]Benchmark Results:[/bold green]\n")
        benchmark.display_results()
        
        # Export results
        output_path = Path('benchmark_results.json')
        benchmark.export_results(output_path)
        
    finally:
        # Clean up
        benchmark.cleanup()


def main():
    """
    CLI interface for benchmarking.
    
    Author: Cazzy Aporbo, MS
    """
    import argparse
    import os
    
    parser = argparse.ArgumentParser(
        description="Python Environment Benchmarking by Cazzy Aporbo, MS"
    )
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of iterations per benchmark')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='Output format')
    
    args = parser.parse_args()
    
    benchmark_tools(iterations=args.iterations)


if __name__ == "__main__":
    main()
