"""
Environment Manager - Core functionality for Python environment management

Author: Cazzy Aporbo, MS
Created: March 23 2025

This module handles the creation and management of Python virtual environments.
I built this after getting frustrated with repetitive environment setup tasks.
"""

import os
import sys
import shutil
import subprocess
import platform
import json
import venv
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@dataclass
class VirtualEnvironment:
    """
    Represents a Python virtual environment.
    
    I use this class to track all my project environments and their metadata.
    """
    name: str
    path: Path
    python_version: str
    created_at: datetime
    packages: List[str]
    is_active: bool = False
    
    def __str__(self) -> str:
        """String representation"""
        status = "Active" if self.is_active else "Inactive"
        return f"{self.name} (Python {self.python_version}) - {status}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'path': str(self.path),
            'python_version': self.python_version,
            'created_at': self.created_at.isoformat(),
            'packages': self.packages,
            'is_active': self.is_active
        }


class EnvironmentManager:
    """
    Manages Python virtual environments with best practices.
    
    After years of environment headaches, I built this to automate
    everything I do manually when setting up a new project.
    
    Author: Cazzy Aporbo, MS
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        Initialize the environment manager.
        
        Args:
            base_dir: Base directory for storing environments
        """
        self.base_dir = base_dir or Path.home() / '.velvet_envs'
        self.base_dir.mkdir(exist_ok=True)
        self.config_file = self.base_dir / 'environments.json'
        self.environments = self._load_environments()
        
    def _load_environments(self) -> Dict[str, VirtualEnvironment]:
        """Load saved environment configurations"""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                data = json.load(f)
                
            envs = {}
            for name, env_data in data.items():
                env_data['path'] = Path(env_data['path'])
                env_data['created_at'] = datetime.fromisoformat(env_data['created_at'])
                envs[name] = VirtualEnvironment(**env_data)
            
            return envs
        except Exception as e:
            console.print(f"[red]Error loading environments: {e}[/red]")
            return {}
    
    def _save_environments(self):
        """Save environment configurations"""
        data = {
            name: env.to_dict() 
            for name, env in self.environments.items()
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_environment(
        self,
        name: str,
        python_version: Optional[str] = None,
        requirements: Optional[Path] = None,
        dev_requirements: Optional[Path] = None,
        project_dir: Optional[Path] = None
    ) -> VirtualEnvironment:
        """
        Create a new virtual environment with my standard setup.
        
        This implements all the best practices I've learned:
        - Consistent naming (.venv)
        - Immediate pip upgrade
        - Separate dev dependencies
        - Automatic .gitignore
        
        Args:
            name: Environment name
            python_version: Specific Python version to use
            requirements: Path to requirements.txt
            dev_requirements: Path to requirements-dev.txt
            project_dir: Project directory (defaults to current)
            
        Returns:
            Created VirtualEnvironment instance
        """
        project_dir = project_dir or Path.cwd()
        env_path = project_dir / '.venv'
        
        console.print(f"[cyan]Creating environment '{name}'...[/cyan]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Create virtual environment
            task = progress.add_task("Creating virtual environment...", total=5)
            
            # Step 1: Create venv
            if python_version:
                python_exe = f"python{python_version}"
                if not shutil.which(python_exe):
                    python_exe = "python3"
            else:
                python_exe = sys.executable
            
            venv.create(env_path, with_pip=True, clear=True)
            progress.advance(task)
            
            # Step 2: Get pip path
            if platform.system() == "Windows":
                pip_path = env_path / "Scripts" / "pip.exe"
                python_path = env_path / "Scripts" / "python.exe"
            else:
                pip_path = env_path / "bin" / "pip"
                python_path = env_path / "bin" / "python"
            
            progress.advance(task)
            
            # Step 3: Upgrade pip
            progress.update(task, description="Upgrading pip...")
            subprocess.run(
                [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True,
                text=True
            )
            progress.advance(task)
            
            # Step 4: Install requirements
            installed_packages = []
            
            if requirements and requirements.exists():
                progress.update(task, description="Installing requirements...")
                result = subprocess.run(
                    [str(pip_path), "install", "-r", str(requirements)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    console.print("[green]✓[/green] Requirements installed")
            
            if dev_requirements and dev_requirements.exists():
                progress.update(task, description="Installing dev requirements...")
                result = subprocess.run(
                    [str(pip_path), "install", "-r", str(dev_requirements)],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    console.print("[green]✓[/green] Dev requirements installed")
            
            progress.advance(task)
            
            # Step 5: Get installed packages
            progress.update(task, description="Listing installed packages...")
            result = subprocess.run(
                [str(pip_path), "list", "--format=json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                packages_data = json.loads(result.stdout)
                installed_packages = [f"{p['name']}=={p['version']}" for p in packages_data]
            
            progress.advance(task)
        
        # Get Python version
        result = subprocess.run(
            [str(python_path), "--version"],
            capture_output=True,
            text=True
        )
        py_version = result.stdout.strip().split()[-1] if result.returncode == 0 else "Unknown"
        
        # Create environment record
        env = VirtualEnvironment(
            name=name,
            path=env_path,
            python_version=py_version,
            created_at=datetime.now(),
            packages=installed_packages,
            is_active=False
        )
        
        self.environments[name] = env
        self._save_environments()
        
        # Create .gitignore if it doesn't exist
        gitignore_path = project_dir / '.gitignore'
        if not gitignore_path.exists():
            with open(gitignore_path, 'w') as f:
                f.write("# Virtual environment\n")
                f.write(".venv/\n")
                f.write("venv/\n")
                f.write("env/\n")
                f.write("\n# Python\n")
                f.write("__pycache__/\n")
                f.write("*.pyc\n")
                f.write("*.pyo\n")
                f.write("\n# Environment variables\n")
                f.write(".env\n")
            console.print("[green]✓[/green] Created .gitignore")
        
        console.print(f"[green]✓ Environment '{name}' created successfully![/green]")
        self.show_activation_instructions(env_path)
        
        return env
    
    def show_activation_instructions(self, env_path: Path):
        """Show how to activate the environment"""
        if platform.system() == "Windows":
            activate_cmd = f"{env_path}\\Scripts\\activate"
        else:
            activate_cmd = f"source {env_path}/bin/activate"
        
        instructions = Panel(
            f"[cyan]To activate this environment:[/cyan]\n\n"
            f"[yellow]{activate_cmd}[/yellow]\n\n"
            f"[cyan]To deactivate:[/cyan]\n\n"
            f"[yellow]deactivate[/yellow]",
            title="[bold]Activation Instructions[/bold]",
            border_style="cyan"
        )
        console.print(instructions)
    
    def list_environments(self, detailed: bool = False) -> None:
        """
        List all managed environments.
        
        I use this to keep track of all my project environments.
        """
        if not self.environments:
            console.print("[yellow]No environments found.[/yellow]")
            return
        
        if detailed:
            for name, env in self.environments.items():
                panel = Panel(
                    f"[cyan]Path:[/cyan] {env.path}\n"
                    f"[cyan]Python:[/cyan] {env.python_version}\n"
                    f"[cyan]Created:[/cyan] {env.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                    f"[cyan]Packages:[/cyan] {len(env.packages)}",
                    title=f"[bold]{name}[/bold]",
                    border_style="green" if env.is_active else "white"
                )
                console.print(panel)
        else:
            table = Table(title="Virtual Environments")
            table.add_column("Name", style="cyan")
            table.add_column("Python", style="yellow")
            table.add_column("Created", style="green")
            table.add_column("Packages", style="magenta")
            table.add_column("Status", style="white")
            
            for name, env in self.environments.items():
                status = "[green]Active[/green]" if env.is_active else "Inactive"
                table.add_row(
                    name,
                    env.python_version,
                    env.created_at.strftime('%Y-%m-%d'),
                    str(len(env.packages)),
                    status
                )
            
            console.print(table)
    
    def delete_environment(self, name: str) -> bool:
        """
        Delete a virtual environment.
        
        I added confirmation because I once deleted the wrong environment
        and lost a day's work setting it up again.
        """
        if name not in self.environments:
            console.print(f"[red]Environment '{name}' not found.[/red]")
            return False
        
        env = self.environments[name]
        
        # Confirmation
        console.print(f"[yellow]Warning: This will delete environment '{name}' at {env.path}[/yellow]")
        confirm = console.input("[cyan]Are you sure? (yes/no): [/cyan]")
        
        if confirm.lower() != 'yes':
            console.print("[green]Deletion cancelled.[/green]")
            return False
        
        # Delete the environment directory
        if env.path.exists():
            shutil.rmtree(env.path)
            console.print(f"[green]✓ Deleted environment directory: {env.path}[/green]")
        
        # Remove from registry
        del self.environments[name]
        self._save_environments()
        
        console.print(f"[green]✓ Environment '{name}' deleted successfully.[/green]")
        return True
    
    def compare_environments(self, env1_name: str, env2_name: str) -> None:
        """
        Compare two environments' packages.
        
        I built this after spending hours figuring out why code worked
        in one environment but not another. Usually it's a version mismatch.
        """
        if env1_name not in self.environments:
            console.print(f"[red]Environment '{env1_name}' not found.[/red]")
            return
        
        if env2_name not in self.environments:
            console.print(f"[red]Environment '{env2_name}' not found.[/red]")
            return
        
        env1 = self.environments[env1_name]
        env2 = self.environments[env2_name]
        
        # Parse packages into dicts
        def parse_packages(packages: List[str]) -> Dict[str, str]:
            result = {}
            for pkg in packages:
                if '==' in pkg:
                    name, version = pkg.split('==')
                    result[name] = version
                else:
                    result[pkg] = 'unknown'
            return result
        
        pkgs1 = parse_packages(env1.packages)
        pkgs2 = parse_packages(env2.packages)
        
        # Find differences
        only_in_1 = set(pkgs1.keys()) - set(pkgs2.keys())
        only_in_2 = set(pkgs2.keys()) - set(pkgs1.keys())
        common = set(pkgs1.keys()) & set(pkgs2.keys())
        version_diff = {
            pkg: (pkgs1[pkg], pkgs2[pkg])
            for pkg in common
            if pkgs1[pkg] != pkgs2[pkg]
        }
        
        # Display comparison
        console.print(f"\n[bold]Comparing {env1_name} vs {env2_name}[/bold]\n")
        
        if only_in_1:
            console.print(f"[cyan]Only in {env1_name}:[/cyan]")
            for pkg in sorted(only_in_1):
                console.print(f"  + {pkg}=={pkgs1[pkg]}")
        
        if only_in_2:
            console.print(f"\n[cyan]Only in {env2_name}:[/cyan]")
            for pkg in sorted(only_in_2):
                console.print(f"  + {pkg}=={pkgs2[pkg]}")
        
        if version_diff:
            console.print("\n[cyan]Version differences:[/cyan]")
            for pkg, (v1, v2) in sorted(version_diff.items()):
                console.print(f"  {pkg}: {v1} → {v2}")
        
        if not (only_in_1 or only_in_2 or version_diff):
            console.print("[green]Environments are identical![/green]")
    
    def export_requirements(self, name: str, output_path: Optional[Path] = None) -> bool:
        """
        Export requirements.txt from an environment.
        
        I use this to create reproducible requirements files.
        """
        if name not in self.environments:
            console.print(f"[red]Environment '{name}' not found.[/red]")
            return False
        
        env = self.environments[name]
        output_path = output_path or Path.cwd() / 'requirements.txt'
        
        with open(output_path, 'w') as f:
            for package in sorted(env.packages):
                f.write(f"{package}\n")
        
        console.print(f"[green]✓ Exported requirements to {output_path}[/green]")
        return True


def main():
    """
    CLI interface for environment management.
    
    Author: Cazzy Aporbo, MS
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Python Environment Manager by Cazzy Aporbo, MS"
    )
    parser.add_argument('action', choices=['create', 'list', 'delete', 'compare', 'export'])
    parser.add_argument('--name', help='Environment name')
    parser.add_argument('--requirements', help='Path to requirements.txt')
    parser.add_argument('--dev-requirements', help='Path to requirements-dev.txt')
    parser.add_argument('--detailed', action='store_true', help='Show detailed info')
    parser.add_argument('--env1', help='First environment for comparison')
    parser.add_argument('--env2', help='Second environment for comparison')
    parser.add_argument('--output', help='Output path for export')
    
    args = parser.parse_args()
    
    manager = EnvironmentManager()
    
    if args.action == 'create':
        if not args.name:
            console.print("[red]Error: --name required for create[/red]")
            return
        
        req_path = Path(args.requirements) if args.requirements else None
        dev_req_path = Path(args.dev_requirements) if args.dev_requirements else None
        
        manager.create_environment(
            args.name,
            requirements=req_path,
            dev_requirements=dev_req_path
        )
    
    elif args.action == 'list':
        manager.list_environments(detailed=args.detailed)
    
    elif args.action == 'delete':
        if not args.name:
            console.print("[red]Error: --name required for delete[/red]")
            return
        manager.delete_environment(args.name)
    
    elif args.action == 'compare':
        if not args.env1 or not args.env2:
            console.print("[red]Error: --env1 and --env2 required for compare[/red]")
            return
        manager.compare_environments(args.env1, args.env2)
    
    elif args.action == 'export':
        if not args.name:
            console.print("[red]Error: --name required for export[/red]")
            return
        
        output = Path(args.output) if args.output else None
        manager.export_requirements(args.name, output)


if __name__ == "__main__":
    main()
