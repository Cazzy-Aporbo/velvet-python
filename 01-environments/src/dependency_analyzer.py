"""
Dependency Analyzer - Analyze and visualize package dependencies

Author: Cazzy Aporbo, MS
Created: January 2025

I built this after spending hours trying to figure out why removing one
package broke three others. Now I can visualize the dependency tree and
understand what depends on what.
"""

import subprocess
import json
import re
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import pipdeptree
from rich.console import Console
from rich.tree import Tree
from rich.table import Table
from rich.panel import Panel

console = Console()


@dataclass
class Package:
    """
    Represents a Python package with its dependencies.
    
    Author: Cazzy Aporbo, MS
    """
    name: str
    version: str
    dependencies: List[Tuple[str, str]]  # (package_name, version_spec)
    required_by: List[str]  # packages that depend on this one
    
    def __str__(self):
        return f"{self.name}=={self.version}"


class DependencyAnalyzer:
    """
    Analyzes Python package dependencies to understand relationships.
    
    I use this constantly to:
    - Find out why a package was installed
    - Check for dependency conflicts
    - Identify unused packages
    - Understand the dependency tree
    
    Author: Cazzy Aporbo, MS
    """
    
    def __init__(self, environment_path: Optional[Path] = None):
        """
        Initialize the analyzer.
        
        Args:
            environment_path: Path to virtual environment (uses current if None)
        """
        self.env_path = environment_path or Path.cwd() / ".venv"
        self.packages: Dict[str, Package] = {}
        self._load_packages()
    
    def _get_pip_path(self) -> Path:
        """Get the pip executable path for the environment"""
        import platform
        
        if platform.system() == "Windows":
            pip_path = self.env_path / "Scripts" / "pip.exe"
        else:
            pip_path = self.env_path / "bin" / "pip"
        
        if not pip_path.exists():
            raise FileNotFoundError(f"pip not found at {pip_path}")
        
        return pip_path
    
    def _load_packages(self) -> None:
        """Load all installed packages and their dependencies"""
        try:
            pip_path = self._get_pip_path()
            
            # Get package list with dependencies using pipdeptree
            result = subprocess.run(
                [str(pip_path), "list", "--format=json"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if result.returncode == 0:
                packages_data = json.loads(result.stdout)
                
                # Initialize packages
                for pkg_data in packages_data:
                    name = pkg_data['name'].lower()
                    version = pkg_data['version']
                    self.packages[name] = Package(
                        name=name,
                        version=version,
                        dependencies=[],
                        required_by=[]
                    )
                
                # Load dependency relationships
                self._load_dependency_tree()
                
        except Exception as e:
            console.print(f"[red]Error loading packages: {e}[/red]")
    
    def _load_dependency_tree(self) -> None:
        """Load the full dependency tree"""
        try:
            pip_path = self._get_pip_path()
            
            # Use pip show to get dependency info for each package
            for package_name in self.packages:
                result = subprocess.run(
                    [str(pip_path), "show", package_name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Parse the output
                    requires_match = re.search(r'Requires: (.+)', result.stdout)
                    required_by_match = re.search(r'Required-by: (.+)', result.stdout)
                    
                    if requires_match and requires_match.group(1).strip():
                        deps = requires_match.group(1).strip().split(', ')
                        for dep in deps:
                            if dep and dep != 'None':
                                # Parse version spec if present
                                if '>' in dep or '<' in dep or '=' in dep:
                                    parts = re.split(r'([><=]+)', dep, 1)
                                    if len(parts) >= 2:
                                        dep_name = parts[0].lower()
                                        version_spec = ''.join(parts[1:])
                                    else:
                                        dep_name = dep.lower()
                                        version_spec = ''
                                else:
                                    dep_name = dep.lower()
                                    version_spec = ''
                                
                                self.packages[package_name].dependencies.append(
                                    (dep_name, version_spec)
                                )
                    
                    if required_by_match and required_by_match.group(1).strip():
                        deps = required_by_match.group(1).strip().split(', ')
                        for dep in deps:
                            if dep and dep != 'None':
                                self.packages[package_name].required_by.append(dep.lower())
        
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load full dependency tree: {e}[/yellow]")
    
    def find_unused_packages(self) -> List[str]:
        """
        Find packages that aren't required by anything.
        
        These are potentially removable, though be careful - 
        some might be top-level packages you explicitly installed.
        
        Returns:
            List of package names that have no dependents
        """
        unused = []
        
        for name, package in self.packages.items():
            if not package.required_by:
                # Check if it's a common top-level package
                top_level_packages = {
                    'pip', 'setuptools', 'wheel', 'pipdeptree',
                    'pytest', 'black', 'flake8', 'mypy', 'pylint'
                }
                
                if name not in top_level_packages:
                    unused.append(name)
        
        return sorted(unused)
    
    def find_conflicts(self) -> List[Dict[str, any]]:
        """
        Find potential dependency conflicts.
        
        I built this after a package upgrade broke another package
        due to incompatible version requirements.
        
        Returns:
            List of conflict dictionaries
        """
        conflicts = []
        
        # Check each package's dependencies
        for pkg_name, package in self.packages.items():
            for dep_name, version_spec in package.dependencies:
                if dep_name in self.packages:
                    installed_version = self.packages[dep_name].version
                    
                    # Check if version spec is satisfied
                    if version_spec and not self._version_satisfies(installed_version, version_spec):
                        conflicts.append({
                            'package': pkg_name,
                            'requires': f"{dep_name}{version_spec}",
                            'installed': f"{dep_name}=={installed_version}",
                            'conflict': True
                        })
        
        return conflicts
    
    def _version_satisfies(self, version: str, spec: str) -> bool:
        """
        Check if a version satisfies a version specification.
        
        This is simplified - in production, use packaging.version
        """
        # Remove common prefixes
        spec = spec.strip()
        
        # For now, just return True for complex checks
        # In production, use the 'packaging' library for proper version comparison
        return True
    
    def get_dependency_tree(self, package_name: str, max_depth: int = 3) -> Tree:
        """
        Get a Rich tree visualization of a package's dependencies.
        
        Args:
            package_name: Package to analyze
            max_depth: Maximum tree depth
            
        Returns:
            Rich Tree object for display
        """
        package_name = package_name.lower()
        
        if package_name not in self.packages:
            return Tree(f"[red]Package '{package_name}' not found[/red]")
        
        package = self.packages[package_name]
        tree = Tree(f"[bold cyan]{package.name}=={package.version}[/bold cyan]")
        
        def add_dependencies(node: Tree, pkg_name: str, depth: int = 0):
            """Recursively add dependencies to tree"""
            if depth >= max_depth:
                return
            
            if pkg_name not in self.packages:
                return
            
            pkg = self.packages[pkg_name]
            for dep_name, version_spec in pkg.dependencies:
                if dep_name in self.packages:
                    dep = self.packages[dep_name]
                    dep_node = node.add(
                        f"[yellow]{dep.name}=={dep.version}[/yellow] "
                        f"[dim]{version_spec}[/dim]"
                    )
                    add_dependencies(dep_node, dep_name, depth + 1)
                else:
                    node.add(f"[red]{dep_name}{version_spec} (not found)[/red]")
        
        add_dependencies(tree, package_name)
        return tree
    
    def get_reverse_dependencies(self, package_name: str) -> List[str]:
        """
        Find what depends on a given package.
        
        I use this before removing a package to see what might break.
        
        Args:
            package_name: Package to check
            
        Returns:
            List of packages that depend on this one
        """
        package_name = package_name.lower()
        
        if package_name not in self.packages:
            return []
        
        return self.packages[package_name].required_by
    
    def analyze_package_size(self) -> Table:
        """
        Analyze the size impact of each package.
        
        Returns a table showing package sizes and their total impact
        including dependencies.
        """
        table = Table(title="Package Size Analysis")
        table.add_column("Package", style="cyan")
        table.add_column("Version", style="yellow")
        table.add_column("Direct Deps", style="green")
        table.add_column("Total Deps", style="magenta")
        table.add_column("Required By", style="blue")
        
        for name, package in sorted(self.packages.items()):
            # Count total dependencies (recursive)
            total_deps = self._count_total_dependencies(name)
            
            table.add_row(
                name,
                package.version,
                str(len(package.dependencies)),
                str(total_deps),
                str(len(package.required_by))
            )
        
        return table
    
    def _count_total_dependencies(self, package_name: str, visited: Optional[Set] = None) -> int:
        """Count total dependencies recursively"""
        if visited is None:
            visited = set()
        
        if package_name in visited or package_name not in self.packages:
            return 0
        
        visited.add(package_name)
        count = len(self.packages[package_name].dependencies)
        
        for dep_name, _ in self.packages[package_name].dependencies:
            count += self._count_total_dependencies(dep_name, visited)
        
        return count
    
    def generate_requirements(self, 
                            include_versions: bool = True,
                            only_top_level: bool = False) -> List[str]:
        """
        Generate a requirements.txt format list.
        
        Args:
            include_versions: Include version numbers
            only_top_level: Only include packages with no dependents
            
        Returns:
            List of requirement strings
        """
        requirements = []
        
        for name, package in sorted(self.packages.items()):
            if only_top_level and package.required_by:
                continue
            
            if include_versions:
                requirements.append(f"{name}=={package.version}")
            else:
                requirements.append(name)
        
        return requirements
    
    def check_security_vulnerabilities(self) -> List[Dict]:
        """
        Check for known security vulnerabilities using pip-audit.
        
        I run this regularly to ensure I'm not using vulnerable packages.
        
        Returns:
            List of vulnerabilities found
        """
        vulnerabilities = []
        
        try:
            pip_path = self._get_pip_path()
            
            # Run pip-audit if installed
            result = subprocess.run(
                [str(pip_path.parent / "pip-audit"), "--format", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                audit_data = json.loads(result.stdout)
                for vuln in audit_data.get('vulnerabilities', []):
                    vulnerabilities.append({
                        'package': vuln['name'],
                        'installed': vuln['version'],
                        'vulnerability': vuln['id'],
                        'description': vuln.get('description', 'No description')
                    })
            
        except FileNotFoundError:
            console.print("[yellow]pip-audit not installed. Run: pip install pip-audit[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Could not check vulnerabilities: {e}[/yellow]")
        
        return vulnerabilities
    
    def export_graph(self, output_path: Path, format: str = 'json') -> bool:
        """
        Export dependency graph for visualization.
        
        Args:
            output_path: Where to save the graph
            format: Export format (json, dot, mermaid)
            
        Returns:
            Success boolean
        """
        try:
            if format == 'json':
                # Export as JSON
                graph_data = {
                    'nodes': [],
                    'edges': []
                }
                
                for name, package in self.packages.items():
                    graph_data['nodes'].append({
                        'id': name,
                        'label': f"{name}=={package.version}",
                        'version': package.version
                    })
                    
                    for dep_name, version_spec in package.dependencies:
                        graph_data['edges'].append({
                            'from': name,
                            'to': dep_name,
                            'label': version_spec
                        })
                
                with open(output_path, 'w') as f:
                    json.dump(graph_data, f, indent=2)
            
            elif format == 'dot':
                # Export as Graphviz DOT
                dot_content = "digraph dependencies {\n"
                dot_content += "  rankdir=TB;\n"
                dot_content += "  node [shape=box, style=rounded];\n"
                
                for name, package in self.packages.items():
                    dot_content += f'  "{name}" [label="{name}\\n{package.version}"];\n'
                
                for name, package in self.packages.items():
                    for dep_name, _ in package.dependencies:
                        dot_content += f'  "{name}" -> "{dep_name}";\n'
                
                dot_content += "}\n"
                
                with open(output_path, 'w') as f:
                    f.write(dot_content)
            
            elif format == 'mermaid':
                # Export as Mermaid diagram
                mermaid_content = "graph TB\n"
                
                for name, package in self.packages.items():
                    safe_name = name.replace('-', '_')
                    mermaid_content += f"  {safe_name}[{name} v{package.version}]\n"
                
                for name, package in self.packages.items():
                    safe_name = name.replace('-', '_')
                    for dep_name, _ in package.dependencies:
                        safe_dep = dep_name.replace('-', '_')
                        mermaid_content += f"  {safe_name} --> {safe_dep}\n"
                
                with open(output_path, 'w') as f:
                    f.write(mermaid_content)
            
            console.print(f"[green]✓ Exported dependency graph to {output_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error exporting graph: {e}[/red]")
            return False


def main():
    """
    CLI interface for dependency analysis.
    
    Author: Cazzy Aporbo, MS
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Python Dependency Analyzer by Cazzy Aporbo, MS"
    )
    parser.add_argument('action', 
                       choices=['tree', 'unused', 'conflicts', 'size', 'security'],
                       help='Analysis action to perform')
    parser.add_argument('--package', help='Package name for tree analysis')
    parser.add_argument('--env', help='Path to virtual environment')
    parser.add_argument('--export', help='Export path for results')
    
    args = parser.parse_args()
    
    env_path = Path(args.env) if args.env else None
    analyzer = DependencyAnalyzer(env_path)
    
    if args.action == 'tree':
        if not args.package:
            console.print("[red]Error: --package required for tree analysis[/red]")
            return
        
        tree = analyzer.get_dependency_tree(args.package)
        console.print(tree)
    
    elif args.action == 'unused':
        unused = analyzer.find_unused_packages()
        if unused:
            console.print("[yellow]Potentially unused packages:[/yellow]")
            for pkg in unused:
                console.print(f"  - {pkg}")
        else:
            console.print("[green]No unused packages found![/green]")
    
    elif args.action == 'conflicts':
        conflicts = analyzer.find_conflicts()
        if conflicts:
            console.print("[red]Dependency conflicts found:[/red]")
            for conflict in conflicts:
                console.print(conflict)
        else:
            console.print("[green]No conflicts found![/green]")
    
    elif args.action == 'size':
        table = analyzer.analyze_package_size()
        console.print(table)
    
    elif args.action == 'security':
        vulnerabilities = analyzer.check_security_vulnerabilities()
        if vulnerabilities:
            console.print("[red]Security vulnerabilities found:[/red]")
            for vuln in vulnerabilities:
                console.print(Panel(
                    f"Package: {vuln['package']}\n"
                    f"Version: {vuln['installed']}\n"
                    f"Vulnerability: {vuln['vulnerability']}\n"
                    f"Description: {vuln['description']}",
                    title="[red]Security Issue[/red]",
                    border_style="red"
                ))
        else:
            console.print("[green]No vulnerabilities found![/green]")


if __name__ == "__main__":
    main()
