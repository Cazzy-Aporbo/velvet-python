"""
Configuration Loader - Handle various Python configuration files

Author: Cazzy Aporbo, MS
Created: January 2025

After dealing with setup.py, setup.cfg, pyproject.toml, requirements.txt,
Pipfile, and every other config format, I built this to handle them all
in one place.
"""

import os
import json
import yaml
import toml
import configparser
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv, dotenv_values
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


@dataclass
class ConfigFile:
    """
    Represents a configuration file.
    
    Author: Cazzy Aporbo, MS
    """
    path: Path
    type: str  # 'toml', 'yaml', 'json', 'ini', 'env', 'requirements'
    content: Dict[str, Any]
    raw_content: str
    modified: datetime
    
    def __str__(self):
        return f"{self.path.name} ({self.type})"


class ConfigLoader:
    """
    Unified configuration loader for Python projects.
    
    This handles all the configuration files you'll encounter in Python:
    - pyproject.toml (modern Python projects)
    - setup.py / setup.cfg (legacy packaging)
    - requirements.txt / requirements-dev.txt (dependencies)
    - .env / .env.example (environment variables)
    - config.yml / config.json (application config)
    - Pipfile / Pipfile.lock (pipenv)
    - poetry.lock (poetry)
    
    Author: Cazzy Aporbo, MS
    """
    
    # Standard config file names to look for
    STANDARD_FILES = {
        'pyproject.toml': 'toml',
        'setup.cfg': 'ini',
        'setup.py': 'python',
        'requirements.txt': 'requirements',
        'requirements-dev.txt': 'requirements',
        'requirements-test.txt': 'requirements',
        '.env': 'env',
        '.env.example': 'env',
        'config.yml': 'yaml',
        'config.yaml': 'yaml',
        'config.json': 'json',
        'Pipfile': 'toml',
        'Pipfile.lock': 'json',
        'poetry.lock': 'toml',
        '.python-version': 'text',
        'tox.ini': 'ini',
        'pytest.ini': 'ini',
        '.coveragerc': 'ini',
    }
    
    def __init__(self, project_dir: Optional[Path] = None):
        """
        Initialize the config loader.
        
        Args:
            project_dir: Project directory to scan (defaults to current)
        """
        self.project_dir = project_dir or Path.cwd()
        self.configs: Dict[str, ConfigFile] = {}
        self._scan_configs()
    
    def _scan_configs(self) -> None:
        """Scan for all configuration files in the project"""
        for filename, file_type in self.STANDARD_FILES.items():
            file_path = self.project_dir / filename
            if file_path.exists():
                try:
                    config = self._load_config_file(file_path, file_type)
                    if config:
                        self.configs[filename] = config
                except Exception as e:
                    console.print(f"[yellow]Warning: Could not load {filename}: {e}[/yellow]")
    
    def _load_config_file(self, path: Path, file_type: str) -> Optional[ConfigFile]:
        """Load a configuration file based on its type"""
        try:
            raw_content = path.read_text(encoding='utf-8')
            content = {}
            
            if file_type == 'toml':
                content = toml.loads(raw_content)
            
            elif file_type == 'yaml':
                content = yaml.safe_load(raw_content)
            
            elif file_type == 'json':
                content = json.loads(raw_content)
            
            elif file_type == 'ini':
                parser = configparser.ConfigParser()
                parser.read(path)
                content = {section: dict(parser.items(section)) 
                          for section in parser.sections()}
            
            elif file_type == 'env':
                content = dotenv_values(path)
            
            elif file_type == 'requirements':
                # Parse requirements file
                lines = raw_content.strip().split('\n')
                requirements = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        requirements.append(line)
                content = {'packages': requirements}
            
            elif file_type == 'text':
                content = {'content': raw_content.strip()}
            
            elif file_type == 'python':
                # For setup.py, we just store the raw content
                # Parsing Python code is complex and potentially dangerous
                content = {'raw': raw_content}
            
            return ConfigFile(
                path=path,
                type=file_type,
                content=content,
                raw_content=raw_content,
                modified=datetime.fromtimestamp(path.stat().st_mtime)
            )
            
        except Exception as e:
            console.print(f"[red]Error loading {path}: {e}[/red]")
            return None
    
    def get_project_metadata(self) -> Dict[str, Any]:
        """
        Extract project metadata from various config files.
        
        I use this to get a unified view of project information
        regardless of which config format is used.
        
        Returns:
            Dictionary with project metadata
        """
        metadata = {
            'name': None,
            'version': None,
            'description': None,
            'author': None,
            'license': None,
            'python_version': None,
            'dependencies': [],
            'dev_dependencies': [],
        }
        
        # Try pyproject.toml first (modern standard)
        if 'pyproject.toml' in self.configs:
            pyproject = self.configs['pyproject.toml'].content
            if 'project' in pyproject:
                project = pyproject['project']
                metadata['name'] = project.get('name')
                metadata['version'] = project.get('version')
                metadata['description'] = project.get('description')
                
                # Author might be a list of dicts
                authors = project.get('authors', [])
                if authors and isinstance(authors, list):
                    metadata['author'] = authors[0].get('name') if isinstance(authors[0], dict) else str(authors[0])
                
                metadata['license'] = project.get('license')
                metadata['python_version'] = project.get('requires-python')
                metadata['dependencies'] = project.get('dependencies', [])
                
                # Dev dependencies might be in optional-dependencies
                optional = project.get('optional-dependencies', {})
                metadata['dev_dependencies'] = optional.get('dev', [])
        
        # Fall back to setup.cfg
        elif 'setup.cfg' in self.configs:
            setup_cfg = self.configs['setup.cfg'].content
            if 'metadata' in setup_cfg:
                meta = setup_cfg['metadata']
                metadata['name'] = meta.get('name')
                metadata['version'] = meta.get('version')
                metadata['description'] = meta.get('description')
                metadata['author'] = meta.get('author')
                metadata['license'] = meta.get('license')
            
            if 'options' in setup_cfg:
                options = setup_cfg['options']
                metadata['python_version'] = options.get('python_requires')
                metadata['dependencies'] = options.get('install_requires', '').split('\n')
        
        # Get Python version from .python-version if not found
        if not metadata['python_version'] and '.python-version' in self.configs:
            metadata['python_version'] = self.configs['.python-version'].content.get('content')
        
        # Get dependencies from requirements.txt if not found
        if not metadata['dependencies'] and 'requirements.txt' in self.configs:
            metadata['dependencies'] = self.configs['requirements.txt'].content.get('packages', [])
        
        # Get dev dependencies from requirements-dev.txt
        if not metadata['dev_dependencies'] and 'requirements-dev.txt' in self.configs:
            metadata['dev_dependencies'] = self.configs['requirements-dev.txt'].content.get('packages', [])
        
        return metadata
    
    def get_environment_variables(self) -> Dict[str, str]:
        """
        Get all environment variables from .env files.
        
        Returns:
            Dictionary of environment variables
        """
        env_vars = {}
        
        # Load .env file
        if '.env' in self.configs:
            env_vars.update(self.configs['.env'].content)
        
        # Also check .env.example for expected variables
        expected_vars = {}
        if '.env.example' in self.configs:
            expected_vars = self.configs['.env.example'].content
        
        # Warn about missing expected variables
        for key in expected_vars:
            if key not in env_vars:
                console.print(f"[yellow]Warning: Expected env var '{key}' not found in .env[/yellow]")
        
        return env_vars
    
    def validate_configs(self) -> List[Dict[str, str]]:
        """
        Validate configuration files for common issues.
        
        I built this after too many times of "why isn't this working?"
        only to find a typo in a config file.
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check for conflicting Python versions
        python_versions = set()
        
        if 'pyproject.toml' in self.configs:
            pyproject = self.configs['pyproject.toml'].content
            if 'project' in pyproject and 'requires-python' in pyproject['project']:
                python_versions.add(pyproject['project']['requires-python'])
        
        if '.python-version' in self.configs:
            python_versions.add(self.configs['.python-version'].content.get('content'))
        
        if len(python_versions) > 1:
            issues.append({
                'type': 'warning',
                'message': f'Multiple Python versions specified: {python_versions}'
            })
        
        # Check for missing required files
        if 'pyproject.toml' not in self.configs and 'setup.py' not in self.configs:
            issues.append({
                'type': 'warning',
                'message': 'No package configuration found (pyproject.toml or setup.py)'
            })
        
        # Check for both Pipfile and requirements.txt
        if 'Pipfile' in self.configs and 'requirements.txt' in self.configs:
            issues.append({
                'type': 'info',
                'message': 'Both Pipfile and requirements.txt found - consider using just one'
            })
        
        # Check .env vs .env.example
        if '.env.example' in self.configs and '.env' not in self.configs:
            issues.append({
                'type': 'error',
                'message': '.env.example exists but .env is missing - create .env from template'
            })
        
        # Check for uncommitted secrets
        if '.env' in self.configs:
            # Check if .gitignore exists and includes .env
            gitignore_path = self.project_dir / '.gitignore'
            if gitignore_path.exists():
                gitignore_content = gitignore_path.read_text()
                if '.env' not in gitignore_content:
                    issues.append({
                        'type': 'error',
                        'message': '.env file exists but not in .gitignore - SECURITY RISK!'
                    })
        
        return issues
    
    def display_config_summary(self) -> None:
        """Display a beautiful summary of all configurations"""
        console.print("\n[bold cyan]Configuration Files Found:[/bold cyan]\n")
        
        for filename, config in self.configs.items():
            # Create a panel for each config file
            content_preview = config.raw_content[:200] + "..." if len(config.raw_content) > 200 else config.raw_content
            
            syntax = Syntax(
                content_preview,
                lexer=config.type if config.type != 'requirements' else 'text',
                theme='monokai',
                line_numbers=False
            )
            
            panel = Panel(
                syntax,
                title=f"[bold]{filename}[/bold]",
                subtitle=f"Modified: {config.modified.strftime('%Y-%m-%d %H:%M')}",
                border_style="cyan"
            )
            console.print(panel)
        
        # Show extracted metadata
        metadata = self.get_project_metadata()
        if metadata['name']:
            console.print("\n[bold green]Project Metadata:[/bold green]")
            for key, value in metadata.items():
                if value:
                    if isinstance(value, list):
                        value = f"{len(value)} items"
                    console.print(f"  [cyan]{key}:[/cyan] {value}")
        
        # Show validation issues
        issues = self.validate_configs()
        if issues:
            console.print("\n[bold yellow]Configuration Issues:[/bold yellow]")
            for issue in issues:
                icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(issue['type'], '•')
                console.print(f"  {icon} {issue['message']}")
    
    def merge_requirements(self) -> List[str]:
        """
        Merge requirements from all sources.
        
        Useful for getting a complete picture of all dependencies.
        
        Returns:
            Unified list of requirements
        """
        all_requirements = set()
        
        # From pyproject.toml
        metadata = self.get_project_metadata()
        all_requirements.update(metadata.get('dependencies', []))
        all_requirements.update(metadata.get('dev_dependencies', []))
        
        # From requirements files
        for filename in ['requirements.txt', 'requirements-dev.txt', 'requirements-test.txt']:
            if filename in self.configs:
                all_requirements.update(
                    self.configs[filename].content.get('packages', [])
                )
        
        # From Pipfile
        if 'Pipfile' in self.configs:
            pipfile = self.configs['Pipfile'].content
            if 'packages' in pipfile:
                all_requirements.update(pipfile['packages'].keys())
            if 'dev-packages' in pipfile:
                all_requirements.update(pipfile['dev-packages'].keys())
        
        return sorted(list(all_requirements))
    
    def export_unified_config(self, output_path: Path, format: str = 'toml') -> bool:
        """
        Export a unified configuration file.
        
        I use this when migrating between config formats.
        
        Args:
            output_path: Where to save the unified config
            format: Output format ('toml', 'json', 'yaml')
            
        Returns:
            Success boolean
        """
        try:
            # Build unified config
            unified = {
                'project': self.get_project_metadata(),
                'environment': self.get_environment_variables(),
                'files': {
                    filename: {
                        'type': config.type,
                        'modified': config.modified.isoformat()
                    }
                    for filename, config in self.configs.items()
                }
            }
            
            # Export based on format
            if format == 'toml':
                with open(output_path, 'w') as f:
                    toml.dump(unified, f)
            
            elif format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(unified, f, indent=2, default=str)
            
            elif format == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.safe_dump(unified, f, default_flow_style=False, default=str)
            
            console.print(f"[green]✓ Exported unified config to {output_path}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error exporting config: {e}[/red]")
            return False


def main():
    """
    CLI interface for config loading.
    
    Author: Cazzy Aporbo, MS
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Python Config Loader by Cazzy Aporbo, MS"
    )
    parser.add_argument('action', 
                       choices=['scan', 'validate', 'metadata', 'export'],
                       help='Action to perform')
    parser.add_argument('--dir', help='Project directory to scan')
    parser.add_argument('--output', help='Output path for export')
    parser.add_argument('--format', 
                       choices=['toml', 'json', 'yaml'],
                       default='toml',
                       help='Export format')
    
    args = parser.parse_args()
    
    project_dir = Path(args.dir) if args.dir else None
    loader = ConfigLoader(project_dir)
    
    if args.action == 'scan':
        loader.display_config_summary()
    
    elif args.action == 'validate':
        issues = loader.validate_configs()
        if issues:
            console.print("[yellow]Validation issues found:[/yellow]")
            for issue in issues:
                console.print(f"  [{issue['type']}] {issue['message']}")
        else:
            console.print("[green]✓ No configuration issues found![/green]")
    
    elif args.action == 'metadata':
        metadata = loader.get_project_metadata()
        console.print("[cyan]Project Metadata:[/cyan]")
        for key, value in metadata.items():
            if value:
                console.print(f"  {key}: {value}")
    
    elif args.action == 'export':
        if not args.output:
            console.print("[red]Error: --output required for export[/red]")
            return
        
        loader.export_unified_config(Path(args.output), args.format)


if __name__ == "__main__":
    main()
