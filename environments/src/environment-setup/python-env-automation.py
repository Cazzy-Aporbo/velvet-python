#!/usr/bin/env python3

"""
Python Environment Automation and Management System
Purpose: Programmatic control of Python environments with automation features
Features: Auto-creation, dependency resolution, conflict detection, CI/CD integration
Use Case: DevOps automation, CI/CD pipelines, enterprise environment management
"""

import os
import sys
import json
import subprocess
import shutil
import hashlib
import logging
import argparse
import platform
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import venv
import site

# Configure logging with detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('env_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# CONFIGURATION AND CONSTANTS

class EnvType(Enum):
    """Supported environment types"""
    VENV = "venv"
    CONDA = "conda"
    VIRTUALENV = "virtualenv"
    POETRY = "poetry"
    PIPENV = "pipenv"

@dataclass
class EnvironmentConfig:
    """Configuration for a Python environment"""
    name: str
    env_type: EnvType
    python_version: str
    path: Optional[str] = None
    project_path: Optional[str] = None
    dependencies: List[str] = None
    dev_dependencies: List[str] = None
    created_at: Optional[str] = None
    last_activated: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        """Initialize optional fields"""
        if self.dependencies is None:
            self.dependencies = []
        if self.dev_dependencies is None:
            self.dev_dependencies = []
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

# ENVIRONMENT MANAGER CLASS

class PythonEnvironmentManager:
    """
    Comprehensive Python environment management system
    Handles creation, activation, migration, and monitoring of environments
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the environment manager"""
        self.base_dir = Path(base_dir or os.path.expanduser("~/.python_envs"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup paths
        self.registry_file = self.base_dir / "registry.json"
        self.templates_dir = self.base_dir / "templates"
        self.backups_dir = self.base_dir / "backups"
        self.logs_dir = self.base_dir / "logs"
        
        # Create directories
        for dir_path in [self.templates_dir, self.backups_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Load or initialize registry
        self.registry = self._load_registry()
        
        # Detect available tools
        self.available_tools = self._detect_available_tools()
        
        logger.info(f"Environment Manager initialized at {self.base_dir}")
        logger.info(f"Available tools: {', '.join(self.available_tools)}")
    
    def _load_registry(self) -> Dict:
        """Load the environment registry from disk"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize empty registry
            registry = {
                "version": "2.0.0",
                "environments": {},
                "projects": {},
                "templates": {}
            }
            self._save_registry(registry)
            return registry
    
    def _save_registry(self, registry: Dict = None):
        """Save the registry to disk"""
        if registry is None:
            registry = self.registry
        with open(self.registry_file, 'w') as f:
            json.dump(registry, f, indent=2, default=str)
    
    def _detect_available_tools(self) -> Set[str]:
        """Detect which environment management tools are available"""
        tools = set()
        
        # Check for each tool
        tool_commands = {
            "venv": [sys.executable, "-m", "venv", "--help"],
            "conda": ["conda", "--version"],
            "virtualenv": ["virtualenv", "--version"],
            "poetry": ["poetry", "--version"],
            "pipenv": ["pipenv", "--version"],
            "mamba": ["mamba", "--version"]
        }
        
        for tool, command in tool_commands.items():
            try:
                subprocess.run(command, capture_output=True, check=True)
                tools.add(tool)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pass
        
        return tools
    
    # ENVIRONMENT CREATION
    
    def create_environment(
        self, 
        name: str, 
        env_type: str = "venv",
        python_version: str = None,
        requirements_file: str = None,
        template: str = None,
        force: bool = False
    ) -> EnvironmentConfig:
        """
        Create a new Python environment with specified configuration
        
        Args:
            name: Environment name (must be unique)
            env_type: Type of environment (venv, conda, etc.)
            python_version: Python version to use
            requirements_file: Path to requirements file
            template: Template name to use
            force: Force recreation if exists
        
        Returns:
            EnvironmentConfig object for the created environment
        """
        logger.info(f"Creating environment: {name} (type: {env_type})")
        
        # Check if environment already exists
        if name in self.registry["environments"] and not force:
            raise ValueError(f"Environment '{name}' already exists. Use force=True to recreate.")
        
        # Determine Python version
        if python_version is None:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        # Create environment configuration
        env_config = EnvironmentConfig(
            name=name,
            env_type=EnvType(env_type),
            python_version=python_version
        )
        
        # Apply template if specified
        if template:
            env_config = self._apply_template(env_config, template)
        
        # Create the environment based on type
        if env_type == "venv":
            env_path = self._create_venv(name, python_version)
        elif env_type == "conda":
            env_path = self._create_conda_env(name, python_version)
        elif env_type == "virtualenv":
            env_path = self._create_virtualenv(name, python_version)
        else:
            raise ValueError(f"Unsupported environment type: {env_type}")
        
        env_config.path = str(env_path)
        
        # Install requirements if provided
        if requirements_file:
            self._install_requirements(env_config, requirements_file)
        
        # Register the environment
        self.registry["environments"][name] = asdict(env_config)
        self._save_registry()
        
        logger.info(f"Successfully created environment: {name}")
        return env_config
    
    def _create_venv(self, name: str, python_version: str) -> Path:
        """Create a venv environment"""
        env_path = self.base_dir / "venv" / name
        env_path.parent.mkdir(exist_ok=True)
        
        # Find Python executable for specified version
        python_exe = self._find_python_executable(python_version)
        
        # Create virtual environment
        logger.debug(f"Creating venv at {env_path} with {python_exe}")
        venv.create(env_path, with_pip=True, clear=True)
        
        # Upgrade pip and install basic tools
        pip_exe = env_path / "bin" / "pip" if platform.system() != "Windows" else env_path / "Scripts" / "pip.exe"
        
        subprocess.run([str(pip_exe), "install", "--upgrade", "pip", "setuptools", "wheel"], 
                      check=True, capture_output=True)
        
        return env_path
    
    def _create_conda_env(self, name: str, python_version: str) -> Path:
        """Create a conda environment"""
        if "conda" not in self.available_tools:
            raise RuntimeError("Conda is not installed or not in PATH")
        
        # Create conda environment
        cmd = ["conda", "create", "-n", name, f"python={python_version}", "-y"]
        logger.debug(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Get environment path
        result = subprocess.run(["conda", "info", "--envs"], 
                               capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if name in line:
                parts = line.split()
                if parts[0] == name:
                    return Path(parts[-1])
        
        raise RuntimeError(f"Failed to locate conda environment: {name}")
    
    def _create_virtualenv(self, name: str, python_version: str) -> Path:
        """Create a virtualenv environment"""
        if "virtualenv" not in self.available_tools:
            raise RuntimeError("virtualenv is not installed")
        
        env_path = self.base_dir / "virtualenv" / name
        env_path.parent.mkdir(exist_ok=True)
        
        python_exe = self._find_python_executable(python_version)
        
        cmd = ["virtualenv", "-p", python_exe, str(env_path)]
        logger.debug(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, capture_output=True)
        
        return env_path
    
    def _find_python_executable(self, version: str) -> str:
        """Find Python executable for specified version"""
        # Try common executable names
        candidates = [
            f"python{version}",
            f"python{version.replace('.', '')}",
            "python3",
            "python"
        ]
        
        for candidate in candidates:
            exe_path = shutil.which(candidate)
            if exe_path:
                # Verify version
                result = subprocess.run([exe_path, "--version"], 
                                       capture_output=True, text=True)
                if version in result.stdout or version in result.stderr:
                    return exe_path
        
        # If not found, use system Python
        logger.warning(f"Python {version} not found, using system Python")
        return sys.executable
    
    # DEPENDENCY MANAGEMENT
    
    def analyze_dependencies(self, env_name: str) -> Dict:
        """
        Analyze dependencies in an environment
        
        Returns:
            Dictionary with dependency analysis including:
            - Total packages
            - Direct dependencies
            - Dependency tree
            - Potential conflicts
            - Security issues
        """
        logger.info(f"Analyzing dependencies for environment: {env_name}")
        
        env_config = self._get_env_config(env_name)
        
        if env_config.env_type == EnvType.VENV:
            return self._analyze_venv_dependencies(env_config)
        elif env_config.env_type == EnvType.CONDA:
            return self._analyze_conda_dependencies(env_config)
        else:
            raise NotImplementedError(f"Dependency analysis not implemented for {env_config.env_type}")
    
    def _analyze_venv_dependencies(self, env_config: EnvironmentConfig) -> Dict:
        """Analyze dependencies in a venv environment"""
        pip_exe = Path(env_config.path) / "bin" / "pip"
        if platform.system() == "Windows":
            pip_exe = Path(env_config.path) / "Scripts" / "pip.exe"
        
        # Get installed packages
        result = subprocess.run([str(pip_exe), "list", "--format=json"], 
                               capture_output=True, text=True)
        packages = json.loads(result.stdout)
        
        # Get dependency tree using pipdeptree if available
        tree = {}
        try:
            subprocess.run([str(pip_exe), "install", "pipdeptree"], 
                          capture_output=True, check=True)
            result = subprocess.run([str(pip_exe.parent / "pipdeptree"), "--json"], 
                                   capture_output=True, text=True)
            tree = json.loads(result.stdout)
        except:
            logger.warning("Could not generate dependency tree")
        
        # Check for outdated packages
        result = subprocess.run([str(pip_exe), "list", "--outdated", "--format=json"], 
                               capture_output=True, text=True)
        outdated = json.loads(result.stdout) if result.returncode == 0 else []
        
        # Security check with pip-audit if available
        vulnerabilities = []
        try:
            subprocess.run([str(pip_exe), "install", "pip-audit"], 
                          capture_output=True, check=True)
            result = subprocess.run([str(pip_exe.parent / "pip-audit"), "--format=json"], 
                                   capture_output=True, text=True)
            audit_result = json.loads(result.stdout)
            vulnerabilities = audit_result.get("vulnerabilities", [])
        except:
            logger.warning("Could not run security audit")
        
        return {
            "total_packages": len(packages),
            "packages": packages,
            "dependency_tree": tree,
            "outdated": outdated,
            "vulnerabilities": vulnerabilities,
            "analysis_date": datetime.now().isoformat()
        }
    
    def _analyze_conda_dependencies(self, env_config: EnvironmentConfig) -> Dict:
        """Analyze dependencies in a conda environment"""
        # Get package list
        result = subprocess.run(["conda", "list", "-n", env_config.name, "--json"],
                               capture_output=True, text=True)
        packages = json.loads(result.stdout)
        
        # Get dependency tree
        result = subprocess.run(["conda", "tree", "-n", env_config.name],
                               capture_output=True, text=True)
        tree_output = result.stdout if result.returncode == 0 else ""
        
        return {
            "total_packages": len(packages),
            "packages": packages,
            "dependency_tree": tree_output,
            "analysis_date": datetime.now().isoformat()
        }
    
    def resolve_conflicts(self, env_name: str, strategy: str = "conservative") -> List[str]:
        """
        Resolve dependency conflicts in an environment
        
        Args:
            env_name: Environment name
            strategy: Resolution strategy (conservative, aggressive, latest)
        
        Returns:
            List of resolution actions taken
        """
        logger.info(f"Resolving conflicts in {env_name} with strategy: {strategy}")
        
        actions = []
        env_config = self._get_env_config(env_name)
        
        if env_config.env_type == EnvType.VENV:
            pip_exe = Path(env_config.path) / "bin" / "pip"
            if platform.system() == "Windows":
                pip_exe = Path(env_config.path) / "Scripts" / "pip.exe"
            
            # Check for conflicts
            result = subprocess.run([str(pip_exe), "check"], 
                                   capture_output=True, text=True)
            
            if result.returncode != 0:
                conflicts = result.stdout
                logger.warning(f"Found conflicts: {conflicts}")
                
                if strategy == "conservative":
                    # Try to fix with minimal changes
                    subprocess.run([str(pip_exe), "install", "--upgrade", "--force-reinstall", 
                                  "--no-deps", "pip", "setuptools", "wheel"], check=True)
                    actions.append("Reinstalled core packages")
                    
                elif strategy == "aggressive":
                    # Reinstall all packages
                    result = subprocess.run([str(pip_exe), "freeze"], 
                                          capture_output=True, text=True)
                    requirements = result.stdout
                    
                    # Uninstall all
                    subprocess.run([str(pip_exe), "freeze", "|", "xargs", str(pip_exe), 
                                  "uninstall", "-y"], shell=True)
                    actions.append("Uninstalled all packages")
                    
                    # Reinstall
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                        f.write(requirements)
                        temp_req = f.name
                    
                    subprocess.run([str(pip_exe), "install", "-r", temp_req], check=True)
                    os.unlink(temp_req)
                    actions.append("Reinstalled all packages")
                    
                elif strategy == "latest":
                    # Upgrade everything to latest
                    subprocess.run([str(pip_exe), "install", "--upgrade", "--force-reinstall",
                                  "-r", "requirements.txt"], check=True)
                    actions.append("Upgraded all packages to latest versions")
        
        return actions
    
    # ENVIRONMENT OPERATIONS
    
    def activate_environment(self, name: str) -> str:
        """
        Generate activation command for an environment
        
        Returns:
            Shell command to activate the environment
        """
        env_config = self._get_env_config(name)
        
        # Update last activated timestamp
        self.registry["environments"][name]["last_activated"] = datetime.now().isoformat()
        self._save_registry()
        
        if env_config.env_type == EnvType.CONDA:
            return f"conda activate {name}"
        else:
            activate_path = Path(env_config.path) / "bin" / "activate"
            if platform.system() == "Windows":
                activate_path = Path(env_config.path) / "Scripts" / "activate.bat"
            return f"source {activate_path}" if platform.system() != "Windows" else str(activate_path)
    
    def clone_environment(self, source_name: str, target_name: str) -> EnvironmentConfig:
        """
        Clone an existing environment
        
        Args:
            source_name: Name of environment to clone
            target_name: Name for the new environment
        
        Returns:
            EnvironmentConfig for the cloned environment
        """
        logger.info(f"Cloning environment: {source_name} -> {target_name}")
        
        source_config = self._get_env_config(source_name)
        
        # Export requirements from source
        requirements = self.export_requirements(source_name)
        
        # Create new environment with same type and Python version
        new_config = self.create_environment(
            target_name,
            env_type=source_config.env_type.value,
            python_version=source_config.python_version
        )
        
        # Install requirements
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements)
            temp_req = f.name
        
        self._install_requirements(new_config, temp_req)
        os.unlink(temp_req)
        
        # Copy metadata
        new_config.metadata = source_config.metadata.copy()
        new_config.metadata["cloned_from"] = source_name
        new_config.metadata["clone_date"] = datetime.now().isoformat()
        
        self.registry["environments"][target_name] = asdict(new_config)
        self._save_registry()
        
        logger.info(f"Successfully cloned environment: {target_name}")
        return new_config
    
    def migrate_environment(
        self, 
        source_name: str, 
        target_type: str,
        target_name: str = None
    ) -> EnvironmentConfig:
        """
        Migrate an environment to a different type
        
        Args:
            source_name: Source environment name
            target_type: Target environment type (venv, conda, etc.)
            target_name: Name for migrated environment (optional)
        
        Returns:
            EnvironmentConfig for the migrated environment
        """
        logger.info(f"Migrating {source_name} from {self._get_env_config(source_name).env_type} to {target_type}")
        
        source_config = self._get_env_config(source_name)
        
        if target_name is None:
            target_name = f"{source_name}_{target_type}"
        
        # Export dependencies
        requirements = self.export_requirements(source_name)
        
        # Create new environment
        new_config = self.create_environment(
            target_name,
            env_type=target_type,
            python_version=source_config.python_version
        )
        
        # Install requirements with fallback handling
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(requirements)
            temp_req = f.name
        
        try:
            self._install_requirements(new_config, temp_req)
        except subprocess.CalledProcessError as e:
            logger.warning(f"Some packages failed to install: {e}")
            # Try installing packages one by one
            self._install_requirements_individually(new_config, requirements)
        finally:
            os.unlink(temp_req)
        
        # Update metadata
        new_config.metadata["migrated_from"] = source_name
        new_config.metadata["migration_date"] = datetime.now().isoformat()
        
        self.registry["environments"][target_name] = asdict(new_config)
        self._save_registry()
        
        logger.info(f"Successfully migrated to: {target_name}")
        return new_config
    
    # IMPORT/EXPORT
    
    def export_requirements(self, env_name: str, format: str = "pip") -> str:
        """
        Export requirements from an environment
        
        Args:
            env_name: Environment name
            format: Export format (pip, conda, poetry)
        
        Returns:
            Requirements string in specified format
        """
        env_config = self._get_env_config(env_name)
        
        if format == "pip":
            if env_config.env_type == EnvType.VENV:
                pip_exe = Path(env_config.path) / "bin" / "pip"
                if platform.system() == "Windows":
                    pip_exe = Path(env_config.path) / "Scripts" / "pip.exe"
                
                result = subprocess.run([str(pip_exe), "freeze"], 
                                       capture_output=True, text=True)
                return result.stdout
                
            elif env_config.env_type == EnvType.CONDA:
                # Export conda env to pip format
                result = subprocess.run(
                    ["conda", "list", "-n", env_name, "--export"],
                    capture_output=True, text=True
                )
                # Convert conda format to pip format
                lines = []
                for line in result.stdout.splitlines():
                    if not line.startswith("#") and "=" in line:
                        parts = line.split("=")
                        if len(parts) >= 2:
                            lines.append(f"{parts[0]}=={parts[1]}")
                return "\n".join(lines)
        
        elif format == "conda":
            if env_config.env_type == EnvType.CONDA:
                result = subprocess.run(
                    ["conda", "env", "export", "-n", env_name],
                    capture_output=True, text=True
                )
                return result.stdout
            else:
                raise ValueError(f"Cannot export {env_config.env_type} to conda format")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def backup_environment(self, env_name: str) -> str:
        """
        Create a backup of an environment
        
        Returns:
            Path to backup file
        """
        logger.info(f"Creating backup of environment: {env_name}")
        
        env_config = self._get_env_config(env_name)
        
        # Create backup directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backups_dir / env_name / timestamp
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Export requirements
        requirements = self.export_requirements(env_name)
        req_file = backup_dir / "requirements.txt"
        req_file.write_text(requirements)
        
        # Save environment configuration
        config_file = backup_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(env_config), f, indent=2, default=str)
        
        # Create tarball
        tarball_path = self.backups_dir / f"{env_name}_{timestamp}.tar.gz"
        subprocess.run(
            ["tar", "-czf", str(tarball_path), "-C", str(backup_dir.parent), timestamp],
            check=True
        )
        
        # Clean up temporary directory
        shutil.rmtree(backup_dir)
        
        logger.info(f"Backup created: {tarball_path}")
        return str(tarball_path)
    
    # MONITORING AND HEALTH CHECKS
    
    def health_check(self, env_name: str = None) -> Dict:
        """
        Perform health check on environment(s)
        
        Args:
            env_name: Specific environment to check (None for all)
        
        Returns:
            Dictionary with health check results
        """
        if env_name:
            env_names = [env_name]
        else:
            env_names = list(self.registry["environments"].keys())
        
        results = {}
        
        for name in env_names:
            logger.info(f"Checking health of environment: {name}")
            
            try:
                env_config = self._get_env_config(name)
                health = {
                    "exists": False,
                    "python_works": False,
                    "pip_works": False,
                    "packages_count": 0,
                    "has_issues": False,
                    "issues": []
                }
                
                # Check if environment path exists
                env_path = Path(env_config.path)
                health["exists"] = env_path.exists()
                
                if not health["exists"]:
                    health["has_issues"] = True
                    health["issues"].append("Environment path does not exist")
                    results[name] = health
                    continue
                
                # Check Python executable
                if env_config.env_type == EnvType.CONDA:
                    python_exe = "python"
                    cmd_prefix = ["conda", "run", "-n", name]
                else:
                    python_exe = env_path / "bin" / "python"
                    if platform.system() == "Windows":
                        python_exe = env_path / "Scripts" / "python.exe"
                    cmd_prefix = []
                
                # Test Python
                try:
                    cmd = cmd_prefix + [str(python_exe), "--version"] if cmd_prefix else [str(python_exe), "--version"]
                    subprocess.run(cmd, capture_output=True, check=True)
                    health["python_works"] = True
                except:
                    health["has_issues"] = True
                    health["issues"].append("Python executable not working")
                
                # Test pip
                if env_config.env_type == EnvType.VENV:
                    pip_exe = env_path / "bin" / "pip"
                    if platform.system() == "Windows":
                        pip_exe = env_path / "Scripts" / "pip.exe"
                    
                    try:
                        result = subprocess.run([str(pip_exe), "list", "--format=json"],
                                              capture_output=True, text=True, check=True)
                        packages = json.loads(result.stdout)
                        health["pip_works"] = True
                        health["packages_count"] = len(packages)
                    except:
                        health["has_issues"] = True
                        health["issues"].append("pip not working")
                
                results[name] = health
                
            except Exception as e:
                results[name] = {
                    "has_issues": True,
                    "issues": [f"Error during health check: {str(e)}"]
                }
        
        return results
    
    def cleanup_old_environments(self, days: int = 30, dry_run: bool = True) -> List[str]:
        """
        Clean up environments not used for specified number of days
        
        Args:
            days: Number of days of inactivity
            dry_run: If True, only show what would be deleted
        
        Returns:
            List of environment names that were (or would be) deleted
        """
        logger.info(f"Cleaning up environments older than {days} days (dry_run={dry_run})")
        
        cutoff_date = datetime.now() - timedelta(days=days)
        to_delete = []
        
        for name, env_data in self.registry["environments"].items():
            last_activated = env_data.get("last_activated")
            if last_activated:
                last_activated_date = datetime.fromisoformat(last_activated)
            else:
                # Use creation date if never activated
                created = env_data.get("created_at")
                if created:
                    last_activated_date = datetime.fromisoformat(created)
                else:
                    continue
            
            if last_activated_date < cutoff_date:
                to_delete.append(name)
                logger.info(f"Environment '{name}' last used on {last_activated_date.date()}")
        
        if not dry_run:
            for name in to_delete:
                self.delete_environment(name)
        
        logger.info(f"{'Would delete' if dry_run else 'Deleted'} {len(to_delete)} environments")
        return to_delete
    
    def delete_environment(self, name: str, force: bool = False):
        """
        Delete an environment
        
        Args:
            name: Environment name
            force: Force deletion without confirmation
        """
        logger.info(f"Deleting environment: {name}")
        
        env_config = self._get_env_config(name)
        
        # Backup before deletion if not forced
        if not force:
            self.backup_environment(name)
        
        # Delete based on type
        if env_config.env_type == EnvType.CONDA:
            subprocess.run(["conda", "remove", "-n", name, "--all", "-y"], check=True)
        else:
            # Delete directory
            env_path = Path(env_config.path)
            if env_path.exists():
                shutil.rmtree(env_path)
        
        # Remove from registry
        del self.registry["environments"][name]
        self._save_registry()
        
        logger.info(f"Deleted environment: {name}")
    
    # UTILITY METHODS
    
    def _get_env_config(self, name: str) -> EnvironmentConfig:
        """Get environment configuration by name"""
        if name not in self.registry["environments"]:
            raise ValueError(f"Environment '{name}' not found")
        
        env_data = self.registry["environments"][name]
        return EnvironmentConfig(
            name=name,
            env_type=EnvType(env_data["env_type"]),
            python_version=env_data["python_version"],
            path=env_data.get("path"),
            project_path=env_data.get("project_path"),
            dependencies=env_data.get("dependencies", []),
            dev_dependencies=env_data.get("dev_dependencies", []),
            created_at=env_data.get("created_at"),
            last_activated=env_data.get("last_activated"),
            metadata=env_data.get("metadata", {})
        )
    
    def _install_requirements(self, env_config: EnvironmentConfig, requirements_file: str):
        """Install requirements in an environment"""
        logger.info(f"Installing requirements in {env_config.name}")
        
        if env_config.env_type == EnvType.VENV:
            pip_exe = Path(env_config.path) / "bin" / "pip"
            if platform.system() == "Windows":
                pip_exe = Path(env_config.path) / "Scripts" / "pip.exe"
            
            subprocess.run([str(pip_exe), "install", "-r", requirements_file], check=True)
            
        elif env_config.env_type == EnvType.CONDA:
            # Try conda first, fall back to pip
            subprocess.run(["conda", "install", "-n", env_config.name, "--file", 
                          requirements_file, "-y"], capture_output=True)
            
            # Use pip for remaining packages
            subprocess.run(["conda", "run", "-n", env_config.name, "pip", "install", 
                          "-r", requirements_file], check=True)
    
    def _install_requirements_individually(self, env_config: EnvironmentConfig, requirements: str):
        """Install requirements one by one (fallback for migration)"""
        logger.info("Installing requirements individually (fallback mode)")
        
        if env_config.env_type == EnvType.VENV:
            pip_exe = Path(env_config.path) / "bin" / "pip"
            if platform.system() == "Windows":
                pip_exe = Path(env_config.path) / "Scripts" / "pip.exe"
            
            for line in requirements.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    try:
                        subprocess.run([str(pip_exe), "install", line], 
                                     check=True, capture_output=True)
                        logger.debug(f"Installed: {line}")
                    except subprocess.CalledProcessError:
                        logger.warning(f"Failed to install: {line}")
    
    def list_environments(self) -> List[Dict]:
        """List all registered environments"""
        environments = []
        for name, env_data in self.registry["environments"].items():
            environments.append({
                "name": name,
                "type": env_data["env_type"],
                "python_version": env_data["python_version"],
                "created": env_data.get("created_at", "Unknown"),
                "last_activated": env_data.get("last_activated", "Never"),
                "path": env_data.get("path", "Unknown")
            })
        return environments
    
    def get_statistics(self) -> Dict:
        """Get environment statistics"""
        stats = {
            "total_environments": len(self.registry["environments"]),
            "by_type": {},
            "by_python_version": {},
            "total_disk_usage": 0,
            "oldest_environment": None,
            "newest_environment": None,
            "most_used_environment": None
        }
        
        # Count by type and version
        for name, env_data in self.registry["environments"].items():
            env_type = env_data["env_type"]
            python_version = env_data["python_version"]
            
            stats["by_type"][env_type] = stats["by_type"].get(env_type, 0) + 1
            stats["by_python_version"][python_version] = stats["by_python_version"].get(python_version, 0) + 1
            
            # Calculate disk usage
            if "path" in env_data and Path(env_data["path"]).exists():
                size = sum(f.stat().st_size for f in Path(env_data["path"]).rglob("*") if f.is_file())
                stats["total_disk_usage"] += size
        
        # Find oldest/newest
        if self.registry["environments"]:
            sorted_by_created = sorted(
                self.registry["environments"].items(),
                key=lambda x: x[1].get("created_at", ""),
            )
            stats["oldest_environment"] = sorted_by_created[0][0] if sorted_by_created else None
            stats["newest_environment"] = sorted_by_created[-1][0] if sorted_by_created else None
            
            # Find most used
            sorted_by_activated = sorted(
                self.registry["environments"].items(),
                key=lambda x: x[1].get("last_activated", ""),
                reverse=True
            )
            stats["most_used_environment"] = sorted_by_activated[0][0] if sorted_by_activated else None
        
        # Convert disk usage to human readable
        stats["total_disk_usage_human"] = self._format_bytes(stats["total_disk_usage"])
        
        return stats
    
    def _format_bytes(self, bytes: int) -> str:
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024.0:
                return f"{bytes:.2f} {unit}"
            bytes /= 1024.0
        return f"{bytes:.2f} PB"

# CLI INTERFACE

def main():
    """Command-line interface for the environment manager"""
    parser = argparse.ArgumentParser(
        description="Python Environment Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new environment
  %(prog)s create myenv --type venv --python 3.10
  
  # List all environments
  %(prog)s list
  
  # Activate an environment
  %(prog)s activate myenv
  
  # Clone an environment
  %(prog)s clone source_env target_env
  
  # Health check all environments
  %(prog)s health
  
  # Clean up old environments
  %(prog)s cleanup --days 60
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create new environment')
    create_parser.add_argument('name', help='Environment name')
    create_parser.add_argument('--type', default='venv', 
                               choices=['venv', 'conda', 'virtualenv'],
                               help='Environment type')
    create_parser.add_argument('--python', help='Python version')
    create_parser.add_argument('--requirements', help='Requirements file')
    create_parser.add_argument('--template', help='Template name')
    create_parser.add_argument('--force', action='store_true', 
                               help='Force recreation if exists')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all environments')
    list_parser.add_argument('--format', choices=['table', 'json'], 
                             default='table', help='Output format')
    
    # Activate command
    activate_parser = subparsers.add_parser('activate', help='Show activation command')
    activate_parser.add_argument('name', help='Environment name')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete environment')
    delete_parser.add_argument('name', help='Environment name')
    delete_parser.add_argument('--force', action='store_true', 
                               help='Skip backup and confirmation')
    
    # Clone command
    clone_parser = subparsers.add_parser('clone', help='Clone environment')
    clone_parser.add_argument('source', help='Source environment name')
    clone_parser.add_argument('target', help='Target environment name')
    
    # Migrate command
    migrate_parser = subparsers.add_parser('migrate', help='Migrate environment type')
    migrate_parser.add_argument('source', help='Source environment name')
    migrate_parser.add_argument('target_type', 
                                choices=['venv', 'conda', 'virtualenv'],
                                help='Target environment type')
    migrate_parser.add_argument('--name', help='New environment name')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export requirements')
    export_parser.add_argument('name', help='Environment name')
    export_parser.add_argument('--format', choices=['pip', 'conda'], 
                               default='pip', help='Export format')
    export_parser.add_argument('--output', help='Output file')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup environment')
    backup_parser.add_argument('name', help='Environment name')
    
    # Health command
    health_parser = subparsers.add_parser('health', help='Health check')
    health_parser.add_argument('name', nargs='?', help='Environment name (optional)')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean old environments')
    cleanup_parser.add_argument('--days', type=int, default=30, 
                                help='Days of inactivity')
    cleanup_parser.add_argument('--dry-run', action='store_true', 
                                help='Show what would be deleted')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dependencies')
    analyze_parser.add_argument('name', help='Environment name')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = PythonEnvironmentManager()
    
    # Execute command
    try:
        if args.command == 'create':
            env = manager.create_environment(
                args.name,
                env_type=args.type,
                python_version=args.python,
                requirements_file=args.requirements,
                template=args.template,
                force=args.force
            )
            print(f"✅ Created environment: {env.name}")
            
        elif args.command == 'list':
            envs = manager.list_environments()
            if args.format == 'json':
                print(json.dumps(envs, indent=2))
            else:
                if envs:
                    # Print table
                    print("\nEnvironments:")
                    print()
                    print(f"{'Name':<20} {'Type':<10} {'Python':<10} {'Created':<20} {'Last Used':<20}")
                    print()
                    for env in envs:
                        created = env['created'][:10] if env['created'] != 'Unknown' else 'Unknown'
                        last_used = env['last_activated'][:10] if env['last_activated'] not in ['Never', 'Unknown'] else env['last_activated']
                        print(f"{env['name']:<20} {env['type']:<10} {env['python_version']:<10} {created:<20} {last_used:<20}")
                else:
                    print("No environments found")
                    
        elif args.command == 'activate':
            cmd = manager.activate_environment(args.name)
            print(f"To activate {args.name}, run:")
            print(f"\n  {cmd}\n")
            
        elif args.command == 'delete':
            manager.delete_environment(args.name, force=args.force)
            print(f"✅ Deleted environment: {args.name}")
            
        elif args.command == 'clone':
            env = manager.clone_environment(args.source, args.target)
            print(f"✅ Cloned {args.source} to {args.target}")
            
        elif args.command == 'migrate':
            env = manager.migrate_environment(
                args.source,
                args.target_type,
                args.name
            )
            print(f"✅ Migrated {args.source} to {env.name} ({args.target_type})")
            
        elif args.command == 'export':
            requirements = manager.export_requirements(args.name, format=args.format)
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(requirements)
                print(f"✅ Exported requirements to {args.output}")
            else:
                print(requirements)
                
        elif args.command == 'backup':
            backup_path = manager.backup_environment(args.name)
            print(f"✅ Backup created: {backup_path}")
            
        elif args.command == 'health':
            results = manager.health_check(args.name)
            for name, health in results.items():
                status = "❌" if health.get('has_issues') else "✅"
                print(f"\n{status} {name}:")
                if health.get('exists'):
                    print(f"  - Path exists: ✓")
                    if health.get('python_works'):
                        print(f"  - Python works: ✓")
                    if health.get('pip_works'):
                        print(f"  - Pip works: ✓")
                        print(f"  - Packages: {health.get('packages_count', 0)}")
                if health.get('issues'):
                    print(f"  - Issues: {', '.join(health['issues'])}")
                    
        elif args.command == 'cleanup':
            to_delete = manager.cleanup_old_environments(
                days=args.days,
                dry_run=args.dry_run
            )
            if to_delete:
                action = "Would delete" if args.dry_run else "Deleted"
                print(f"{action} {len(to_delete)} environment(s):")
                for name in to_delete:
                    print(f"  - {name}")
            else:
                print("No environments to clean up")
                
        elif args.command == 'stats':
            stats = manager.get_statistics()
            print("\nEnvironment Statistics:")
            print()
            print(f"Total environments: {stats['total_environments']}")
            print(f"Total disk usage: {stats['total_disk_usage_human']}")
            print(f"\nBy type:")
            for env_type, count in stats['by_type'].items():
                print(f"  - {env_type}: {count}")
            print(f"\nBy Python version:")
            for version, count in stats['by_python_version'].items():
                print(f"  - {version}: {count}")
            if stats['oldest_environment']:
                print(f"\nOldest: {stats['oldest_environment']}")
            if stats['newest_environment']:
                print(f"Newest: {stats['newest_environment']}")
            if stats['most_used_environment']:
                print(f"Most used: {stats['most_used_environment']}")
                
        elif args.command == 'analyze':
            analysis = manager.analyze_dependencies(args.name)
            print(f"\nDependency Analysis for {args.name}:")
            print()
            print(f"Total packages: {analysis['total_packages']}")
            if analysis.get('outdated'):
                print(f"Outdated packages: {len(analysis['outdated'])}")
                for pkg in analysis['outdated'][:5]:  # Show first 5
                    print(f"  - {pkg['name']}: {pkg['version']} → {pkg['latest_version']}")
                if len(analysis['outdated']) > 5:
                    print(f"  ... and {len(analysis['outdated']) - 5} more")
            if analysis.get('vulnerabilities'):
                print(f"⚠️  Security vulnerabilities: {len(analysis['vulnerabilities'])}")
                
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()