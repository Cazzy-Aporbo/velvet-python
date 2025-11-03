#!/usr/bin/env python3
"""
Advanced Environment Management System for Velvet Python
Author: Cazandra Aporbo
Version: 3.0.0

More comprehensive environment setup with automatic dependency resolution,
virtual environment management, and intelligent package installation.
Handles conda, pip, poetry, and system-level dependencies.
"""

import os
import sys
import subprocess
import json
import platform
import venv
import site
import logging
import hashlib
import tempfile
import shutil
import urllib.request
import urllib.error
import ssl
import re
import importlib.util
import importlib.metadata
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('environment_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PackageSpec:
    """Specification for a package with version constraints and metadata."""
    name: str
    version: Optional[str] = None
    extras: List[str] = field(default_factory=list)
    markers: Optional[str] = None
    source: str = "pypi"  # pypi, conda, github, local
    optional: bool = False
    gpu_only: bool = False
    description: str = ""
    
    def to_pip_spec(self) -> str:
        """Convert to pip installation specification."""
        spec = self.name
        if self.extras:
            spec += f"[{','.join(self.extras)}]"
        if self.version:
            spec += self.version
        if self.markers:
            spec += f"; {self.markers}"
        return spec


@dataclass
class EnvironmentConfig:
    """Configuration for a complete Python environment."""
    name: str
    python_version: str
    packages: List[PackageSpec]
    system_packages: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    post_install_scripts: List[str] = field(default_factory=list)
    cuda_version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage."""
        return {
            'name': self.name,
            'python_version': self.python_version,
            'packages': [
                {
                    'name': p.name,
                    'version': p.version,
                    'extras': p.extras,
                    'source': p.source,
                    'optional': p.optional
                } for p in self.packages
            ],
            'system_packages': self.system_packages,
            'environment_variables': self.environment_variables,
            'cuda_version': self.cuda_version,
            'created_at': self.created_at.isoformat()
        }


class DependencyResolver:
    """Resolves package dependencies and handles conflicts."""
    
    def __init__(self):
        self.dependency_graph = defaultdict(set)
        self.version_constraints = defaultdict(list)
        self.resolved_versions = {}
        self.conflict_resolution_strategy = "latest"  # latest, stable, minimal
        
    def add_package(self, package: PackageSpec):
        """Add a package to the dependency graph."""
        try:
            # Get package metadata from PyPI
            metadata = self._fetch_package_metadata(package.name)
            if metadata:
                deps = metadata.get('requires_dist', [])
                for dep in deps:
                    dep_name = self._parse_dependency_name(dep)
                    self.dependency_graph[package.name].add(dep_name)
                    self.version_constraints[dep_name].append(dep)
        except Exception as e:
            logger.warning(f"Could not fetch metadata for {package.name}: {e}")
    
    def _fetch_package_metadata(self, package_name: str) -> Optional[Dict]:
        """Fetch package metadata from PyPI JSON API."""
        url = f"https://pypi.org/pypi/{package_name}/json"
        try:
            context = ssl.create_default_context()
            with urllib.request.urlopen(url, context=context, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get('info', {})
        except Exception as e:
            logger.debug(f"Error fetching metadata for {package_name}: {e}")
            return None
    
    def _parse_dependency_name(self, dep_spec: str) -> str:
        """Extract package name from dependency specification."""
        # Handle specs like 'package>=1.0,<2.0; python_version >= "3.6"'
        match = re.match(r'^([a-zA-Z0-9_-]+)', dep_spec)
        return match.group(1) if match else dep_spec
    
    def resolve_dependencies(self, packages: List[PackageSpec]) -> List[PackageSpec]:
        """Resolve all dependencies and return ordered installation list."""
        # Build dependency graph
        for package in packages:
            self.add_package(package)
        
        # Topological sort for installation order
        resolved = []
        visited = set()
        
        def visit(name: str):
            if name in visited:
                return
            visited.add(name)
            for dep in self.dependency_graph.get(name, []):
                visit(dep)
            # Create PackageSpec for dependency
            spec = PackageSpec(name=name)
            if name in self.version_constraints:
                # Parse version constraints and choose appropriate version
                spec.version = self._resolve_version_constraint(
                    self.version_constraints[name]
                )
            resolved.append(spec)
        
        # Visit all packages
        for package in packages:
            visit(package.name)
        
        return resolved
    
    def _resolve_version_constraint(self, constraints: List[str]) -> str:
        """Resolve multiple version constraints to a single specification."""
        # Simplified version resolution
        # In production, use packaging.specifiers for proper resolution
        if not constraints:
            return ""
        
        # Extract version specifications
        specs = []
        for constraint in constraints:
            match = re.search(r'([><=!]+[\d.]+)', constraint)
            if match:
                specs.append(match.group(1))
        
        # Combine specifications (simplified)
        if specs:
            return ','.join(specs)
        return ""


class PackageInstaller:
    """Handles package installation with multiple backends."""
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.pip_cache_dir = Path.home() / '.cache' / 'pip'
        self.installed_packages = set()
        self._lock = threading.Lock()
        
    def install_packages(self, packages: List[PackageSpec], 
                        parallel: bool = True,
                        upgrade: bool = False) -> Dict[str, bool]:
        """Install multiple packages with optional parallelization."""
        results = {}
        
        if parallel and len(packages) > 1:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._install_single, pkg, upgrade): pkg
                    for pkg in packages
                }
                
                for future in as_completed(futures):
                    pkg = futures[future]
                    try:
                        success = future.result()
                        results[pkg.name] = success
                    except Exception as e:
                        logger.error(f"Failed to install {pkg.name}: {e}")
                        results[pkg.name] = False
        else:
            for pkg in packages:
                results[pkg.name] = self._install_single(pkg, upgrade)
        
        return results
    
    def _install_single(self, package: PackageSpec, upgrade: bool = False) -> bool:
        """Install a single package."""
        # Check if already installed
        if not upgrade and self._is_installed(package):
            logger.info(f"{package.name} is already installed")
            return True
        
        # Choose installation method based on source
        if package.source == "conda":
            return self._install_conda(package, upgrade)
        elif package.source == "github":
            return self._install_github(package, upgrade)
        else:
            return self._install_pip(package, upgrade)
    
    def _is_installed(self, package: PackageSpec) -> bool:
        """Check if a package is installed."""
        try:
            # Try to import the package
            spec = importlib.util.find_spec(package.name.replace('-', '_'))
            if spec is not None:
                # Check version if specified
                if package.version:
                    try:
                        installed_version = importlib.metadata.version(package.name)
                        # Simplified version check
                        return True  # Would need proper version comparison
                    except:
                        pass
                return True
        except (ImportError, ModuleNotFoundError):
            pass
        return False
    
    def _install_pip(self, package: PackageSpec, upgrade: bool) -> bool:
        """Install package using pip."""
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if upgrade:
            cmd.append("--upgrade")
        
        if self.use_cache:
            cmd.extend(["--cache-dir", str(self.pip_cache_dir)])
        
        # Add package specification
        cmd.append(package.to_pip_spec())
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info(f"Successfully installed {package.name}")
                with self._lock:
                    self.installed_packages.add(package.name)
                return True
            else:
                logger.error(f"Failed to install {package.name}: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"Installation of {package.name} timed out")
            return False
        except Exception as e:
            logger.error(f"Error installing {package.name}: {e}")
            return False
    
    def _install_conda(self, package: PackageSpec, upgrade: bool) -> bool:
        """Install package using conda."""
        # Check if conda is available
        conda_exe = shutil.which("conda")
        if not conda_exe:
            logger.warning("Conda not found, falling back to pip")
            return self._install_pip(package, upgrade)
        
        cmd = [conda_exe, "install", "-y"]
        
        if upgrade:
            cmd.append("--update-deps")
        
        # Add package specification
        pkg_spec = package.name
        if package.version:
            pkg_spec += package.version
        cmd.append(pkg_spec)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Conda installation failed for {package.name}: {e}")
            return False
    
    def _install_github(self, package: PackageSpec, upgrade: bool) -> bool:
        """Install package from GitHub."""
        # Format: git+https://github.com/user/repo.git@branch
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if upgrade:
            cmd.append("--upgrade")
        
        # Construct GitHub URL
        if package.name.startswith("git+"):
            url = package.name
        else:
            # Assume format: user/repo or user/repo@branch
            url = f"git+https://github.com/{package.name}.git"
            if package.version:
                url += f"@{package.version}"
        
        cmd.append(url)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"GitHub installation failed for {package.name}: {e}")
            return False


class VirtualEnvironmentManager:
    """Manages Python virtual environments."""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.home() / '.velvet_envs'
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.environments = {}
        
    def create_environment(self, name: str, python_version: Optional[str] = None) -> Path:
        """Create a new virtual environment."""
        env_path = self.base_dir / name
        
        if env_path.exists():
            logger.warning(f"Environment {name} already exists")
            return env_path
        
        # Find Python executable for specified version
        python_exe = self._find_python_executable(python_version)
        
        # Create virtual environment
        logger.info(f"Creating virtual environment: {name}")
        venv.create(env_path, with_pip=True, system_site_packages=False)
        
        # Upgrade pip, setuptools, wheel
        self._upgrade_base_packages(env_path)
        
        self.environments[name] = env_path
        return env_path
    
    def _find_python_executable(self, version: Optional[str]) -> str:
        """Find Python executable for specified version."""
        if not version:
            return sys.executable
        
        # Try common Python executable names
        candidates = [
            f"python{version}",
            f"python{version.replace('.', '')}",
            "python3",
            "python"
        ]
        
        for candidate in candidates:
            exe = shutil.which(candidate)
            if exe:
                # Verify version
                try:
                    result = subprocess.run(
                        [exe, "--version"],
                        capture_output=True,
                        text=True
                    )
                    if version in result.stdout:
                        return exe
                except:
                    continue
        
        logger.warning(f"Python {version} not found, using current Python")
        return sys.executable
    
    def _upgrade_base_packages(self, env_path: Path):
        """Upgrade pip, setuptools, and wheel in the environment."""
        pip_exe = env_path / "bin" / "pip" if platform.system() != "Windows" else env_path / "Scripts" / "pip.exe"
        
        packages = ["pip", "setuptools", "wheel"]
        for pkg in packages:
            try:
                subprocess.run(
                    [str(pip_exe), "install", "--upgrade", pkg],
                    capture_output=True,
                    timeout=60
                )
            except Exception as e:
                logger.warning(f"Failed to upgrade {pkg}: {e}")
    
    def activate_environment(self, name: str) -> Dict[str, str]:
        """Get environment variables to activate an environment."""
        env_path = self.environments.get(name) or self.base_dir / name
        
        if not env_path.exists():
            raise ValueError(f"Environment {name} does not exist")
        
        # Prepare environment variables
        env_vars = os.environ.copy()
        
        if platform.system() == "Windows":
            scripts_dir = env_path / "Scripts"
            env_vars["PATH"] = f"{scripts_dir};{env_vars['PATH']}"
            env_vars["VIRTUAL_ENV"] = str(env_path)
        else:
            bin_dir = env_path / "bin"
            env_vars["PATH"] = f"{bin_dir}:{env_vars['PATH']}"
            env_vars["VIRTUAL_ENV"] = str(env_path)
        
        # Unset PYTHONHOME if set
        env_vars.pop("PYTHONHOME", None)
        
        return env_vars
    
    def remove_environment(self, name: str):
        """Remove a virtual environment."""
        env_path = self.environments.get(name) or self.base_dir / name
        
        if env_path.exists():
            logger.info(f"Removing environment: {name}")
            shutil.rmtree(env_path)
            if name in self.environments:
                del self.environments[name]
        else:
            logger.warning(f"Environment {name} does not exist")


class EnvironmentSetup:
    """Main class for comprehensive environment setup."""
    
    def __init__(self):
        self.venv_manager = VirtualEnvironmentManager()
        self.installer = PackageInstaller()
        self.resolver = DependencyResolver()
        self.configs = {}
        
    def create_ml_environment(self) -> EnvironmentConfig:
        """Create a comprehensive machine learning environment."""
        packages = [
            # Core scientific computing
            PackageSpec("numpy", ">=1.24.0", description="Numerical computing"),
            PackageSpec("pandas", ">=2.0.0", description="Data manipulation"),
            PackageSpec("scipy", ">=1.10.0", description="Scientific computing"),
            
            # Machine learning frameworks
            PackageSpec("scikit-learn", ">=1.3.0", description="Classical ML"),
            PackageSpec("xgboost", ">=2.0.0", description="Gradient boosting"),
            PackageSpec("lightgbm", ">=4.0.0", description="Gradient boosting"),
            PackageSpec("catboost", ">=1.2", description="Gradient boosting"),
            
            # Deep learning
            PackageSpec("torch", ">=2.0.0", gpu_only=True, description="PyTorch"),
            PackageSpec("tensorflow", ">=2.13.0", gpu_only=True, description="TensorFlow"),
            PackageSpec("transformers", ">=4.30.0", description="Transformers"),
            PackageSpec("accelerate", ">=0.20.0", description="Training acceleration"),
            
            # Computer vision
            PackageSpec("opencv-python", ">=4.8.0", description="Computer vision"),
            PackageSpec("pillow", ">=10.0.0", description="Image processing"),
            PackageSpec("albumentations", ">=1.3.0", description="Image augmentation"),
            
            # NLP
            PackageSpec("spacy", ">=3.6.0", description="NLP library"),
            PackageSpec("nltk", ">=3.8.0", description="Natural language toolkit"),
            PackageSpec("gensim", ">=4.3.0", description="Topic modeling"),
            
            # Visualization
            PackageSpec("matplotlib", ">=3.7.0", description="Plotting"),
            PackageSpec("seaborn", ">=0.12.0", description="Statistical plots"),
            PackageSpec("plotly", ">=5.15.0", description="Interactive plots"),
            PackageSpec("bokeh", ">=3.2.0", description="Interactive visualization"),
            
            # Jupyter ecosystem
            PackageSpec("jupyter", ">=1.0.0", description="Jupyter notebooks"),
            PackageSpec("jupyterlab", ">=4.0.0", description="JupyterLab"),
            PackageSpec("ipywidgets", ">=8.0.0", description="Interactive widgets"),
            PackageSpec("nbconvert", ">=7.0.0", description="Notebook conversion"),
            
            # Development tools
            PackageSpec("black", ">=23.0.0", description="Code formatter"),
            PackageSpec("mypy", ">=1.4.0", description="Type checking"),
            PackageSpec("pytest", ">=7.4.0", description="Testing framework"),
            PackageSpec("pre-commit", ">=3.3.0", description="Git hooks"),
            
            # Data formats
            PackageSpec("pyarrow", ">=12.0.0", description="Arrow format"),
            PackageSpec("fastparquet", ">=2023.7.0", description="Parquet files"),
            PackageSpec("openpyxl", ">=3.1.0", description="Excel files"),
            PackageSpec("xlrd", ">=2.0.0", description="Excel reading"),
            
            # Database connectors
            PackageSpec("sqlalchemy", ">=2.0.0", description="SQL toolkit"),
            PackageSpec("pymongo", ">=4.4.0", description="MongoDB driver"),
            PackageSpec("redis", ">=4.6.0", description="Redis client"),
            PackageSpec("psycopg2-binary", ">=2.9.0", description="PostgreSQL"),
            
            # API and web
            PackageSpec("requests", ">=2.31.0", description="HTTP library"),
            PackageSpec("httpx", ">=0.24.0", description="Async HTTP"),
            PackageSpec("fastapi", ">=0.100.0", description="Web framework"),
            PackageSpec("uvicorn", ">=0.23.0", extras=["standard"], description="ASGI server"),
            
            # Cloud SDKs
            PackageSpec("boto3", ">=1.28.0", description="AWS SDK"),
            PackageSpec("google-cloud-storage", ">=2.10.0", description="GCS"),
            PackageSpec("azure-storage-blob", ">=12.17.0", description="Azure Storage"),
            
            # MLOps
            PackageSpec("mlflow", ">=2.5.0", description="ML lifecycle"),
            PackageSpec("wandb", ">=0.15.0", description="Experiment tracking"),
            PackageSpec("optuna", ">=3.3.0", description="Hyperparameter optimization"),
            PackageSpec("ray", ">=2.6.0", extras=["tune"], description="Distributed computing"),
            
            # Time series
            PackageSpec("statsmodels", ">=0.14.0", description="Statistical models"),
            PackageSpec("prophet", ">=1.1.0", description="Time series forecasting"),
            PackageSpec("pmdarima", ">=2.0.0", description="Auto-ARIMA"),
            
            # Geospatial
            PackageSpec("geopandas", ">=0.13.0", optional=True, description="Geospatial data"),
            PackageSpec("folium", ">=0.14.0", optional=True, description="Maps"),
            
            # Audio processing
            PackageSpec("librosa", ">=0.10.0", optional=True, description="Audio analysis"),
            PackageSpec("soundfile", ">=0.12.0", optional=True, description="Audio I/O"),
        ]
        
        config = EnvironmentConfig(
            name="ml_complete",
            python_version="3.11",
            packages=packages,
            system_packages=["ffmpeg", "graphviz", "tesseract"],
            environment_variables={
                "PYTHONPATH": "${PYTHONPATH}:.",
                "CUDA_VISIBLE_DEVICES": "0",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "TOKENIZERS_PARALLELISM": "false"
            },
            cuda_version="11.8"
        )
        
        return config
    
    def create_data_engineering_environment(self) -> EnvironmentConfig:
        """Create a data engineering focused environment."""
        packages = [
            # Data processing
            PackageSpec("pandas", ">=2.0.0"),
            PackageSpec("polars", ">=0.18.0"),
            PackageSpec("dask", ">=2023.7.0", extras=["complete"]),
            PackageSpec("vaex", ">=4.17.0"),
            
            # ETL tools
            PackageSpec("apache-airflow", ">=2.7.0"),
            PackageSpec("prefect", ">=2.11.0"),
            PackageSpec("dagster", ">=1.4.0"),
            PackageSpec("luigi", ">=3.4.0"),
            
            # Streaming
            PackageSpec("kafka-python", ">=2.0.0"),
            PackageSpec("confluent-kafka", ">=2.2.0"),
            PackageSpec("pulsar-client", ">=3.2.0"),
            
            # Big data
            PackageSpec("pyspark", ">=3.4.0"),
            PackageSpec("pyarrow", ">=12.0.0"),
            PackageSpec("databricks-sdk", ">=0.8.0"),
            
            # Data quality
            PackageSpec("great-expectations", ">=0.17.0"),
            PackageSpec("pandera", ">=0.16.0"),
            PackageSpec("pydantic", ">=2.3.0"),
            
            # Storage
            PackageSpec("delta-lake-reader", ">=0.2.0"),
            PackageSpec("pyiceberg", ">=0.4.0"),
            PackageSpec("minio", ">=7.1.0"),
        ]
        
        config = EnvironmentConfig(
            name="data_engineering",
            python_version="3.10",
            packages=packages,
            system_packages=["java-11-openjdk", "hadoop", "spark"],
            environment_variables={
                "SPARK_HOME": "/opt/spark",
                "HADOOP_HOME": "/opt/hadoop",
                "JAVA_HOME": "/usr/lib/jvm/java-11-openjdk"
            }
        )
        
        return config
    
    def setup_environment(self, config: EnvironmentConfig, 
                         force_reinstall: bool = False) -> bool:
        """Set up a complete environment from configuration."""
        logger.info(f"Setting up environment: {config.name}")
        
        # Create virtual environment
        env_path = self.venv_manager.create_environment(
            config.name, 
            config.python_version
        )
        
        # Activate environment
        env_vars = self.venv_manager.activate_environment(config.name)
        
        # Install system packages if on Linux
        if platform.system() == "Linux" and config.system_packages:
            self._install_system_packages(config.system_packages)
        
        # Resolve dependencies
        resolved_packages = self.resolver.resolve_dependencies(config.packages)
        
        # Install packages
        results = self.installer.install_packages(
            resolved_packages,
            parallel=True,
            upgrade=force_reinstall
        )
        
        # Set environment variables
        self._set_environment_variables(config.environment_variables, env_path)
        
        # Run post-install scripts
        for script in config.post_install_scripts:
            self._run_script(script, env_vars)
        
        # Save configuration
        self._save_config(config, env_path)
        
        # Generate report
        successful = sum(1 for v in results.values() if v)
        total = len(results)
        logger.info(f"Installation complete: {successful}/{total} packages installed")
        
        return successful == total
    
    def _install_system_packages(self, packages: List[str]):
        """Install system-level packages using apt/yum/brew."""
        if platform.system() == "Linux":
            # Detect package manager
            if shutil.which("apt-get"):
                cmd = ["sudo", "apt-get", "install", "-y"] + packages
            elif shutil.which("yum"):
                cmd = ["sudo", "yum", "install", "-y"] + packages
            else:
                logger.warning("No supported package manager found")
                return
            
            try:
                subprocess.run(cmd, check=True, timeout=300)
                logger.info(f"Installed system packages: {', '.join(packages)}")
            except Exception as e:
                logger.error(f"Failed to install system packages: {e}")
        
        elif platform.system() == "Darwin" and shutil.which("brew"):
            # macOS with Homebrew
            for pkg in packages:
                try:
                    subprocess.run(["brew", "install", pkg], check=True, timeout=300)
                except:
                    logger.warning(f"Failed to install {pkg} via Homebrew")
    
    def _set_environment_variables(self, variables: Dict[str, str], env_path: Path):
        """Set environment variables in activation script."""
        if platform.system() == "Windows":
            activate_script = env_path / "Scripts" / "activate.bat"
            for key, value in variables.items():
                with open(activate_script, "a") as f:
                    f.write(f"\nset {key}={value}")
        else:
            activate_script = env_path / "bin" / "activate"
            for key, value in variables.items():
                with open(activate_script, "a") as f:
                    f.write(f'\nexport {key}="{value}"')
    
    def _run_script(self, script: str, env_vars: Dict[str, str]):
        """Run a post-installation script."""
        try:
            subprocess.run(
                script,
                shell=True,
                env=env_vars,
                check=True,
                timeout=300
            )
            logger.info(f"Executed script: {script}")
        except Exception as e:
            logger.error(f"Script execution failed: {e}")
    
    def _save_config(self, config: EnvironmentConfig, env_path: Path):
        """Save environment configuration to JSON."""
        config_file = env_path / "environment_config.json"
        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {config_file}")
    
    def auto_setup_best_environment(self) -> bool:
        """Automatically set up the best environment based on system capabilities."""
        logger.info("Auto-detecting system capabilities...")
        
        # Detect GPU
        has_gpu = self._detect_gpu()
        
        # Detect available memory
        available_memory = self._get_available_memory()
        
        # Choose appropriate configuration
        if has_gpu and available_memory > 16:
            logger.info("Setting up full ML environment with GPU support")
            config = self.create_ml_environment()
        elif available_memory > 8:
            logger.info("Setting up ML environment without GPU packages")
            config = self.create_ml_environment()
            # Filter out GPU-only packages
            config.packages = [p for p in config.packages if not p.gpu_only]
        else:
            logger.info("Setting up lightweight data engineering environment")
            config = self.create_data_engineering_environment()
        
        return self.setup_environment(config)
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        # Check for NVIDIA GPU
        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    logger.info(f"GPU detected: {result.stdout.strip()}")
                    return True
            except:
                pass
        
        # Check for AMD GPU (ROCm)
        if shutil.which("rocm-smi"):
            logger.info("AMD GPU detected")
            return True
        
        # Check for Apple Silicon
        if platform.system() == "Darwin" and platform.processor() == "arm":
            logger.info("Apple Silicon GPU detected")
            return True
        
        logger.info("No GPU detected")
        return False
    
    def _get_available_memory(self) -> int:
        """Get available system memory in GB."""
        try:
            if platform.system() == "Linux":
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            return kb // (1024 * 1024)
            elif platform.system() == "Darwin":
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    bytes_mem = int(result.stdout.strip())
                    return bytes_mem // (1024 ** 3)
            elif platform.system() == "Windows":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                c_ulonglong = ctypes.c_ulonglong
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ('dwLength', ctypes.c_ulong),
                        ('dwMemoryLoad', ctypes.c_ulong),
                        ('ullTotalPhys', c_ulonglong),
                        ('ullAvailPhys', c_ulonglong),
                        ('ullTotalPageFile', c_ulonglong),
                        ('ullAvailPageFile', c_ulonglong),
                        ('ullTotalVirtual', c_ulonglong),
                        ('ullAvailVirtual', c_ulonglong),
                        ('ullAvailExtendedVirtual', c_ulonglong),
                    ]
                memstat = MEMORYSTATUSEX()
                memstat.dwLength = ctypes.sizeof(memstat)
                kernel32.GlobalMemoryStatusEx(ctypes.byref(memstat))
                return memstat.ullTotalPhys // (1024 ** 3)
        except Exception as e:
            logger.warning(f"Could not detect memory: {e}")
        
        return 8  # Default assumption


def main():
    """Main entry point for environment setup."""
    print("Velvet Python Environment Setup")
    print("Author: Cazandra Aporbo")
    print("-" * 50)
    
    setup = EnvironmentSetup()
    
    # Parse command line arguments (simplified)
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "auto":
            print("\nAuto-detecting and setting up best environment...")
            success = setup.auto_setup_best_environment()
        elif command == "ml":
            print("\nSetting up complete ML environment...")
            config = setup.create_ml_environment()
            success = setup.setup_environment(config)
        elif command == "data":
            print("\nSetting up data engineering environment...")
            config = setup.create_data_engineering_environment()
            success = setup.setup_environment(config)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python environment_setup.py [auto|ml|data]")
            sys.exit(1)
    else:
        # Default to auto setup
        print("\nRunning automatic environment setup...")
        success = setup.auto_setup_best_environment()
    
    if success:
        print("\n✅ Environment setup completed successfully!")
        print("Activate the environment and start developing!")
    else:
        print("\n❌ Environment setup encountered issues.")
        print("Check environment_setup.log for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
