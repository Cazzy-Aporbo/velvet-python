# Advanced Python Environment Troubleshooting & Performance Optimization Guide

## Table of Contents
1. [Common Environment Issues & Solutions](#common-issues)
2. [Performance Optimization Techniques](#performance-optimization)
3. [Debugging Environment Problems](#debugging-environment)
4. [Migration Strategies](#migration-strategies)
5. [Disaster Recovery](#disaster-recovery)
6. [Advanced Diagnostics](#advanced-diagnostics)

---

## Common Environment Issues & Solutions {#common-issues}

### 1. Package Conflict Resolution

#### Symptom: `ERROR: pip's dependency resolver does not currently take into account all the packages that are installed`

**Root Cause**: Incompatible version requirements between packages

**Solutions**:

```bash
# Solution 1: Use pip's new resolver (pip >= 20.3)
pip install --use-feature=2020-resolver package_name

# Solution 2: Install packages in specific order
pip install numpy==1.21.0
pip install pandas==1.3.0
pip install scikit-learn==1.0.0

# Solution 3: Use pip-tools for dependency resolution
pip install pip-tools
echo "numpy>=1.21,<1.22" > requirements.in
echo "pandas>=1.3,<1.4" >> requirements.in
pip-compile requirements.in  # Generates requirements.txt with resolved versions
pip-sync requirements.txt     # Installs exact versions

# Solution 4: Use poetry for better dependency management
poetry init
poetry add numpy@^1.21
poetry install
```

#### Advanced Conflict Resolution Script

```python
#!/usr/bin/env python3
"""
Advanced package conflict resolver
Analyzes and resolves complex dependency conflicts
"""

import subprocess
import json
import re
from typing import Dict, List, Set, Tuple

class ConflictResolver:
    def __init__(self, env_path: str):
        self.env_path = env_path
        self.pip_exe = f"{env_path}/bin/pip"
    
    def analyze_conflicts(self) -> Dict:
        """Analyze package conflicts in environment"""
        conflicts = {
            'version_conflicts': [],
            'missing_dependencies': [],
            'circular_dependencies': [],
            'incompatible_versions': []
        }
        
        # Run pip check
        result = subprocess.run(
            [self.pip_exe, 'check'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Parse error output
            for line in result.stdout.splitlines():
                if 'has requirement' in line:
                    conflicts['version_conflicts'].append(line)
                elif 'but you have' in line:
                    conflicts['incompatible_versions'].append(line)
        
        return conflicts
    
    def resolve_conflicts(self, strategy: str = 'conservative'):
        """
        Resolve conflicts with different strategies:
        - conservative: Minimal changes
        - aggressive: Update all to latest compatible
        - freeze: Lock current working versions
        """
        conflicts = self.analyze_conflicts()
        
        if strategy == 'conservative':
            # Try to find minimal version changes
            self._resolve_conservative(conflicts)
        elif strategy == 'aggressive':
            # Update everything to latest compatible
            self._resolve_aggressive(conflicts)
        elif strategy == 'freeze':
            # Lock current versions that work
            self._freeze_working_versions()
    
    def _resolve_conservative(self, conflicts):
        """Conservative resolution - minimal changes"""
        for conflict in conflicts['version_conflicts']:
            # Extract package names and version requirements
            match = re.search(r'(\S+) (\S+) has requirement (\S+)(.*)', conflict)
            if match:
                package = match.group(1)
                required = match.group(3)
                
                # Try installing the required version
                try:
                    subprocess.run(
                        [self.pip_exe, 'install', f'{required}'],
                        check=True
                    )
                    print(f"Resolved: Installed {required}")
                except:
                    print(f"Failed to resolve: {required}")
    
    def _resolve_aggressive(self, conflicts):
        """Aggressive resolution - update all"""
        # Get all installed packages
        result = subprocess.run(
            [self.pip_exe, 'list', '--format=json'],
            capture_output=True,
            text=True
        )
        packages = json.loads(result.stdout)
        
        # Update all packages to latest compatible versions
        for pkg in packages:
            try:
                subprocess.run(
                    [self.pip_exe, 'install', '--upgrade', pkg['name']],
                    check=True,
                    timeout=30
                )
            except:
                pass
    
    def _freeze_working_versions(self):
        """Freeze current working versions"""
        subprocess.run(
            [self.pip_exe, 'freeze', '>', 'requirements-frozen.txt'],
            shell=True
        )
        print("Frozen current versions to requirements-frozen.txt")

# Usage
resolver = ConflictResolver('/path/to/env')
conflicts = resolver.analyze_conflicts()
resolver.resolve_conflicts('conservative')
```

---

### 2. Binary Package Installation Issues

#### Symptom: `error: Microsoft Visual C++ 14.0 is required` (Windows) or `error: no acceptable C compiler found` (Linux/Mac)

**Solutions**:

```bash
# Linux - Install build tools
sudo apt-get install build-essential python3-dev
# or
sudo yum install gcc gcc-c++ python3-devel

# macOS - Install Xcode Command Line Tools
xcode-select --install

# Windows - Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Alternative: Use pre-compiled wheels
pip install --only-binary :all: package_name

# Or use conda which includes compiled binaries
conda install package_name
```

---

### 3. Memory Issues During Package Installation

#### Symptom: `MemoryError` or `killed` during pip install

**Solutions**:

```bash
# Solution 1: Limit pip's memory usage
pip install --no-cache-dir package_name

# Solution 2: Install packages one at a time
cat requirements.txt | xargs -n 1 -L 1 pip install

# Solution 3: Use swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Solution 4: Use lightweight alternatives
# Instead of pandas, try:
pip install modin[ray]  # Parallel pandas
# Instead of scikit-learn:
pip install scikit-learn-intelex  # Intel optimized

# Solution 5: Increase pip timeout and retries
pip install --timeout=1000 --retries=10 package_name
```

---

## Performance Optimization Techniques {#performance-optimization}

### 1. Environment Size Optimization

```bash
# Measure environment size
du -sh /path/to/env

# Clean pip cache
pip cache purge

# Remove unnecessary files
find /path/to/env -name "*.pyc" -delete
find /path/to/env -name "__pycache__" -type d -exec rm -rf {} +
find /path/to/env -name "*.pyo" -delete

# Strip debug symbols from compiled extensions
find /path/to/env -name "*.so" -exec strip {} \;

# Use pip-autoremove to remove unused dependencies
pip install pip-autoremove
pip-autoremove package_name -y
```

### 2. Installation Speed Optimization

```bash
# Use faster package index
pip install --index-url https://mirror.example.com/pypi/simple package_name

# Parallel installation (experimental)
pip install --use-feature=fast-deps package_name

# Pre-download packages for offline installation
pip download -r requirements.txt -d ./offline_packages
pip install --no-index --find-links ./offline_packages -r requirements.txt

# Use binary wheels only (no compilation)
pip install --only-binary :all: -r requirements.txt

# Cache wheels for reuse
export PIP_WHEEL_DIR=$HOME/.cache/pip/wheels
export PIP_FIND_LINKS=$HOME/.cache/pip/wheels
pip wheel -r requirements.txt
```

### 3. Import Time Optimization

```python
# Profile import times
python -X importtime -c "import your_package" 2>import_times.txt

# Lazy loading for faster startup
# Instead of:
import heavy_library

# Use:
def get_heavy_library():
    global _heavy_library
    if '_heavy_library' not in globals():
        import heavy_library as _heavy_library
    return _heavy_library

# Or use importlib for dynamic imports
import importlib

def process_data():
    pandas = importlib.import_module('pandas')
    return pandas.DataFrame()
```

### 4. Memory Usage Optimization

```python
#!/usr/bin/env python3
"""
Monitor and optimize environment memory usage
"""

import psutil
import os
import sys

def check_package_memory():
    """Check memory usage of imported packages"""
    import tracemalloc
    
    tracemalloc.start()
    
    # Import packages to measure
    packages_to_check = ['numpy', 'pandas', 'sklearn']
    memory_usage = {}
    
    for package in packages_to_check:
        snapshot_before = tracemalloc.take_snapshot()
        try:
            __import__(package)
            snapshot_after = tracemalloc.take_snapshot()
            
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            total_size = sum(stat.size_diff for stat in top_stats)
            memory_usage[package] = total_size / 1024 / 1024  # MB
        except ImportError:
            memory_usage[package] = None
    
    tracemalloc.stop()
    return memory_usage

# Optimize pandas memory usage
import pandas as pd

def optimize_dataframe(df):
    """Optimize DataFrame memory usage"""
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB")
    return df
```

---

## Debugging Environment Problems {#debugging-environment}

### 1. Comprehensive Environment Diagnostic Script

```bash
#!/bin/bash
# Complete environment diagnostic script

ENV_PATH=${1:-$VIRTUAL_ENV}

if [ -z "$ENV_PATH" ]; then
    echo "Usage: $0 /path/to/environment"
    exit 1
fi

echo "=== Python Environment Diagnostics ==="
echo "Environment: $ENV_PATH"
echo "Date: $(date)"
echo ""

# Check environment structure
echo "=== Environment Structure ==="
if [ -d "$ENV_PATH/bin" ]; then
    echo "✓ bin directory exists"
else
    echo "✗ bin directory missing"
fi

if [ -f "$ENV_PATH/bin/python" ]; then
    echo "✓ Python executable exists"
    echo "  Python version: $($ENV_PATH/bin/python --version 2>&1)"
else
    echo "✗ Python executable missing"
fi

if [ -f "$ENV_PATH/bin/pip" ]; then
    echo "✓ pip exists"
    echo "  pip version: $($ENV_PATH/bin/pip --version)"
else
    echo "✗ pip missing"
fi

# Check for corruption
echo ""
echo "=== Checking for Corruption ==="
find "$ENV_PATH" -name "*.so" -o -name "*.dylib" -o -name "*.dll" | while read lib; do
    if ! file "$lib" | grep -q "shared object\|shared library\|DLL"; then
        echo "✗ Corrupted library: $lib"
    fi
done

# Check Python path and imports
echo ""
echo "=== Python Path Configuration ==="
$ENV_PATH/bin/python -c "
import sys
import site
print('sys.prefix:', sys.prefix)
print('sys.exec_prefix:', sys.exec_prefix)
print('sys.path:')
for p in sys.path:
    print('  -', p)
print('site-packages:', site.getsitepackages())
"

# Check package health
echo ""
echo "=== Package Health Check ==="
$ENV_PATH/bin/pip check 2>&1 | head -20

# Check for duplicate packages
echo ""
echo "=== Checking for Duplicate Packages ==="
$ENV_PATH/bin/pip list | awk '{print $1}' | sort | uniq -d

# Check disk usage
echo ""
echo "=== Disk Usage Analysis ==="
du -sh "$ENV_PATH" 2>/dev/null
du -sh "$ENV_PATH"/* 2>/dev/null | sort -rh | head -10

# Check for security issues
echo ""
echo "=== Security Check ==="
if command -v safety &> /dev/null; then
    $ENV_PATH/bin/pip list --format=freeze | safety check --stdin
else
    echo "Safety not installed, skipping security check"
fi

# Check permissions
echo ""
echo "=== Permission Check ==="
find "$ENV_PATH" ! -user $(whoami) -print 2>/dev/null | head -10
if [ $? -eq 0 ]; then
    echo "✓ All files owned by current user"
fi

# Performance metrics
echo ""
echo "=== Performance Metrics ==="
echo "Import time for common packages:"
for pkg in numpy pandas requests; do
    if $ENV_PATH/bin/python -c "import $pkg" 2>/dev/null; then
        TIME=$($ENV_PATH/bin/python -c "
import time
start = time.time()
import $pkg
print(f'{time.time() - start:.3f}s')
" 2>&1)
        echo "  $pkg: $TIME"
    fi
done

echo ""
echo "=== Diagnostic Complete ==="
```

### 2. Environment Repair Script

```python
#!/usr/bin/env python3
"""
Automatic environment repair tool
Attempts to fix common environment issues
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnvironmentRepair:
    def __init__(self, env_path: str):
        self.env_path = Path(env_path)
        self.pip_exe = self.env_path / "bin" / "pip"
        self.python_exe = self.env_path / "bin" / "python"
        self.issues_found = []
        self.fixes_applied = []
    
    def diagnose(self):
        """Run all diagnostic checks"""
        logger.info("Starting environment diagnosis...")
        
        self._check_structure()
        self._check_executables()
        self._check_packages()
        self._check_permissions()
        self._check_symlinks()
        
        return self.issues_found
    
    def repair(self, auto_fix=False):
        """Attempt to repair found issues"""
        if not self.issues_found:
            logger.info("No issues found to repair")
            return
        
        for issue in self.issues_found:
            if auto_fix or self._confirm_fix(issue):
                fix_method = getattr(self, f"_fix_{issue['type']}", None)
                if fix_method:
                    try:
                        fix_method(issue)
                        self.fixes_applied.append(issue)
                        logger.info(f"Fixed: {issue['description']}")
                    except Exception as e:
                        logger.error(f"Failed to fix {issue['type']}: {e}")
    
    def _check_structure(self):
        """Check environment directory structure"""
        required_dirs = ['bin', 'lib', 'include']
        for dir_name in required_dirs:
            dir_path = self.env_path / dir_name
            if not dir_path.exists():
                self.issues_found.append({
                    'type': 'missing_directory',
                    'description': f"Missing directory: {dir_name}",
                    'path': str(dir_path)
                })
    
    def _check_executables(self):
        """Check Python and pip executables"""
        if not self.python_exe.exists():
            self.issues_found.append({
                'type': 'missing_python',
                'description': "Python executable missing",
                'path': str(self.python_exe)
            })
        
        if not self.pip_exe.exists():
            self.issues_found.append({
                'type': 'missing_pip',
                'description': "pip executable missing",
                'path': str(self.pip_exe)
            })
    
    def _check_packages(self):
        """Check package integrity"""
        try:
            result = subprocess.run(
                [str(self.pip_exe), 'check'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                self.issues_found.append({
                    'type': 'package_conflicts',
                    'description': "Package dependency conflicts detected",
                    'details': result.stdout
                })
        except Exception as e:
            logger.error(f"Failed to check packages: {e}")
    
    def _check_permissions(self):
        """Check file permissions"""
        for root, dirs, files in os.walk(self.env_path):
            for file in files:
                file_path = Path(root) / file
                if not os.access(file_path, os.R_OK):
                    self.issues_found.append({
                        'type': 'permission_error',
                        'description': f"Permission issue: {file_path}",
                        'path': str(file_path)
                    })
    
    def _check_symlinks(self):
        """Check for broken symlinks"""
        for root, dirs, files in os.walk(self.env_path):
            for file in files:
                file_path = Path(root) / file
                if file_path.is_symlink() and not file_path.exists():
                    self.issues_found.append({
                        'type': 'broken_symlink',
                        'description': f"Broken symlink: {file_path}",
                        'path': str(file_path)
                    })
    
    def _fix_missing_directory(self, issue):
        """Create missing directory"""
        Path(issue['path']).mkdir(parents=True, exist_ok=True)
    
    def _fix_missing_pip(self, issue):
        """Reinstall pip"""
        subprocess.run(
            [str(self.python_exe), '-m', 'ensurepip', '--upgrade'],
            check=True
        )
    
    def _fix_package_conflicts(self, issue):
        """Fix package conflicts"""
        # Try conservative fix first
        subprocess.run(
            [str(self.pip_exe), 'install', '--upgrade', 'pip', 'setuptools', 'wheel'],
            check=True
        )
        
        # Re-check
        result = subprocess.run(
            [str(self.pip_exe), 'check'],
            capture_output=True
        )
        
        if result.returncode != 0:
            # More aggressive fix
            subprocess.run(
                [str(self.pip_exe), 'install', '--force-reinstall', '-r', 'requirements.txt'],
                check=False
            )
    
    def _fix_permission_error(self, issue):
        """Fix permission issues"""
        os.chmod(issue['path'], 0o755)
    
    def _fix_broken_symlink(self, issue):
        """Remove broken symlinks"""
        os.unlink(issue['path'])
    
    def _confirm_fix(self, issue):
        """Ask user for confirmation"""
        response = input(f"Fix {issue['description']}? (y/n): ")
        return response.lower() == 'y'

# Usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python repair_env.py /path/to/environment [--auto]")
        sys.exit(1)
    
    repair_tool = EnvironmentRepair(sys.argv[1])
    issues = repair_tool.diagnose()
    
    if issues:
        print(f"\nFound {len(issues)} issues:")
        for issue in issues:
            print(f"  - {issue['description']}")
        
        auto_fix = '--auto' in sys.argv
        repair_tool.repair(auto_fix=auto_fix)
        
        print(f"\nApplied {len(repair_tool.fixes_applied)} fixes")
    else:
        print("No issues found!")
```

---

## Migration Strategies {#migration-strategies}

### 1. Migrating from System Python to Virtual Environment

```bash
#!/bin/bash
# Migrate system-installed packages to virtual environment

# Step 1: Export system packages
pip freeze --user > system_packages.txt

# Step 2: Create new virtual environment
python3 -m venv migration_env
source migration_env/bin/activate

# Step 3: Install packages with error handling
while read package; do
    echo "Installing $package..."
    pip install "$package" || echo "Failed: $package" >> failed_packages.txt
done < system_packages.txt

# Step 4: Verify migration
pip list > migrated_packages.txt
diff system_packages.txt migrated_packages.txt
```

### 2. Conda to Venv Migration

```python
#!/usr/bin/env python3
"""
Migrate from Conda to venv environment
"""

import subprocess
import json
import re
from pathlib import Path

def export_conda_packages(env_name):
    """Export packages from conda environment"""
    result = subprocess.run(
        ['conda', 'list', '-n', env_name, '--json'],
        capture_output=True,
        text=True
    )
    packages = json.loads(result.stdout)
    
    # Convert to pip format
    pip_requirements = []
    for pkg in packages:
        if pkg['channel'] != 'pypi':
            # Conda package - try to find pip equivalent
            pip_requirements.append(f"{pkg['name']}=={pkg['version']}")
        else:
            # Already from pip
            pip_requirements.append(f"{pkg['name']}=={pkg['version']}")
    
    return pip_requirements

def create_venv_from_conda(conda_env, venv_path):
    """Create venv with same packages as conda env"""
    # Export packages
    requirements = export_conda_packages(conda_env)
    
    # Create venv
    subprocess.run(['python3', '-m', 'venv', venv_path], check=True)
    
    # Install packages
    pip_exe = Path(venv_path) / 'bin' / 'pip'
    
    # Save requirements
    req_file = Path('requirements_from_conda.txt')
    req_file.write_text('\n'.join(requirements))
    
    # Install with fallback
    failed = []
    for req in requirements:
        try:
            subprocess.run(
                [str(pip_exe), 'install', req],
                check=True,
                capture_output=True,
                timeout=60
            )
        except:
            failed.append(req)
    
    if failed:
        print(f"Failed to install: {failed}")
        # Try finding alternatives
        for pkg in failed:
            pkg_name = pkg.split('==')[0]
            try:
                # Try without version
                subprocess.run(
                    [str(pip_exe), 'install', pkg_name],
                    check=True
                )
            except:
                print(f"Could not install {pkg_name} - manual intervention needed")
    
    print(f"Migration complete! Activate with: source {venv_path}/bin/activate")

# Usage
create_venv_from_conda('my_conda_env', '/path/to/new_venv')
```

---

## Disaster Recovery {#disaster-recovery}

### 1. Environment Backup System

```python
#!/usr/bin/env python3
"""
Complete environment backup and restore system
"""

import os
import tarfile
import json
import subprocess
from datetime import datetime
from pathlib import Path

class EnvironmentBackup:
    def __init__(self, backup_dir='/var/backups/python-envs'):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def backup_environment(self, env_path, backup_name=None):
        """Create complete backup of environment"""
        env_path = Path(env_path)
        
        if not backup_name:
            backup_name = f"{env_path.name}_{datetime.now():%Y%m%d_%H%M%S}"
        
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        metadata_path = self.backup_dir / f"{backup_name}.json"
        
        # Collect metadata
        metadata = {
            'environment_name': env_path.name,
            'backup_date': datetime.now().isoformat(),
            'python_version': self._get_python_version(env_path),
            'packages': self._get_packages(env_path),
            'size_bytes': sum(f.stat().st_size for f in env_path.rglob('*') if f.is_file()),
            'file_count': len(list(env_path.rglob('*')))
        }
        
        # Create tarball
        with tarfile.open(backup_path, 'w:gz') as tar:
            tar.add(env_path, arcname=env_path.name)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Backup created: {backup_path}")
        print(f"Metadata saved: {metadata_path}")
        
        return backup_path
    
    def restore_environment(self, backup_name, restore_path=None):
        """Restore environment from backup"""
        backup_path = self.backup_dir / f"{backup_name}.tar.gz"
        metadata_path = self.backup_dir / f"{backup_name}.json"
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if not restore_path:
            restore_path = Path.cwd() / metadata['environment_name']
        else:
            restore_path = Path(restore_path)
        
        # Extract backup
        with tarfile.open(backup_path, 'r:gz') as tar:
            tar.extractall(restore_path.parent)
        
        print(f"Environment restored to: {restore_path}")
        print(f"Python version: {metadata['python_version']}")
        print(f"Packages: {len(metadata['packages'])}")
        
        return restore_path
    
    def list_backups(self):
        """List all available backups"""
        backups = []
        
        for metadata_file in self.backup_dir.glob('*.json'):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                backups.append({
                    'name': metadata_file.stem,
                    'environment': metadata['environment_name'],
                    'date': metadata['backup_date'],
                    'python': metadata['python_version'],
                    'size': metadata['size_bytes']
                })
        
        return backups
    
    def _get_python_version(self, env_path):
        """Get Python version from environment"""
        python_exe = env_path / 'bin' / 'python'
        try:
            result = subprocess.run(
                [str(python_exe), '--version'],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except:
            return 'unknown'
    
    def _get_packages(self, env_path):
        """Get installed packages"""
        pip_exe = env_path / 'bin' / 'pip'
        try:
            result = subprocess.run(
                [str(pip_exe), 'list', '--format=json'],
                capture_output=True,
                text=True
            )
            return json.loads(result.stdout)
        except:
            return []

# Automated backup script
def automated_backup():
    """Run as cron job for regular backups"""
    backup_system = EnvironmentBackup()
    
    # Find all virtual environments
    env_locations = [
        Path.home() / '.virtualenvs',
        Path.home() / 'envs',
        Path('/opt/python-envs')
    ]
    
    for location in env_locations:
        if location.exists():
            for env_path in location.iterdir():
                if env_path.is_dir() and (env_path / 'bin' / 'python').exists():
                    try:
                        backup_system.backup_environment(env_path)
                    except Exception as e:
                        print(f"Failed to backup {env_path}: {e}")
```

### 2. Emergency Recovery Procedures

```bash
#!/bin/bash
# Emergency environment recovery script

RECOVERY_MODE=$1
ENV_NAME=$2

case $RECOVERY_MODE in
    "corrupted")
        echo "Attempting to recover corrupted environment..."
        
        # Backup corrupted environment first
        mv $ENV_NAME ${ENV_NAME}_corrupted_$(date +%Y%m%d)
        
        # Try to recreate from requirements
        if [ -f requirements.txt ]; then
            python3 -m venv $ENV_NAME
            source $ENV_NAME/bin/activate
            pip install -r requirements.txt
        else
            echo "No requirements.txt found, attempting to extract from corrupted env..."
            if [ -f ${ENV_NAME}_corrupted_*/bin/pip ]; then
                ${ENV_NAME}_corrupted_*/bin/pip freeze > recovered_requirements.txt
                python3 -m venv $ENV_NAME
                source $ENV_NAME/bin/activate
                pip install -r recovered_requirements.txt
            fi
        fi
        ;;
    
    "rollback")
        echo "Rolling back to previous version..."
        
        # Find most recent backup
        LATEST_BACKUP=$(ls -t /var/backups/python-envs/${ENV_NAME}_*.tar.gz | head -1)
        
        if [ -f "$LATEST_BACKUP" ]; then
            tar -xzf $LATEST_BACKUP
            echo "Rolled back to: $LATEST_BACKUP"
        else
            echo "No backup found!"
        fi
        ;;
    
    "rebuild")
        echo "Rebuilding environment from scratch..."
        
        # Clean rebuild
        rm -rf $ENV_NAME
        python3 -m venv $ENV_NAME
        source $ENV_NAME/bin/activate
        
        # Install essentials
        pip install --upgrade pip setuptools wheel
        
        # Attempt to install from various requirement files
        for req_file in requirements.txt requirements-prod.txt Pipfile; do
            if [ -f $req_file ]; then
                echo "Installing from $req_file..."
                pip install -r $req_file || pipenv install
            fi
        done
        ;;
    
    *)
        echo "Usage: $0 {corrupted|rollback|rebuild} environment_name"
        exit 1
        ;;
esac
```

---

## Advanced Diagnostics {#advanced-diagnostics}

### 1. Performance Profiling

```python
#!/usr/bin/env python3
"""
Profile environment performance and identify bottlenecks
"""

import time
import sys
import importlib
import tracemalloc
import cProfile
import pstats
from pathlib import Path

class EnvironmentProfiler:
    def __init__(self, env_path):
        self.env_path = Path(env_path)
        sys.path.insert(0, str(self.env_path / 'lib' / 'python3.10' / 'site-packages'))
    
    def profile_imports(self, packages):
        """Profile import times for packages"""
        results = {}
        
        for package in packages:
            # Clear any cached imports
            if package in sys.modules:
                del sys.modules[package]
            
            start = time.perf_counter()
            try:
                importlib.import_module(package)
                import_time = time.perf_counter() - start
                results[package] = {
                    'time': import_time,
                    'status': 'success'
                }
            except Exception as e:
                results[package] = {
                    'time': 0,
                    'status': f'failed: {e}'
                }
        
        return results
    
    def profile_memory(self, packages):
        """Profile memory usage of packages"""
        results = {}
        
        for package in packages:
            tracemalloc.start()
            snapshot_before = tracemalloc.take_snapshot()
            
            try:
                if package in sys.modules:
                    del sys.modules[package]
                
                importlib.import_module(package)
                
                snapshot_after = tracemalloc.take_snapshot()
                top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                
                total_memory = sum(stat.size_diff for stat in top_stats)
                results[package] = {
                    'memory_mb': total_memory / 1024 / 1024,
                    'top_consumers': [
                        {
                            'file': stat.traceback[0].filename,
                            'line': stat.traceback[0].lineno,
                            'size_mb': stat.size_diff / 1024 / 1024
                        }
                        for stat in top_stats[:5]
                    ]
                }
            except Exception as e:
                results[package] = {'error': str(e)}
            
            tracemalloc.stop()
        
        return results
    
    def profile_startup(self):
        """Profile complete environment startup"""
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Simulate typical startup sequence
        common_imports = [
            'os', 'sys', 'json', 'datetime',
            'numpy', 'pandas', 'requests'
        ]
        
        for module in common_imports:
            try:
                importlib.import_module(module)
            except:
                pass
        
        profiler.disable()
        
        # Generate statistics
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        return stats

# Usage
profiler = EnvironmentProfiler('/path/to/env')

# Profile import times
import_times = profiler.profile_imports(['numpy', 'pandas', 'sklearn'])
for pkg, info in import_times.items():
    print(f"{pkg}: {info['time']:.3f}s - {info['status']}")

# Profile memory usage
memory_usage = profiler.profile_memory(['numpy', 'pandas'])
for pkg, info in memory_usage.items():
    if 'memory_mb' in info:
        print(f"{pkg}: {info['memory_mb']:.2f} MB")
```

### 2. Dependency Graph Analysis

```python
#!/usr/bin/env python3
"""
Analyze and visualize package dependencies
"""

import subprocess
import json
from collections import defaultdict
import graphviz

def create_dependency_graph(env_path):
    """Create visual dependency graph"""
    pip_exe = f"{env_path}/bin/pip"
    
    # Get dependency information
    result = subprocess.run(
        [pip_exe, 'list', '--format=json'],
        capture_output=True,
        text=True
    )
    packages = json.loads(result.stdout)
    
    # Build dependency graph
    graph = defaultdict(list)
    
    for package in packages:
        pkg_name = package['name']
        
        # Get package dependencies
        result = subprocess.run(
            [pip_exe, 'show', pkg_name],
            capture_output=True,
            text=True
        )
        
        for line in result.stdout.splitlines():
            if line.startswith('Requires:'):
                deps = line.split(':')[1].strip()
                if deps:
                    for dep in deps.split(','):
                        graph[pkg_name].append(dep.strip())
    
    # Create visualization
    dot = graphviz.Digraph(comment='Package Dependencies')
    dot.attr(rankdir='LR')
    
    # Add nodes and edges
    for package, deps in graph.items():
        dot.node(package, package)
        for dep in deps:
            dot.edge(package, dep)
    
    # Save graph
    dot.render('dependencies', format='png', cleanup=True)
    
    # Find circular dependencies
    circular = find_circular_dependencies(graph)
    if circular:
        print("Warning: Circular dependencies detected:")
        for cycle in circular:
            print(f"  {' -> '.join(cycle)}")
    
    return graph

def find_circular_dependencies(graph):
    """Find circular dependencies in graph"""
    def dfs(node, visited, rec_stack, path):
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                result = dfs(neighbor, visited, rec_stack, path[:])
                if result:
                    return result
            elif neighbor in rec_stack:
                # Found cycle
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
        
        rec_stack.remove(node)
        return None
    
    visited = set()
    cycles = []
    
    for node in graph:
        if node not in visited:
            rec_stack = set()
            cycle = dfs(node, visited, rec_stack, [])
            if cycle:
                cycles.append(cycle)
    
    return cycles
```

---

## Best Practices Summary

### Environment Hygiene Checklist

```markdown
□ Regular Updates
  - Update pip, setuptools, wheel monthly
  - Security patches weekly
  - Major updates quarterly

□ Monitoring
  - Environment size < 2GB
  - Package count < 200
  - Import time < 5s for main packages
  - No circular dependencies

□ Backup Strategy
  - Daily backups for production
  - Weekly for development
  - Before major updates
  - Retention: 30 days

□ Security
  - Run pip-audit weekly
  - Check licenses quarterly
  - Verify package signatures
  - No packages from untrusted sources

□ Performance
  - Use binary wheels when possible
  - Clean cache regularly
  - Remove unused packages
  - Optimize import order

□ Documentation
  - Maintain requirements.txt
  - Document Python version
  - List system dependencies
  - Include setup instructions
```

### Quick Commands Reference

```bash
# Emergency fixes
pip install --force-reinstall --no-deps package_name
pip install --upgrade --force-reinstall package_name
pip install --ignore-installed package_name

# Performance
pip install --use-feature=fast-deps package_name
pip install --no-compile package_name  # Skip bytecode
pip install --no-binary :all: package_name  # Force source

# Debugging
python -m pip debug
python -m pip config debug
python -X importtime -c "import package"
python -m trace -t script.py

# Cleanup
pip cache purge
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name '*.pyc' -delete
```