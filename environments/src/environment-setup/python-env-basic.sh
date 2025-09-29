#!/bin/bash

# Basic Python Virtual Environment Management
# Purpose: Introduction to creating and managing Python virtual environments
# Prerequisites: Python 3.6+ installed on your system
# Use Case: Individual developers working on single or multiple Python projects

# ==============================================================================
# UNDERSTANDING PYTHON VIRTUAL ENVIRONMENTS
# ==============================================================================
# Virtual environments are isolated Python installations that allow you to:
# - Install packages without affecting system Python
# - Have different package versions for different projects
# - Avoid permission issues when installing packages
# - Keep your projects reproducible and portable

# ==============================================================================
# METHOD 1: USING VENV (Built into Python 3.3+)
# ==============================================================================

# Check your Python version first
echo "Checking Python installation..."
python3 --version
# or on some systems:
python --version

# Create a basic virtual environment
# Syntax: python3 -m venv [environment_name]
python3 -m venv myproject_env

# This creates a directory structure:
# myproject_env/
# ├── bin/          (or Scripts/ on Windows)
# │   ├── activate  (activation script)
# │   ├── pip       (isolated pip)
# │   └── python    (python interpreter)
# ├── include/      (C headers for compiling packages)
# ├── lib/          (installed packages)
# └── pyvenv.cfg    (configuration file)

# Activate the virtual environment
# On Linux/Mac:
source myproject_env/bin/activate
# On Windows:
# myproject_env\Scripts\activate.bat
# On Windows PowerShell:
# myproject_env\Scripts\Activate.ps1

# Your prompt changes to show the active environment:
# (myproject_env) user@machine:~$

# Verify you're in the virtual environment
which python  # Should show path to myproject_env/bin/python
which pip     # Should show path to myproject_env/bin/pip
python --version

# Install packages in the isolated environment
pip install requests numpy pandas

# List installed packages
pip list

# Create a requirements file to track dependencies
pip freeze > requirements.txt
# This creates a file listing all packages and versions:
# requests==2.28.1
# numpy==1.23.5
# pandas==1.5.2

# Deactivate the virtual environment
deactivate
# Prompt returns to normal, system Python is active again

# ==============================================================================
# METHOD 2: USING VIRTUALENV (More features, needs installation)
# ==============================================================================

# Install virtualenv globally (one time only)
pip install virtualenv

# Create environment with specific Python version
virtualenv -p python3.9 myapp_env

# Create environment with system site packages access
# Useful when you need system-installed packages like tkinter
virtualenv --system-site-packages gui_project_env

# Create environment with custom prompt name
virtualenv --prompt="(DataScience) " datascience_env

# Activate and use same as venv
source myapp_env/bin/activate

# ==============================================================================
# ORGANIZING MULTIPLE PROJECTS
# ==============================================================================

# Best practice: Create a dedicated directory for all environments
mkdir -p ~/python_environments
cd ~/python_environments

# Create environments with descriptive names
python3 -m venv web_scraper_env
python3 -m venv ml_project_env
python3 -m venv api_backend_env

# Alternative: Keep environment in project directory
cd ~/projects/my_website
python3 -m venv .venv  # Hidden directory with dot prefix
# Add .venv/ to .gitignore to avoid committing environment

# ==============================================================================
# SWITCHING BETWEEN ENVIRONMENTS
# ==============================================================================

# Function to quickly switch environments (add to ~/.bashrc or ~/.zshrc)
activate_env() {
    # First deactivate any current environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        deactivate
    fi
    
    # Activate the requested environment
    if [ -d "$HOME/python_environments/$1" ]; then
        source "$HOME/python_environments/$1/bin/activate"
        echo "Activated: $1"
    else
        echo "Environment not found: $1"
        echo "Available environments:"
        ls -1 "$HOME/python_environments"
    fi
}

# Usage:
# activate_env web_scraper_env
# activate_env ml_project_env

# ==============================================================================
# RECREATING ENVIRONMENTS FROM REQUIREMENTS
# ==============================================================================

# Create new environment from existing requirements.txt
python3 -m venv new_project_env
source new_project_env/bin/activate
pip install -r requirements.txt

# Upgrade pip itself in the environment
pip install --upgrade pip

# Install packages from different requirement files
pip install -r requirements/base.txt      # Core dependencies
pip install -r requirements/dev.txt       # Development tools
pip install -r requirements/test.txt      # Testing frameworks

# ==============================================================================
# CLEANING UP AND MAINTENANCE
# ==============================================================================

# Remove a virtual environment (just delete the directory)
rm -rf myproject_env

# Check size of environments
du -sh ~/python_environments/*

# Find all virtual environments on system
find ~ -type d -name "bin" -exec test -e "{}/activate" \; -print 2>/dev/null | sed 's|/bin||'

# ==============================================================================
# TROUBLESHOOTING COMMON ISSUES
# ==============================================================================

# Issue: Permission denied when installing packages
# Solution: Never use sudo with pip in virtual environment
# The environment should have full user permissions

# Issue: Wrong Python version in environment
# Check Python version before creating environment
which python3.9  # Find specific version
/usr/bin/python3.9 -m venv specific_version_env

# Issue: Environment not activating
# Check if activation script exists and has permissions
ls -la myproject_env/bin/activate
chmod +x myproject_env/bin/activate

# Issue: Package conflicts
# Create fresh environment rather than trying to fix
deactivate
rm -rf problematic_env
python3 -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt

# ==============================================================================
# BEST PRACTICES
# ==============================================================================

echo "Virtual Environment Best Practices:"
echo "1. One environment per project"
echo "2. Name environments descriptively"
echo "3. Keep requirements.txt updated"
echo "4. Don't commit environments to git"
echo "5. Document Python version needed"
echo "6. Use consistent naming convention"
echo "7. Regularly update pip and setuptools"