#!/bin/bash

# Comprehensive Conda Environment Management
# Purpose: Professional conda environment creation and management
# Prerequisites: Anaconda or Miniconda installed
# Use Case: Data scientists and researchers needing complex package management

# ==============================================================================
# CONDA VS PIP VIRTUAL ENVIRONMENTS
# ==============================================================================
# Conda advantages:
# - Manages non-Python dependencies (C libraries, R, Julia)
# - Better at resolving complex dependencies
# - Can install precompiled binaries (faster)
# - Handles different Python versions easily
# - Includes many data science packages pre-optimized
#
# When to use Conda:
# - Data science/ML projects
# - Scientific computing
# - Projects needing specific CUDA versions
# - Cross-language projects (Python + R)

# ==============================================================================
# INSTALLING CONDA (If not already installed)
# ==============================================================================

# Option 1: Miniconda (Minimal, recommended)
# Download from: https://docs.conda.io/en/latest/miniconda.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# Option 2: Anaconda (Full distribution with many packages)
# Download from: https://www.anaconda.com/products/distribution
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
bash Anaconda3-2023.03-Linux-x86_64.sh -b -p $HOME/anaconda3

# Initialize conda for your shell
# This adds conda to PATH and enables 'conda activate'
~/miniconda3/bin/conda init bash
# or for zsh:
~/miniconda3/bin/conda init zsh

# Restart shell or source profile
source ~/.bashrc

# ==============================================================================
# BASIC CONDA ENVIRONMENT OPERATIONS
# ==============================================================================

# Check conda installation
conda --version
conda info

# Update conda itself
conda update -n base conda -y

# List all environments
conda env list
# or
conda info --envs

# ==============================================================================
# CREATING CONDA ENVIRONMENTS
# ==============================================================================

# Create basic environment with default Python
conda create --name myproject

# Create with specific Python version
conda create --name py39_project python=3.9

# Create with multiple packages
conda create --name datascience python=3.10 numpy pandas matplotlib jupyter

# Create from specific channel (conda-forge is community maintained)
conda create --name geo_project -c conda-forge python=3.9 geopandas folium

# Create with pip installed in environment (recommended)
conda create --name webdev python=3.10 pip

# Clone an existing environment
conda create --name experiment_copy --clone production_env

# ==============================================================================
# ADVANCED ENVIRONMENT CREATION WITH UNIQUE NAMING
# ==============================================================================

# Function to create uniquely named environments with metadata
create_conda_env() {
    local project_name=$1
    local python_version=${2:-3.10}
    local env_type=${3:-dev}  # dev, prod, test, exp
    
    # Generate unique environment name with timestamp
    local timestamp=$(date +%Y%m%d)
    local env_name="${project_name}_${env_type}_py${python_version//.}_${timestamp}"
    
    echo "Creating environment: $env_name"
    echo "Python version: $python_version"
    echo "Environment type: $env_type"
    
    # Create environment with metadata file
    conda create --name "$env_name" python="$python_version" pip -y
    
    # Activate and add metadata
    conda activate "$env_name"
    
    # Create metadata file in environment
    cat > "$CONDA_PREFIX/.env_metadata.json" << EOF
{
    "created": "$(date -Iseconds)",
    "project": "$project_name",
    "python_version": "$python_version",
    "env_type": "$env_type",
    "creator": "$(whoami)",
    "hostname": "$(hostname)"
}
EOF
    
    echo "Environment created successfully: $env_name"
    echo "Metadata stored in: $CONDA_PREFIX/.env_metadata.json"
}

# Usage examples:
# create_conda_env "ml_classifier" "3.9" "dev"
# Creates: ml_classifier_dev_py39_20231215

# create_conda_env "web_api" "3.11" "prod"
# Creates: web_api_prod_py311_20231215

# ==============================================================================
# ACTIVATING AND SWITCHING ENVIRONMENTS
# ==============================================================================

# Activate an environment
conda activate datascience

# Your prompt changes to show active environment:
# (datascience) user@machine:~$

# Check current environment
echo $CONDA_DEFAULT_ENV
conda info --envs  # Current env marked with asterisk

# Deactivate current environment
conda deactivate

# Switch directly between environments
conda activate env1
conda activate env2  # Automatically deactivates env1

# ==============================================================================
# ENVIRONMENT CONFIGURATION WITH YAML FILES
# ==============================================================================

# Create environment from YAML file (recommended for reproducibility)
cat > environment.yml << 'EOF'
name: ml_project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - numpy=1.23.5
  - pandas=1.5.2
  - scikit-learn=1.2.0
  - matplotlib=3.6.2
  - jupyter=1.0.0
  - pip=22.3.1
  - pip:
    - tensorflow==2.11.0
    - transformers==4.25.1
variables:
  PROJECT_ROOT: /path/to/project
  DATA_DIR: /path/to/data
EOF

# Create environment from YAML
conda env create -f environment.yml

# Update existing environment from YAML
conda env update --name ml_project --file environment.yml --prune

# Export current environment to YAML
conda env export > environment_backup.yml

# Export without build specific info (more portable)
conda env export --no-builds > environment_portable.yml

# Export only explicitly installed packages
conda env export --from-history > environment_minimal.yml

# ==============================================================================
# MANAGING PACKAGES IN ENVIRONMENTS
# ==============================================================================

# Always activate environment first
conda activate myproject

# Install packages from conda
conda install numpy pandas

# Install specific version
conda install scipy=1.9.3

# Install from specific channel
conda install -c conda-forge streamlit

# Install multiple packages with versions
conda install numpy=1.23 pandas=1.5 "matplotlib>=3.5,<3.7"

# Search for packages
conda search tensorflow

# Update packages
conda update numpy
conda update --all  # Update all packages

# Remove packages
conda remove package_name

# Install pip packages (when not available in conda)
pip install some-pip-only-package

# ==============================================================================
# MIXING CONDA AND PIP (Best Practices)
# ==============================================================================

# IMPORTANT: Follow this order to avoid conflicts:
# 1. Install as many packages as possible with conda first
# 2. Then install pip packages
# 3. Don't use conda after pip in the same environment

# Good practice example:
conda create -n mixed_env python=3.9
conda activate mixed_env
# First: Install all conda packages
conda install numpy pandas scikit-learn jupyter
# Then: Install pip-only packages
pip install transformers datasets

# Create requirements files for both
conda list --export > conda-requirements.txt
pip freeze > pip-requirements.txt

# ==============================================================================
# ENVIRONMENT ISOLATION AND JUPYTER
# ==============================================================================

# Install Jupyter in base environment
conda activate base
conda install jupyter nb_conda_kernels

# Install ipykernel in each project environment
conda activate ml_project
conda install ipykernel
python -m ipykernel install --user --name ml_project --display-name "ML Project (Python 3.9)"

# Now Jupyter can see all environments as kernels
jupyter notebook
# Kernel menu will show all environments

# Remove a kernel
jupyter kernelspec uninstall ml_project

# ==============================================================================
# ADVANCED ENVIRONMENT MANAGEMENT
# ==============================================================================

# Set environment variables for an environment
conda activate myproject
conda env config vars set MY_DATABASE_URL=postgresql://localhost/mydb
conda env config vars set SECRET_KEY=your-secret-key

# List environment variables
conda env config vars list

# Remove environment variable
conda env config vars unset MY_DATABASE_URL

# These variables are automatically set when environment is activated

# ==============================================================================
# PERFORMANCE OPTIMIZATION
# ==============================================================================

# Configure conda for faster operations
# Use libmamba solver (faster dependency resolution)
conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Configure parallel downloads
conda config --set default_threads 4

# Use mamba (faster conda alternative)
conda install mamba -n base -c conda-forge
# Then use mamba instead of conda:
mamba install numpy pandas

# Clean up cached packages to save space
conda clean --all -y
# Removes:
# - Package cache
# - Temporary files
# - Old package versions

# ==============================================================================
# BACKUP AND MIGRATION
# ==============================================================================

# Backup all environments
backup_all_conda_envs() {
    local backup_dir="$HOME/conda_backups/$(date +%Y%m%d)"
    mkdir -p "$backup_dir"
    
    echo "Backing up all conda environments to $backup_dir"
    
    for env in $(conda env list | grep -E "^[a-zA-Z]" | awk '{print $1}'); do
        if [ "$env" != "base" ]; then
            echo "Backing up $env..."
            conda env export -n "$env" > "$backup_dir/${env}.yml"
        fi
    done
    
    echo "Backup complete. Files saved to $backup_dir"
}

# Restore environments from backup
restore_conda_envs() {
    local backup_dir=$1
    
    for yaml_file in "$backup_dir"/*.yml; do
        if [ -f "$yaml_file" ]; then
            echo "Restoring from $yaml_file..."
            conda env create -f "$yaml_file"
        fi
    done
}

# ==============================================================================
# REMOVING AND CLEANING ENVIRONMENTS
# ==============================================================================

# Remove an environment
conda remove --name old_project --all

# Remove environments older than N days
cleanup_old_envs() {
    local days=${1:-30}
    echo "Finding environments older than $days days..."
    
    # This is a safety example - uncomment to actually delete
    for env_path in $(conda info --envs | grep -E "^[^#]" | awk '{print $2}'); do
        if [ -d "$env_path" ] && [ "$env_path" != "$CONDA_PREFIX" ]; then
            age=$(find "$env_path" -maxdepth 0 -mtime +$days 2>/dev/null)
            if [ -n "$age" ]; then
                env_name=$(basename "$env_path")
                echo "Would remove: $env_name (older than $days days)"
                # conda remove --name "$env_name" --all -y
            fi
        fi
    done
}

# ==============================================================================
# TROUBLESHOOTING CONDA ENVIRONMENTS
# ==============================================================================

# Fix "CommandNotFoundError: Your shell has not been properly configured"
conda init bash
source ~/.bashrc

# Fix corrupted environment
conda update -n base conda
conda update --all

# Reset conda configuration
conda config --show-sources
conda config --remove-key channels  # Reset channels

# Solve environment conflicts
conda install --freeze-installed numpy=1.23
# or use strict channel priority
conda config --set channel_priority strict

# Check environment health
conda list --revisions  # Show environment history
conda install --revision 2  # Rollback to revision 2

# ==============================================================================
# PROJECT STRUCTURE BEST PRACTICES
# ==============================================================================

echo "Recommended project structure with conda:"
echo "
project/
├── environment.yml          # Conda environment specification
├── requirements.txt         # Pip requirements (if needed)
├── .env                    # Environment variables (git ignored)
├── scripts/
│   ├── setup_env.sh       # Environment setup script
│   └── activate_env.sh    # Custom activation script
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
├── data/                  # Data files (git ignored)
└── README.md             # Setup instructions
"