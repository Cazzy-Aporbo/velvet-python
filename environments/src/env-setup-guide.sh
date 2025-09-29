#!/bin/bash

# Complete Python Environment Setup and Integration Guide
# Purpose: Comprehensive guide for setting up and integrating all environment management tools
# Scope: From initial setup to advanced workflows and team collaboration
# Use Case: Complete reference for individuals and teams

# ==============================================================================
# INITIAL SYSTEM SETUP
# ==============================================================================

echo "==================================================================="
echo "Python Environment Management - Complete Setup Guide"
echo "==================================================================="

# Step 1: Check and Install Python Versions
setup_python_versions() {
    echo "Setting up multiple Python versions..."
    
    # Check current Python installation
    echo "Current Python installations:"
    which -a python python3 python3.8 python3.9 python3.10 python3.11 2>/dev/null
    
    # Platform-specific installation
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Linux detected - Using apt/deadsnakes PPA for Python versions"
        
        # Add deadsnakes PPA for Ubuntu/Debian
        sudo add-apt-repository ppa:deadsnakes/ppa -y
        sudo apt update
        
        # Install multiple Python versions
        for version in 3.8 3.9 3.10 3.11; do
            echo "Installing Python $version..."
            sudo apt install -y python${version} python${version}-venv python${version}-dev
        done
        
        # Install python-is-python3 for convenience
        sudo apt install -y python-is-python3
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS detected - Using Homebrew for Python versions"
        
        # Check if Homebrew is installed
        if ! command -v brew &> /dev/null; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        # Install pyenv for Python version management
        brew install pyenv
        
        # Add pyenv to shell
        echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
        echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
        echo 'eval "$(pyenv init -)"' >> ~/.zshrc
        
        # Install Python versions
        for version in 3.8.16 3.9.16 3.10.11 3.11.3; do
            echo "Installing Python $version..."
            pyenv install $version
        done
        
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "Windows detected - Please use Python installer from python.org"
        echo "Or use Windows Store to install multiple Python versions"
        echo "Alternatively, use WSL2 for Linux-like environment"
    fi
    
    echo "Python versions setup complete!"
}

# Step 2: Install Conda/Miniconda
setup_conda() {
    echo "Setting up Conda environment manager..."
    
    # Check if conda is already installed
    if command -v conda &> /dev/null; then
        echo "Conda is already installed at: $(which conda)"
        conda --version
        return 0
    fi
    
    # Download and install Miniconda
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # Check if Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            CONDA_INSTALLER="Miniconda3-latest-MacOSX-arm64.sh"
        else
            CONDA_INSTALLER="Miniconda3-latest-MacOSX-x86_64.sh"
        fi
    else
        echo "Please download Miniconda manually from https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
    
    # Download installer
    wget "https://repo.anaconda.com/miniconda/$CONDA_INSTALLER"
    
    # Install Miniconda
    bash "$CONDA_INSTALLER" -b -p "$HOME/miniconda3"
    
    # Initialize conda for shell
    "$HOME/miniconda3/bin/conda" init bash
    "$HOME/miniconda3/bin/conda" init zsh 2>/dev/null
    
    # Clean up installer
    rm "$CONDA_INSTALLER"
    
    echo "Conda installation complete! Please restart your shell or run:"
    echo "source ~/.bashrc"
}

# Step 3: Install Additional Tools
setup_additional_tools() {
    echo "Installing additional environment management tools..."
    
    # Install virtualenv
    pip install --user virtualenv virtualenvwrapper
    
    # Install pipenv
    pip install --user pipenv
    
    # Install poetry
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Install useful development tools
    pip install --user \
        black \          # Code formatter
        flake8 \         # Linter
        mypy \           # Type checker
        pytest \         # Testing framework
        ipython \        # Enhanced Python shell
        jupyter \        # Jupyter notebooks
        pipdeptree \     # Dependency tree viewer
        pip-autoremove \ # Clean unused dependencies
        pip-audit       # Security audit tool
    
    echo "Additional tools installed!"
}

# ==============================================================================
# SHELL CONFIGURATION
# ==============================================================================

# Configure shell for optimal environment management
configure_shell() {
    echo "Configuring shell for environment management..."
    
    # Detect shell
    SHELL_RC="$HOME/.bashrc"
    if [ -n "$ZSH_VERSION" ]; then
        SHELL_RC="$HOME/.zshrc"
    fi
    
    # Backup existing configuration
    cp "$SHELL_RC" "${SHELL_RC}.backup.$(date +%Y%m%d)"
    
    # Add environment management configuration
    cat >> "$SHELL_RC" << 'EOF'

# ==============================================================================
# Python Environment Management Configuration
# ==============================================================================

# Conda configuration (if installed)
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Virtualenvwrapper configuration
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/projects
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
if [ -f "$HOME/.local/bin/virtualenvwrapper.sh" ]; then
    source "$HOME/.local/bin/virtualenvwrapper.sh"
fi

# Poetry configuration
export PATH="$HOME/.poetry/bin:$PATH"

# Pyenv configuration (macOS)
if command -v pyenv 1>/dev/null 2>&1; then
    eval "$(pyenv init -)"
fi

# Custom environment functions
export PYTHON_ENV_DIR="$HOME/.python_envs"

# Function to activate environment based on current directory
auto_activate_env() {
    if [ -f ".python-version" ]; then
        env_name=$(cat .python-version)
        if [ -n "$env_name" ]; then
            # Try conda first
            if conda env list | grep -q "^$env_name "; then
                conda activate "$env_name"
            # Try venv
            elif [ -d "$PYTHON_ENV_DIR/venv/$env_name" ]; then
                source "$PYTHON_ENV_DIR/venv/$env_name/bin/activate"
            # Try local .venv
            elif [ -d ".venv" ]; then
                source .venv/bin/activate
            fi
        fi
    elif [ -f ".env" ] && grep -q "VIRTUAL_ENV=" .env; then
        source .env
    fi
}

# Quick environment creation function
mkenv() {
    local name=${1:-$(basename $(pwd))}
    local python_version=${2:-3.10}
    
    echo "Creating environment: $name with Python $python_version"
    
    # Detect project type and create appropriate environment
    if [ -f "environment.yml" ] || [ -f "environment.yaml" ]; then
        echo "Found environment.yml - creating conda environment"
        conda env create -f environment.yml -n "$name"
    elif [ -f "Pipfile" ]; then
        echo "Found Pipfile - using pipenv"
        pipenv install
    elif [ -f "pyproject.toml" ] && grep -q "poetry" pyproject.toml; then
        echo "Found poetry project"
        poetry install
    elif [ -f "requirements.txt" ]; then
        echo "Found requirements.txt - creating venv"
        python${python_version} -m venv "$PYTHON_ENV_DIR/venv/$name"
        source "$PYTHON_ENV_DIR/venv/$name/bin/activate"
        pip install -r requirements.txt
    else
        echo "No requirements found - creating basic venv"
        python${python_version} -m venv "$PYTHON_ENV_DIR/venv/$name"
        source "$PYTHON_ENV_DIR/venv/$name/bin/activate"
    fi
    
    echo "$name" > .python-version
    echo "Environment $name created and activated!"
}

# List all environments from all sources
lsenv() {
    echo "=== Conda Environments ==="
    conda env list 2>/dev/null | grep -v "^#" | grep -v "^$" || echo "No conda environments"
    
    echo -e "\n=== Virtualenv Environments ==="
    if [ -d "$WORKON_HOME" ]; then
        ls -1 "$WORKON_HOME" 2>/dev/null || echo "No virtualenv environments"
    fi
    
    echo -e "\n=== Local venv Environments ==="
    if [ -d "$PYTHON_ENV_DIR/venv" ]; then
        ls -1 "$PYTHON_ENV_DIR/venv" 2>/dev/null || echo "No venv environments"
    fi
    
    echo -e "\n=== Project-local Environments ==="
    find ~/projects -maxdepth 2 -name ".venv" -o -name "venv" 2>/dev/null | \
        xargs -I {} dirname {} | xargs -I {} basename {} || echo "No project environments"
}

# Remove environment
rmenv() {
    local name=$1
    if [ -z "$name" ]; then
        echo "Usage: rmenv <environment_name>"
        return 1
    fi
    
    echo "Removing environment: $name"
    
    # Try conda
    if conda env list | grep -q "^$name "; then
        conda remove -n "$name" --all -y
    # Try venv
    elif [ -d "$PYTHON_ENV_DIR/venv/$name" ]; then
        rm -rf "$PYTHON_ENV_DIR/venv/$name"
    # Try virtualenv
    elif [ -d "$WORKON_HOME/$name" ]; then
        rmvirtualenv "$name"
    else
        echo "Environment not found: $name"
        return 1
    fi
    
    echo "Environment $name removed!"
}

# Compare two environments
diffenv() {
    local env1=$1
    local env2=$2
    
    if [ -z "$env1" ] || [ -z "$env2" ]; then
        echo "Usage: diffenv <env1> <env2>"
        return 1
    fi
    
    echo "Comparing $env1 and $env2..."
    
    # Create temporary files for comparison
    local temp1=$(mktemp)
    local temp2=$(mktemp)
    
    # Get package lists
    if conda env list | grep -q "^$env1 "; then
        conda list -n "$env1" --export > "$temp1"
    elif [ -d "$PYTHON_ENV_DIR/venv/$env1" ]; then
        "$PYTHON_ENV_DIR/venv/$env1/bin/pip" freeze > "$temp1"
    fi
    
    if conda env list | grep -q "^$env2 "; then
        conda list -n "$env2" --export > "$temp2"
    elif [ -d "$PYTHON_ENV_DIR/venv/$env2" ]; then
        "$PYTHON_ENV_DIR/venv/$env2/bin/pip" freeze > "$temp2"
    fi
    
    # Compare
    diff -u "$temp1" "$temp2" | grep -E "^[+-][^+-]" | sort
    
    # Cleanup
    rm -f "$temp1" "$temp2"
}

# Environment health check
checkenv() {
    local env_name=${1:-$CONDA_DEFAULT_ENV}
    if [ -z "$env_name" ] && [ -n "$VIRTUAL_ENV" ]; then
        env_name=$(basename "$VIRTUAL_ENV")
    fi
    
    if [ -z "$env_name" ]; then
        echo "No active environment detected"
        return 1
    fi
    
    echo "Checking environment: $env_name"
    echo "================================"
    
    # Check Python
    echo -n "Python: "
    python --version
    
    # Check pip
    echo -n "Pip: "
    pip --version
    
    # Count packages
    echo -n "Installed packages: "
    pip list --format=freeze | wc -l
    
    # Check for outdated packages
    echo -n "Outdated packages: "
    pip list --outdated --format=json | jq length 2>/dev/null || echo "unknown"
    
    # Check for broken packages
    echo -n "Package conflicts: "
    pip check 2>&1 | grep -c "has requirement" || echo "none"
    
    # Disk usage
    if [ -n "$VIRTUAL_ENV" ]; then
        echo -n "Disk usage: "
        du -sh "$VIRTUAL_ENV" | cut -f1
    fi
}

# Aliases for common operations
alias activate='source activate'
alias deactivate='conda deactivate 2>/dev/null || deactivate 2>/dev/null'
alias piplist='pip list --format=columns'
alias pipout='pip list --outdated'
alias pipup='pip install --upgrade pip setuptools wheel'
alias pipreq='pip freeze > requirements.txt'
alias pipinst='pip install -r requirements.txt'

# Auto-activate on directory change (optional - uncomment to enable)
# cd() {
#     builtin cd "$@"
#     auto_activate_env
# }

# Color output for environment status
env_prompt() {
    if [ -n "$CONDA_DEFAULT_ENV" ]; then
        echo -e "\033[36m(conda:$CONDA_DEFAULT_ENV)\033[0m"
    elif [ -n "$VIRTUAL_ENV" ]; then
        echo -e "\033[32m(venv:$(basename $VIRTUAL_ENV))\033[0m"
    fi
}

# Add environment indicator to prompt (bash)
if [ -n "$BASH_VERSION" ]; then
    PS1='$(env_prompt) '$PS1
fi

echo "Python environment management configured!"
echo "Commands available: mkenv, lsenv, rmenv, diffenv, checkenv"
EOF
    
    echo "Shell configuration complete!"
    echo "Please run: source $SHELL_RC"
}

# ==============================================================================
# PROJECT TEMPLATES
# ==============================================================================

# Create project templates for different use cases
create_project_templates() {
    echo "Creating project templates..."
    
    TEMPLATE_DIR="$HOME/.python_envs/templates"
    mkdir -p "$TEMPLATE_DIR"
    
    # Data Science Template
    cat > "$TEMPLATE_DIR/datascience_template.yml" << 'EOF'
name: datascience_project
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - numpy=1.23.*
  - pandas=1.5.*
  - matplotlib=3.6.*
  - seaborn=0.12.*
  - scikit-learn=1.2.*
  - jupyter=1.0.*
  - notebook=6.5.*
  - ipykernel=6.19.*
  - statsmodels=0.13.*
  - scipy=1.9.*
  - pip
  - pip:
    - plotly==5.11.*
    - streamlit==1.17.*
    - mlflow==2.1.*
variables:
  PYTHONPATH: ./src
  DATA_DIR: ./data
EOF
    
    # Web Development Template
    cat > "$TEMPLATE_DIR/webdev_requirements.txt" << 'EOF'
# Web Frameworks
django==4.2.0
flask==2.3.0
fastapi==0.95.0
uvicorn[standard]==0.21.0

# Database
sqlalchemy==2.0.0
alembic==1.10.0
psycopg2-binary==2.9.5
redis==4.5.0

# API Tools
requests==2.28.2
httpx==0.23.3
pydantic==1.10.7

# Testing
pytest==7.3.0
pytest-cov==4.0.0
pytest-django==4.5.2
pytest-asyncio==0.21.0

# Development Tools
black==23.3.0
flake8==6.0.0
mypy==1.2.0
pre-commit==3.2.0
EOF
    
    # Machine Learning Template
    cat > "$TEMPLATE_DIR/ml_requirements.txt" << 'EOF'
# Core ML Libraries
torch==2.0.0
tensorflow==2.12.0
transformers==4.28.0
datasets==2.11.0

# ML Tools
scikit-learn==1.2.2
xgboost==1.7.5
lightgbm==3.3.5
optuna==3.1.0

# Experiment Tracking
mlflow==2.3.0
wandb==0.15.0
tensorboard==2.12.0

# Data Processing
numpy==1.24.2
pandas==2.0.0
polars==0.17.0
dask==2023.4.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.0
EOF
    
    # Create project initialization script
    cat > "$TEMPLATE_DIR/init_project.sh" << 'EOF'
#!/bin/bash
# Initialize a new Python project with best practices

PROJECT_NAME=$1
PROJECT_TYPE=${2:-basic}  # basic, datascience, webdev, ml

if [ -z "$PROJECT_NAME" ]; then
    echo "Usage: init_project.sh <project_name> [project_type]"
    exit 1
fi

# Create project structure
mkdir -p "$PROJECT_NAME"/{src,tests,docs,data,notebooks,scripts}
cd "$PROJECT_NAME"

# Create initial files
touch README.md
touch .gitignore
touch .env.example
touch requirements.txt
touch setup.py

# Git ignore file
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environments
.env
.python-version

# IDEs
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Project specific
/data/
*.log
*.db
*.sqlite3

# Jupyter
.ipynb_checkpoints/
GITIGNORE

# README template
cat > README.md << 'README'
# PROJECT_NAME

## Description
Brief description of your project.

## Installation

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```python
# Example usage
```

## Project Structure
```
PROJECT_NAME/
├── src/          # Source code
├── tests/        # Test files
├── docs/         # Documentation
├── data/         # Data files (git ignored)
├── notebooks/    # Jupyter notebooks
├── scripts/      # Utility scripts
└── README.md
```

## Testing
```bash
pytest tests/
```

## License
MIT
README

sed -i "s/PROJECT_NAME/$PROJECT_NAME/g" README.md

# Create setup.py
cat > setup.py << SETUP
from setuptools import setup, find_packages

setup(
    name="$PROJECT_NAME",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Add your dependencies here
    ],
    python_requires=">=3.8",
)
SETUP

# Create environment based on type
case $PROJECT_TYPE in
    datascience)
        cp ~/.python_envs/templates/datascience_template.yml environment.yml
        sed -i "s/datascience_project/$PROJECT_NAME/g" environment.yml
        conda env create -f environment.yml
        echo "Data Science environment created!"
        ;;
    webdev)
        cp ~/.python_envs/templates/webdev_requirements.txt requirements.txt
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        echo "Web Development environment created!"
        ;;
    ml)
        cp ~/.python_envs/templates/ml_requirements.txt requirements.txt
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
        echo "Machine Learning environment created!"
        ;;
    *)
        python -m venv .venv
        source .venv/bin/activate
        echo "Basic environment created!"
        ;;
esac

# Initialize git
git init
git add .
git commit -m "Initial commit"

echo "Project $PROJECT_NAME initialized successfully!"
echo "To activate: source .venv/bin/activate (or conda activate $PROJECT_NAME)"
EOF
    
    chmod +x "$TEMPLATE_DIR/init_project.sh"
    
    echo "Project templates created in $TEMPLATE_DIR"
}

# ==============================================================================
# TEAM COLLABORATION SETUP
# ==============================================================================

# Setup for team environment management
setup_team_collaboration() {
    echo "Setting up team collaboration tools..."
    
    # Create shared environment specifications
    cat > "team_environment_guide.md" << 'EOF'
# Team Python Environment Guidelines

## Standard Environment Setup

### 1. Environment Naming Convention
```
<project>_<purpose>_<python_version>
Examples:
- api_dev_py310
- ml_training_py39
- web_prod_py311
```

### 2. Required Files in Repository

#### environment.yml (for Conda users)
```yaml
name: project_name
dependencies:
  - python=3.10
  - pip
  - pip:
    - -r requirements.txt
```

#### requirements.txt (for pip/venv users)
```
# Always pin versions for production
package==1.2.3
```

#### .python-version (for pyenv/auto-activation)
```
3.10.11
```

#### Makefile (for standardized commands)
```makefile
.PHONY: setup test clean

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	.venv/bin/pytest tests/

clean:
	rm -rf .venv __pycache__
```

## Environment Synchronization

### Developer Workflow
1. Pull latest changes
2. Check requirements.txt for updates
3. Update environment:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version-file: '.python-version'
    cache: 'pip'

- name: Install dependencies
  run: |
    pip install -r requirements.txt
```

## Dependency Management Rules

1. **Production dependencies** → requirements.txt
2. **Development dependencies** → requirements-dev.txt
3. **Testing dependencies** → requirements-test.txt

### Adding new dependencies:
```bash
# Install and freeze
pip install new-package
pip freeze | grep new-package >> requirements.txt

# Or use pip-tools
pip-compile requirements.in
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Package conflicts
```bash
# Create fresh environment
rm -rf .venv
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

#### Issue: Different OS compatibility
```bash
# Use environment markers in requirements.txt
psutil==5.9.0 ; sys_platform == 'linux'
pywin32==305 ; sys_platform == 'win32'
```

#### Issue: Reproducibility
```bash
# Lock all dependencies
pip freeze > requirements-lock.txt

# Install exact versions
pip install -r requirements-lock.txt
```

## Best Practices

1. ✅ Always use virtual environments
2. ✅ Pin package versions for production
3. ✅ Document Python version requirement
4. ✅ Keep requirements files up to date
5. ✅ Use .gitignore for environment directories
6. ✅ Test in clean environments regularly
7. ❌ Never commit .venv or node_modules
8. ❌ Avoid sudo pip install
9. ❌ Don't mix conda and pip unnecessarily
EOF
    
    # Create pre-commit hooks for environment validation
    cat > ".pre-commit-config.yaml" << 'EOF'
repos:
  - repo: local
    hooks:
      - id: check-requirements
        name: Check requirements.txt
        entry: scripts/check_requirements.sh
        language: script
        files: requirements.*\.txt$
        
      - id: validate-python-version
        name: Validate Python version
        entry: scripts/validate_python.sh
        language: script
        files: \.python-version$
EOF
    
    # Create validation scripts
    mkdir -p scripts
    
    cat > "scripts/check_requirements.sh" << 'EOF'
#!/bin/bash
# Validate requirements files

echo "Checking requirements files..."

# Check for unpinned versions in production requirements
if [ -f "requirements.txt" ]; then
    unpinned=$(grep -E "^[^#].*[^=<>!]$" requirements.txt | grep -v "^\s*$")
    if [ -n "$unpinned" ]; then
        echo "ERROR: Unpinned packages found in requirements.txt:"
        echo "$unpinned"
        exit 1
    fi
fi

echo "Requirements check passed!"
EOF
    
    chmod +x scripts/check_requirements.sh
    
    echo "Team collaboration setup complete!"
}

# ==============================================================================
# DOCKER INTEGRATION
# ==============================================================================

# Create Docker templates for Python environments
setup_docker_integration() {
    echo "Setting up Docker integration..."
    
    # Multi-stage Dockerfile template
    cat > "Dockerfile.template" << 'EOF'
# Multi-stage Python Docker build
# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "app.py"]
EOF
    
    # Docker Compose template
    cat > "docker-compose.yml.template" << 'EOF'
version: '3.8'

services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - PYTHONUNBUFFERED=1
      - ENV=development
    volumes:
      - .:/app
      - venv:/opt/venv  # Persist venv between restarts
    ports:
      - "8000:8000"
    command: python manage.py runserver 0.0.0.0:8000
    
  # Development environment with hot reload
  dev:
    build:
      context: .
      target: builder  # Use builder stage for development
    volumes:
      - .:/app
      - venv:/opt/venv
    command: /bin/bash
    stdin_open: true
    tty: true

volumes:
  venv:
EOF
    
    # Create build script
    cat > "scripts/docker_build.sh" << 'EOF'
#!/bin/bash
# Build and run Python app in Docker

PROJECT_NAME=${1:-python-app}
ENV_TYPE=${2:-production}

echo "Building Docker image for $PROJECT_NAME ($ENV_TYPE)..."

# Build image
docker build -t "$PROJECT_NAME:latest" .

if [ "$ENV_TYPE" == "development" ]; then
    # Run with volume mounts for development
    docker run -it --rm \
        -v "$(pwd):/app" \
        -p 8000:8000 \
        "$PROJECT_NAME:latest" \
        /bin/bash
else
    # Run production container
    docker run -d \
        --name "$PROJECT_NAME" \
        -p 8000:8000 \
        --restart unless-stopped \
        "$PROJECT_NAME:latest"
fi
EOF
    
    chmod +x scripts/docker_build.sh
    
    echo "Docker integration setup complete!"
}

# ==============================================================================
# MAIN SETUP ORCHESTRATION
# ==============================================================================

# Complete setup wizard
run_setup_wizard() {
    echo "======================================"
    echo "Python Environment Setup Wizard"
    echo "======================================"
    echo ""
    echo "This wizard will help you set up a complete Python environment management system."
    echo ""
    
    # Menu options
    PS3="Please select what to set up: "
    options=(
        "Complete Setup (Recommended)"
        "Python Versions Only"
        "Conda/Miniconda Only"
        "Shell Configuration Only"
        "Project Templates Only"
        "Team Collaboration Tools"
        "Docker Integration"
        "Custom Selection"
        "Exit"
    )
    
    select opt in "${options[@]}"
    do
        case $opt in
            "Complete Setup (Recommended)")
                echo "Running complete setup..."
                setup_python_versions
                setup_conda
                setup_additional_tools
                configure_shell
                create_project_templates
                setup_team_collaboration
                setup_docker_integration
                echo ""
                echo "✅ Complete setup finished!"
                echo ""
                echo "Next steps:"
                echo "1. Restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
                echo "2. Test with: mkenv test_env"
                echo "3. List environments: lsenv"
                echo "4. Create a project: ~/.python_envs/templates/init_project.sh myproject"
                break
                ;;
            "Python Versions Only")
                setup_python_versions
                break
                ;;
            "Conda/Miniconda Only")
                setup_conda
                break
                ;;
            "Shell Configuration Only")
                configure_shell
                break
                ;;
            "Project Templates Only")
                create_project_templates
                break
                ;;
            "Team Collaboration Tools")
                setup_team_collaboration
                break
                ;;
            "Docker Integration")
                setup_docker_integration
                break
                ;;
            "Custom Selection")
                echo "Select components to install:"
                read -p "Install Python versions? (y/n): " install_python
                read -p "Install Conda? (y/n): " install_conda
                read -p "Configure shell? (y/n): " config_shell
                read -p "Create templates? (y/n): " create_templates
                read -p "Setup team tools? (y/n): " setup_team
                read -p "Setup Docker? (y/n): " setup_docker
                
                [[ $install_python == "y" ]] && setup_python_versions
                [[ $install_conda == "y" ]] && setup_conda
                [[ $config_shell == "y" ]] && configure_shell
                [[ $create_templates == "y" ]] && create_project_templates
                [[ $setup_team == "y" ]] && setup_team_collaboration
                [[ $setup_docker == "y" ]] && setup_docker_integration
                
                echo "Custom setup complete!"
                break
                ;;
            "Exit")
                echo "Setup wizard cancelled."
                exit 0
                ;;
            *) echo "Invalid option $REPLY";;
        esac
    done
}

# ==============================================================================
# QUICK REFERENCE CARD
# ==============================================================================

print_quick_reference() {
    cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════════════╗
║                    PYTHON ENVIRONMENT QUICK REFERENCE                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ CREATING ENVIRONMENTS                                                         ║
║ venv:        python -m venv myenv                                           ║
║ conda:       conda create -n myenv python=3.10                              ║
║ virtualenv:  virtualenv myenv                                               ║
║ quick:       mkenv myproject 3.10                                           ║
║                                                                              ║
║ ACTIVATING ENVIRONMENTS                                                      ║
║ venv:        source myenv/bin/activate                                      ║
║ conda:       conda activate myenv                                           ║
║ Windows:     myenv\Scripts\activate.bat                                     ║
║                                                                              ║
║ DEACTIVATING                                                                 ║
║ venv:        deactivate                                                     ║
║ conda:       conda deactivate                                               ║
║                                                                              ║
║ LISTING PACKAGES                                                            ║
║ pip:         pip list / pip freeze                                          ║
║ conda:       conda list                                                     ║
║                                                                              ║
║ INSTALLING PACKAGES                                                         ║
║ pip:         pip install package_name                                       ║
║ conda:       conda install package_name                                     ║
║ from file:   pip install -r requirements.txt                                ║
║                                                                              ║
║ SAVING DEPENDENCIES                                                         ║
║ pip:         pip freeze > requirements.txt                                  ║
║ conda:       conda env export > environment.yml                             ║
║                                                                              ║
║ CUSTOM COMMANDS (after setup)                                               ║
║ mkenv:       Create new environment                                         ║
║ lsenv:       List all environments                                          ║
║ rmenv:       Remove environment                                             ║
║ diffenv:     Compare two environments                                       ║
║ checkenv:    Health check current environment                               ║
║                                                                              ║
║ PROJECT INITIALIZATION                                                      ║
║ ~/.python_envs/templates/init_project.sh <name> [type]                      ║
║ Types: basic, datascience, webdev, ml                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

EOF
}

# ==============================================================================
# SCRIPT EXECUTION
# ==============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed
    echo "Welcome to Python Environment Management Setup!"
    echo ""
    
    # Check for command line arguments
    if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
        print_quick_reference
        exit 0
    elif [ "$1" == "--quick" ]; then
        # Quick setup without prompts
        setup_python_versions
        setup_conda
        configure_shell
        create_project_templates
        echo "Quick setup complete! Restart your shell to apply changes."
        exit 0
    else
        # Run interactive setup wizard
        run_setup_wizard
        print_quick_reference
    fi
else
    # Script is being sourced
    echo "Python Environment Setup functions loaded."
    echo "Available functions:"
    echo "  - setup_python_versions"
    echo "  - setup_conda"
    echo "  - configure_shell"
    echo "  - create_project_templates"
    echo "  - setup_team_collaboration"
    echo "  - setup_docker_integration"
    echo "  - run_setup_wizard"
    echo "  - print_quick_reference"
fi