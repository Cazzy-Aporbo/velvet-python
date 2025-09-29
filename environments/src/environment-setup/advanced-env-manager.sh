#!/bin/bash

# Advanced Python Environment Management System
# Purpose: Unified management of conda, venv, and virtualenv environments
# Features: Auto-detection, smart switching, project tracking, health monitoring
# Use Case: Teams managing multiple projects with different requirement types

# ==============================================================================
# ENVIRONMENT MANAGER CONFIGURATION
# ==============================================================================

# Set up environment management directory structure
ENV_BASE_DIR="$HOME/.python_envs"
ENV_REGISTRY="$ENV_BASE_DIR/registry.json"
ENV_LOGS="$ENV_BASE_DIR/logs"
CONDA_ENV_DIR="$HOME/miniconda3/envs"
VENV_ENV_DIR="$ENV_BASE_DIR/venv"
PROJECT_MAP="$ENV_BASE_DIR/project_mapping.json"

# Initialize environment management system
init_env_manager() {
    echo "Initializing Python Environment Manager..."
    
    # Create directory structure
    mkdir -p "$ENV_BASE_DIR"
    mkdir -p "$ENV_LOGS"
    mkdir -p "$VENV_ENV_DIR"
    
    # Initialize registry if not exists
    if [ ! -f "$ENV_REGISTRY" ]; then
        echo '{"environments": [], "version": "1.0.0"}' > "$ENV_REGISTRY"
    fi
    
    # Initialize project mapping
    if [ ! -f "$PROJECT_MAP" ]; then
        echo '{"projects": {}}' > "$PROJECT_MAP"
    fi
    
    # Install required tools
    pip install --user jq yq python-json-logger
    
    echo "Environment Manager initialized at $ENV_BASE_DIR"
}

# ==============================================================================
# UNIFIED ENVIRONMENT DETECTION SYSTEM
# ==============================================================================

# Detect all Python environments on system
detect_all_environments() {
    echo "Scanning for Python environments..."
    
    local all_envs=()
    
    # Detect conda environments
    if command -v conda &> /dev/null; then
        echo "Detecting conda environments..."
        while IFS= read -r line; do
            if [[ $line != "#"* ]] && [[ -n $line ]]; then
                env_name=$(echo "$line" | awk '{print $1}')
                env_path=$(echo "$line" | awk '{print $2}')
                if [ -n "$env_path" ]; then
                    all_envs+=("conda:$env_name:$env_path")
                fi
            fi
        done < <(conda env list | tail -n +3)
    fi
    
    # Detect venv environments in standard location
    echo "Detecting venv environments..."
    if [ -d "$VENV_ENV_DIR" ]; then
        for env_dir in "$VENV_ENV_DIR"/*; do
            if [ -f "$env_dir/bin/activate" ]; then
                env_name=$(basename "$env_dir")
                all_envs+=("venv:$env_name:$env_dir")
            fi
        done
    fi
    
    # Detect virtualenv environments in common project locations
    echo "Scanning project directories..."
    for project_dir in ~/projects/* ~/Documents/*/; do
        if [ -f "$project_dir/.venv/bin/activate" ]; then
            project_name=$(basename "$project_dir")
            all_envs+=("venv:${project_name}_venv:$project_dir/.venv")
        fi
        if [ -f "$project_dir/venv/bin/activate" ]; then
            project_name=$(basename "$project_dir")
            all_envs+=("venv:${project_name}_venv:$project_dir/venv")
        fi
    done
    
    # Update registry with discovered environments
    echo "Found ${#all_envs[@]} environments"
    
    # Create JSON array of environments
    local json_envs="[]"
    for env in "${all_envs[@]}"; do
        IFS=':' read -r type name path <<< "$env"
        json_envs=$(echo "$json_envs" | jq ". += [{
            \"type\": \"$type\",
            \"name\": \"$name\",
            \"path\": \"$path\",
            \"discovered\": \"$(date -Iseconds)\"
        }]")
    done
    
    # Save to registry
    echo "{\"environments\": $json_envs, \"last_scan\": \"$(date -Iseconds)\"}" > "$ENV_REGISTRY"
    
    # Display summary
    echo "Environment Summary:"
    echo "$json_envs" | jq -r '.[] | "\(.type):\t\(.name)"' | column -t
}

# ==============================================================================
# INTELLIGENT ENVIRONMENT CREATION
# ==============================================================================

# Smart environment creation based on project requirements
create_smart_env() {
    local project_name=$1
    local project_path=${2:-$(pwd)}
    
    echo "Analyzing project requirements for optimal environment setup..."
    
    # Check for existing requirement files
    local env_type="venv"  # default
    local python_version="3.10"
    local packages=()
    
    # Detect environment type from project files
    if [ -f "$project_path/environment.yml" ] || [ -f "$project_path/environment.yaml" ]; then
        env_type="conda"
        echo "Found environment.yml - using conda"
        
        # Extract Python version from YAML
        if [ -f "$project_path/environment.yml" ]; then
            python_version=$(grep -E "python[=>]" "$project_path/environment.yml" | sed -E 's/.*python[=>]+([0-9.]+).*/\1/' | head -1)
        fi
        
    elif [ -f "$project_path/Pipfile" ]; then
        env_type="pipenv"
        echo "Found Pipfile - using pipenv"
        
    elif [ -f "$project_path/pyproject.toml" ]; then
        env_type="poetry"
        echo "Found pyproject.toml - checking for poetry"
        
    elif [ -f "$project_path/requirements.txt" ]; then
        env_type="venv"
        echo "Found requirements.txt - using venv"
        
        # Try to detect Python version from requirements
        if grep -q "python_version" "$project_path/requirements.txt"; then
            python_version=$(grep "python_version" "$project_path/requirements.txt" | sed -E 's/.*"([0-9.]+)".*/\1/')
        fi
    fi
    
    # Detect if scientific packages are needed (prefer conda)
    if grep -qE "numpy|scipy|pandas|tensorflow|torch|sklearn" "$project_path/requirements*.txt" 2>/dev/null; then
        echo "Detected scientific packages - recommending conda"
        read -p "Use conda for better scientific package management? [Y/n]: " use_conda
        if [[ ! $use_conda =~ [nN] ]]; then
            env_type="conda"
        fi
    fi
    
    # Generate unique environment name
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local env_name="${project_name}_${env_type}_${timestamp}"
    
    echo "Creating $env_type environment: $env_name"
    echo "Python version: $python_version"
    
    # Create environment based on type
    case $env_type in
        conda)
            create_conda_environment "$env_name" "$python_version" "$project_path"
            ;;
        venv)
            create_venv_environment "$env_name" "$python_version" "$project_path"
            ;;
        pipenv)
            create_pipenv_environment "$env_name" "$project_path"
            ;;
        poetry)
            create_poetry_environment "$env_name" "$project_path"
            ;;
    esac
    
    # Register environment with project
    register_project_env "$project_name" "$env_name" "$env_type" "$project_path"
}

# Create conda environment with project integration
create_conda_environment() {
    local env_name=$1
    local python_version=$2
    local project_path=$3
    
    # Create environment
    if [ -f "$project_path/environment.yml" ]; then
        # Create from YAML file
        conda env create -f "$project_path/environment.yml" -n "$env_name"
    else
        # Create basic environment
        conda create -n "$env_name" python="$python_version" pip -y
        
        # Install requirements if exists
        if [ -f "$project_path/requirements.txt" ]; then
            conda activate "$env_name"
            pip install -r "$project_path/requirements.txt"
            conda deactivate
        fi
    fi
    
    # Create activation script with project settings
    cat > "$ENV_BASE_DIR/activate_${env_name}.sh" << EOF
#!/bin/bash
# Auto-generated activation script for $env_name
export PROJECT_ROOT="$project_path"
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"
conda activate "$env_name"
cd "$project_path"
echo "Activated $env_name for project at $project_path"
EOF
    chmod +x "$ENV_BASE_DIR/activate_${env_name}.sh"
}

# Create venv environment with project integration  
create_venv_environment() {
    local env_name=$1
    local python_version=$2
    local project_path=$3
    
    local env_path="$VENV_ENV_DIR/$env_name"
    
    # Find Python executable
    local python_exe="python$python_version"
    if ! command -v "$python_exe" &> /dev/null; then
        python_exe="python3"
        echo "Warning: Python $python_version not found, using $python_exe"
    fi
    
    # Create virtual environment
    "$python_exe" -m venv "$env_path"
    
    # Activate and install requirements
    source "$env_path/bin/activate"
    pip install --upgrade pip setuptools wheel
    
    if [ -f "$project_path/requirements.txt" ]; then
        pip install -r "$project_path/requirements.txt"
    fi
    
    # Install development tools
    pip install ipython jupyter black flake8 pytest
    
    deactivate
    
    # Create activation script
    cat > "$ENV_BASE_DIR/activate_${env_name}.sh" << EOF
#!/bin/bash
# Auto-generated activation script for $env_name
export PROJECT_ROOT="$project_path"
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"
source "$env_path/bin/activate"
cd "$project_path"
echo "Activated $env_name for project at $project_path"
EOF
    chmod +x "$ENV_BASE_DIR/activate_${env_name}.sh"
}

# ==============================================================================
# INTELLIGENT ENVIRONMENT SWITCHING
# ==============================================================================

# Smart environment switcher with auto-detection
smart_activate() {
    local target=$1
    
    # If no target specified, try to auto-detect from current directory
    if [ -z "$target" ]; then
        target=$(detect_project_env)
    fi
    
    # Deactivate current environment if any
    if [ -n "$VIRTUAL_ENV" ]; then
        deactivate 2>/dev/null
    fi
    if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
        conda deactivate 2>/dev/null
    fi
    
    # Check if target is a registered environment
    local env_info=$(jq -r ".environments[] | select(.name == \"$target\")" "$ENV_REGISTRY")
    
    if [ -n "$env_info" ]; then
        local env_type=$(echo "$env_info" | jq -r '.type')
        local env_path=$(echo "$env_info" | jq -r '.path')
        
        case $env_type in
            conda)
                echo "Activating conda environment: $target"
                conda activate "$target"
                ;;
            venv|virtualenv)
                echo "Activating venv environment: $target"
                source "$env_path/bin/activate"
                ;;
        esac
        
        # Log activation
        echo "$(date -Iseconds) - Activated $target ($env_type)" >> "$ENV_LOGS/activation.log"
        
    else
        echo "Environment not found: $target"
        echo "Available environments:"
        list_environments
        return 1
    fi
}

# Detect environment from current project directory
detect_project_env() {
    local current_dir=$(pwd)
    
    # Check project mapping
    local project_env=$(jq -r ".projects.\"$current_dir\"" "$PROJECT_MAP" 2>/dev/null)
    
    if [ "$project_env" != "null" ] && [ -n "$project_env" ]; then
        echo "$project_env"
        return
    fi
    
    # Check for local .python-version file
    if [ -f ".python-version" ]; then
        cat ".python-version"
        return
    fi
    
    # Check for local .env file with ENV_NAME
    if [ -f ".env" ]; then
        grep "^ENV_NAME=" .env | cut -d= -f2
        return
    fi
    
    echo ""
}

# ==============================================================================
# ENVIRONMENT HEALTH MONITORING
# ==============================================================================

# Check health of all environments
check_env_health() {
    echo "Running environment health checks..."
    
    local unhealthy=()
    
    # Read all environments from registry
    while IFS= read -r env_json; do
        local name=$(echo "$env_json" | jq -r '.name')
        local type=$(echo "$env_json" | jq -r '.type')
        local path=$(echo "$env_json" | jq -r '.path')
        
        echo -n "Checking $name ($type)... "
        
        # Check if environment exists
        if [ ! -d "$path" ]; then
            echo "ERROR: Path not found"
            unhealthy+=("$name:missing")
            continue
        fi
        
        # Check if activation script exists
        case $type in
            conda)
                if conda env list | grep -q "^$name "; then
                    echo "OK"
                else
                    echo "ERROR: Not in conda list"
                    unhealthy+=("$name:not-registered")
                fi
                ;;
            venv|virtualenv)
                if [ -f "$path/bin/activate" ]; then
                    # Check Python executable
                    if [ -x "$path/bin/python" ]; then
                        echo "OK"
                    else
                        echo "ERROR: Python not executable"
                        unhealthy+=("$name:python-broken")
                    fi
                else
                    echo "ERROR: Activate script missing"
                    unhealthy+=("$name:activation-broken")
                fi
                ;;
        esac
    done < <(jq -c '.environments[]' "$ENV_REGISTRY")
    
    # Report summary
    echo ""
    if [ ${#unhealthy[@]} -eq 0 ]; then
        echo "✅ All environments are healthy!"
    else
        echo "⚠️  Found ${#unhealthy[@]} unhealthy environments:"
        for issue in "${unhealthy[@]}"; do
            echo "  - $issue"
        done
        
        read -p "Remove unhealthy environments from registry? [y/N]: " remove_unhealthy
        if [[ $remove_unhealthy =~ [yY] ]]; then
            cleanup_unhealthy_envs "${unhealthy[@]}"
        fi
    fi
}

# ==============================================================================
# ENVIRONMENT COMPARISON AND MIGRATION
# ==============================================================================

# Compare two environments
compare_envs() {
    local env1=$1
    local env2=$2
    
    echo "Comparing environments: $env1 vs $env2"
    
    # Create temp files for package lists
    local temp1="/tmp/${env1}_packages.txt"
    local temp2="/tmp/${env2}_packages.txt"
    
    # Get package lists based on environment type
    get_env_packages "$env1" > "$temp1"
    get_env_packages "$env2" > "$temp2"
    
    echo ""
    echo "Packages only in $env1:"
    comm -23 <(sort "$temp1") <(sort "$temp2") | sed 's/^/  + /'
    
    echo ""
    echo "Packages only in $env2:"
    comm -13 <(sort "$temp1") <(sort "$temp2") | sed 's/^/  - /'
    
    echo ""
    echo "Common packages with different versions:"
    join <(sort "$temp1") <(sort "$temp2") | while read pkg ver1 ver2; do
        if [ "$ver1" != "$ver2" ]; then
            echo "  * $pkg: $ver1 → $ver2"
        fi
    done
    
    rm -f "$temp1" "$temp2"
}

# Get package list for any environment type
get_env_packages() {
    local env_name=$1
    local env_info=$(jq -r ".environments[] | select(.name == \"$env_name\")" "$ENV_REGISTRY")
    local env_type=$(echo "$env_info" | jq -r '.type')
    
    case $env_type in
        conda)
            conda list -n "$env_name" --export | grep -v "^#" | cut -d= -f1,2 | tr '=' ' '
            ;;
        venv|virtualenv)
            local env_path=$(echo "$env_info" | jq -r '.path')
            "$env_path/bin/pip" list --format=freeze | cut -d= -f1,2 | tr '=' ' '
            ;;
    esac
}

# ==============================================================================
# ENVIRONMENT TEMPLATES AND PRESETS
# ==============================================================================

# Create environment from template
create_from_template() {
    local template_name=$1
    local new_env_name=$2
    
    # Template definitions
    case $template_name in
        datascience)
            echo "Creating Data Science environment..."
            conda create -n "$new_env_name" python=3.10 \
                numpy pandas matplotlib seaborn scikit-learn \
                jupyter notebook ipython statsmodels -y
            ;;
        
        webdev)
            echo "Creating Web Development environment..."
            python3 -m venv "$VENV_ENV_DIR/$new_env_name"
            source "$VENV_ENV_DIR/$new_env_name/bin/activate"
            pip install django flask fastapi uvicorn requests \
                beautifulsoup4 selenium pytest black flake8
            deactivate
            ;;
        
        ml-pytorch)
            echo "Creating PyTorch ML environment..."
            conda create -n "$new_env_name" python=3.9 \
                pytorch torchvision torchaudio cudatoolkit=11.6 \
                -c pytorch -c conda-forge -y
            conda activate "$new_env_name"
            pip install transformers datasets tensorboard
            conda deactivate
            ;;
        
        ml-tensorflow)
            echo "Creating TensorFlow ML environment..."
            conda create -n "$new_env_name" python=3.9 tensorflow-gpu -y
            conda activate "$new_env_name"
            pip install keras-tuner tensorboard-plugin-profile
            conda deactivate
            ;;
        
        minimal)
            echo "Creating minimal environment..."
            python3 -m venv "$VENV_ENV_DIR/$new_env_name"
            source "$VENV_ENV_DIR/$new_env_name/bin/activate"
            pip install --upgrade pip setuptools wheel
            deactivate
            ;;
        
        *)
            echo "Unknown template: $template_name"
            echo "Available templates:"
            echo "  - datascience: NumPy, Pandas, Jupyter, sklearn"
            echo "  - webdev: Django, Flask, FastAPI, testing tools"
            echo "  - ml-pytorch: PyTorch with CUDA support"
            echo "  - ml-tensorflow: TensorFlow with GPU support"
            echo "  - minimal: Just pip and setuptools"
            return 1
            ;;
    esac
    
    # Register the new environment
    detect_all_environments
    echo "Environment $new_env_name created from template: $template_name"
}

# ==============================================================================
# PROJECT AND ENVIRONMENT MAPPING
# ==============================================================================

# Register project-environment mapping
register_project_env() {
    local project_name=$1
    local env_name=$2
    local env_type=$3
    local project_path=$4
    
    # Update project mapping
    local current_mapping=$(cat "$PROJECT_MAP")
    local updated_mapping=$(echo "$current_mapping" | jq \
        ".projects.\"$project_path\" = {
            \"name\": \"$project_name\",
            \"environment\": \"$env_name\",
            \"type\": \"$env_type\",
            \"created\": \"$(date -Iseconds)\"
        }")
    
    echo "$updated_mapping" > "$PROJECT_MAP"
    
    # Create .python-version file in project
    echo "$env_name" > "$project_path/.python-version"
    
    echo "Registered $env_name for project $project_name at $project_path"
}

# ==============================================================================
# BATCH OPERATIONS
# ==============================================================================

# Update all environments
update_all_envs() {
    echo "Updating all environments..."
    
    local count=0
    while IFS= read -r env_json; do
        local name=$(echo "$env_json" | jq -r '.name')
        local type=$(echo "$env_json" | jq -r '.type')
        
        echo "Updating $name ($type)..."
        
        case $type in
            conda)
                conda update -n "$name" --all -y
                ;;
            venv|virtualenv)
                local env_path=$(echo "$env_json" | jq -r '.path')
                "$env_path/bin/pip" install --upgrade pip setuptools wheel
                "$env_path/bin/pip" list --outdated --format=json | \
                    jq -r '.[] | .name' | \
                    xargs -n1 "$env_path/bin/pip" install --upgrade
                ;;
        esac
        
        ((count++))
    done < <(jq -c '.environments[]' "$ENV_REGISTRY")
    
    echo "Updated $count environments"
}

# ==============================================================================
# INTERACTIVE ENVIRONMENT MENU
# ==============================================================================

# Interactive menu for environment management
env_menu() {
    while true; do
        clear
        echo "╔════════════════════════════════════════════╗"
        echo "║     Python Environment Manager Menu        ║"
        echo "╚════════════════════════════════════════════╝"
        echo ""
        echo "1. List all environments"
        echo "2. Create new environment"
        echo "3. Activate environment"
        echo "4. Delete environment"
        echo "5. Compare environments"
        echo "6. Check environment health"
        echo "7. Update all environments"
        echo "8. Create from template"
        echo "9. Scan for environments"
        echo "0. Exit"
        echo ""
        read -p "Select option: " choice
        
        case $choice in
            1) list_environments; read -p "Press Enter to continue..." ;;
            2) 
                read -p "Project name: " proj_name
                create_smart_env "$proj_name"
                read -p "Press Enter to continue..."
                ;;
            3)
                read -p "Environment name: " env_name
                smart_activate "$env_name"
                break  # Exit menu after activation
                ;;
            4)
                read -p "Environment to delete: " env_name
                delete_environment "$env_name"
                read -p "Press Enter to continue..."
                ;;
            5)
                read -p "First environment: " env1
                read -p "Second environment: " env2
                compare_envs "$env1" "$env2"
                read -p "Press Enter to continue..."
                ;;
            6) check_env_health; read -p "Press Enter to continue..." ;;
            7) update_all_envs; read -p "Press Enter to continue..." ;;
            8)
                echo "Available templates: datascience, webdev, ml-pytorch, ml-tensorflow, minimal"
                read -p "Template name: " template
                read -p "New environment name: " new_name
                create_from_template "$template" "$new_name"
                read -p "Press Enter to continue..."
                ;;
            9) detect_all_environments; read -p "Press Enter to continue..." ;;
            0) break ;;
            *) echo "Invalid option"; sleep 1 ;;
        esac
    done
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

# List all registered environments
list_environments() {
    echo "Registered Python Environments:"
    echo "================================"
    jq -r '.environments[] | "\(.type)\t\(.name)\t\(.path)"' "$ENV_REGISTRY" | \
        column -t -s $'\t' -N "Type,Name,Path"
}

# Delete environment
delete_environment() {
    local env_name=$1
    
    # Get environment info
    local env_info=$(jq -r ".environments[] | select(.name == \"$env_name\")" "$ENV_REGISTRY")
    if [ -z "$env_info" ]; then
        echo "Environment not found: $env_name"
        return 1
    fi
    
    local env_type=$(echo "$env_info" | jq -r '.type')
    local env_path=$(echo "$env_info" | jq -r '.path')
    
    read -p "Are you sure you want to delete $env_name? [y/N]: " confirm
    if [[ ! $confirm =~ [yY] ]]; then
        echo "Cancelled"
        return
    fi
    
    # Delete environment
    case $env_type in
        conda)
            conda remove -n "$env_name" --all -y
            ;;
        venv|virtualenv)
            rm -rf "$env_path"
            ;;
    esac
    
    # Remove from registry
    local updated_registry=$(jq ".environments = [.environments[] | select(.name != \"$env_name\")]" "$ENV_REGISTRY")
    echo "$updated_registry" > "$ENV_REGISTRY"
    
    echo "Environment $env_name deleted"
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Initialize if not already done
if [ ! -d "$ENV_BASE_DIR" ]; then
    init_env_manager
fi

# Add useful aliases
alias envs='list_environments'
alias enva='smart_activate'
alias envc='create_smart_env'
alias envd='delete_environment'
alias envm='env_menu'
alias envh='check_env_health'

echo "Python Environment Manager loaded!"
echo "Use 'envm' for interactive menu or 'envs' to list environments"