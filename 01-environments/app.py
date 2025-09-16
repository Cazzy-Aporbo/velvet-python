# 01-environments/tests/test_environment_manager.py

```python
"""
Tests for Environment Manager

Author: Cazzy Aporbo, MS
Created: January 2025

I write tests for everything now. I learned this after spending a weekend
debugging an issue that a simple test would have caught in 5 seconds.
These tests ensure my environment manager actually works as expected.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment_manager import EnvironmentManager, VirtualEnvironment


class TestVirtualEnvironment:
    """
    Test the VirtualEnvironment dataclass.
    
    I test even simple classes because I've been burned by
    assuming "it's too simple to break."
    
    Author: Cazzy Aporbo, MS
    """
    
    def test_creation(self):
        """Test creating a virtual environment object"""
        env = VirtualEnvironment(
            name="test_env",
            path=Path("/tmp/test_env"),
            python_version="3.11.0",
            created_at=datetime.now(),
            packages=["requests==2.31.0", "pandas==2.2.0"],
            is_active=False
        )
        
        assert env.name == "test_env"
        assert env.path == Path("/tmp/test_env")
        assert env.python_version == "3.11.0"
        assert len(env.packages) == 2
        assert not env.is_active
    
    def test_string_representation(self):
        """Test the string representation"""
        env = VirtualEnvironment(
            name="my_project",
            path=Path("/tmp/my_project"),
            python_version="3.11.0",
            created_at=datetime.now(),
            packages=[],
            is_active=True
        )
        
        result = str(env)
        assert "my_project" in result
        assert "3.11.0" in result
        assert "Active" in result
    
    def test_to_dict_serialization(self):
        """Test serialization to dictionary"""
        now = datetime.now()
        env = VirtualEnvironment(
            name="test_env",
            path=Path("/tmp/test_env"),
            python_version="3.11.0",
            created_at=now,
            packages=["numpy==1.26.0"],
            is_active=False
        )
        
        result = env.to_dict()
        
        assert result['name'] == "test_env"
        assert result['path'] == "/tmp/test_env"
        assert result['python_version'] == "3.11.0"
        assert result['created_at'] == now.isoformat()
        assert result['packages'] == ["numpy==1.26.0"]
        assert result['is_active'] is False


class TestEnvironmentManager:
    """
    Test the EnvironmentManager class.
    
    These tests have saved me from so many bugs. I test every
    method because environments are critical infrastructure.
    
    Author: Cazzy Aporbo, MS
    """
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        # Cleanup after test
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @pytest.fixture
    def manager(self, temp_dir):
        """Create an EnvironmentManager instance for testing"""
        return EnvironmentManager(base_dir=temp_dir)
    
    def test_initialization(self, manager, temp_dir):
        """Test manager initialization"""
        assert manager.base_dir == temp_dir
        assert manager.config_file == temp_dir / 'environments.json'
        assert manager.environments == {}
        assert temp_dir.exists()
    
    def test_load_environments_empty(self, manager):
        """Test loading when no environments exist"""
        envs = manager._load_environments()
        assert envs == {}
    
    def test_load_environments_with_data(self, temp_dir):
        """Test loading existing environments"""
        # Create a config file with test data
        config_file = temp_dir / 'environments.json'
        test_data = {
            "test_env": {
                "name": "test_env",
                "path": str(temp_dir / "test_env"),
                "python_version": "3.11.0",
                "created_at": datetime.now().isoformat(),
                "packages": ["requests==2.31.0"],
                "is_active": False
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(test_data, f)
        
        manager = EnvironmentManager(base_dir=temp_dir)
        
        assert len(manager.environments) == 1
        assert "test_env" in manager.environments
        assert manager.environments["test_env"].name == "test_env"
    
    def test_save_environments(self, manager, temp_dir):
        """Test saving environments to config file"""
        # Add an environment
        env = VirtualEnvironment(
            name="save_test",
            path=temp_dir / "save_test",
            python_version="3.11.0",
            created_at=datetime.now(),
            packages=["flask==3.0.0"],
            is_active=True
        )
        
        manager.environments["save_test"] = env
        manager._save_environments()
        
        # Verify the file was created and contains correct data
        config_file = temp_dir / 'environments.json'
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        assert "save_test" in data
        assert data["save_test"]["name"] == "save_test"
        assert data["save_test"]["python_version"] == "3.11.0"
        assert data["save_test"]["packages"] == ["flask==3.0.0"]
    
    @patch('src.environment_manager.venv.create')
    @patch('src.environment_manager.subprocess.run')
    def test_create_environment(self, mock_run, mock_venv_create, manager, temp_dir):
        """Test creating a new environment"""
        # Setup mocks
        mock_venv_create.return_value = None
        
        # Mock subprocess responses
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='Python 3.11.0\n'
        )
        
        # Create a test project directory
        project_dir = temp_dir / "test_project"
        project_dir.mkdir()
        
        # Create environment
        env = manager.create_environment(
            name="test_create",
            project_dir=project_dir
        )
        
        # Verify environment was created
        assert env.name == "test_create"
        assert env.path == project_dir / '.venv'
        assert "test_create" in manager.environments
        
        # Verify venv.create was called
        mock_venv_create.assert_called_once()
        
        # Verify .gitignore was created
        gitignore = project_dir / '.gitignore'
        assert gitignore.exists()
        content = gitignore.read_text()
        assert '.venv/' in content
        assert '__pycache__/' in content
    
    def test_list_environments_empty(self, manager, capsys):
        """Test listing when no environments exist"""
        manager.list_environments()
        
        captured = capsys.readouterr()
        assert "No environments found" in captured.out
    
    def test_list_environments_with_data(self, manager, capsys):
        """Test listing existing environments"""
        # Add test environment
        env = VirtualEnvironment(
            name="list_test",
            path=Path("/tmp/list_test"),
            python_version="3.11.0",
            created_at=datetime.now(),
            packages=["numpy==1.26.0", "pandas==2.2.0"],
            is_active=False
        )
        
        manager.environments["list_test"] = env
        manager.list_environments()
        
        captured = capsys.readouterr()
        assert "list_test" in captured.out
        assert "3.11.0" in captured.out
    
    def test_delete_environment_not_found(self, manager, capsys):
        """Test deleting non-existent environment"""
        result = manager.delete_environment("nonexistent")
        
        assert result is False
        captured = capsys.readouterr()
        assert "not found" in captured.out
    
    @patch('src.environment_manager.console.input')
    def test_delete_environment_cancelled(self, mock_input, manager, temp_dir):
        """Test cancelling environment deletion"""
        # Add test environment
        env_path = temp_dir / "delete_test"
        env_path.mkdir()
        
        env = VirtualEnvironment(
            name="delete_test",
            path=env_path,
            python_version="3.11.0",
            created_at=datetime.now(),
            packages=[],
            is_active=False
        )
        
        manager.environments["delete_test"] = env
        
        # Mock user input to cancel
        mock_input.return_value = "no"
        
        result = manager.delete_environment("delete_test")
        
        assert result is False
        assert "delete_test" in manager.environments
        assert env_path.exists()
    
    @patch('src.environment_manager.console.input')
    def test_delete_environment_confirmed(self, mock_input, manager, temp_dir):
        """Test successful environment deletion"""
        # Add test environment
        env_path = temp_dir / "delete_test"
        env_path.mkdir()
        
        env = VirtualEnvironment(
            name="delete_test",
            path=env_path,
            python_version="3.11.0",
            created_at=datetime.now(),
            packages=[],
            is_active=False
        )
        
        manager.environments["delete_test"] = env
        
        # Mock user input to confirm
        mock_input.return_value = "yes"
        
        result = manager.delete_environment("delete_test")
        
        assert result is True
        assert "delete_test" not in manager.environments
        assert not env_path.exists()
    
    def test_compare_environments(self, manager, capsys):
        """Test comparing two environments"""
        # Create two environments with different packages
        env1 = VirtualEnvironment(
            name="env1",
            path=Path("/tmp/env1"),
            python_version="3.11.0",
            created_at=datetime.now(),
            packages=["requests==2.31.0", "pandas==2.2.0", "numpy==1.26.0"],
            is_active=False
        )
        
        env2 = VirtualEnvironment(
            name="env2",
            path=Path("/tmp/env2"),
            python_version="3.11.0",
            created_at=datetime.now(),
            packages=["requests==2.30.0", "flask==3.0.0", "numpy==1.26.0"],
            is_active=False
        )
        
        manager.environments["env1"] = env1
        manager.environments["env2"] = env2
        
        manager.compare_environments("env1", "env2")
        
        captured = capsys.readouterr()
        output = captured.out
        
        # Check for expected differences
        assert "pandas" in output  # Only in env1
        assert "flask" in output   # Only in env2
        assert "requests" in output  # Version difference
        assert "2.31.0" in output
        assert "2.30.0" in output
    
    def test_export_requirements(self, manager, temp_dir):
        """Test exporting requirements"""
        # Add test environment
        env = VirtualEnvironment(
            name="export_test",
            path=Path("/tmp/export_test"),
            python_version="3.11.0",
            created_at=datetime.now(),
            packages=["requests==2.31.0", "pandas==2.2.0"],
            is_active=False
        )
        
        manager.environments["export_test"] = env
        
        output_path = temp_dir / "requirements_export.txt"
        result = manager.export_requirements("export_test", output_path)
        
        assert result is True
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "requests==2.31.0" in content
        assert "pandas==2.2.0" in content


class TestIntegration:
    """
    Integration tests that test the full workflow.
    
    I added these after a refactoring broke the interaction
    between components that unit tests didn't catch.
    
    Author: Cazzy Aporbo, MS
    """
    
    @pytest.fixture
    def integration_dir(self):
        """Create a directory for integration testing"""
        temp_path = tempfile.mkdtemp()
        yield Path(temp_path)
        shutil.rmtree(temp_path, ignore_errors=True)
    
    @patch('src.environment_manager.venv.create')
    @patch('src.environment_manager.subprocess.run')
    def test_full_workflow(self, mock_run, mock_venv_create, integration_dir):
        """Test complete environment lifecycle"""
        # Setup mocks
        mock_venv_create.return_value = None
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='Python 3.11.0\n'
        )
        
        manager = EnvironmentManager(base_dir=integration_dir)
        
        # Create an environment
        project_dir = integration_dir / "my_project"
        project_dir.mkdir()
        
        env = manager.create_environment(
            name="integration_test",
            project_dir=project_dir
        )
        
        assert env is not None
        assert env.name == "integration_test"
        
        # List environments
        assert len(manager.environments) == 1
        
        # Export requirements
        req_file = integration_dir / "requirements.txt"
        manager.export_requirements("integration_test", req_file)
        assert req_file.exists()
        
        # Delete environment (mock confirmation)
        with patch('src.environment_manager.console.input', return_value="yes"):
            result = manager.delete_environment("integration_test")
            assert result is True
        
        # Verify deletion
        assert len(manager.environments) == 0


# Performance tests
class TestPerformance:
    """
    Performance tests to ensure operations remain fast.
    
    I added these after environment creation started taking
    too long and impacting productivity.
    
    Author: Cazzy Aporbo, MS
    """
    
    def test_load_large_environment_list(self, temp_dir):
        """Test loading many environments doesn't slow down"""
        import time
        
        # Create config with many environments
        config_file = temp_dir / 'environments.json'
        test_data = {}
        
        for i in range(100):
            test_data[f"env_{i}"] = {
                "name": f"env_{i}",
                "path": str(temp_dir / f"env_{i}"),
                "python_version": "3.11.0",
                "created_at": datetime.now().isoformat(),
                "packages": [f"package{j}==1.0.0" for j in range(50)],
                "is_active": False
            }
        
        with open(config_file, 'w') as f:
            json.dump(test_data, f)
        
        # Measure load time
        start_time = time.time()
        manager = EnvironmentManager(base_dir=temp_dir)
        load_time = time.time() - start_time
        
        # Should load in under 1 second even with 100 environments
        assert load_time < 1.0
        assert len(manager.environments) == 100


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=src", "--cov-report=term-missing"])
```

Now let's create the beautiful Streamlit app with our pastel aesthetic:

# 01-environments/app.py

```python
"""
Interactive Environment Manager Dashboard

Author: Cazzy Aporbo, MS
Created: January 2025

This Streamlit app provides a beautiful interface for managing Python environments.
I built this because command-line tools are great, but sometimes you want
a visual overview of your development setup.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import json
import subprocess
import platform
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.environment_manager import EnvironmentManager, VirtualEnvironment
from src.dependency_analyzer import DependencyAnalyzer

# Page configuration with our pastel theme
st.set_page_config(
    page_title="Velvet Python - Environment Manager",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for our pastel aesthetic
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #FFE4E1 0%, #F0F8FF 100%);
    }
    
    /* Headers with gradient text */
    h1 {
        background: linear-gradient(135deg, #8B7D8B 0%, #DDA0DD 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Georgia', serif;
        font-weight: 600;
    }
    
    h2 {
        color: #8B7D8B;
        font-family: 'Georgia', serif;
        border-bottom: 2px solid #E6E6FA;
        padding-bottom: 10px;
    }
    
    h3 {
        color: #706B70;
        font-family: 'Georgia', serif;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 228, 225, 0.5) 0%, rgba(237, 229, 255, 0.5) 100%);
        border: 1px solid #DDA0DD;
        padding: 15px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(139, 125, 139, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #E6E6FA 0%, #DDA0DD 100%);
        color: #4A4A4A;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(139, 125, 139, 0.3);
    }
    
    /* Success/Error messages */
    .success-box {
        background: linear-gradient(135deg, #98FB98 0%, #90EE90 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #228B22;
        margin: 10px 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #FFB6C1 0%, #FFA07A 100%);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #DC143C;
        margin: 10px 0;
    }
    
    /* Info boxes */
    .info-card {
        background: linear-gradient(135deg, #FFF0F5 0%, #FFEFD5 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #DDA0DD;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #FFF0F5 0%, #FFE4E1 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #FFE4E1 0%, #E6E6FA 100%);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #706B70;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: white;
        border-radius: 8px;
    }
    
    /* Dataframe styling */
    .dataframe {
        background: white;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #E6E6FA 0%, #F0E6FF 100%);
        border-radius: 10px;
        color: #4A4A4A;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'manager' not in st.session_state:
    st.session_state.manager = EnvironmentManager()

if 'show_success' not in st.session_state:
    st.session_state.show_success = False

if 'success_message' not in st.session_state:
    st.session_state.success_message = ""

# Header with gradient
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; margin-bottom: 0;">🎀 Velvet Python</h1>
    <p style="color: #8B7D8B; font-size: 1.2rem; font-style: italic;">Environment Management Dashboard</p>
    <p style="color: #706B70; font-size: 0.9rem;">by Cazzy Aporbo, MS</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="info-card">
        <h3 style="margin-top: 0;">Quick Actions</h3>
        <p style="color: #706B70; font-size: 0.9em;">
        Manage your Python environments with style
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    total_envs = len(st.session_state.manager.environments)
    active_envs = sum(1 for env in st.session_state.manager.environments.values() if env.is_active)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Environments", total_envs, delta=None)
    with col2:
        st.metric("Active", active_envs, delta=None)
    
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigate",
        ["Dashboard", "Create Environment", "Compare Environments", "Tools & Tips"],
        label_visibility="collapsed"
    )

# Main content area
if page == "Dashboard":
    # Dashboard view
    st.markdown("## Environment Overview")
    
    if not st.session_state.manager.environments:
        st.markdown("""
        <div class="info-card">
            <h3>No environments yet!</h3>
            <p>Get started by creating your first environment using the sidebar.</p>
            <p style="color: #8B7D8B;">
            I remember my first Python environment. It was a mess, but we all start somewhere!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Environment cards
        for name, env in st.session_state.manager.environments.items():
            with st.expander(f"**{name}** - Python {env.python_version}", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**Path:** `{env.path}`")
                    st.markdown(f"**Created:** {env.created_at.strftime('%B %d, %Y at %I:%M %p')}")
                    st.markdown(f"**Status:** {'🟢 Active' if env.is_active else '⚪ Inactive'}")
                
                with col2:
                    st.markdown(f"**Packages:** {len(env.packages)} installed")
                    if st.button(f"View Packages", key=f"view_{name}"):
                        st.session_state[f"show_packages_{name}"] = True
                    
                    if st.session_state.get(f"show_packages_{name}", False):
                        packages_df = pd.DataFrame(
                            [pkg.split('==') for pkg in env.packages if '==' in pkg],
                            columns=['Package', 'Version']
                        )
                        st.dataframe(packages_df, height=200)
                
                with col3:
                    if st.button("Delete", key=f"delete_{name}", type="secondary"):
                        if st.session_state.manager.delete_environment(name):
                            st.rerun()
        
        # Visualization section
        st.markdown("---")
        st.markdown("### Environment Statistics")
        
        # Create visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Python version distribution
            versions = [env.python_version for env in st.session_state.manager.environments.values()]
            version_counts = pd.Series(versions).value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=version_counts.index,
                    values=version_counts.values,
                    hole=0.4,
                    marker=dict(colors=['#FFE4E1', '#E6E6FA', '#F0E6FF', '#FFF0F5', '#FFEFD5'])
                )
            ])
            fig.update_layout(
                title="Python Versions",
                height=300,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Package count per environment
            env_data = []
            for name, env in st.session_state.manager.environments.items():
                env_data.append({
                    'Environment': name,
                    'Packages': len(env.packages)
                })
            
            if env_data:
                df = pd.DataFrame(env_data)
                fig = go.Figure(data=[
                    go.Bar(
                        x=df['Environment'],
                        y=df['Packages'],
                        marker=dict(
                            color=df['Packages'],
                            colorscale=[[0, '#FFE4E1'], [0.5, '#E6E6FA'], [1, '#DDA0DD']],
                            showscale=False
                        )
                    )
                ])
                fig.update_layout(
                    title="Package Count by Environment",
                    height=300,
                    xaxis_title="Environment",
                    yaxis_title="Number of Packages",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

elif page == "Create Environment":
    st.markdown("## Create New Environment")
    
    st.markdown("""
    <div class="info-card">
        <p><strong>Best Practice:</strong> I always use descriptive names for my environments. 
        Instead of 'env1', use 'blog-api' or 'data-analysis'. Future you will thank you!</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("create_env_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            env_name = st.text_input(
                "Environment Name",
                placeholder="my-awesome-project",
                help="Choose a descriptive name"
            )
            
            project_dir = st.text_input(
                "Project Directory",
                value=str(Path.cwd()),
                help="Where to create the .venv folder"
            )
        
        with col2:
            python_version = st.selectbox(
                "Python Version",
                ["System Default", "3.10", "3.11", "3.12"],
                help="Specific Python version to use"
            )
            
            requirements_file = st.text_input(
                "Requirements File (optional)",
                placeholder="requirements.txt",
                help="Path to requirements.txt"
            )
        
        dev_requirements = st.text_input(
            "Dev Requirements File (optional)",
            placeholder="requirements-dev.txt",
            help="Path to development requirements"
        )
        
        submitted = st.form_submit_button("Create Environment", type="primary")
        
        if submitted:
            if not env_name:
                st.error("Please provide an environment name!")
            else:
                with st.spinner(f"Creating environment '{env_name}'... This might take a minute."):
                    try:
                        req_path = Path(requirements_file) if requirements_file else None
                        dev_req_path = Path(dev_requirements) if dev_requirements else None
                        py_version = None if python_version == "System Default" else python_version
                        
                        env = st.session_state.manager.create_environment(
                            name=env_name,
                            python_version=py_version,
                            requirements=req_path,
                            dev_requirements=dev_req_path,
                            project_dir=Path(project_dir)
                        )
                        
                        st.success(f"Environment '{env_name}' created successfully!")
                        
                        # Show activation instructions
                        if platform.system() == "Windows":
                            activate_cmd = f"{env.path}\\Scripts\\activate"
                        else:
                            activate_cmd = f"source {env.path}/bin/activate"
                        
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>Next Steps:</h4>
                            <p>1. Open your terminal</p>
                            <p>2. Navigate to: <code>{project_dir}</code></p>
                            <p>3. Activate: <code>{activate_cmd}</code></p>
                            <p>4. Start coding! 🎉</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        time.sleep(2)
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error creating environment: {str(e)}")

elif page == "Compare Environments":
    st.markdown("## Compare Environments")
    
    if len(st.session_state.manager.environments) < 2:
        st.markdown("""
        <div class="info-card">
            <p>You need at least 2 environments to compare.</p>
            <p style="color: #8B7D8B;">
            I use this feature all the time to figure out why code works in one environment but not another.
            Usually it's a version mismatch!
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        col1, col2 = st.columns(2)
        
        env_names = list(st.session_state.manager.environments.keys())
        
        with col1:
            env1 = st.selectbox("First Environment", env_names, key="env1_select")
        
        with col2:
            env2 = st.selectbox("Second Environment", env_names, key="env2_select")
        
        if st.button("Compare", type="primary"):
            if env1 == env2:
                st.warning("Please select different environments to compare!")
            else:
                # Get package lists
                packages1 = st.session_state.manager.environments[env1].packages
                packages2 = st.session_state.manager.environments[env2].packages
                
                # Parse packages
                def parse_packages(packages):
                    result = {}
                    for pkg in packages:
                        if '==' in pkg:
                            name, version = pkg.split('==')
                            result[name] = version
                    return result
                
                pkgs1 = parse_packages(packages1)
                pkgs2 = parse_packages(packages2)
                
                # Find differences
                only_in_1 = set(pkgs1.keys()) - set(pkgs2.keys())
                only_in_2 = set(pkgs2.keys()) - set(pkgs1.keys())
                common = set(pkgs1.keys()) & set(pkgs2.keys())
                version_diff = {
                    pkg: (pkgs1[pkg], pkgs2[pkg])
                    for pkg in common
                    if pkgs1[pkg] != pkgs2[pkg]
                }
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"### Only in {env1}")
                    if only_in_1:
                        for pkg in sorted(only_in_1):
                            st.markdown(f"- {pkg} ({pkgs1[pkg]})")
                    else:
                        st.markdown("*No unique packages*")
                
                with col2:
                    st.markdown(f"### Only in {env2}")
                    if only_in_2:
                        for pkg in sorted(only_in_2):
                            st.markdown(f"- {pkg} ({pkgs2[pkg]})")
                    else:
                        st.markdown("*No unique packages*")
                
                with col3:
                    st.markdown("### Version Differences")
                    if version_diff:
                        for pkg, (v1, v2) in sorted(version_diff.items()):
                            st.markdown(f"- **{pkg}**")
                            st.markdown(f"  - {env1}: {v1}")
                            st.markdown(f"  - {env2}: {v2}")
                    else:
                        st.markdown("*No version differences*")

elif page == "Tools & Tips":
    st.markdown("## Tools & Tips")
    
    # Tool comparison
    st.markdown("### Tool Comparison")
    
    tools_data = {
        'Tool': ['venv', 'virtualenv', 'pipenv', 'poetry', 'conda', 'uv'],
        'Speed': ['⭐⭐⭐', '⭐⭐⭐', '⭐', '⭐⭐', '⭐', '⭐⭐⭐⭐⭐'],
        'Ease of Use': ['⭐⭐⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐⭐⭐⭐'],
        'Features': ['⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐⭐⭐', '⭐⭐⭐'],
        'My Rating': ['⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐⭐']
    }
    
    df = pd.DataFrame(tools_data)
    st.dataframe(df, use_container_width=True)
    
    # Tips section
    st.markdown("### My Best Practices")
    
    tips = [
        ("Always use virtual environments", 
         "Even for 'simple' scripts. I learned this after breaking system Python."),
        
        ("Pin your dependencies", 
         "Use exact versions (requests==2.31.0) not ranges (requests>=2.30.0)."),
        
        ("Separate dev and prod requirements", 
         "Keep testing tools out of production. Use requirements-dev.txt."),
        
        ("Document Python version", 
         "Add .python-version file or specify in README. Saves debugging time."),
        
        ("Use .env for secrets", 
         "Never commit credentials. Use python-dotenv for configuration."),
        
        ("Automate environment setup", 
         "Create a setup.sh script. Your future self will thank you."),
    ]
    
    for i, (title, desc) in enumerate(tips):
        with st.expander(f"**Tip {i+1}:** {title}"):
            st.markdown(f"""
            <div class="info-card">
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Common problems
    st.markdown("### Common Problems & Solutions")
    
    problems = {
        "pip: command not found": "Use `python -m pip` or `python3 -m pip` instead.",
        "Virtual environment not activating": "On Windows, run `Set-ExecutionPolicy RemoteSigned` first.",
        "Wrong Python version": "Specify version explicitly: `python3.11 -m venv .venv`",
        "Dependency conflicts": "Use pip-tools or poetry for better dependency resolution.",
        "Can't install packages": "Upgrade pip first: `pip install --upgrade pip`"
    }
    
    for problem, solution in problems.items():
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #FFE4E1 0%, #FFB6C1 100%); 
                    padding: 15px; border-radius: 10px; margin: 10px 0;">
            <strong>Problem:</strong> {problem}<br>
            <strong>Solution:</strong> {solution}
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8B7D8B; padding: 2rem 0;">
    <p>Built with passion by Cazzy Aporbo, MS</p>
    <p style="font-size: 0.9em;">Part of the Velvet Python learning journey</p>
</div>
""", unsafe_allow_html=True)
```

This completes the `01-environments` module with:

1. **Comprehensive tests** - Unit, integration, and performance tests
2. **Beautiful Streamlit app** - Full pastel aesthetic with gradients and styled components
3. **Interactive features** - Create, compare, and manage environments visually
4. **Educational content** - Tips, best practices, and problem solutions
5. **Professional polish** - Consistent styling throughout

The app provides a gorgeous interface for environment management while maintaining the educational focus and personal touch you wanted.
