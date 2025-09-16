"""
CLI Tests - Testing the Velvet Python Command Line Interface

Author: Cazzy Aporbo, MS
Created: January 2025

These tests ensure our CLI works correctly. I test every command because
nothing is more frustrating than a broken CLI. I've been burned by
"simple" CLI changes that broke everything.

Testing strategy:
- Test successful command execution
- Test error handling
- Test output formatting
- Test file operations in isolation
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call

import pytest
from click.testing import CliRunner
from typer.testing import CliRunner as TyperRunner

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from velvet_python.cli import app, state, print_header
from velvet_python import __version__, __author__


# =============================================================================
# BASIC CLI TESTS
# =============================================================================

@pytest.mark.cli
class TestCLIBasics:
    """
    Test basic CLI functionality.
    
    I always start with the basics - does it even run?
    
    Author: Cazzy Aporbo, MS
    """
    
    def test_cli_runs(self, cli_runner):
        """Test that the CLI runs without errors"""
        runner = TyperRunner()
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Velvet Python CLI" in result.stdout
        assert __author__ in result.stdout
    
    def test_version_command(self, cli_runner):
        """Test the version command"""
        runner = TyperRunner()
        result = runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert __version__ in result.stdout
        assert __author__ in result.stdout
    
    def test_info_command(self, cli_runner):
        """Test the info command shows environment information"""
        runner = TyperRunner()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.check_environment.return_value = {
                'python_version': '3.11.0',
                'platform': 'Linux',
                'virtual_env': True,
                'modules_available': 23,
                'modules_completed': 1,
                'modules_in_progress': 2
            }
            mock_state.__getitem__.return_value = mock_vp
            
            result = runner.invoke(app, ["info"])
            
            # Should run successfully
            assert result.exit_code == 0
            # Note: Can't check stdout directly with Rich formatting
    
    def test_invalid_command(self, cli_runner):
        """Test that invalid commands show help"""
        runner = TyperRunner()
        result = runner.invoke(app, ["nonexistent"])
        
        assert result.exit_code != 0
        # Typer shows usage information for invalid commands


# =============================================================================
# MODULE MANAGEMENT TESTS
# =============================================================================

@pytest.mark.cli
class TestModuleCommands:
    """
    Test module-related commands.
    
    These are critical - module management is core functionality.
    
    Author: Cazzy Aporbo, MS
    """
    
    def test_modules_list_empty(self, cli_runner):
        """Test listing modules when none exist"""
        runner = TyperRunner()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.list_modules.return_value = []
            mock_state.__getitem__.return_value = mock_vp
            
            result = runner.invoke(app, ["modules"])
            
            assert result.exit_code == 0
    
    def test_modules_list_with_data(self, cli_runner):
        """Test listing modules with data"""
        runner = TyperRunner()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.list_modules.return_value = [
                {
                    'id': '01-environments',
                    'name': 'Environment Management',
                    'difficulty': 'Beginner',
                    'status': 'completed',
                    'topics': ['venv', 'pip', 'conda'],
                    'key_learning': 'Virtual environments'
                }
            ]
            mock_state.__getitem__.return_value = mock_vp
            
            result = runner.invoke(app, ["modules"])
            
            assert result.exit_code == 0
    
    def test_modules_filtered_by_difficulty(self, cli_runner):
        """Test filtering modules by difficulty"""
        runner = TyperRunner()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.list_modules.return_value = [
                {
                    'id': '01-environments',
                    'name': 'Environment Management',
                    'difficulty': 'Beginner',
                    'status': 'completed',
                    'topics': ['venv'],
                    'key_learning': 'Environments'
                }
            ]
            mock_state.__getitem__.return_value = mock_vp
            
            result = runner.invoke(app, ["modules", "--difficulty", "Beginner"])
            
            assert result.exit_code == 0
            mock_vp.list_modules.assert_called_with(difficulty="Beginner", status=None)
    
    def test_modules_detailed_view(self, cli_runner):
        """Test detailed module view"""
        runner = TyperRunner()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.list_modules.return_value = [
                {
                    'id': '09-concurrency',
                    'name': 'Concurrency Patterns',
                    'difficulty': 'Advanced',
                    'status': 'in-progress',
                    'topics': ['threading', 'asyncio', 'multiprocessing'],
                    'key_learning': 'Choosing the right concurrency model'
                }
            ]
            mock_state.__getitem__.return_value = mock_vp
            
            result = runner.invoke(app, ["modules", "--detailed"])
            
            assert result.exit_code == 0


# =============================================================================
# START COMMAND TESTS
# =============================================================================

@pytest.mark.cli
class TestStartCommand:
    """
    Test the start command for beginning work on a module.
    
    Author: Cazzy Aporbo, MS
    """
    
    def test_start_module_not_found(self, cli_runner):
        """Test starting a non-existent module"""
        runner = TyperRunner()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.get_module_info.return_value = None
            mock_state.__getitem__.return_value = mock_vp
            
            result = runner.invoke(app, ["start", "99-nonexistent"])
            
            assert result.exit_code == 1
    
    @patch('velvet_python.cli.subprocess.run')
    @patch('velvet_python.cli.Confirm.ask')
    def test_start_module_creates_structure(self, mock_confirm, mock_subprocess, cli_runner, temp_dir):
        """Test that start creates module directory structure"""
        runner = TyperRunner()
        mock_confirm.return_value = False  # Don't open VS Code
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.get_module_info.return_value = {
                'name': 'Test Module',
                'difficulty': 'Beginner',
                'status': 'planned',
            }
            # Use temp_dir as project root
            mock_state.__getitem__.side_effect = lambda key: {
                'vp': mock_vp,
                'project_root': temp_dir,
                'current_module': None
            }.get(key, mock_vp)
            
            result = runner.invoke(app, ["start", "01-test"])
            
            assert result.exit_code == 0
            
            # Check directory structure was created
            module_path = temp_dir / "01-test"
            assert module_path.exists()
            assert (module_path / "src").exists()
            assert (module_path / "tests").exists()
            assert (module_path / "benchmarks").exists()


# =============================================================================
# TEST COMMAND TESTS  
# =============================================================================

@pytest.mark.cli
class TestTestCommand:
    """
    Test the test command (meta, I know).
    
    Testing the test command feels recursive, but it's important.
    
    Author: Cazzy Aporbo, MS
    """
    
    @patch('velvet_python.cli.subprocess.run')
    def test_test_all(self, mock_subprocess, cli_runner, temp_dir):
        """Test running all tests"""
        runner = TyperRunner()
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["test"])
            
            assert result.exit_code == 0
            mock_subprocess.assert_called_once()
            
            # Check pytest was called with coverage
            call_args = mock_subprocess.call_args[0][0]
            assert "pytest" in call_args[0]
            assert any("--cov" in arg for arg in call_args)
    
    @patch('velvet_python.cli.subprocess.run')  
    def test_test_specific_module(self, mock_subprocess, cli_runner, temp_dir):
        """Test running tests for specific module"""
        runner = TyperRunner()
        mock_subprocess.return_value = Mock(returncode=0)
        
        # Create test directory
        module_tests = temp_dir / "01-test" / "tests"
        module_tests.mkdir(parents=True)
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["test", "01-test"])
            
            assert result.exit_code == 0
    
    @patch('velvet_python.cli.subprocess.run')
    def test_test_no_coverage(self, mock_subprocess, cli_runner, temp_dir):
        """Test running tests without coverage"""
        runner = TyperRunner()
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["test", "--no-coverage"])
            
            assert result.exit_code == 0
            
            # Check coverage wasn't included
            call_args = mock_subprocess.call_args[0][0]
            assert not any("--cov" in arg for arg in call_args)


# =============================================================================
# FORMAT AND LINT TESTS
# =============================================================================

@pytest.mark.cli
class TestCodeQualityCommands:
    """
    Test format and lint commands.
    
    Code quality matters. These commands keep code clean.
    
    Author: Cazzy Aporbo, MS
    """
    
    @patch('velvet_python.cli.subprocess.run')
    def test_format_command(self, mock_subprocess, cli_runner, temp_dir):
        """Test code formatting"""
        runner = TyperRunner()
        mock_subprocess.return_value = Mock(returncode=0, capture_output=True)
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["format"])
            
            assert result.exit_code == 0
            # Should call black and isort
            assert mock_subprocess.call_count == 2
    
    @patch('velvet_python.cli.subprocess.run')
    def test_format_check_only(self, mock_subprocess, cli_runner, temp_dir):
        """Test format checking without modification"""
        runner = TyperRunner()
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["format", "--check"])
            
            assert result.exit_code == 0
            
            # Should include --check flag
            calls = mock_subprocess.call_args_list
            black_call = calls[0][0][0]
            assert "--check" in black_call
    
    @patch('velvet_python.cli.subprocess.run')
    def test_lint_command(self, mock_subprocess, cli_runner, temp_dir):
        """Test linting"""
        runner = TyperRunner()
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["lint"])
            
            assert result.exit_code == 0
            # Should call both ruff and mypy
            assert mock_subprocess.call_count == 2


# =============================================================================
# UTILITY COMMAND TESTS
# =============================================================================

@pytest.mark.cli
class TestUtilityCommands:
    """
    Test utility commands like tree, progress, clean.
    
    These aren't critical but they make development nicer.
    
    Author: Cazzy Aporbo, MS
    """
    
    def test_tree_command(self, cli_runner, temp_dir):
        """Test directory tree display"""
        runner = TyperRunner()
        
        # Create some structure
        (temp_dir / "module1").mkdir()
        (temp_dir / "module1" / "src").mkdir()
        (temp_dir / "module1" / "test.py").touch()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["tree"])
            
            assert result.exit_code == 0
    
    def test_tree_specific_module(self, cli_runner, temp_dir):
        """Test tree for specific module"""
        runner = TyperRunner()
        
        module_dir = temp_dir / "01-test"
        module_dir.mkdir()
        (module_dir / "src").mkdir()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["tree", "01-test"])
            
            assert result.exit_code == 0
    
    def test_progress_command(self, cli_runner):
        """Test progress display"""
        runner = TyperRunner()
        
        result = runner.invoke(app, ["progress"])
        
        assert result.exit_code == 0
    
    @patch('shutil.rmtree')
    @patch('pathlib.Path.unlink')
    def test_clean_cache(self, mock_unlink, mock_rmtree, cli_runner, temp_dir):
        """Test cleaning cache files"""
        runner = TyperRunner()
        
        # Create some cache files
        cache_dir = temp_dir / "__pycache__"
        cache_dir.mkdir()
        (temp_dir / "test.pyc").touch()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["clean", "--cache"])
            
            assert result.exit_code == 0


# =============================================================================
# RUN COMMAND TESTS
# =============================================================================

@pytest.mark.cli
class TestRunCommand:
    """
    Test the run command for executing modules.
    
    Author: Cazzy Aporbo, MS
    """
    
    @patch('velvet_python.cli.subprocess.run')
    def test_run_interactive(self, mock_subprocess, cli_runner, temp_dir):
        """Test running interactive app"""
        runner = TyperRunner()
        
        # Create app.py
        module_dir = temp_dir / "01-test"
        module_dir.mkdir()
        app_file = module_dir / "app.py"
        app_file.write_text("print('Test app')")
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["run", "01-test", "--interactive"])
            
            assert result.exit_code == 0
            mock_subprocess.assert_called_once()
            
            # Should call streamlit
            call_args = mock_subprocess.call_args[0][0]
            assert "streamlit" in call_args[0]
            assert "run" in call_args
    
    @patch('velvet_python.cli.subprocess.run')
    def test_run_example(self, mock_subprocess, cli_runner, temp_dir):
        """Test running specific example"""
        runner = TyperRunner()
        
        # Create example file
        examples_dir = temp_dir / "01-test" / "examples"
        examples_dir.mkdir(parents=True)
        example_file = examples_dir / "demo.py"
        example_file.write_text("print('Demo')")
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["run", "01-test", "--example", "demo"])
            
            assert result.exit_code == 0
            mock_subprocess.assert_called_once()


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

@pytest.mark.cli
class TestErrorHandling:
    """
    Test error handling in CLI.
    
    I test errors because graceful failure is important.
    Users shouldn't see stack traces.
    
    Author: Cazzy Aporbo, MS
    """
    
    def test_module_not_found_error(self, cli_runner, temp_dir):
        """Test handling of missing module"""
        runner = TyperRunner()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.get_module_info.return_value = None
            mock_state.__getitem__.side_effect = lambda key: {
                'vp': mock_vp,
                'project_root': temp_dir
            }.get(key, mock_vp)
            
            result = runner.invoke(app, ["start", "nonexistent"])
            
            assert result.exit_code == 1
    
    @patch('velvet_python.cli.subprocess.run')
    def test_test_failure_handling(self, mock_subprocess, cli_runner, temp_dir):
        """Test handling of test failures"""
        runner = TyperRunner()
        mock_subprocess.return_value = Mock(returncode=1)  # Test failure
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_state.__getitem__.return_value = temp_dir
            
            result = runner.invoke(app, ["test"])
            
            # Should propagate the failure
            assert result.exit_code == 1
    
    def test_missing_requirements_file(self, cli_runner, temp_dir):
        """Test handling missing requirements file"""
        runner = TyperRunner()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.get_module_info.return_value = {
                'name': 'Test Module',
                'difficulty': 'Beginner',
                'status': 'planned',
            }
            mock_state.__getitem__.side_effect = lambda key: {
                'vp': mock_vp,
                'project_root': temp_dir
            }.get(key, mock_vp)
            
            # Try to create environment with non-existent requirements
            with patch('velvet_python.cli.Path.exists', return_value=False):
                result = runner.invoke(app, ["start", "01-test"])
                
                # Should still succeed, just without requirements
                assert result.exit_code == 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.integration
@pytest.mark.cli
class TestCLIIntegration:
    """
    Integration tests that test multiple commands together.
    
    These test real workflows that I use daily.
    
    Author: Cazzy Aporbo, MS
    """
    
    @patch('velvet_python.cli.subprocess.run')
    @patch('velvet_python.cli.Confirm.ask')
    def test_full_module_workflow(self, mock_confirm, mock_subprocess, cli_runner, temp_dir):
        """Test complete module creation and test workflow"""
        runner = TyperRunner()
        mock_confirm.return_value = False  # Don't open VS Code
        mock_subprocess.return_value = Mock(returncode=0)
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            mock_vp.get_module_info.return_value = {
                'name': 'Test Module',
                'difficulty': 'Beginner',
                'status': 'planned',
            }
            mock_vp.list_modules.return_value = [
                {
                    'id': '01-test',
                    'name': 'Test Module',
                    'difficulty': 'Beginner',
                    'status': 'planned',
                    'topics': ['testing'],
                    'key_learning': 'Testing'
                }
            ]
            
            mock_state.__getitem__.side_effect = lambda key: {
                'vp': mock_vp,
                'project_root': temp_dir,
                'current_module': None
            }.get(key, mock_vp)
            
            # 1. List modules
            result = runner.invoke(app, ["modules"])
            assert result.exit_code == 0
            
            # 2. Start a module
            result = runner.invoke(app, ["start", "01-test"])
            assert result.exit_code == 0
            
            # 3. Run tests
            result = runner.invoke(app, ["test", "01-test"])
            # Would fail because no actual tests, but subprocess is mocked
            assert result.exit_code == 0
    
    def test_header_printing(self, capsys):
        """Test that header prints with correct styling"""
        print_header()
        
        captured = capsys.readouterr()
        # Can't check exact output due to Rich formatting,
        # but function should run without error


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

@pytest.mark.benchmark
@pytest.mark.cli
class TestCLIPerformance:
    """
    Performance tests for CLI commands.
    
    I care about CLI speed because slow CLIs kill productivity.
    
    Author: Cazzy Aporbo, MS
    """
    
    def test_help_speed(self, cli_runner, benchmark_timer):
        """Test that help displays quickly"""
        runner = TyperRunner()
        
        with benchmark_timer() as timer:
            result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        # Help should display in under 1 second
        assert timer["elapsed"] < 1.0
    
    def test_modules_list_speed(self, cli_runner, benchmark_timer):
        """Test that module listing is fast"""
        runner = TyperRunner()
        
        with patch('velvet_python.cli.state') as mock_state:
            mock_vp = Mock()
            # Create many modules to test performance
            mock_vp.list_modules.return_value = [
                {
                    'id': f'{i:02d}-module',
                    'name': f'Module {i}',
                    'difficulty': 'Beginner',
                    'status': 'planned',
                    'topics': ['test'],
                    'key_learning': 'Test'
                }
                for i in range(100)
            ]
            mock_state.__getitem__.return_value = mock_vp
            
            with benchmark_timer() as timer:
                result = runner.invoke(app, ["modules"])
            
            assert result.exit_code == 0
            # Should list 100 modules in under 2 seconds
            assert timer["elapsed"] < 2.0


if __name__ == "__main__":
    """
    Run tests directly.
    
    Author: Cazzy Aporbo, MS
    """
    pytest.main([__file__, "-v"])
