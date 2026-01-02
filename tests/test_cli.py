import pytest
from unittest.mock import patch, MagicMock
import sys
import shutil
from pathlib import Path

# Need to import these to patch them where they are used
import cli
import application_state
from application_state import state

@pytest.fixture
def mock_sys_argv(monkeypatch):
    def _set_argv(args):
        monkeypatch.setattr(sys, 'argv', ['delta'] + args)
    return _set_argv

@pytest.fixture
def mock_processing(monkeypatch):
    """Mocks core processing functions used by cli."""
    mock_proc = MagicMock(return_value={"success": True, "backup_id": "bak_1", "message": "OK"})
    monkeypatch.setattr("cli.process_request", mock_proc)
    
    # Also need to mock get_available_backups for undo/review tests
    mock_backups = MagicMock(return_value=[
        {"session_id": "sess_1", "timestamp": "2023-01-01", "files": ["a.py"]}
    ])
    monkeypatch.setattr("cli.get_available_backups", mock_backups)
    
    mock_undo = MagicMock()
    monkeypatch.setattr("cli.undo_last_changes", mock_undo)
    
    mock_report = MagicMock()
    monkeypatch.setattr("cli.open_diff_report", mock_report)
    
    return {
        "process_request": mock_proc,
        "get_available_backups": mock_backups,
        "undo_last_changes": mock_undo,
        "open_diff_report": mock_report
    }

@pytest.fixture(autouse=True)
def mock_persistence_paths(tmp_path):
    """Patch persistence paths to temporary files."""
    with patch("application_state.PRESETS_PATH", str(tmp_path / "presets.json")):
        yield

@pytest.fixture(autouse=True)
def clean_state():
    """Reset app state."""
    application_state.init_app_state()
    yield
    application_state.init_app_state()

def test_run_command_basic(mock_sys_argv, mock_processing, temp_cwd):
    (temp_cwd / "main.py").touch()
    
    mock_sys_argv(["run", "Refactor Code", "main.py"])
    cli.run_cli()
    
    mock_processing["process_request"].assert_called_once()
    args = mock_processing["process_request"].call_args[1]
    
    assert args["prompt"] == "Refactor Code"
    files = args["files"]
    assert len(files) == 1
    assert "main.py" in files[0]

def test_run_command_options(mock_sys_argv, mock_processing, temp_cwd):
    (temp_cwd / "main.py").touch()
    
    mock_sys_argv([
        "run", "Fix", "main.py", 
        "--model", "gpt-4",
        "--tries", "5",
        "--recurse", "1",
        "--timeout", "30",
        "--validate", "pytest",
        "--verify",
        "--review"
    ])
    
    cli.run_cli()
    
    args = mock_processing["process_request"].call_args[1]
    assert args["max_retries"] == 5
    assert args["recursion_limit"] == 1
    assert args["validation_timeout"] == 30.0
    assert args["validation_cmd"] == "pytest"
    assert args["verify"] is True
    
    # Review should be triggered if success
    mock_processing["open_diff_report"].assert_called()

def test_ask_command(mock_sys_argv, mock_processing, temp_cwd):
    (temp_cwd / "main.py").touch()
    mock_sys_argv(["ask", "Why?", "main.py"])
    cli.run_cli()
    
    args = mock_processing["process_request"].call_args[1]
    assert args["ask_mode"] is True
    assert args["files"] # should have files

def test_plan_command(mock_sys_argv, mock_processing, temp_cwd):
    mock_sys_argv(["plan", "Plan it", "*.py"])
    (temp_cwd / "script.py").touch()
    
    cli.run_cli()
    
    args = mock_processing["process_request"].call_args[1]
    assert args["plan_mode"] is True
    assert args["ask_mode"] is False

def test_state_management(mock_sys_argv, temp_cwd):
    (temp_cwd / "a.txt").touch()
    (temp_cwd / "b.txt").touch()
    
    # ADD
    mock_sys_argv(["add", "*.txt"])
    cli.run_cli()
    assert len(state.selected_files) == 2
    
    # REMOVE
    mock_sys_argv(["remove", "a.txt"])
    cli.run_cli()
    assert len(state.selected_files) == 1
    assert "b.txt" in str(list(state.selected_files)[0])
    
    # CLEAR
    mock_sys_argv(["clear"])
    cli.run_cli()
    assert len(state.selected_files) == 0

def test_add_preset(mock_sys_argv, temp_cwd):
    (temp_cwd / "core.py").touch()
    mock_sys_argv(["add", "core.py", "-p", "core_files"])
    
    cli.run_cli()
    
    assert "core_files" in state.presets
    assert len(state.presets["core_files"]["files"]) == 1
    assert "core.py" in state.presets["core_files"]["files"][0]

def test_run_with_preset(mock_sys_argv, mock_processing, temp_cwd):
    # Setup state manually and save it
    state.presets["my_preset"] = {"files": ["saved.py"]}
    application_state.save_presets()

    (temp_cwd / "saved.py").touch()
    
    mock_sys_argv(["run", "do it", "-p", "my_preset"])
    cli.run_cli()
    
    args = mock_processing["process_request"].call_args[1]
    assert len(args["files"]) == 1
    assert "saved.py" in args["files"][0]

def test_undo_command(mock_sys_argv, mock_processing):
    mock_sys_argv(["undo"])
    cli.run_cli()
    mock_processing["undo_last_changes"].assert_called_once()

def test_backups_command(mock_sys_argv, mock_processing, capsys):
    mock_sys_argv(["backups"])
    cli.run_cli()
    mock_processing["get_available_backups"].assert_called()
    captured = capsys.readouterr()
    assert "Available backups" in captured.out
    assert "sess_1" in captured.out

def test_review_command(mock_sys_argv, mock_processing):
    # Review latest
    mock_sys_argv(["review"])
    cli.run_cli()
    mock_processing["open_diff_report"].assert_called_with("sess_1")
    
    # Review range
    # session list in mock is length 1. 0..1 means index 0 to 1 -> [0]
    mock_sys_argv(["review", "0..1"])
    cli.run_cli()
    # called with list of session ids
    assert mock_processing["open_diff_report"].call_args[0][0] == ["sess_1"]

def test_run_failure_exit(mock_sys_argv, mock_processing, temp_cwd):
    (temp_cwd / "f.py").touch()
    mock_processing["process_request"].return_value = {"success": False, "message": "Failed"}
    
    mock_sys_argv(["run", "fail", "f.py"])
    
    with pytest.raises(SystemExit) as e:
        cli.run_cli()
    assert e.type == SystemExit
    assert e.value.code == 1

def test_missing_files_warning(mock_sys_argv, temp_cwd, capsys):
    # No files, no saved state
    mock_sys_argv(["run", "whatever"])
    
    with pytest.raises(SystemExit) as e:
        cli.run_cli()
        
    captured = capsys.readouterr()
    assert "Error: No files specified" in captured.err