import pytest
import shutil
import tempfile
import os
import stat
from pathlib import Path
from unittest.mock import patch

# Helper for Windows permission removal
def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

@pytest.fixture
def temp_cwd():
    """Create a temporary directory and change CWD to it."""
    orig_cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp()
    os.chdir(temp_dir)
    yield Path(temp_dir)
    os.chdir(orig_cwd)
    shutil.rmtree(temp_dir, onerror=remove_readonly)

@pytest.fixture
def mock_settings(monkeypatch):
    """Mock core settings."""
    import core
    # Patch default settings dict
    settings = {
        "backup_enabled": True,
        "use_git_backup": False,
        "git_backup_branch": "delta-backup-test"
    }
    monkeypatch.setattr(core, "_settings", settings)
    monkeypatch.setattr(core.config, "backup_enabled", True)
    return settings

@pytest.fixture(autouse=True)
def mock_app_data(tmp_path):
    """Redirect all AppData writes to a temp directory."""
    temp_app_data = tmp_path / "deltatool_test_appdata"
    temp_app_data.mkdir(parents=True, exist_ok=True)
    (temp_app_data / "sessions").mkdir(exist_ok=True)
    
    with patch("core.APP_DATA_DIR", temp_app_data), \
         patch("core.SETTINGS_PATH", temp_app_data / "settings.json"), \
         patch("core.BACKUP_DIR", str(temp_app_data / "backups")), \
         patch("application_state.APP_DATA_DIR", temp_app_data), \
         patch("application_state.SESSIONS_DIR", temp_app_data / "sessions"), \
         patch("application_state.FILESET_PATH", str(temp_app_data / "filesets.json")), \
         patch("application_state.PROMPT_HISTORY_PATH", str(temp_app_data / "prompt_history.json")), \
         patch("application_state.CWD_HISTORY_PATH", str(temp_app_data / "cwd_history.json")), \
         patch("application_state.PRESETS_PATH", str(temp_app_data / "selection_presets.json")):
         yield temp_app_data

def stub_to_diff(text: str) -> str:
    """
    Convert a safe stub format to the actual diff format used by the tool.

    This allows us to write tests with the tool itself.

    Stub format:
    filename
    [SEARCH]
    code
    [REPLACE]
    code
    [END]
    """

    text = text.strip()

    # First, handle the wrapping code blocks if implied, or just raw replacement
    # Using specific replacements to avoid creating actual diff markers in this source code
    text = text.replace("[SEARCH]", "<<<<<<< SEARCH")
    text = text.replace("[REPLACE]", "=======")
    text = text.replace("[END]", ">>>>>>> REPLACE")

    text = text.replace("[PLAN]", "<<<<<<< PLAN")
    text = text.replace("[PLAN_END]", ">>>>>>> END")
    return text

def wrap_in_code_block(content: str) -> str:
    """Wrap content in markdown code block."""
    return f"```\n{content}\n```"
