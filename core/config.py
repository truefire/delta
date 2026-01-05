"""Configuration and constants for Delta Tool."""
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

# Constants
DEFAULT_HIDDEN = {
    ".git", ".svn", ".hg", ".DS_Store", "Thumbs.db",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".vscode", ".idea", ".vs",
    "venv", ".venv", "env", "node_modules", "site-packages", # Python/Node
    "dist", "build", "target", "out", "bin", "obj", # Build artifacts
    "vendor", "coverage"
}

logger = logging.getLogger(__name__)

# Basic path helpers needed for config loading
def get_app_data_dir() -> Path:
    """Get the application data directory for the current platform."""
    home = Path.home()
    if sys.platform == "win32":
        return Path(os.getenv("APPDATA") or home / "AppData" / "Roaming") / "deltatool"
    elif sys.platform == "darwin":
        return home / "Library" / "Application Support" / "deltatool"
    return Path(os.getenv("XDG_CONFIG_HOME") or home / ".config") / "deltatool"

APP_DATA_DIR = get_app_data_dir()
APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR = str(APP_DATA_DIR / "backups")

# _TOOL_DIR is relative to this file (core/config.py) -> parent (core) -> parent (deltatool)
_TOOL_DIR = Path(__file__).parent.parent.resolve()

SETTINGS_FILENAME = "settings.json"
DEFAULT_SETTINGS_FILENAME = "default_settings.json"
SETTINGS_PATH = APP_DATA_DIR / SETTINGS_FILENAME
DEFAULT_SETTINGS_PATH = _TOOL_DIR / DEFAULT_SETTINGS_FILENAME

def load_json_file(path: Path | str, default: Any = None) -> Any:
    """Load a JSON file safely."""
    try:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        pass
    return default

def save_json_file(path: Path | str, data: Any, indent: int = 2) -> bool:
    """Save data to a JSON file safely."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON {path}: {e}")
        return False

def _load_settings() -> dict:
    """Load settings from settings.json."""
    if not SETTINGS_PATH.exists() and DEFAULT_SETTINGS_PATH.exists():
        try:
            shutil.copy2(DEFAULT_SETTINGS_PATH, SETTINGS_PATH)
        except Exception: pass

    data = load_json_file(SETTINGS_PATH)
    if data: return data
    
    data = load_json_file(DEFAULT_SETTINGS_PATH)
    return data or {}

def _save_settings(settings: dict) -> None:
    """Save settings to settings.json."""
    save_json_file(SETTINGS_PATH, settings)

_settings = _load_settings()

API_KEY = (
    _settings.get("api_key") 
    or os.environ.get("DELTATOOL_API_KEY") 
    or os.environ.get("OPENROUTER_API_KEY") 
    or os.environ.get("OPENAI_API_KEY") 
    or ""
)
API_BASE_URL = _settings.get("api_base_url", "https://openrouter.ai/api/v1")
TOKENS_PER_CHAR_ESTIMATE = _settings.get("tokens_per_char_estimate", 4)
DEFAULT_MODEL = _settings.get("default_model", "google/gemini-3-pro-preview")
AVAILABLE_MODELS = _settings.get("available_models", {
    "google/gemini-3-pro-preview": {"input": 2.00, "output": 12.00},
})
MAX_PROMPT_HISTORY = _settings.get("max_prompt_history", 50)
QUEUE_POLL_INTERVAL_MS = _settings.get("queue_poll_interval_ms", 100)

def update_core_settings(api_key: str, base_url: str, models: dict, git_branch: str) -> None:
    """Update and save core API settings."""
    global API_KEY, API_BASE_URL
    
    _settings["api_key"] = api_key
    _settings["api_base_url"] = base_url
    _settings["available_models"] = models
    _settings["git_backup_branch"] = git_branch
    
    _save_settings(_settings)
    
    API_KEY = api_key
    API_BASE_URL = base_url
    
    AVAILABLE_MODELS.clear()
    AVAILABLE_MODELS.update(models)
    
    config.set_git_backup_branch(git_branch)

class DeltaToolConfig:
    """Configuration for the delta tool."""

    def __init__(self):
        self.model = _settings.get("default_model", DEFAULT_MODEL)
        self.backup_enabled = _settings.get("backup_enabled", True)
        self.use_git_backup = _settings.get("use_git_backup", False)
        self.git_backup_branch = _settings.get("git_backup_branch", "delta-backup")
        self.block_on_fail = _settings.get("block_on_fail", False)

        fm = _settings.get("focus_mode", "off")
        if isinstance(fm, bool):
            fm = "flash" if fm else "off"
        self.focus_mode = fm
        self.focus_trigger = _settings.get("focus_trigger", "task")

        self.auto_review = _settings.get("auto_review", False)
        self.persist_session = _settings.get("persist_session", True)
        self.theme = _settings.get("theme", "light")

        self.default_tries = _settings.get("default_tries", 2)
        self.default_recurse = _settings.get("default_recurse", 0)
        self.default_timeout = _settings.get("default_timeout", 10)
        self.dig_max_turns = _settings.get("dig_max_turns", _settings.get("filedig_max_turns", 200))
        self.default_ambiguous_mode = _settings.get("default_ambiguous_mode", "replace_all")

        self.diff_fuzzy_lines_threshold = _settings.get("diff_fuzzy_lines_threshold", 0.95)
        self.diff_fuzzy_max_bad_lines = _settings.get("diff_fuzzy_max_bad_lines", 1)

        self.validation_failure_behavior = _settings.get("validation_failure_behavior", "correct")

        self.verify_changes = _settings.get("verify_changes", False)
        self.require_approval = _settings.get("require_approval", False)
        self.validate_at_start = _settings.get("validate_at_start", False)
        self.add_new_files = _settings.get("add_new_files", True)

        self.output_sharding_limit = _settings.get("output_sharding_limit", 60000)
        self.max_shards = _settings.get("max_shards", 5)
        self.sharding_ratio = _settings.get("sharding_ratio", 0.9)
        self.extra_system_prompt = _settings.get("extra_system_prompt", "")
        self.allow_rewrite = _settings.get("allow_rewrite", False)
        self.has_seen_tutorial = _settings.get("has_seen_tutorial", False)

    def set_has_seen_tutorial(self, seen: bool) -> None:
        self.has_seen_tutorial = seen
        _settings["has_seen_tutorial"] = seen
        _save_settings(_settings)

    def set_allow_rewrite(self, enabled: bool) -> None:
        self.allow_rewrite = enabled
        _settings["allow_rewrite"] = enabled
        _save_settings(_settings)

    def set_extra_system_prompt(self, prompt: str) -> None:
        self.extra_system_prompt = prompt
        _settings["extra_system_prompt"] = prompt
        _save_settings(_settings)

    def set_verify_changes(self, enabled: bool) -> None:
        self.verify_changes = enabled
        _settings["verify_changes"] = enabled
        _save_settings(_settings)

    def set_require_approval(self, enabled: bool) -> None:
        self.require_approval = enabled
        _settings["require_approval"] = enabled
        _save_settings(_settings)

    def set_validate_at_start(self, enabled: bool) -> None:
        self.validate_at_start = enabled
        _settings["validate_at_start"] = enabled
        _save_settings(_settings)

    def set_add_new_files(self, enabled: bool) -> None:
        self.add_new_files = enabled
        _settings["add_new_files"] = enabled
        _save_settings(_settings)

    def set_auto_review(self, enabled: bool) -> None:
        self.auto_review = enabled
        _settings["auto_review"] = enabled
        _save_settings(_settings)

    def set_persist_session(self, enabled: bool) -> None:
        self.persist_session = enabled
        _settings["persist_session"] = enabled
        _save_settings(_settings)

    def set_theme(self, theme: str) -> None:
        self.theme = theme
        _settings["theme"] = theme
        _save_settings(_settings)

    def set_model(self, model_name: str) -> None:
        self.model = model_name
        _settings["default_model"] = model_name
        _save_settings(_settings)

    def set_backup_enabled(self, enabled: bool) -> None:
        self.backup_enabled = enabled
        _settings["backup_enabled"] = enabled
        _save_settings(_settings)

    def set_use_git_backup(self, enabled: bool) -> None:
        self.use_git_backup = enabled
        _settings["use_git_backup"] = enabled
        _save_settings(_settings)

    def set_git_backup_branch(self, branch: str) -> None:
        self.git_backup_branch = branch
        _settings["git_backup_branch"] = branch
        _save_settings(_settings)

    def set_block_on_fail(self, enabled: bool) -> None:
        self.block_on_fail = enabled
        _settings["block_on_fail"] = enabled
        _save_settings(_settings)

    def set_focus_mode(self, mode: str) -> None:
        self.focus_mode = mode
        _settings["focus_mode"] = mode
        _save_settings(_settings)

    def set_focus_trigger(self, trigger: str) -> None:
        self.focus_trigger = trigger
        _settings["focus_trigger"] = trigger
        _save_settings(_settings)

    def set_default_tries(self, tries: int) -> None:
        self.default_tries = tries
        _settings["default_tries"] = tries
        _save_settings(_settings)

    def set_default_recurse(self, recurse: int) -> None:
        self.default_recurse = recurse
        _settings["default_recurse"] = recurse
        _save_settings(_settings)

    def set_default_timeout(self, timeout: float) -> None:
        self.default_timeout = timeout
        _settings["default_timeout"] = timeout
        _save_settings(_settings)

    def set_dig_max_turns(self, turns: int) -> None:
        self.dig_max_turns = turns
        _settings["dig_max_turns"] = turns
        _settings.pop("filedig_max_turns", None) # Clean up legacy
        _save_settings(_settings)

    def set_default_ambiguous_mode(self, mode: str) -> None:
        self.default_ambiguous_mode = mode
        _settings["default_ambiguous_mode"] = mode
        _save_settings(_settings)

    def set_diff_fuzzy_lines_threshold(self, threshold: float) -> None:
        self.diff_fuzzy_lines_threshold = threshold
        _settings["diff_fuzzy_lines_threshold"] = threshold
        _save_settings(_settings)

    def set_diff_fuzzy_max_bad_lines(self, count: int) -> None:
        self.diff_fuzzy_max_bad_lines = count
        _settings["diff_fuzzy_max_bad_lines"] = count
        _save_settings(_settings)

    def set_output_sharding_limit(self, limit: int) -> None:
        self.output_sharding_limit = limit
        _settings["output_sharding_limit"] = limit
        _save_settings(_settings)

    def set_max_shards(self, count: int) -> None:
        self.max_shards = count
        _settings["max_shards"] = count
        _save_settings(_settings)

    def set_sharding_ratio(self, ratio: float) -> None:
        self.sharding_ratio = ratio
        _settings["sharding_ratio"] = ratio
        _save_settings(_settings)

    def set_validation_failure_behavior(self, behavior: str) -> None:
        self.validation_failure_behavior = behavior
        _settings["validation_failure_behavior"] = behavior
        _save_settings(_settings)

config = DeltaToolConfig()

# Helper placed here to resolve circular dependencies
def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text."""
    return len(text) // TOKENS_PER_CHAR_ESTIMATE