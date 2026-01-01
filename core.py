"""Core functionality for the delta tool - LLM-powered file modification."""

import base64
import difflib
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from pattern import code_block_pattern, diff_example, search_block_pattern, plan_block_pattern

logger = logging.getLogger(__name__)

_TOOL_DIR = Path(__file__).parent.resolve()


def is_image_file(path: Path | str) -> bool:
    """Check if the path looks like an image file."""
    # Ensure common types are recognized even if OS map is incomplete
    if str(path).lower().endswith(".webp"):
        return True

    guess, _ = mimetypes.guess_type(str(path))
    return guess is not None and guess.startswith("image/")


def get_display_path(path: Path | str, cwd: Path | None = None) -> str:
    """Get relative path for display, normalized to forward slashes."""
    if cwd is None:
        cwd = Path.cwd()
    p = Path(path)
    try:
        return str(p.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def encode_image(image_path: str | Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(image_path: str | Path) -> str:
    """Get the MIME type for an image file."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type:
        return mime_type
    
    # Fallback/Specifics
    if Path(image_path).suffix.lower() == ".webp":
        return "image/webp"
    return "application/octet-stream"


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

# Use a specific directory for backups to keep root clean
BACKUP_DIR = str(APP_DATA_DIR / "backups")

# Settings can be in the install directory (default) or app data
SETTINGS_FILENAME = "settings.json"
DEFAULT_SETTINGS_FILENAME = "default_settings.json"
SETTINGS_PATH = APP_DATA_DIR / SETTINGS_FILENAME
DEFAULT_SETTINGS_PATH = _TOOL_DIR / DEFAULT_SETTINGS_FILENAME


def _load_settings() -> dict:
    """Load settings from settings.json."""
    if not SETTINGS_PATH.exists() and DEFAULT_SETTINGS_PATH.exists():
        try:
            shutil.copy2(DEFAULT_SETTINGS_PATH, SETTINGS_PATH)
        except Exception: pass

    for path in [SETTINGS_PATH, DEFAULT_SETTINGS_PATH]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")
    return {}


def _save_settings(settings: dict) -> None:
    """Save settings to settings.json."""
    try:
        # Always save to AppData
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save settings.json: {e}")


def update_core_settings(api_key: str, base_url: str, models: dict, git_branch: str) -> None:
    """Update and save core API settings."""
    global API_KEY, API_BASE_URL, AVAILABLE_MODELS
    
    _settings["api_key"] = api_key
    _settings["api_base_url"] = base_url
    _settings["available_models"] = models
    _settings["git_backup_branch"] = git_branch
    
    _save_settings(_settings)
    
    # Update globals
    API_KEY = api_key
    API_BASE_URL = base_url
    
    # Update dict in place
    AVAILABLE_MODELS.clear()
    AVAILABLE_MODELS.update(models)
    
    # Update config
    config.set_git_backup_branch(git_branch)


# Load settings at module import time
_settings = _load_settings()


def load_cwd_data(filepath: Path | str) -> Any:
    """Load data associated with the current CWD."""
    try:
        path = Path(filepath)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get(str(Path.cwd()))
    except Exception:
        pass
    return None


def save_cwd_data(filepath: Path | str, value: Any, indent: int = 2) -> None:
    """Save data associated with current CWD."""
    path = Path(filepath)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        data = {}
        
    data[str(Path.cwd())] = value
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        logger.warning(f"Warning saving {path.name}: {e}")


class DeltaToolError(Exception):
    """Base exception for Delta Tool errors."""
    pass


class CancelledError(DeltaToolError):
    """Raised when an operation is cancelled by the user."""
    pass


class DiffApplicationError(DeltaToolError):
    """Raised when diff application fails."""
    pass


class ValidationError(DeltaToolError):
    """Raised when file or input validation fails."""
    pass


class GenerationError(DeltaToolError):
    """Raised when LLM generation fails."""
    pass


# Constants loaded from settings with fallback defaults
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

# Application constants
QUEUE_POLL_INTERVAL_MS = _settings.get("queue_poll_interval_ms", 100)
MAX_PROMPT_HISTORY = _settings.get("max_prompt_history", 50)

# Type alias for output functions
OutputFunc = Callable[[str], None]


class DeltaToolConfig:
    """Configuration for the delta tool."""

    def __init__(self):
        self.model = _settings.get("default_model", DEFAULT_MODEL)
        self.backup_enabled = _settings.get("backup_enabled", True)
        self.use_git_backup = _settings.get("use_git_backup", False)
        self.git_backup_branch = _settings.get("git_backup_branch", "delta-backup")
        self.block_on_fail = _settings.get("block_on_fail", False)

        # Determine focus mode (migrate bool to string)
        fm = _settings.get("focus_mode", "off")
        if isinstance(fm, bool):
            fm = "flash" if fm else "off"
        self.focus_mode = fm
        self.focus_trigger = _settings.get("focus_trigger", "task")

        self.auto_review = _settings.get("auto_review", False)
        self.persist_session = _settings.get("persist_session", True)
        self.theme = _settings.get("theme", "light")

        # Execution defaults
        self.default_tries = _settings.get("default_tries", 2)
        self.default_recurse = _settings.get("default_recurse", 0)
        self.default_timeout = _settings.get("default_timeout", 10)
        self.default_ambiguous_mode = _settings.get("default_ambiguous_mode", "replace_all")

        self.diff_fuzzy_lines_threshold = _settings.get("diff_fuzzy_lines_threshold", 0.95)
        self.diff_fuzzy_max_bad_lines = _settings.get("diff_fuzzy_max_bad_lines", 1)

        # New global settings
        self.verify_changes = _settings.get("verify_changes", False)
        self.require_approval = _settings.get("require_approval", False)
        self.validate_at_start = _settings.get("validate_at_start", False)
        self.add_new_files = _settings.get("add_new_files", True)

        self.output_sharding_limit = _settings.get("output_sharding_limit", 60000)
        self.max_shards = _settings.get("max_shards", 5)
        self.sharding_ratio = _settings.get("sharding_ratio", 0.9)
        self.extra_system_prompt = _settings.get("extra_system_prompt", "")

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
        # Persist the choice to settings.json
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


# Global configuration instance
config = DeltaToolConfig()


def ensure_temp_dir() -> Path:
    """Ensure the temp directory exists and return its path."""
    temp_dir = APP_DATA_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


# Cache for heavy IO checks
_io_cache: dict[str, Any] = {
    "git_installed": None,
    "git_repo_path": None,
    "git_is_repo": False,
    "last_check": 0.0
}
IO_CACHE_INTERVAL = 10.0


def io_throttled_update() -> None:
    """Run periodic IO checks (e.g. git status)."""
    t = time.time()
    if t - _io_cache["last_check"] > IO_CACHE_INTERVAL:
        _io_cache["git_installed"] = shutil.which("git") is not None
        
        cwd = str(Path.cwd())
        _io_cache["git_repo_path"] = cwd
        _io_cache["git_is_repo"] = (Path(cwd) / ".git").exists()
        
        _io_cache["last_check"] = t


def force_io_cache_refresh() -> None:
    """Force a refresh of the IO cache."""
    _io_cache["last_check"] = 0.0
    io_throttled_update()


def is_git_installed() -> bool:
    """Check if git is installed and in PATH."""
    io_throttled_update()
    if _io_cache["git_installed"] is None:
        force_io_cache_refresh()
    return _io_cache["git_installed"]


def is_git_repo(path: Path | None = None) -> bool:
    """Check if the path (or CWD) is a valid git repo root."""
    p = path or Path.cwd()
    
    # Use cache if path is CWD (or None)
    if path is None or str(path) == str(Path.cwd()):
        io_throttled_update()
        if _io_cache["git_repo_path"] == str(Path.cwd()):
            return _io_cache["git_is_repo"]
            
    # Fallback for explicit path or cache miss
    return (p / ".git").exists()


def init_git_repo(path: Path | None = None) -> tuple[bool, str]:
    """Initialize a git repo in the path (or CWD)."""
    p = path or Path.cwd()
    success, msg = run_command("git init", cwd=p)
    if success:
        force_io_cache_refresh()
    return success, msg


class GitShadowHandler:
    """Handles git operations for the shadow backup branch."""
    
    def __init__(self, branch_name=None):
        self.branch = branch_name or config.git_backup_branch
        self.env = os.environ.copy()
        # Use a separate index file so we don't mess up the user's 'git status'
        self.env["GIT_INDEX_FILE"] = str(APP_DATA_DIR / "delta_git_index")

    def _run(self, args: list[str]) -> tuple[bool, str]:
        """Run a git command with the shadow index."""
        try:
            result = subprocess.run(
                ["git"] + args, 
                env=self.env, 
                cwd=Path.cwd(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            output = result.stdout.strip() if result.stdout else result.stderr.strip()
            return result.returncode == 0, output
        except Exception as e:
            return False, str(e)

    def is_available(self) -> bool:
        """Check if git is available and cwd is a repo."""
        return is_git_installed() and is_git_repo()

    def ensure_init(self) -> bool:
        """Ensure the shadow branch exists and handle init."""
        if not self.is_available():
            return False
            
        # Check if branch exists
        ok, _ = self._run(["rev-parse", "--verify", self.branch])
        if not ok:
            # Create branch history implicitly by operations
            pass
        return True

    def commit_files(self, file_paths: list[str], message: str) -> str | None:
        """Commit specific files to the shadow branch. Returns commit hash."""
        if not self.is_available():
            return None

        # 1. Reset our shadow index to HEAD (to capture full project context)
        #    This makes the backup mirror the current branch state plus our changes.
        self._run(["read-tree", "HEAD"])
        
        # 2. Sync index with Working Directory
        #    We use 'add -A' to capture all workspace changes (tracked & untracked).
        #    This ensures we don't accidentally revert changes to files that were
        #    modified in previous steps (but not committed to HEAD) just because
        #    they aren't the focus of the current operation.
        self._run(["add", "-A"])
        
        # 3. Write tree object
        ok, tree_oid = self._run(["write-tree"])
        if not ok: return False
        
        # 4. Get parent commit (current tip of shadow branch)
        ok, parent_oid = self._run(["rev-parse", self.branch])
        
        # Check if we actually have changes to commit
        if ok and parent_oid:
            ok_ptree, parent_tree = self._run(["rev-parse", f"{self.branch}^{{tree}}"])
            if ok_ptree and parent_tree == tree_oid:
                return parent_oid

        # 5. Create commit object
        commit_args = ["commit-tree", tree_oid, "-m", message]
        if ok and parent_oid:
            commit_args.extend(["-p", parent_oid])
            
        ok, commit_oid = self._run(commit_args)
        if not ok: return None
        
        # 6. Move branch pointer
        self._run(["update-ref", f"refs/heads/{self.branch}", commit_oid])
        return commit_oid

    def restore_files(self, file_paths: list[str], revision: str = "HEAD") -> None:
        """Restore files from the shadow branch to the working directory."""
        if not self.is_available(): return
        
        target = f"{self.branch}{revision}" if revision != "HEAD" else self.branch
        if revision.startswith("~") or revision.startswith("^"):
             target = f"{self.branch}{revision}"
             
        args = ["checkout", target, "--"] + file_paths
        self._run(args)

    def restore_snapshot(self, commit_hash: str) -> bool:
        """Restore the entire working directory to the state of a specific commit."""
        if not self.is_available(): return False
        # Checkout the specific commit's tree into the working directory
        ok, _ = self._run(["checkout", commit_hash, "--", "."])
        return ok

    def get_history(self, limit: int = 20) -> list[dict]:
        """Get list of commits from shadow branch."""
        if not self.is_available(): return []
        
        # Format: hash|timestamp|msg
        args = ["log", self.branch, f"-n{limit}", "--pretty=format:%H|%ad|%s", "--date=iso"]
        ok, out = self._run(args)
        if not ok or not out: return []
        
        history = []
        for line in out.splitlines():
            try:
                parts = line.split("|", 2)
                if len(parts) == 3:
                    history.append({
                        "session_id": parts[0], # Using hash as session_id
                        "timestamp": parts[1],
                        "message": parts[2],
                        "files": [], # Populated on demand
                        "source": "git"
                    })
            except Exception: pass
        return history
        
    def get_commit_files(self, commit_hash: str) -> list[str]:
        """Get files changed in a commit."""
        ok, out = self._run(["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash])
        if ok and out:
            return out.splitlines()
        return []


class BackupManager:
    """Manages file backups and undo operations."""
    
    def __init__(self, backup_dir: str = BACKUP_DIR):
        self.backup_dir = Path(backup_dir)
        self._current_session: str | None = None
        self._project_dir: Path = Path.cwd()
        self._active_manifest: dict = {"created": [], "modified": []}
    
    def _ensure_backup_dir(self) -> None:
        """Create backup directory if it doesn't exist."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        # Create .gitignore to prevent backup tracking
        gitignore = self.backup_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")
    
    def _get_project_hash(self) -> str:
        return hashlib.md5(str(Path.cwd()).encode()).hexdigest()[:8]

    def _get_manifest_path(self, session_id: str) -> Path:
        return self.backup_dir / f"{session_id}_manifest.json"

    def _save_manifest(self, session_id: str) -> None:
        try:
            path = self._get_manifest_path(session_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._active_manifest, f)
        except Exception:
            pass

    def _load_manifest(self, session_id: str) -> dict:
        try:
            path = self._get_manifest_path(session_id)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"created": [], "modified": []}

    def start_session(self) -> str:
        """Start a new backup session and return the session ID."""
        project_hash = self._get_project_hash()
        self._project_dir = Path.cwd()
        self._current_session = f"{project_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self._active_manifest = {"created": [], "modified": []}
        self._ensure_backup_dir()
        self._save_manifest(self._current_session)
        return self._current_session
    
    def backup_file(self, filepath: Path) -> Path | None:
        """Create a backup of a file before modification."""
        if not config.backup_enabled or not filepath.exists():
            return None
        
        session_id = self._current_session
        if not session_id:
            session_id = self.start_session()
        
        try:
            rel_path = filepath.relative_to(self._project_dir)
        except ValueError:
            rel_path = str(filepath).replace(":", "")

        safe_name = str(rel_path).replace(os.sep, "__").replace("/", "__")
        backup_path = self.backup_dir / f"{session_id}_{safe_name}.bak"
        
        try:
            shutil.copy2(filepath, backup_path)
            if str(rel_path) not in self._active_manifest["modified"]:
                self._active_manifest["modified"].append(str(rel_path))
                self._save_manifest(session_id)
            return backup_path
        except Exception:
            return None

    def register_created_file(self, filepath: Path) -> None:
        """Register a file that is being created in this session."""
        if not config.backup_enabled: return
        session_id = self._current_session
        if not session_id:
            session_id = self.start_session()
            
        try:
            rel_path = filepath.relative_to(self._project_dir)
            if str(rel_path) not in self._active_manifest["created"]:
                self._active_manifest["created"].append(str(rel_path))
                self._save_manifest(session_id)
        except Exception:
            pass
    
    def get_sessions(self) -> list[str]:
        """Get list of available backup sessions for current project."""
        if not self.backup_dir.exists():
            return []
        
        project_hash = self._get_project_hash()
        sessions = set()
        
        # Glob patterns to check
        for pattern in ["*_manifest.json", "*.bak"]:
            for f in self.backup_dir.glob(pattern):
                parts = f.stem.split("_", 4)
                if len(parts) >= 4 and parts[0] == project_hash:
                    sessions.add(f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}")
        
        return sorted(sessions, reverse=True)
    
    def get_session_files(self, session_id: str) -> list[tuple[Path, Path]]:
        """Get list of (backup_path, original_path) keys (files modified)."""
        if not self.backup_dir.exists():
            return []
        
        files = []
        # Find .bak files
        for backup_file in self.backup_dir.glob(f"{session_id}_*.bak"):
            name_without_session = backup_file.stem[len(session_id) + 1:]
            original_rel = name_without_session.replace("__", os.sep)
            original_path = Path.cwd() / original_rel
            files.append((backup_file, original_path))
        return files
    
    def undo_session(self, session_id: str) -> dict[str, str]:
        """Restore files from a backup session."""
        results = {}
        
        # 1. Load manifest if exists to delete created files
        manifest = self._load_manifest(session_id)
        
        # Delete created files
        for rel_path in manifest.get("created", []):
            full_path = Path.cwd() / rel_path
            try:
                if full_path.exists():
                    full_path.unlink()
                    _clean_empty_dirs(full_path.parent, Path.cwd())
                    results[str(full_path)] = "deleted (was created)"
                else:
                    results[str(full_path)] = "already missing"
            except Exception as e:
                results[str(full_path)] = f"error deleting: {e}"

        # 2. Restore modified files from .bak
        session_files = self.get_session_files(session_id)
        for backup_path, original_path in session_files:
            try:
                if backup_path.exists():
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, original_path)
                    results[str(original_path)] = "restored"
                else:
                    results[str(original_path)] = "backup not found"
            except Exception as e:
                results[str(original_path)] = f"error restoring: {e}"
        
        return results
    
    def delete_session(self, session_id: str) -> int:
        """Delete backups for a session."""
        count = 0
        # Delete .bak files
        for backup_path, _ in self.get_session_files(session_id):
            try:
                backup_path.unlink()
                count += 1
            except Exception:
                pass
        
        # Delete manifest
        try:
            manifest_path = self._get_manifest_path(session_id)
            if manifest_path.exists():
                manifest_path.unlink()
        except Exception:
            pass
            
        return count
    
    def rollback_to_session(self, target_session_id: str) -> dict[str, str]:
        """Rollback project state to the backup recorded in target_session_id.
        
        This undoes the target session and all sessions that occurred after it.
        """
        sessions = self.get_sessions()
        try:
            target_idx = sessions.index(target_session_id)
        except ValueError:
            return {"error": f"Session {target_session_id} not found"}
        
        # sessions list is [Newest, ..., Target, ..., Oldest]
        # We want to undo sessions[0] ... sessions[target_idx]
        
        sessions_to_undo = sessions[:target_idx+1]
        results = {}
        
        for session_id in sessions_to_undo:
            # Undo
            undo_res = self.undo_session(session_id)
            for file_path, status in undo_res.items():
                results[file_path] = status
            
            # Delete session backup as it is now invalidated history
            self.delete_session(session_id)
            
        return results

    def cleanup_old_backups(self, keep_sessions: int = 10) -> int:
        """Remove old backup sessions, keeping only the most recent.
        
        Args:
            keep_sessions: Number of sessions to keep.
            
        Returns:
            Number of sessions deleted.
        """
        sessions = self.get_sessions()
        deleted = 0
        
        for session_id in sessions[keep_sessions:]:
            self.delete_session(session_id)
            deleted += 1
        
        return deleted
    
    def clear_all_backups(self) -> int:
        """Clear all backup sessions for the current project.
        
        Returns:
            Number of sessions deleted.
        """
        sessions = self.get_sessions()
        deleted = 0
        
        for session_id in sessions:
            self.delete_session(session_id)
            deleted += 1
        
        return deleted


# Global backup manager
backup_manager = BackupManager()


def build_file_contents(filenames: list[str]) -> str:
    """Read and format file contents for the prompt.

    Files are sorted by path for consistent ordering, which helps with API caching.
    Uses file_cache for efficient repeated reads.
    """
    parts = []
    cwd = Path.cwd()

    # Sort files for consistent ordering (helps with API caching)
    sorted_filenames = sorted(filenames, key=lambda f: str(Path(f).resolve()))

    for filename in sorted_filenames:
        if is_image_file(filename):
            continue

        try:
            p = Path(filename)
            if p.exists() and p.is_file():
                # Use file cache for efficient repeated reads
                content = file_cache.get_or_read(str(p.resolve()))

                display_path = get_display_path(p, cwd)

                parts.append(f"\n--- START OF FILE: {display_path} ---\n{content}\n--- END OF FILE: {display_path} ---\n")
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
    return "".join(parts)


def build_system_message(filenames: list[str], ask_mode: bool = False, plan_mode: bool = False) -> str:
    """Build the system message for the LLM.

    Args:
        filenames: List of file paths being processed.
        ask_mode: If True, omit diff format instructions (for Q&A mode).
        plan_mode: If True, provide planning mode instructions.

    Returns:
        System message string for the LLM.
    """
    # Sort filenames for consistent ordering (helps with API caching)
    sorted_names = sorted(filenames)

    msg = ""
    if plan_mode:
        msg = f"""You have been given the following files: {", ".join(sorted_names)}

You are in Planning Mode.
Create a detailed implementation plan for the user's request.
Break it down into a sequence of tractable sub-tasks.
For each task, provide a clear 'Title' and a specific 'Prompt' for a code-editing LLM.

Output strictly using this format for every step:
<<<<<<< PLAN
Title: <short title>
Prompt: <detailed instructions for the LLM>
>>>>>>> END
"""
    elif ask_mode:
        msg = f"""You have been given the following files: {", ".join(sorted_names)}

You are in ask mode - answer questions about the code without making changes.
Provide helpful explanations, analysis, and insights about the codebase.
"""
    else:
        msg = f"""You have been given the following files: {", ".join(sorted_names)}

Use the following diff format to specify changes to files, including the surrounding backticks.
If you are writing a new file, leave the original text blank.
Ensure the filename is on its own line, not nestled with the backticks.
The original text must match exactly with no differences -- This means no annotations of any kind. 

{diff_example}

Ensure you surround your diff with triple backticks on their own lines.
Include a brief human-readable overview of the changes you plan to make at the start, AND a recap of the changes you made at the end.
"""

    if config.extra_system_prompt:
        msg += f"\n\n=== Custom Instructions ===\n{config.extra_system_prompt}"

    return msg


def _calc_tiered_cost(token_count: int, pricing: dict, key_prefix: str) -> float:
    """Helper to calculate tiered cost."""
    base_cost = (token_count / 1_000_000) * pricing.get(key_prefix, 0)
    over_key = f"{key_prefix}_over_200k"
    
    if over_key in pricing and token_count > 200_000:
        # Recalculate with tiering
        base_tokens = 200_000
        over_tokens = token_count - 200_000
        cost = (base_tokens / 1_000_000) * pricing[key_prefix]
        cost += (over_tokens / 1_000_000) * pricing[over_key]
        return cost
        
    return base_cost


def _get_model_pricing(model_name: str) -> dict | None:
    """Retrieve pricing for the model, or None if unknown."""
    return AVAILABLE_MODELS.get(model_name)


def _calculate_cost(input_tokens: int, output_tokens: int, model_name: str) -> tuple[float, str]:
    """Calculate the cost for the given token counts and model.
    
    Returns:
        Tuple of (total_cost, formatted_cost_string)
    """
    pricing = _get_model_pricing(model_name)
    if not pricing:
        return 0.0, " | Cost: (unknown model pricing)"
    
    input_cost = _calc_tiered_cost(input_tokens, pricing, "input")
    output_cost = _calc_tiered_cost(output_tokens, pricing, "output")
    
    total_cost = input_cost + output_cost
    return total_cost, f" | Est. Cost: ${total_cost:.4f}"




def _create_openai_client():
    from openai import OpenAI
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def generate(
    in_filenames: list[str],
    prompt: str,
    preuploaded_files: list | None = None,
    output_func_override: OutputFunc | None = None,
    raw_stream_output_func: OutputFunc | None = None,
    conversation_history: list[dict] | None = None,
    ask_mode: bool = False,
    plan_mode: bool = False,
    cancel_event: threading.Event | None = None,
) -> str:
    """Generate LLM response for the given files and prompt."""
    output_func = output_func_override or print
    client = _create_openai_client()
    
    system_message = build_system_message(in_filenames, ask_mode=ask_mode, plan_mode=plan_mode)
    
    # Process Files
    text_files = [f for f in in_filenames if not is_image_file(f)]
    image_files = [f for f in in_filenames if is_image_file(f)]
    file_contents_str = build_file_contents(text_files)

    # Construct User Message for API (including files)
    # File contents come first for better API caching (static content as prefix)
    header = "File Contents"
    egress_msg = f"{header}:\n{file_contents_str}"
    if image_files:
        egress_msg += f"\n\n(Plus {len(image_files)} image file(s) attached)"
    egress_msg += f"\n\nRequest:\n{prompt}"

    # Egress Content (API) - includes file content
    api_user_content = egress_msg
    if image_files:
        api_user_content = [{"type": "text", "text": egress_msg}]
        for img_path in image_files:
            try:
                api_user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{get_mime_type(img_path)};base64,{encode_image(img_path)}"}
                })
            except Exception as e:
                logger.error(f"Failed to encode image {img_path}: {e}")

    # History Content (Clean) - excludes file content
    history_user_content = prompt
    if image_files:
        history_user_content = [{"type": "text", "text": prompt}]
        for img_path in image_files:
            try:
                history_user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{get_mime_type(img_path)};base64,{encode_image(img_path)}"}
                })
            except Exception: pass

    # Assemble base messages list
    base_messages = [{"role": "system", "content": system_message}]
    if conversation_history:
        base_messages.extend(conversation_history)
    base_messages.append({"role": "user", "content": api_user_content})
    
    # Generation Loop (Sharding)
    full_result = ""
    start_time = time.time()
    last_progress_time = start_time
    total_input_tokens = 0
    total_output_tokens = 0
    
    # Create a working copy of messages for sharding loop
    current_messages = list(base_messages)
    
    extra_body = {}
    if "claude" in config.model.lower() or "anthropic" in config.model.lower():
        extra_body["provider"] = {"order": ["Anthropic"]}

    # Calculate soft limit for speculative sharding (ratio of max tokens converted to chars)
    soft_limit_chars = int((config.output_sharding_limit * TOKENS_PER_CHAR_ESTIMATE) * config.sharding_ratio)

    logger.info(f"Generating with model: {config.model}")

    try:
        for shard_idx in range(config.max_shards):
            shard_response = ""
            shard_finish_reason = None
            shard_input_tokens = 0
            shard_output_tokens = 0

            stream = client.chat.completions.create(
                model=config.model, 
                messages=current_messages, 
                stream=True,
                max_tokens=config.output_sharding_limit,
                extra_body=extra_body or None
            )

            for chunk in stream:
                if cancel_event and cancel_event.is_set():
                    output_func("\nGeneration cancelled.\n")
                    logger.info("Generation cancelled by user")
                    raise CancelledError("Cancelled by user")

                if not chunk.choices:
                    continue

                # Content
                content = chunk.choices[0].delta.content
                if content:
                    shard_response += content
                    if raw_stream_output_func:
                        raw_stream_output_func(content, end="")
                    elif time.time() - last_progress_time > 1:
                        last_progress_time = time.time()
                        output_func(".", end="", flush=True)

                    # Speculative Sharding: Check if we're near the limit and have a clean break
                    if len(shard_response) > soft_limit_chars:
                        if shard_response.endswith("\n>>>>>>> REPLACE\n") or shard_response.endswith("\n```\n"):
                            shard_finish_reason = "length"
                            # Break stream loop to trigger sharding in outer loop
                            break
                
                # Finish condition
                if chunk.choices[0].finish_reason:
                    shard_finish_reason = chunk.choices[0].finish_reason
                
                # Usage
                if hasattr(chunk, 'usage') and chunk.usage:
                    shard_input_tokens = getattr(chunk.usage, 'prompt_tokens', shard_input_tokens)
                    shard_output_tokens = getattr(chunk.usage, 'completion_tokens', shard_output_tokens)
            
            full_result += shard_response
            
            # Estimate if no usage returned
            if shard_input_tokens == 0:
                shard_input_tokens = len(str(current_messages)) // TOKENS_PER_CHAR_ESTIMATE
            if shard_output_tokens == 0:
                shard_output_tokens = len(shard_response) // TOKENS_PER_CHAR_ESTIMATE
            
            total_input_tokens += shard_input_tokens
            total_output_tokens += shard_output_tokens

            # Check for sharding requirement
            if shard_finish_reason == "length":
                output_func(f" [Shard {shard_idx + 1} limit. Continuing...]", end="", flush=True)
                logger.debug(f"Shard {shard_idx+1} limit reached. Continuing...")
                
                # Append partial result as assistant message
                current_messages.append({"role": "assistant", "content": shard_response})
                # Append user instruction to continue
                current_messages.append({
                    "role": "user", 
                    "content": "Output limit reached. Please continue generating EXACTLY where you left off. Do not repeat the last sentence, just continue the stream of text."
                })
                # Loop continues to next shard
            else:
                # Stop reason is "stop" or other, we are done
                break
        else:
            # Loop finished without breaking -> Max shards hit
            logger.error("Error: Maximum shard limit reached.")
            raise GenerationError(f"Generation exceeded max shards ({config.max_shards})")

    except Exception as e:
        if not isinstance(e, (CancelledError, GenerationError)):
            logger.exception(f"Error during stream processing: {e}")
        raise e

    if conversation_history is not None:
        conversation_history.append({"role": "user", "content": history_user_content})
        conversation_history.append({"role": "assistant", "content": full_result})

    # Metrics
    elapsed_time = time.time() - start_time
    _, cost_str = _calculate_cost(total_input_tokens, total_output_tokens, config.model)
    
    logger.info(f"Tokens: {total_input_tokens} in / {total_output_tokens} out{cost_str}")
    logger.info(f"Time: {elapsed_time:.2f}s")
    return full_result


def is_path_within_cwd(file_path: Path, cwd: Path) -> bool:
    """Check if file_path is within the current working directory."""
    try:
        # Resolve to standard paths to handle symlinks/normalization
        f = file_path.resolve()
        c = cwd.resolve()
        return f.is_relative_to(c)
    except ValueError:
        return False


def _validate_paths(filenames: list[str | Path]) -> tuple[list[str], list[str]]:
    """Internal helper to validate paths against CWD."""
    try:
        cwd = Path.cwd().resolve(strict=True)
    except FileNotFoundError:
        return [], ["Error: Invalid CWD"]

    validated, errors = [], []
    for f in filenames:
        try:
            p = Path(f)
            if not p.exists() or not p.is_file():
                errors.append(f"Invalid file: '{f}'")
                continue
            
            abs_path = p.resolve(strict=True)
            if is_path_within_cwd(abs_path, cwd):
                validated.append(str(abs_path))
            else:
                errors.append(f"Outside CWD: '{f}'")
        except Exception as e:
            errors.append(f"Error checking '{f}': {e}")
            
    return validated, errors


def validate_files(filenames: list[str | Path]) -> tuple[list[str], str | None]:
    """Validate files exist and are within CWD. Returns (valid_paths, error_msg)."""
    validated, errors = _validate_paths(filenames)
    return validated, "\n".join(errors) if errors else None


def _get_line_ending(text: str) -> str | None:
    """Determine the line ending sequence of text."""
    if text.endswith("\r\n"):
        return "\r\n"
    elif text.endswith("\n"):
        return "\n"
    return None


def _split_line_content_and_ending(line: str) -> tuple[str, str]:
    """Split a line into its content and line ending."""
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    elif line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""


def _find_best_fuzzy_match(content: str, search_block: str, line_threshold: float, max_bad_lines: int) -> tuple[int, int] | None:
    """Find the best fuzzy match for search_block within content using line-by-line comparison."""
    search_lines = search_block.splitlines()
    if not search_lines:
        return None
        
    n_search = len(search_lines)
    content_lines = content.splitlines(keepends=True)
    n_content = len(content_lines)
    
    if n_content < n_search:
        return None

    # Calculate byte offsets for mapping back to string indices
    line_offsets = [0]
    for line in content_lines:
        line_offsets.append(line_offsets[-1] + len(line))
    
    search_lines_stripped = [s.strip() for s in search_lines]
    content_lines_stripped = [c.strip() for c in content_lines]
    
    best_bad_count = max_bad_lines + 1
    best_total_score = -1.0
    best_window_start = -1
    
    # Optimization: If allowing 0 bad lines, we can check first line strictness to speed up
    check_first_line = (max_bad_lines == 0)

    for i in range(n_content - n_search + 1):
        if check_first_line:
             if abs(len(content_lines_stripped[i]) - len(search_lines_stripped[0])) > max(len(search_lines_stripped[0]), 5) * 0.5:
                 continue

        current_bad_lines = 0
        current_total_score = 0.0
        possible = True
        
        for j in range(n_search):
            s_line = search_lines_stripped[j]
            c_line = content_lines_stripped[i+j]
            
            if s_line == c_line:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, c_line, s_line).ratio()
            
            current_total_score += score
            
            if score < line_threshold:
                possible = False
                break

            if score < 1.0:
                current_bad_lines += 1
            
            if current_bad_lines > max_bad_lines:
                possible = False
                break
        
        if possible:
            # Rank candidates: fewer bad lines is best, then higher score
            is_better = False
            if best_window_start == -1:
                is_better = True
            elif current_bad_lines < best_bad_count:
                is_better = True
            elif current_bad_lines == best_bad_count and current_total_score > best_total_score:
                is_better = True
            
            if is_better:
                best_bad_count = current_bad_lines
                best_total_score = current_total_score
                best_window_start = i

    if best_window_start != -1:
        start_line = best_window_start
        end_line = start_line + n_search
        logger.debug(f"Fuzzy match found at line {start_line + 1}: score={best_total_score:.2f}, bad_lines={best_bad_count}")
        return line_offsets[start_line], line_offsets[end_line]
        
    return None


def build_tolerant_regex(text: str) -> str:
    """Build a regex pattern that tolerates whitespace differences.
    
    Args:
        text: The text to build a regex pattern for.
        
    Returns:
        A regex pattern string.
    """
    if not text:
        return ""

    lines = text.splitlines(keepends=True)
    pattern_parts = []
    overall_ending = _get_line_ending(text)

    for i, line_str in enumerate(lines):
        is_last_line = i == len(lines) - 1
        content, line_ending = _split_line_content_and_ending(line_str)

        # Build regex for content
        if content.strip() == "":
            content_regex = r"\s*"
        else:
            content_regex = r"[ \t\f\v]*" + re.escape(content.lstrip())

        # Build regex for line ending
        ending_regex = ""
        if line_ending:
            if is_last_line and overall_ending == line_ending:
                ending_regex = r"(?:" + re.escape(line_ending) + r")?"
            else:
                ending_regex = re.escape(line_ending)

        pattern_parts.append(content_regex + ending_regex)

    return "".join(pattern_parts)


def reconcile_path(filename_str: str) -> str:
    """Fix common LLM path hallucinations (e.g., including CWD name in path)."""
    # Remove common prefixes LLMs like to add
    for prefix in ["file:", "filename:"]:
        if filename_str.strip().lower().startswith(prefix):
            filename_str = filename_str.strip()[len(prefix):].strip()
            
    clean_filename = filename_str.strip().strip('"\'`').lstrip("/\\")
    cwd = Path.cwd()

    # 1. Exact match
    if (cwd / clean_filename).exists():
        return clean_filename

    parts = clean_filename.replace("\\", "/").split("/")
    
    # 2. Path starts with CWD name (e.g. "project/src/main.py" when in "project")
    if len(parts) > 1 and parts[0] == cwd.name:
        candidate = "/".join(parts[1:])
        if (cwd / candidate).exists():
            return candidate
        return candidate

    # 3. Fallback: Check if basename exists in CWD
    if len(parts) > 1:
        basename = parts[-1]
        if (cwd / basename).exists():
            return basename

    return clean_filename


def parse_plan(text: str) -> list[tuple[str, str]]:
    """Parse plan blocks from the response string.
    
    Returns:
        List of (Title, Prompt) tuples.
    """
    tasks = []
    for match in plan_block_pattern.finditer(text):
        title = match.group(1).strip()
        prompt = match.group(2).strip()
        if title and prompt:
            tasks.append((title, prompt))
    return tasks


def _is_valid_filename(text: str) -> bool:
    """Check if a text string is a valid filename."""
    if not text:
        return False
    
    # Must not be a search block marker
    if text.strip().startswith("<<<<<<<"):
        return False

    clean = reconcile_path(text)
    if not clean:
        return False

    # 1. Existence check (Strongest signal)
    try:
        p = Path.cwd() / clean
        if p.exists() and p.is_file():
            return True
    except Exception:
        pass

    # 2. Structural checks for new/unknown files
    # Reject strings with spaces (likely natural language)
    if " " in clean:
        return False
        
    # Reject strings ending in colon (common LLM artifact)
    if clean.endswith(":"):
        return False

    # Reject common language identifiers if that's all there is
    if clean.lower() in {"bash", "python", "javascript", "typescript", "html", "css", "json", "yaml", "xml", "diff", "sh", "zsh"}:
        return False

    return True


def parse_diffs(diff_string: str) -> list[dict]:
    """Parse diff blocks from the response string."""
    parsed_diffs = []

    for block_match in code_block_pattern.finditer(diff_string):
        before_block = block_match.group(1)       # Text outside/above ```
        info_string = block_match.group(2)        # Text on same line as ```
        content = block_match.group(3)            # All lines inside ```

        if not content:
            continue

        lines = content.split('\n', 1)
        first_line_inside = lines[0].strip()
        rest_of_inside = lines[1] if len(lines) > 1 else ""

        # Identify filename candidates in priority order:
        # 1. Line below (first line inside block)
        # 2. Line above (before block)
        # 3. Same line (info string)
        
        candidates = []
        
        # Candidate 1: Below
        candidates.append({
            "text": first_line_inside,
            "content": rest_of_inside
        })
        
        # Candidate 2: Above
        if before_block:
            candidates.append({
                "text": before_block.strip().split("\n")[-1].strip(),
                "content": content
            })
            
        # Candidate 3: Same Line
        if info_string:
            parts = info_string.strip().split()
            if parts:
                candidates.append({
                    "text": parts[-1],
                    "content": content
                })

        filename = None
        diff_content = content
        
        for cand in candidates:
            if _is_valid_filename(cand["text"]):
                filename = reconcile_path(cand["text"])
                diff_content = cand["content"]
                break
        
        if not filename:
            continue

        for hunk_match in search_block_pattern.finditer(diff_content):
            original = hunk_match.group(1)
            parsed_diffs.append({
                "filename": filename,
                "original": original if original and original.strip() else "",
                "new": hunk_match.group(2),
            })

    return parsed_diffs


def _load_file_state(filename: str, file_states: dict, simulated_states: dict) -> None:
    """Load initial file state if not already loaded."""
    if filename in file_states:
        return

    p = Path.cwd() / filename
    try:
        if p.is_file():
            if is_image_file(p):
                raise ValueError(f"{filename}: Cannot apply text diffs to image file.")
            content = p.read_text(encoding="utf-8")
            file_states[filename] = (True, content)
            simulated_states[filename] = content
        elif p.exists():
            raise ValueError(f"{filename}: Path exists but is not a regular file.")
        else:
            if is_image_file(filename):
                raise ValueError(f"{filename}: Cannot generate new image file via text diffs.")
            file_states[filename] = (False, None)
            simulated_states[filename] = ""
    except ValueError:
        raise
    except Exception as e:
        raise IOError(f"Error accessing file '{filename}' during validation: {e}") from e


def _apply_single_diff(
    diff_info: dict,
    file_states: dict,
    simulated_states: dict,
    diff_counts: defaultdict,
    ambiguous_mode: str = "replace_all",
) -> None:
    """Apply a single diff to the simulated file state."""
    filename = diff_info["filename"]
    original = diff_info["original"]
    new = diff_info["new"]

    current_content = simulated_states[filename]
    is_existing_file, _ = file_states[filename]

    # Case: Creating a new file
    if not is_existing_file and diff_counts[filename] == 0:
        if original == "":
            simulated_states[filename] = new
            diff_counts[filename] += 1
            return
        raise ValueError(f"{filename}: File not found and original text not blank.")

    # Case: Modifying existing file
    if original == "":
        raise ValueError(f"{filename}: Original text cannot be blank for an existing file.")

    pattern = re.compile(build_tolerant_regex(original))
    matches = list(pattern.finditer(current_content))
    match_count = len(matches)

    if match_count == 0 and config.diff_fuzzy_max_bad_lines > 0:
        # Fallback to Fuzzy Line Match
        logger.debug(f"Exact match failed for {filename}. Attempting fuzzy match...")
        fuzzy_match = _find_best_fuzzy_match(
            current_content, 
            original, 
            config.diff_fuzzy_lines_threshold, 
            config.diff_fuzzy_max_bad_lines
        )
        if fuzzy_match:
            start, end = fuzzy_match
            simulated_states[filename] = current_content[:start] + new + current_content[end:]
            diff_counts[filename] += 1
            return

    if match_count == 0:
        raise ValueError(f"Original text not found in file: {filename}")
    
    if match_count == 1:
        start, end = matches[0].span()
        simulated_states[filename] = current_content[:start] + new + current_content[end:]
    else:
        # Ambiguous matches logic
        if ambiguous_mode == "fail":
            raise ValueError(f"{filename}: Original text ambiguous ({match_count} matches) in file.")
        elif ambiguous_mode == "replace_all":
            # Apply to all matches in reverse order to keep indices valid
            running_content = current_content
            for m in reversed(matches):
                start, end = m.span()
                running_content = running_content[:start] + new + running_content[end:]
            simulated_states[filename] = running_content
        elif ambiguous_mode == "ignore":
            pass

    diff_counts[filename] += 1


def _clean_empty_dirs(path: Path, root: Path) -> None:
    """Recursively delete empty directories up to root."""
    try:
        if path == root: return
        if path.exists() and path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            _clean_empty_dirs(path.parent, root)
    except Exception:
        pass


def apply_diffs(diff_string: str, create_backup: bool = True, ambiguous_mode: str = "replace_all", original_prompt: str | None = None, confirmation_callback: Callable[[dict[str, int], dict[str, str]], bool] | None = None) -> tuple[dict[str, str], str | None]:
    """Parse and apply diffs from the response string.
    
    Args:
        diff_string: The raw diff string from the LLM.
        create_backup: If True, create backups before modifying files.
        ambiguous_mode: How to handle ambiguous matches ("replace_all", "ignore", "fail").
        confirmation_callback: Optional callback(diff_counts, simulated_states) -> bool. If returns False, aborts.

    Returns:
        Tuple of (Dictionary mapping filenames to the number of diffs applied, session_id).
        
    Raises:
        ValueError: If no diffs found or diff application fails.
        IOError: If file writing fails.
    """
    parsed_diffs = parse_diffs(diff_string)
    logger.debug(f"Diff parsing complete. Found {len(parsed_diffs)} blocks.")

    if not parsed_diffs:
        error_msg = "No diffs found in response."
        raise ValueError(error_msg)

    file_states: dict[str, tuple[bool, str | None]] = {}
    simulated_states: dict[str, str] = {}
    diff_counts: defaultdict[str, int] = defaultdict(int)

    # Simulate all diffs first (validation phase)
    for diff_info in parsed_diffs:
        filename = diff_info["filename"]
        _load_file_state(filename, file_states, simulated_states)
        _apply_single_diff(diff_info, file_states, simulated_states, diff_counts, ambiguous_mode=ambiguous_mode)

    if confirmation_callback:
        if not confirmation_callback(diff_counts, simulated_states):
            raise CancelledError("Cancelled by confirmation callback")

    # Start backup session
    session_id = None
    git_backup_active = config.use_git_backup and config.backup_enabled
    
    files_to_modify = []
    
    if git_backup_active:
        for filename in simulated_states:
            if filename in diff_counts:
                files_to_modify.append(filename)
        
        git_handler = GitShadowHandler()
        if git_handler.is_available():
            # Snapshot BEFORE changes
            snap_msg = "Snapshot: Pre-modification context for request"
            if original_prompt:
                snap_msg += f"\n\nRequest: {original_prompt}"

            git_handler.commit_files(
                files_to_modify, 
                snap_msg
            )
        else:
            git_backup_active = False # Fallback if not repo

    if config.backup_enabled and not git_backup_active:
        session_id = backup_manager.start_session()

    # Apply changes to disk
    applied: dict[str, str] = {}
    for filename, final_content in simulated_states.items():
        if filename not in diff_counts:
            continue

        p = Path.cwd() / filename
        is_existing_file, _ = file_states[filename]

        try:
            # Check if content actually changed
            if is_existing_file:
                _, original_content = file_states[filename]
                if final_content == original_content:
                    continue

            # Create backup before modifying (FileSystem mode)
            if config.backup_enabled and not git_backup_active:
                if is_existing_file:
                    backup_manager.backup_file(p)
                else:
                    backup_manager.register_created_file(p)
            
            if not is_existing_file:
                p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(final_content, encoding="utf-8")
            
            if not is_existing_file:
                applied[filename] = "Created"
            else:
                applied[filename] = f"{diff_counts[filename]} diff(s)"

        except Exception as e:
            raise IOError(f"Error writing file '{p}' during application phase: {e}\n\nRaw response: {diff_string}") from e

    # Post-modification git snaphot
    if git_backup_active and applied:
        msg = f"Delta: Applied changes to {len(applied)} files"
        if original_prompt:
            msg += f"\n\nRequest: {original_prompt}"

        commit_id = git_handler.commit_files(
            files_to_modify,
            msg
        )
        session_id = commit_id or "GIT_LATEST"

    return applied, session_id


def undo_last_changes() -> dict[str, str]:
    """Undo the most recent set of changes.
    
    Returns:
        Dictionary of filename -> status message.
    """
    if config.use_git_backup and config.backup_enabled:
        git = GitShadowHandler()
        if git.is_available():
            # Undo means reverting the last changes.
            # Assuming HEAD is "Delta: ...", we want to revert to HEAD~1 ("Snapshot: ...")
            
            files = git.get_commit_files(git.branch)
            if not files:
                return {"error": "No changes found in git history to undo"}
                
            # Restore them from HEAD~1
            git.restore_files(files, "~1")
            
            # Create Undo commit
            git.commit_files(files, "Delta: Reverted last change")
            
            return {f: "Restored via Git" for f in files}

    sessions = backup_manager.get_sessions()
    if not sessions:
        return {"error": "No backup sessions found"}
    
    return backup_manager.undo_session(sessions[0])


def get_available_backups() -> list[dict]:
    """Get list of available backup sessions (Git and File) with details.
    
    Returns:
        List of dicts with 'session_id', 'timestamp', 'files', 'source' keys.
    """
    all_backups = []
    
    # 1. File Backups
    sessions = backup_manager.get_sessions()
    for session_id in sessions:
        files = backup_manager.get_session_files(session_id)
        
        # Parse timestamp from session_id (format: projecthash_YYYYMMDD_HHMMSS_microseconds)
        parts = session_id.split("_")
        try:
            if len(parts) >= 3:
                # New format with project hash
                dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp = session_id
        except ValueError:
            timestamp = session_id
        
        all_backups.append({
            "session_id": session_id,
            "timestamp": timestamp,
            "sort_key": timestamp,
            "files": [str(orig) for _, orig in files],
            "source": "file",
            "message": "File Backup"
        })

    # 2. Git Backups
    git = GitShadowHandler()
    if git.is_available():
        history = git.get_history()
        for item in history:
            # item has: session_id (hash), timestamp (ISO), message, source='git'
            item["sort_key"] = item["timestamp"]
            # Populate files for them
            files = git.get_commit_files(item["session_id"])
            item["files"] = files
            all_backups.append(item)
    
    # Sort combined list by timestamp descending
    all_backups.sort(key=lambda x: x.get("sort_key", ""), reverse=True)
    return all_backups


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text.
    
    Args:
        text: The text to estimate tokens for.
        
    Returns:
        Estimated token count.
    """
    return len(text) // TOKENS_PER_CHAR_ESTIMATE


def calculate_input_cost(token_count: int, model_name: str) -> tuple[float, str]:
    """Calculate the input cost for a given token count.
    
    Args:
        token_count: Number of input tokens.
        model_name: Name of the model.
        
    Returns:
        Tuple of (cost, formatted_string).
    """
    pricing = _get_model_pricing(model_name)
    if not pricing:
        return 0.0, "???"
    
    cost = _calc_tiered_cost(token_count, pricing, "input")
    
    return cost, f"${cost:.4f}"


# Cache for file statistics (lines, tokens) to reduce IO in UI loops
_stats_cache: dict[str, tuple[int, int, str]] = {}


def clear_stats_cache() -> None:
    """Clear the file statistics cache."""
    _stats_cache.clear()


class FileCache:
    """Cache for file contents to avoid repeated disk reads."""
    
    def __init__(self):
        self._cache: dict[str, tuple[str, float]] = {}  # path -> (content, mtime)
    
    def get(self, filepath: str) -> str | None:
        """Get cached file content if still valid.
        
        Args:
            filepath: Path to the file.
            
        Returns:
            Cached content or None if not cached/stale.
        """
        if filepath not in self._cache:
            return None
        
        cached_content, cached_mtime = self._cache[filepath]
        
        try:
            current_mtime = Path(filepath).stat().st_mtime
            if current_mtime == cached_mtime:
                return cached_content
        except Exception:
            pass
        
        # Cache is stale
        del self._cache[filepath]
        return None
    
    def set(self, filepath: str, content: str) -> None:
        """Cache file content.
        
        Args:
            filepath: Path to the file.
            content: File content to cache.
        """
        try:
            mtime = Path(filepath).stat().st_mtime
            self._cache[filepath] = (content, mtime)
        except Exception:
            pass
    
    def invalidate(self, filepath: str) -> None:
        """Remove a file from the cache."""
        self._cache.pop(filepath, None)
        try:
            # Also invalidate stats cache
            abs_key = str(Path(filepath).resolve())
            if abs_key in _stats_cache:
                del _stats_cache[abs_key]
        except Exception:
            pass
    
    def clear(self) -> None:
        """Clear all cached content."""
        self._cache.clear()
    
    def get_or_read(self, filepath: str) -> str:
        """Get from cache or read from disk and cache.
        
        Args:
            filepath: Path to the file.
            
        Returns:
            File content.
            
        Raises:
            IOError: If file cannot be read.
        """
        cached = self.get(filepath)
        if cached is not None:
            return cached
        
        try:
            content = Path(filepath).read_text(encoding="utf-8", errors="replace")
            self.set(filepath, content)
            return content
        except Exception as e:
            raise IOError(f"Failed to read {filepath}: {e}") from e


# Global file cache
file_cache = FileCache()


def run_command(cmd: str, timeout: float = 10.0, cwd: Path | None = None) -> tuple[bool, str]:
    """Run a shell command with timeout.
    
    Returns:
        Tuple of (success, output).
    """
    import subprocess
    if cwd is None:
        cwd = Path.cwd()
        
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        
        return result.returncode == 0, result.stdout or ""
        
    except subprocess.TimeoutExpired:
        return False, "(timeout reached)"
    except Exception as e:
        return False, f"Error running command: {e}"


def get_file_stats(path: Path | str) -> tuple[int, int, str]:
    """Get lines, tokens, and display info for a file."""
    p = Path(path)
    
    # Check cache first
    try:
        key = str(p)
        if key in _stats_cache:
            return _stats_cache[key]
    except Exception:
        key = str(p)

    if not p.is_file():
        return 0, 0, ""
    
    if is_image_file(p):
        res = (0, 1000, f"{p.name} (IMG)")
        _stats_cache[key] = res
        return res
        
    try:
        content = file_cache.get_or_read(str(p))
        lines = len(content.splitlines())
        tokens = estimate_tokens(content)
        res = (lines, tokens, f"{p.name} ({lines}|{tokens})")
        _stats_cache[key] = res
        return res
    except Exception:
        return 0, 0, f"{p.name} (?)"


def restore_git_backup(commit_hash: str) -> dict[str, str]:
    """Restore project state to a specific git commit hash."""
    git = GitShadowHandler()
    if git.restore_snapshot(commit_hash):
        return {"git": f"Restored snapshot {commit_hash[:8]}"}
    return {"error": "Failed to restore git snapshot"}


def _run_validation(
    validation_cmd: str,
    validation_timeout: float,
) -> tuple[bool, str]:
    """Run validation command and return (success, error_message)."""
    logger.info(f"Validating: {validation_cmd}")
    ok, output = run_command(validation_cmd, validation_timeout)
    if ok:
        logger.info("Validation passed.")
        return True, ""
    error_msg = output if output else "Unknown validation error"
    logger.error(f"Validation failed:\n{error_msg}")
    return False, error_msg


def _get_git_changes() -> list[tuple[str, list[str], list[str]]]:
    """Get all uncommitted changes from git.
    
    Returns:
        List of (display_path, original_lines, new_lines).
    """
    import subprocess
    changes = []
    
    # helper to run git command
    def git_cmd(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True, cwd=Path.cwd())
        except subprocess.CalledProcessError:
            return ""

    # 1. Modified/Deleted files (Staged and Unstaged) vs HEAD
    # --name-status gives "M path", "D path", "A path"
    status_out = git_cmd(["git", "diff", "HEAD", "--name-status"])
    
    for line in status_out.splitlines():
        if not line.strip(): continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2: continue
        status, rel_path = parts
        
        orig_lines = []
        new_lines = []
        
        # Get Original Content from HEAD
        if status != "A": # If not added
            try:
                content = git_cmd(["git", "show", f"HEAD:{rel_path}"])
                orig_lines = content.splitlines(keepends=True)
            except Exception: pass
            
        # Get New Content from Disk
        if status != "D": # If not deleted
            p = Path.cwd() / rel_path
            if p.exists():
                new_lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                
        changes.append((rel_path, orig_lines, new_lines))
        
    # 2. Untracked files
    untracked_out = git_cmd(["git", "ls-files", "--others", "--exclude-standard"])
    for rel_path in untracked_out.splitlines():
        if not rel_path.strip(): continue
        
        orig_lines = [] # Untracked means new
        p = Path.cwd() / rel_path
        if p.exists():
            try:
                # Basic binary check
                with open(p, 'rb') as f:
                    if b'\0' in f.read(1024): continue
                new_lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                changes.append((f"{rel_path} (untracked)", orig_lines, new_lines))
            except Exception: pass
            
    return changes


def open_diff_report(session_ids: str | list[str] | None = None, use_git: bool = False, compare_session_id: str | None = None, diff_against_disk: bool = False) -> None:
    """Generate and open an HTML diff report.
    
    Args:
        session_ids: Unique ID string or list of IDs for backup sessions. 
                     If a list is provided, aggregates changes (oldest backup vs current).
        use_git: If True, ignore sessions and show diff of local git workspace vs HEAD.
        compare_session_id: If provided, uses this session's start state as "New" instead of disk.
        diff_against_disk: If True, compares the session base against current disk state. 
                           If False, compares session base against session result (Git only).
    """
    try:
        full_diff_text = ""
        project_root = Path.cwd()
        title_extra = "Git Local"
        
        change_sets = [] # (display_name, orig_lines, new_lines)

        if use_git:
            change_sets = _get_git_changes()
        
        elif session_ids:
            # Determine if this is a Git operation or a File operation
            # If session_ids contains git hashes (len ~40, hex), use Git logic.
            # If config.use_git_backup is on, prioritize git unless it looks like a file ID.
            
            is_git_mode = False
            first_id = session_ids[0] if isinstance(session_ids, list) else session_ids
            
            # Simple heuristic: Git hashes are hex. File IDs have underscores.
            if "_" not in first_id:
                is_git_mode = True
            elif first_id == "GIT_LATEST":
                is_git_mode = True
            
            if is_git_mode:
                 # Handle Git-based diffing
                git = GitShadowHandler()
                
                if session_ids == "GIT_LATEST" or session_ids == ["GIT_LATEST"]:
                    title_extra = "Latest Session"
                    # Diff changed files in HEAD vs HEAD~1 (Snapshot vs Post-Delta)
                    changed_files = git.get_commit_files(git.branch)
                    for rel_path in changed_files:
                        ok, content = git._run(["show", f"{git.branch}~1:{rel_path}"])
                        orig_lines = content.splitlines(keepends=True) if ok else []

                        if diff_against_disk:
                            p = project_root / rel_path
                            if p.exists():
                                new_lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                            else:
                                new_lines = []
                        else:
                            ok_new, content_new = git._run(["show", f"{git.branch}:{rel_path}"])
                            new_lines = content_new.splitlines(keepends=True) if ok_new else []

                        change_sets.append((rel_path, orig_lines, new_lines))
            
                else:
                    # Iterate provided commit hashes
                    if isinstance(session_ids, str): session_ids = [session_ids]

                    title_extra = session_ids[0][:8] if len(session_ids) == 1 else f"Range ({len(session_ids)} commits)"

                    for sid in session_ids:
                        files = git.get_commit_files(sid)
                        for rel_path in files:
                            if any(c[0] == rel_path for c in change_sets): continue
                            
                            # Compare commit base (Snapshot) vs Disk/Result
                            # Note: git-backup structure is commit (changes) -> parent (snapshot).
                            # So commit~1 gives the 'before' state of that commit.
                            ok, content = git._run(["show", f"{sid}~1:{rel_path}"])
                            orig_lines = content.splitlines(keepends=True) if ok else []
                            
                            if diff_against_disk:
                                p = project_root / rel_path
                                if p.exists():
                                    new_lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                                else:
                                    new_lines = []
                            else:
                                ok_new, content_new = git._run(["show", f"{sid}:{rel_path}"])
                                new_lines = content_new.splitlines(keepends=True) if ok_new else []
                                
                            change_sets.append((rel_path, orig_lines, new_lines))

            else:
                # Handle File-based diffing
                if not session_ids: return
                if isinstance(session_ids, str):
                    session_ids = [session_ids]
                
                title_extra = session_ids[0] if len(session_ids) == 1 else f"Range ({len(session_ids)} sessions)"

            # Aggregate files across all sessions.
            # For each file, we want the ORIGINAL version from the EARLIEST session involved.
            # Logic:
            # 1. Gather all file paths touched in these sessions.
            # 2. For each path, find the oldest session in the list that has a backup (.bak) for it.
            # 3. Use that .bak as "Original".
            # 4. Use current disk as "New". (If file deleted, new is empty).
            # 5. Also handle created files (in manifest but no .bak).
            
            # Map: rel_path -> oldest_backup_path
            files_map: dict[str, Path] = {}
            created_files: set[str] = set()
            
            # session_ids usually come newest->oldest from get_sessions(), but let's be safe.
            # We want to iterate getting mapping, preferencing older sessions.
            # Let's assume the user passed them in arbitrary order, so we need to know timestamp/order.
            # backup_manager.get_sessions() returns newest to oldest.
            all_known = backup_manager.get_sessions()
            
            # Sort provided IDs by age (oldest first) based on their index in all_known (reversed)
            # Higher index in all_known = older.
            def age_key(sid):
                try: return all_known.index(sid)
                except ValueError: return -1
                
            sorted_sids = sorted(session_ids, key=age_key, reverse=True) # Oldest first
            
            for sid in sorted_sids:
                # 1. Backups (Modified files)
                # If we already have a backup for this file from an older session, keep it (because we sorted oldest first).
                # Actually, wait. "Oldest first" means the first one we see is the oldest.
                # So we simply set if not present.
                for backup_path, original_path in backup_manager.get_session_files(sid):
                    # Key by relative path to project
                    try:
                        rel = str(original_path.relative_to(project_root)).replace("\\", "/")
                        if rel not in files_map and rel not in created_files:
                            files_map[rel] = backup_path
                    except ValueError: pass

                # 2. Created files
                # If a file was created in an older session, it's "new" relative to start of that session.
                # But for the Aggregate Diff (Start of Range -> Now), if it exists now, it exists.
                # If it was created in T1 (old) and modified in T2 (new), we want T1's "pre-creation" state (empty).
                # So if it's in 'modified' (backed up) in a later session (T2), we'd have a T2 .bak.
                # But we want T1's state (empty).
                # The 'created' list in manifest implies: "At the start of this session, this file did not exist."
                manifest = backup_manager._load_manifest(sid)
                for rel in manifest.get("created", []):
                    rel = rel.replace("\\", "/")
                    if rel not in files_map:
                        # Mark as explicitly created (no backup file means original is empty)
                        created_files.add(rel)

            # Build Change Sets
            # Union of files_map keys and created_files
            all_rels = set(files_map.keys()) | created_files
            
            for rel_path in sorted(list(all_rels)):
                full_path = project_root / rel_path
                
                # ORIGINAL CONTENT
                if rel_path in files_map:
                    # It has a backup
                    valid_bak = files_map[rel_path]
                    if valid_bak.exists():
                        orig_lines = valid_bak.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                    else:
                        orig_lines = []
                else:
                    # It was a created file (or in created set), so original is empty
                    orig_lines = []

                # NEW CONTENT
                new_lines = []
                content_found = False

                if compare_session_id:
                    # Find content starting from compare_session_id up to newest
                    try:
                        start_idx = all_known.index(compare_session_id)
                        # Scan towards newest (index 0)
                        for i in range(start_idx, -1, -1):
                            scan_sid = all_known[i]
                            
                            # Check backup
                            scan_files = backup_manager.get_session_files(scan_sid)
                            for bak_path, orig_path in scan_files:
                                try:
                                    sc_rel = str(orig_path.relative_to(project_root)).replace("\\", "/")
                                    if sc_rel == rel_path:
                                        if bak_path.exists():
                                            new_lines = bak_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                                        content_found = True
                                        break
                                except ValueError: pass
                            if content_found: break

                            # Check created
                            scan_manifest = backup_manager._load_manifest(scan_sid)
                            if any(c.replace("\\", "/") == rel_path for c in scan_manifest.get("created", [])):
                                # Created in this session -> Start of session was empty.
                                new_lines = []
                                content_found = True
                                break
                    except ValueError:
                        pass # Session ID not found
                
                if not content_found:
                    # Current Disk
                    if full_path.exists():
                        new_lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                    else:
                        new_lines = [] # Deleted locally?

                change_sets.append((rel_path, orig_lines, new_lines))

        # Generate Diff Text
        for display_path, orig_lines, new_lines in change_sets:
            try:
                diff = difflib.unified_diff(
                    orig_lines, 
                    new_lines, 
                    fromfile=f"a/{display_path}", 
                    tofile=f"b/{display_path}",
                    n=3
                )
                full_diff_text += "".join(diff)
            except Exception as e:
                logger.error(f"Error diffing {display_path}: {e}")

        if not full_diff_text.strip():
            full_diff_text = "No textual changes detected."

        # 3. HTML Template
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <title>Delta Diff - {title_extra}</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css" />
            <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/diff2html/bundles/css/diff2html.min.css" />
            <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/diff2html/bundles/js/diff2html-ui.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background: #f4f4f4; }}
                .d2h-file-header {{ background-color: #f0f0f0 !important; }}
                h2 {{ color: #333; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h2>Review: {title_extra}</h2>
            <div id="diff_target"></div>
            <script>
                document.addEventListener('DOMContentLoaded', function () {{
                    var diffString = `{full_diff_text.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$').replace('</', '<\\/')}`;
                    var targetElement = document.getElementById('diff_target');
                    var configuration = {{
                        drawFileList: true,
                        fileListToggle: false,
                        fileListStartVisible: false,
                        fileContentToggle: true,
                        matching: 'lines',
                        outputFormat: 'side-by-side',
                        synchronisedScroll: true,
                        highlight: true,
                        renderNothingWhenEmpty: false,
                    }};
                    var diff2htmlUi = new Diff2HtmlUI(targetElement, diffString, configuration);
                    diff2htmlUi.draw();
                    diff2htmlUi.highlightCode();
                }});
            </script>
        </body>
        </html>
        """

        fd, path = tempfile.mkstemp(suffix=".html", prefix=f"delta_review_")
        with os.fdopen(fd, 'w', encoding="utf-8") as f:
            f.write(html_content)
        
        webbrowser.open(f"file://{path}")

    except Exception as e:
        logger.error(f"Failed to open diff report: {e}")


def _execute_attempt(
    attempt: int,
    max_retries: int,
    validated_files: list[str],
    current_prompt: str,
    original_prompt: str,
    history: list[dict],
    output_func: OutputFunc,
    stream_func: OutputFunc | None,
    cancel_event: threading.Event | None,
    validation_cmd: str,
    validation_timeout: float,
    verify: bool,
    ambiguous_mode: str,
    allow_new_files: bool,
    on_file_added: Callable[[Path], None] | None,
    on_diff_failure: Callable[[str, str], None] | None,
    on_validation_failure: Callable[[str], None] | None,
    on_validation_start: Callable[[str], None] | None = None,
    on_validation_success: Callable[[], None] | None = None,
    confirmation_callback: Callable[[dict[str, int], dict[str, str]], bool] | None = None,
) -> tuple[bool, str | None, str | None, str | None]:
    """Execute a single modification attempt.
    
    Returns:
        (success, backup_id, user_error_message, llm_error_prompt)
    """
    logger.info(f"--- Attempt {attempt}/{max_retries} ---")

    # Generate LLM response
    try:
        result_diff = generate(
            validated_files, current_prompt,
            output_func_override=output_func,
            raw_stream_output_func=stream_func,
            conversation_history=history,
            cancel_event=cancel_event
        )
    except CancelledError:
        raise
    except Exception as e:
        return False, None, f"Generation failed: {e}", None

    if cancel_event and cancel_event.is_set():
        raise CancelledError("Cancelled")

    # Apply diffs
    backup_id = None
    try:
        diffs, session_backup_id = apply_diffs(
            result_diff, 
            ambiguous_mode=ambiguous_mode, 
            original_prompt=original_prompt,
            confirmation_callback=confirmation_callback
        )
        if session_backup_id:
            backup_id = session_backup_id

        # Track modified/created files
        for fname in diffs:
            fpath = Path.cwd() / fname
            file_cache.invalidate(str(fpath))
            resolved = fpath.resolve()
            if allow_new_files and str(resolved) not in validated_files:
                if on_file_added:
                    on_file_added(resolved)
                validated_files.append(str(resolved))

        # Output summary
        logger.info(f"Applied diffs to {len(diffs)} file(s).")
        for f, status in diffs.items():
            logger.info(f"> {f}: {status}")

    except CancelledError:
        raise
    except Exception as e:
        # Diff parsing/application failed
        logger.error(f"Diff application failed: {e}")
        if on_diff_failure:
            on_diff_failure(str(e), result_diff)

        # Cleanup history for retry
        if history and len(history) >= 2:
            history.pop()  # Assistant
            history.pop()  # User

        return False, backup_id, f"Diff application failed: {e}", None

    # Run validation (Shell)
    if validation_cmd:
        if on_validation_start:
            on_validation_start(validation_cmd)
        
        valid, error_msg = _run_validation(validation_cmd, validation_timeout)
        if not valid:
            if on_validation_failure:
                on_validation_failure(error_msg)
            return False, backup_id, f"Validation error: {error_msg}", f"Fix this error:\n\n{error_msg}"
        
        if on_validation_success:
            on_validation_success()

    # Run verification (LLM)
    if verify:
        verification_prompt = (
            f"I have applied changes to satisfy this request: '{original_prompt}'\n\n"
            "Here are the current file contents. "
            "Does this successfully satisfy the original request? "
            "Output 'YES' if satisfied, or a brief list of remaining issues if not."
        )
        logger.info("Verifying changes with LLM...")
        
        try:
            # We iterate on existing files to build verification context
            verify_response = generate(
                validated_files, 
                verification_prompt,
                output_func_override=output_func,
                ask_mode=True,
                cancel_event=cancel_event
            )
            
            # Naive partial match for YES
            first_words = verify_response[:50].upper()
            if "YES" not in first_words.split() and "YES." not in first_words.split():
                critique = verify_response
                logger.error(f"Verification Failed.\nCritique: {critique[:300]}...")
                if on_validation_failure:
                    on_validation_failure(f"Verification failed: {critique[:100]}...")
                return False, backup_id, f"Verification failed: {critique}", f"The previous attempt failed verification: {critique}. Please fix."
            
            logger.info("Verification Passed.")
        except CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            # If verification blows up, do we fail the attempt? 
            # Let's fail to be safe.
            return False, backup_id, f"Verification process error: {e}", f"Verification process error: {e}"

    return True, backup_id, None, None


def process_request(
    files: list[str],
    prompt: str,
    history: list[dict],
    output_func: OutputFunc,
    stream_func: OutputFunc | None = None,
    cancel_event: threading.Event | None = None,
    validation_cmd: str = "",
    validation_timeout: float = 10.0,
    max_retries: int = 2,
    recursion_limit: int = 0,
    ambiguous_mode: str = "replace_all",
    ask_mode: bool = False,
    plan_mode: bool = False,
    allow_new_files: bool = True,
    on_file_added: Callable[[Path], None] | None = None,
    on_diff_failure: Callable[[str, str], None] | None = None,
    on_validation_failure: Callable[[str], None] | None = None,
    verify: bool = False,
    validate_at_start: bool = False,
    on_validation_start: Callable[[str], None] | None = None,
    on_validation_success: Callable[[], None] | None = None,
    confirmation_callback: Callable[[dict[str, int], dict[str, str]], bool] | None = None,
) -> dict:
    """
    Unified process loop for modifying files based on a prompt.

    The loop has two levels:
    - Outer loop: recursion iterations (runs the full prompt->validate cycle)
    - Inner loop: retry attempts within each iteration (for failures)

    Args:
        files: List of file paths to include in context
        prompt: The user's request
        history: Conversation history for multi-turn
        output_func: Function to output status messages
        stream_func: Optional function for streaming raw LLM output
        cancel_event: Event to signal cancellation
        validation_cmd: Shell command to validate changes (optional)
        validation_timeout: Timeout for validation in seconds
        max_retries: Max attempts per iteration before failing/continuing
        recursion_limit: Number of additional iterations (0 = single pass)
        ambiguous_mode: How to handle ambiguous diff matches
        ask_mode: If True, just answer questions without modifying files
        allow_new_files: Whether to allow creation of new files
        on_file_added: Callback when a new file is created
        on_diff_failure: Callback when diff application fails
        on_validation_failure: Callback when validation fails
        verify: If True, ask LLM to verify changes satisfy the prompt

    Returns:
        Dict with keys: success (bool), backup_id (str|None), message (str)
    """
    def is_cancelled():
        return cancel_event and cancel_event.is_set()

    def make_result(success: bool, backup_id: str | None, message: str) -> dict:
        return {"success": success, "backup_id": backup_id, "message": message}

    # Validate input files
    validated_files, err = validate_files(files)
    if err:
        logger.error(f"File validation failed: {err}")
        return make_result(False, None, f"File validation failed: {err}")

    # Run pre-validation if requested (only for modifications)
    if validate_at_start and validation_cmd and not (ask_mode or plan_mode):
        output_func("Running pre-validation...")
        if on_validation_start:
            on_validation_start(validation_cmd)

        valid, error_msg = _run_validation(validation_cmd, validation_timeout)
        if not valid:
            if on_validation_failure:
                on_validation_failure(f"Pre-validation failed: {error_msg}")
            
            logger.info("Pre-validation failed. Proceeding with error in context.")
            prompt = f"The existing code is failing validation:\n\n{error_msg}\n\nTask: {prompt}\n\nPlease fix the existing errors and complete the task."
        else:
            if on_validation_success:
                on_validation_success()

    # Ask mode or Plan mode: no modifications, just generate response
    if ask_mode or plan_mode:
        try:
            if plan_mode:
                logger.info("--- Planning Mode ---")
            else:
                logger.info("--- Ask Mode ---")

            generate(validated_files, prompt, output_func_override=output_func,
                    raw_stream_output_func=stream_func, conversation_history=history,
                    ask_mode=True, plan_mode=plan_mode, cancel_event=cancel_event)
            return make_result(True, None, "Planning complete." if plan_mode else "Ask mode complete.")
        except CancelledError as e:
            return make_result(False, None, str(e) or "Cancelled.")
        except Exception as e:
            return make_result(False, None, str(e))

    # Modification loop
    backup_id = None
    num_iterations = recursion_limit + 1
    is_multi_iteration = num_iterations > 1
    last_error_msg = "Failed after retries."

    for iteration in range(num_iterations):
        current_prompt = prompt

        if is_cancelled():
            return make_result(False, backup_id, "Cancelled.")

        if is_multi_iteration:
            logger.info(f"=== Iteration {iteration + 1}/{num_iterations} ===")

        iteration_succeeded = False

        # Retry loop within this iteration
        for attempt in range(1, max_retries + 1):
            if is_cancelled():
                return make_result(False, backup_id, "Cancelled.")

            try:
                success, sess_backup_id, user_fail_msg, llm_fail_prompt = _execute_attempt(
                    attempt, max_retries, validated_files,
                    current_prompt, prompt, history,
                    output_func, stream_func, cancel_event,
                    validation_cmd, validation_timeout, verify,
                    ambiguous_mode, allow_new_files,
                    on_file_added, on_diff_failure, on_validation_failure,
                    on_validation_start, on_validation_success,
                    confirmation_callback
                )
                
                if sess_backup_id:
                    backup_id = sess_backup_id

                if success:
                    iteration_succeeded = True
                    break
                
                if user_fail_msg:
                    last_error_msg = user_fail_msg

                # Failure case
                if attempt < max_retries:
                    logger.warning("Retrying...")
                    if llm_fail_prompt:
                        current_prompt = llm_fail_prompt
                    # Else: silent retry (keep current_prompt)
                    continue
                
                # If we're here, it failed and we are out of retries
                logger.warning("Max retries reached.")

            except CancelledError as e:
                return make_result(False, backup_id, str(e) or "Cancelled.")
            except Exception as e:
                return make_result(False, backup_id, f"Execution failed: {e}")

        if not iteration_succeeded:
            return make_result(False, backup_id, last_error_msg)

        # Re-validate files before next iteration (file list may have changed)
        if is_multi_iteration and iteration < num_iterations - 1:
            validated_files, _ = validate_files(validated_files)

    return make_result(True, backup_id, "Task complete.")
