"""Application state and data structures."""
import json
import logging
import os
import queue
import time
import shutil
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

import core
from core import APP_DATA_DIR, config, AVAILABLE_MODELS, MAX_PROMPT_HISTORY, file_cache
from widgets import ChatBubble

# Persistence files
FILESET_PATH = str(APP_DATA_DIR / "filesets.json")
PROMPT_HISTORY_PATH = str(APP_DATA_DIR / "prompt_history.json")
CWD_HISTORY_PATH = str(APP_DATA_DIR / "cwd_history.json")
PRESETS_PATH = str(APP_DATA_DIR / "selection_presets.json")
SESSIONS_DIR = APP_DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_HIDDEN = {
    ".git", ".svn", ".hg", ".DS_Store", "Thumbs.db",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
    ".vscode", ".idea", ".vs",
    "venv", ".venv", "env", "node_modules", "site-packages",
    "dist", "build", "target", "out", "bin", "obj",
    "vendor", "coverage"
}

def to_relative(path: Path) -> Path:
    """Convert path to relative to CWD if possible."""
    try:
        return path.resolve().relative_to(Path.cwd())
    except ValueError:
        return path

@dataclass
class ChatSession:
    """Manages state for a single chat session."""
    id: int
    history: list = field(default_factory=list)
    bubbles: list = field(default_factory=list)
    current_bubble: Any = None
    anchors: list = field(default_factory=list)

    last_prompt: str = ""
    backup_id: str = None
    current_response_chars: int = 0
    request_start_time: float = 0.0
    execution_start_time: float = None

    is_generating: bool = False
    is_queued: bool = False
    is_ask_mode: bool = False
    is_debug: bool = False
    is_planning: bool = False  # True if this is a planning generation session
    failed: bool = False  # True if last request failed
    completed: bool = False  # True if last request completed successfully

    # Input state
    input_text: str = ""
    scroll_to_bottom: bool = False

    # Cancellation
    cancel_event: threading.Event = field(default_factory=threading.Event)

    # Grouping
    group_id: int | None = None

    # Files added during this session
    session_added_files: set = field(default_factory=set)

    # Approval state
    waiting_for_approval: bool = False
    approval_event: threading.Event = field(default_factory=threading.Event)
    approval_result: bool = False

    def to_dict(self) -> dict:
        """Serialize session state to a dictionary."""
        return {
            "history": self.history,
            "session_added_files": [str(p) for p in self.session_added_files],
            "last_prompt": self.last_prompt,
            "is_ask_mode": self.is_ask_mode,
            "is_debug": self.is_debug,
            "is_planning": self.is_planning,
            "failed": self.failed,
            "completed": self.completed,
            "backup_id": self.backup_id,
            "group_id": self.group_id,
        }

    def from_dict(self, data: dict) -> None:
        """Restore session state from a dictionary."""
        self.history = data.get("history", [])
        self.session_added_files = {Path(p) for p in data.get("session_added_files", [])}
        self.last_prompt = data.get("last_prompt", "")
        self.is_ask_mode = data.get("is_ask_mode", False)
        self.is_debug = data.get("is_debug", False)
        self.is_planning = data.get("is_planning", False)
        self.failed = data.get("failed", False)
        self.completed = data.get("completed", False)
        self.backup_id = data.get("backup_id")
        self.group_id = data.get("group_id")

    def recover_lost_prompt(self) -> None:
        """Recover last_prompt into input_text if it wasn't saved content history."""
        if self.input_text or not self.last_prompt:
            return

        last_history_content = ""
        if self.history:
            for msg in reversed(self.history):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        try:
                            parts = [p["text"] for p in content if p.get("type") == "text"]
                            content = "\n".join(parts)
                        except Exception:
                            content = ""
                    else:
                        content = str(content)
                    last_history_content = content
                    break
        
        if self.last_prompt != last_history_content:
            self.input_text = self.last_prompt

@dataclass
class AppState:
    """Global application state."""
    # Sessions
    sessions: dict = field(default_factory=dict)  # id -> ChatSession
    active_session_id: int = None
    next_session_id: int = 1
    next_group_id: int = 1

    # File selection
    file_paths: list = field(default_factory=list)
    file_tree: dict = field(default_factory=dict)
    is_scanning: bool = False
    selected_files: set = field(default_factory=set)  # Files in the sidebar list
    file_checked: dict = field(default_factory=dict)  # Path -> bool, whether file is checked for context
    file_exists_cache: dict = field(default_factory=dict) # Cache for file existence check
    rolling_check_idx: int = 0
    frame_count: int = 0
    folder_states: dict = field(default_factory=dict)
    presets: dict = field(default_factory=dict)

    # Queue system
    gui_queue: queue.Queue = field(default_factory=queue.Queue)
    impl_queue: list = field(default_factory=list)
    impl_history: list = field(default_factory=list)
    current_impl_sid: int = None
    queue_blocked: bool = False

    # UI state
    prompt_history: list = field(default_factory=list)
    cwd_history: list = field(default_factory=list)
    prompt_history_idx: int = -1
    logs: deque = field(default_factory=lambda: deque(maxlen=2000))
    show_debug_logs: bool = False

    chat_input_height: float = 80.0

    # Settings UI state
    model_idx: int = 0
    backup_enabled: bool = config.backup_enabled
    use_git_backup: bool = config.use_git_backup
    auto_review: bool = config.auto_review
    block_on_fail: bool = config.block_on_fail
    focus_mode_idx: int = 0
    focus_trigger_idx: int = 0
    max_tries: str = str(config.default_tries)
    recursions: str = str(config.default_recurse)
    timeout: str = str(config.default_timeout)
    output_sharding_limit: str = str(config.output_sharding_limit)
    sharding_ratio: str = str(config.sharding_ratio)
    max_shards: str = str(config.max_shards)
    validation_cmd: str = ""
    validate_at_start: bool = config.validate_at_start
    validate_command_enabled: bool = True
    add_new_files: bool = config.add_new_files
    persist_session: bool = config.persist_session
    verify_changes: bool = config.verify_changes
    require_approval: bool = config.require_approval
    ambiguous_mode_idx: int = 0

    # Fuzzy Match settings
    diff_fuzzy_lines_threshold: str = "0.95"
    diff_fuzzy_max_bad_lines: str = "0"

    # Context manager state
    show_context_manager: bool = False
    context_search_text: str = ""
    last_context_search: str = None
    view_tree: dict = field(default_factory=dict)
    view_tree_dirty: bool = True

    # Performance Caching
    stats_dirty: bool = True
    cached_sorted_files: list = field(default_factory=list)
    cached_total_lines: int = 0
    cached_total_tokens: int = 0
    cached_cost_str: str = "..."

    # Group creation popup
    show_create_group_popup: bool = False
    new_group_name: str = ""

    # Sessions / Saves
    show_sessions_window: bool = False
    selected_save_name: str | None = None
    new_save_name: str = ""
    last_input_time: float = 0.0
    input_dirty: bool = False

    # File dialogs
    show_backup_history: bool = False
    file_dialog: Any = None

    # API Settings Popup
    show_api_settings_popup: bool = False
    api_settings_inputs: dict = field(default_factory=dict)
    api_settings_error: str = ""

    can_use_git: bool = False

    # System Prompt Popup
    show_system_prompt_popup: bool = False
    temp_system_prompt: str = ""

    # Git validation popups
    show_no_git_popup: bool = False
    show_no_repo_popup: bool = False
    pending_prompt: str = ""
    pending_is_planning: bool = False
    pending_ask_mode: bool = False

    # Missing files popup
    show_missing_files_popup: bool = False
    missing_files_list: list = field(default_factory=list)

    # Tab closing popup
    show_close_tab_popup: bool = False
    session_to_close_id: int | None = None
    restart_requested: bool = False

    # Exit confirmation
    show_exit_confirmation_popup: bool = False
    pending_exit_action: str | None = None  # "exit" or "restart"

    # Caches
    backup_list: list = None
    drag_data: Any = None

# Global state instance
state: AppState = AppState()

def init_app_state():
    """Initialize or reset the global app state."""
    # Reset the existing state object in-place to preserve references
    # held by other modules (gui.py, cli.py)
    new_state = AppState()
    state.__dict__.clear()
    state.__dict__.update(new_state.__dict__)

def create_session() -> ChatSession:
    """Create a new chat session."""
    session_id = state.next_session_id
    state.next_session_id += 1
    session = ChatSession(id=session_id)
    state.sessions[session_id] = session
    return session

def get_active_session() -> ChatSession | None:
    """Get the currently active session."""
    if state.active_session_id is None:
        return None
    return state.sessions.get(state.active_session_id)

def close_session(session_id: int) -> None:
    """Close a session."""
    if session_id in state.sessions:
        del state.sessions[session_id]
    if state.active_session_id == session_id:
        if state.sessions:
            state.active_session_id = next(iter(state.sessions.keys()))
        else:
            state.active_session_id = None

def unqueue_session(session_id: int) -> None:
    """Remove a session from the execution queue."""
    if session_id in state.impl_queue:
        state.impl_queue.remove(session_id)
    
    session = state.sessions.get(session_id)
    if session:
        session.is_queued = False

def log_message(text: str) -> None:
    """Log a message using standard logging."""
    logging.info(text)

class GuiLogHandler(logging.Handler):
    """Pushes logs to the GUI queue."""
    def emit(self, record):
        try:
            msg = self.format(record)
            if state and state.gui_queue:
                state.gui_queue.put({
                    "type": "log_entry",
                    "level": record.levelname,
                    "message": msg,
                    "timestamp": record.created
                })
        except Exception:
            self.handleError(record)

def setup_logging(enable_gui=True):
    """Configure application logging to file and GUI."""
    log_dir = APP_DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "delta.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Silence noisy libraries
    for lib in ["urllib3", "httpcore", "httpx", "openai", "PIL", "git"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    
    # Remove existing handlers to avoid duplicates on restart
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    # File Handler
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=3, encoding='utf-8')
    file_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_fmt)
    root_logger.addHandler(file_handler)

    # GUI Handler
    if enable_gui:
        gui_handler = GuiLogHandler()
        gui_fmt = logging.Formatter('%(message)s')
        gui_handler.setFormatter(gui_fmt)
        root_logger.addHandler(gui_handler)

def rebuild_session_bubbles(session: ChatSession):
    """Rebuild ChatBubbles from history."""
    session.bubbles = []
    for i, msg in enumerate(session.history):
        role = msg.get("role", "user")
        
        bubble = ChatBubble(role, i)
        content = msg.get("content", "")
        
        if isinstance(content, list):
            text_parts = [p["text"] for p in content if p.get("type") == "text"]
            content = "\n".join(text_parts)
            
        bubble.update(content)
        bubble.flush()
        session.bubbles.append(bubble)

def save_state(name: str):
    """Save the current program state to a JSON file."""
    if not name: name = "autosave"
    
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()
    if not safe_name: safe_name = "unnamed_save"
    
    filename = SESSIONS_DIR / f"{safe_name}.json"
    
    sessions_data = []
    for sid, sess in state.sessions.items():
        sess_data = sess.to_dict()
        sess_data["__id__"] = sid 
        sess_data["__input_text__"] = sess.input_text
        sessions_data.append(sess_data)
        
    data = {
        "timestamp": time.time(),
        "active_session_id": state.active_session_id,
        "next_session_id": state.next_session_id,
        "next_group_id": state.next_group_id,
        "sessions": sessions_data,
        "chat_input_height": state.chat_input_height
    }
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log_message(f"Error saving state: {e}")

def load_state(name: str):
    """Load program state from a save file."""
    filename = SESSIONS_DIR / f"{name}.json"
    if not filename.exists():
        log_message(f"Save file not found: {filename}")
        return
        
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        state.sessions.clear()
        
        state.next_session_id = data.get("next_session_id", 1)
        state.next_group_id = data.get("next_group_id", 1)
        state.chat_input_height = data.get("chat_input_height", 80.0)
        
        sessions_list = data.get("sessions", [])
        sessions_list.sort(key=lambda x: x.get("__id__", 0))
        
        for s_data in sessions_list:
            sid = s_data.get("__id__")
            if sid is None: continue
            
            new_sess = ChatSession(id=sid)
            new_sess.from_dict(s_data)
            new_sess.input_text = s_data.get("__input_text__", "")
            new_sess.recover_lost_prompt()
            
            rebuild_session_bubbles(new_sess)
            state.sessions[sid] = new_sess
            
        state.active_session_id = data.get("active_session_id")
        
        if state.active_session_id not in state.sessions and state.sessions:
            state.active_session_id = next(iter(state.sessions.keys()))
            
        if not state.sessions:
            s = create_session()
            state.active_session_id = s.id
            
        log_message(f"Loaded save: {name}")
        
    except Exception as e:
        log_message(f"Error loading save: {e}")

def load_individual_session(save_name: str, session_idx: int):
    """Load a single session from a save into the current state."""
    filename = SESSIONS_DIR / f"{save_name}.json"
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        sessions_list = data.get("sessions", [])
        if session_idx < 0 or session_idx >= len(sessions_list):
            return
            
        s_data = sessions_list[session_idx]
        
        new_sess = create_session()
        new_sess.from_dict(s_data)
        new_sess.input_text = s_data.get("__input_text__", "")
        new_sess.recover_lost_prompt()
        new_sess.group_id = None
        
        rebuild_session_bubbles(new_sess)
        
        state.active_session_id = new_sess.id
        log_message("Imported session from save.")
        
    except Exception as e:
        log_message(f"Error importing session: {e}")

def delete_save(name: str):
    """Delete a save file."""
    filename = SESSIONS_DIR / f"{name}.json"
    try:
        if filename.exists():
            filename.unlink()
            if state.selected_save_name == name:
                state.selected_save_name = None
    except Exception as e:
        log_message(f"Error deleting save: {e}")

def get_saves_list() -> list[dict]:
    """Get list of available saves."""
    saves = []
    if not SESSIONS_DIR.exists():
        return []
        
    for p in SESSIONS_DIR.glob("*.json"):
        try:
            stats = p.stat()
            name = p.stem
            saves.append({
                "name": name,
                "mtime": stats.st_mtime,
                "path": p
            })
        except Exception: pass
        
    saves.sort(key=lambda x: x["mtime"], reverse=True)
    return saves

def _load_json(path: str, default: Any) -> Any:
    try:
        if Path(path).exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _save_json(path: str, data: Any) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass

def load_prompt_history():
    state.prompt_history = _load_json(PROMPT_HISTORY_PATH, [])

def save_prompt_history():
    if MAX_PROMPT_HISTORY > 0:
        data = state.prompt_history[-MAX_PROMPT_HISTORY:]
    else:
        data = state.prompt_history
    _save_json(PROMPT_HISTORY_PATH, data)

def load_cwd_history():
    state.cwd_history = _load_json(CWD_HISTORY_PATH, [])

def save_cwd_history():
    _save_json(CWD_HISTORY_PATH, state.cwd_history)

def add_to_cwd_history(path_str: str):
    try:
        abs_path = str(Path(path_str).resolve())
        if abs_path in state.cwd_history:
            state.cwd_history.remove(abs_path)
        state.cwd_history.insert(0, abs_path)
        if len(state.cwd_history) > 10:
            state.cwd_history = state.cwd_history[:10]
        save_cwd_history()
    except Exception:
        pass

def change_working_directory(new_path: str):
    """Change the current working directory and reset state."""
    if new_path and Path(new_path).exists():
        try:
            os.chdir(new_path)
            add_to_cwd_history(new_path)

            state.selected_files = set()
            state.file_checked = {}
            state.sessions = {}
            state.active_session_id = None
            state.backup_list = None
            state.validation_cmd = ""
            
            file_cache.clear()
            core.force_io_cache_refresh()
            refresh_project_files()
            load_fileset()
            
            initial_session = create_session()
            state.active_session_id = initial_session.id
            
            log_message(f"Changed working directory to: {new_path}")
        except Exception as e:
            log_message(f"Error changing directory: {e}")

def load_fileset():
    data = _load_json(FILESET_PATH, {})
    cwd = str(Path.cwd())
    if data.get("cwd") == cwd:
        state.selected_files = {to_relative(Path(p)) for p in data.get("files", [])}
        state.validation_cmd = data.get("validation_cmd", "")
        state.validate_command_enabled = data.get("validate_command_enabled", True)

        state.file_checked = {}
        checked_state = data.get("checked", {})
        for p_str, is_checked in checked_state.items():
            try:
                p = to_relative(Path(p_str))
                if p in state.selected_files:
                    state.file_checked[p] = is_checked
            except Exception:
                pass

def save_fileset():
    data = {
        "cwd": str(Path.cwd()),
        "files": [str(p) for p in state.selected_files],
        "validation_cmd": state.validation_cmd,
        "validate_command_enabled": state.validate_command_enabled,
        "checked": {str(p): v for p, v in state.file_checked.items() if p in state.selected_files}
    }
    _save_json(FILESET_PATH, data)

def load_presets():
    state.presets = _load_json(PRESETS_PATH, {})

def save_presets():
    _save_json(PRESETS_PATH, state.presets)

def build_file_tree(paths: list[Path], root: Path) -> dict:
    """Build a file tree structure from a list of paths."""
    tree = {}
    for file_path in paths:
        try:
            rel = file_path.relative_to(root)
            parts = rel.parts
        except ValueError:
            parts = file_path.parts

        current_dict = tree
        target_node = tree

        for part in parts[:-1]:
            if part not in current_dict:
                current_dict[part] = {"_files": [], "_children": {}}
            target_node = current_dict[part]
            current_dict = target_node["_children"]

        if "_files" not in target_node:
            target_node["_files"] = []
            if "_children" not in target_node:
                target_node["_children"] = {}
        target_node["_files"].append((parts[-1], file_path))
    return tree

def _scan_worker():
    """Worker function for file scanning."""
    try:
        cwd = Path.cwd()
        new_paths = []

        for item in cwd.rglob("*"):
            if item.is_file():
                try:
                    rel = item.relative_to(cwd)
                except ValueError:
                    rel = item

                parts = rel.parts
                if any(p in DEFAULT_HIDDEN or p.startswith('.') for p in parts):
                    continue
                new_paths.append(rel)

        new_paths.sort(key=lambda p: str(p).lower())
        
        new_tree = build_file_tree(new_paths, cwd)
        
        state.file_paths = new_paths
        state.file_tree = new_tree
        state.view_tree_dirty = True
    except Exception as e:
        log_message(f"Error scanning files: {e}")
    finally:
        state.is_scanning = False

def refresh_project_files():
    """Scan and refresh project files in background."""
    core.clear_stats_cache()
    state.stats_dirty = True
    if state.is_scanning:
        return

    state.is_scanning = True
    threading.Thread(target=_scan_worker, daemon=True).start()

def toggle_file_selection(path: Path, selected: bool):
    """Toggle file selection."""
    path = to_relative(path)
    if selected:
        state.selected_files.add(path)
    else:
        state.selected_files.discard(path)
    state.stats_dirty = True
    save_fileset()

def toggle_folder_selection(folder_path: Path, selected: bool):
    """Toggle all files in a folder."""
    folder_path = to_relative(folder_path)
    folder_str = str(folder_path)
    is_root = str(folder_path) == "."
    
    for f in state.file_paths:
        if is_root or str(f).startswith(folder_str):
            if selected:
                state.selected_files.add(f)
            else:
                state.selected_files.discard(f)
    state.stats_dirty = True
    save_fileset()

def sync_settings_from_config():
    """Sync UI state from config."""
    models = list(AVAILABLE_MODELS.keys())
    state.model_idx = models.index(config.model) if config.model in models else 0
    state.backup_enabled = config.backup_enabled
    state.use_git_backup = config.use_git_backup
    state.auto_review = config.auto_review
    state.verify_changes = config.verify_changes
    state.require_approval = config.require_approval
    state.validate_at_start = config.validate_at_start
    state.add_new_files = config.add_new_files
    state.persist_session = config.persist_session
    state.block_on_fail = config.block_on_fail
    state.max_tries = str(config.default_tries)
    state.recursions = str(config.default_recurse)
    state.timeout = str(config.default_timeout)
    state.output_sharding_limit = str(config.output_sharding_limit)
    state.max_shards = str(config.max_shards)
    state.sharding_ratio = str(config.sharding_ratio)

    state.diff_fuzzy_lines_threshold = str(config.diff_fuzzy_lines_threshold)
    state.diff_fuzzy_max_bad_lines = str(config.diff_fuzzy_max_bad_lines)
    
    modes = ["replace_all", "ignore", "fail"]
    if config.default_ambiguous_mode in modes:
        state.ambiguous_mode_idx = modes.index(config.default_ambiguous_mode)

    focus_modes = ["off", "flash", "yank"]
    fm = config.focus_mode if isinstance(config.focus_mode, str) else ("flash" if config.focus_mode else "off")
    state.focus_mode_idx = focus_modes.index(fm) if fm in focus_modes else 0

    focus_triggers = ["task", "queue"]
    ft = config.focus_trigger if config.focus_trigger else "task"
    state.focus_trigger_idx = focus_triggers.index(ft) if ft in focus_triggers else 0

def sync_config_from_settings():
    """Sync config from UI state."""
    models = list(AVAILABLE_MODELS.keys())
    config.model = models[state.model_idx]
    config.set_backup_enabled(state.backup_enabled)
    config.set_auto_review(state.auto_review)
    config.set_verify_changes(state.verify_changes)
    config.set_require_approval(state.require_approval)
    config.set_validate_at_start(state.validate_at_start)
    config.set_add_new_files(state.add_new_files)
    config.set_persist_session(state.persist_session)
    config.set_block_on_fail(state.block_on_fail)

    try:
        config.set_default_tries(int(state.max_tries))
    except ValueError: pass
    try:
        config.set_default_recurse(int(state.recursions))
    except ValueError: pass
    try:
        config.set_default_timeout(float(state.timeout))
    except ValueError: pass

    try:
        config.output_sharding_limit = int(state.output_sharding_limit)
    except ValueError: pass
    try:
        config.max_shards = int(state.max_shards)
    except ValueError: pass
    try:
        config.sharding_ratio = float(state.sharding_ratio)
    except ValueError: pass

    try:
        val = float(state.diff_fuzzy_lines_threshold)
        config.set_diff_fuzzy_lines_threshold(max(0.0, min(1.0, val)))
    except ValueError: pass

    try:
        val = int(state.diff_fuzzy_max_bad_lines)
        config.set_diff_fuzzy_max_bad_lines(max(0, val))
    except ValueError: pass

    focus_modes = ["off", "flash", "yank"]
    config.focus_mode = focus_modes[state.focus_mode_idx]

    focus_triggers = ["task", "queue"]
    config.set_focus_trigger(focus_triggers[state.focus_trigger_idx])
