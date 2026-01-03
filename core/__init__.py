"""Facade for deltatool.core to maintain original API."""

from .config import (
    config, API_KEY, APP_DATA_DIR, BACKUP_DIR, 
    AVAILABLE_MODELS, API_BASE_URL, TOKENS_PER_CHAR_ESTIMATE,
    MAX_PROMPT_HISTORY, QUEUE_POLL_INTERVAL_MS, update_core_settings, 
    load_json_file, save_json_file,
    _TOOL_DIR, estimate_tokens, SETTINGS_PATH
)

from .fs import (
    file_cache, scan_directory, validate_files, run_command, 
    is_image_file, is_binary_file, get_display_path, get_file_stats, 
    create_askpass_wrapper, open_path_in_os, open_terminal_in_os,
    clear_stats_cache, load_cwd_data, save_cwd_data, ensure_temp_dir
)

from .llm import (
    generate, calculate_input_cost, build_system_message, 
    build_file_contents, CancelledError, GenerationError
)

from .backups import (
    backup_manager, GitShadowHandler, undo_last_changes, 
    get_available_backups, is_git_installed, is_git_repo, 
    init_git_repo, restore_git_backup, force_io_cache_refresh
)

from .patching import (
    apply_diffs, parse_diffs, parse_plan, build_tolerant_regex,
    reconcile_path
)

from .workflow import (
    process_request, run_filedig_agent
)

from .analysis import (
    open_diff_report, _get_git_changes
)

import os
# Re-export exceptions
class DeltaToolError(Exception): pass
class DiffApplicationError(Exception): pass
class ValidationError(Exception): pass