"""Settings panel logic."""
import json
from pathlib import Path
from imgui_bundle import imgui, hello_imgui

import sys
import core
from core import config, AVAILABLE_MODELS, update_core_settings, is_git_installed, is_git_repo, open_diff_report, get_available_backups, undo_last_changes
from application_state import (
    state, sync_config_from_settings, sync_settings_from_config, save_fileset, 
    change_working_directory, log_message, refresh_project_files
)
from styles import STYLE, apply_imgui_theme
from .common import render_tooltip, check_for_updates
from .tutorial import register_area


def toggle_theme():
    """Toggle between light and dark theme."""
    new_theme = "light" if STYLE.dark else "dark"
    config.set_theme(new_theme)
    STYLE.load(new_theme)
    apply_imgui_theme(STYLE.dark)


def open_api_settings():
    """Open the API settings popup."""
    state.show_api_settings_popup = True
    # Use sys.modules to correctly get core.config module, bypassing core.config object shadowing
    cfg = sys.modules["core.config"]
    state.api_settings_inputs = {
        "api_key": cfg.API_KEY,
        "base_url": cfg.API_BASE_URL,
        "models": json.dumps(core.AVAILABLE_MODELS, indent=2),
        "git_branch": config.git_backup_branch
    }
    state.api_settings_error = ""


def render_api_settings_popup():
    if state.show_api_settings_popup:
        imgui.open_popup("API Settings")
        state.show_api_settings_popup = False
        
    if imgui.begin_popup_modal("API Settings", None, imgui.WindowFlags_.always_auto_resize)[0]:
        inputs = state.api_settings_inputs
        imgui.text("API Configuration")
        imgui.separator()
        
        imgui.text("API Key:")
        imgui.push_item_width(400)
        flags = 0 if inputs.get("_show_key") else imgui.InputTextFlags_.password
        _, inputs["api_key"] = imgui.input_text("##apikey", inputs.get("api_key", ""), flags)
        imgui.pop_item_width()
        render_tooltip("Your API key (e.g., sk-...) for the provider.")

        imgui.same_line()
        if imgui.small_button("Show" if not inputs.get("_show_key") else "Hide"):
            inputs["_show_key"] = not inputs.get("_show_key")
        
        imgui.text("Base URL:")
        imgui.push_item_width(400)
        _, inputs["base_url"] = imgui.input_text("##baseurl", inputs.get("base_url", ""))
        imgui.pop_item_width()
        render_tooltip("The API endpoint URL. Default: https://openrouter.ai/api/v1")

        imgui.text("Git Backup Branch:")
        imgui.push_item_width(400)
        _, inputs["git_branch"] = imgui.input_text("##gitbranch", inputs.get("git_branch", ""))
        imgui.pop_item_width()
        render_tooltip("Name of the hidden branch used for storing shadow backups.")

        imgui.separator()
        imgui.text("Available Models (JSON):")
        imgui.text_colored(STYLE.get_imvec4("fg_dim"), "Format: {\"model_id\": {\"input\": cost, \"output\": cost}, ...} (USD per 1M tokens)")
        
        _, inputs["models"] = imgui.input_text_multiline("##models", inputs.get("models", ""), imgui.ImVec2(500, 200))
        
        if state.api_settings_error:
            imgui.text_colored(STYLE.get_imvec4("btn_cncl"), state.api_settings_error)
            
        imgui.separator()
        
        if imgui.button("Save", imgui.ImVec2(120, 0)):
            try:
                models_dict = json.loads(inputs["models"])
                if not isinstance(models_dict, dict):
                    raise ValueError("Models must be a JSON object/dict")
                
                update_core_settings(
                    inputs["api_key"].strip(),
                    inputs["base_url"].strip(),
                    models_dict,
                    inputs["git_branch"].strip()
                )
                
                sync_settings_from_config()

                imgui.close_current_popup()
                log_message("API settings saved.")

                if not config.has_seen_tutorial:
                    state.show_tutorial_offer = True
                
            except Exception as e:
                state.api_settings_error = f"Error: {e}"
                
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(120, 0)):
            imgui.close_current_popup()
            
        imgui.end_popup()

def render_system_prompt_popup():
    if state.show_system_prompt_popup:
        imgui.open_popup("System Instructions")

    if imgui.begin_popup_modal("System Instructions", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text("Global Custom Instructions:")
        imgui.text_colored(STYLE.get_imvec4("fg_dim"), "These instructions are appended to the system prompt for every request.")
        
        changed, state.temp_system_prompt = imgui.input_text_multiline(
            "##sys_prompt_input", 
            state.temp_system_prompt, 
            imgui.ImVec2(500, 300)
        )

        imgui.separator()

        if imgui.button("Save", imgui.ImVec2(120, 0)):
            config.set_extra_system_prompt(state.temp_system_prompt.strip())
            state.show_system_prompt_popup = False
            imgui.close_current_popup()
            log_message("System prompt updated.")

        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(120, 0)):
            state.show_system_prompt_popup = False
            imgui.close_current_popup()
            
        imgui.end_popup()


def render_settings_panel():
    p_min = imgui.get_window_pos()
    s_size = imgui.get_window_size()
    register_area("settings", p_min, imgui.ImVec2(p_min.x + s_size.x, p_min.y + s_size.y))

    models = list(AVAILABLE_MODELS.keys())
    focus_modes = ["Off", "Flash", "Yank"]
    ambiguous_modes = ["Replace all", "Ignore", "Fail"]

    imgui.begin_group()
    cwd = str(Path.cwd())
    if len(cwd) > 35:
        cwd = "..." + cwd[-32:]
    imgui.text(f"CWD: {cwd}")

    imgui.same_line()
    if imgui.small_button("Change"):
        import tkinter as tk
        from tkinter import filedialog
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            new_path = filedialog.askdirectory()
            root.destroy()
            
            if new_path:
                change_working_directory(new_path)
        except Exception as e:
            log_message(f"Error opening file dialog: {e}")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Select a new project root directory.")

    if imgui.begin_popup_context_item("cwd_change_ctx"):
        if state.cwd_history:
            imgui.text_disabled("Recent Directories")
            imgui.separator()
            for hist_path in state.cwd_history:
                if imgui.selectable(hist_path, False)[0]:
                    change_working_directory(hist_path)
        else:
            imgui.text_disabled("No recent directories")
        imgui.end_popup()

    imgui.same_line()
    if imgui.small_button("CMD"):
        core.open_terminal_in_os(Path.cwd())

    if imgui.is_item_hovered():
        imgui.set_tooltip("Open terminal in CWD.\nRight-click for File Explorer.")

    if imgui.begin_popup_context_item("cmd_explore_ctx"):
        if imgui.selectable("Explore", False)[0]:
            core.open_path_in_os(Path.cwd())
        imgui.end_popup()
    imgui.end_group()
    register_area("cwd_area", imgui.get_item_rect_min(), imgui.get_item_rect_max())

    imgui.separator()

    imgui.text("Model:")
    imgui.same_line(80)
    imgui.set_next_item_width(-1)
    changed, state.model_idx = imgui.combo("##model", state.model_idx, models)
    render_tooltip("Select the LLM model to use.")
    if changed:
        sync_config_from_settings()

    imgui.separator()

    imgui.text("Theme:")
    imgui.same_line(80)
    imgui.set_next_item_width(-1)
    
    themes = ["Light", "Dark"]
    current_theme_idx = 1 if config.theme == "dark" else 0
    changed, new_theme_idx = imgui.combo("##theme", current_theme_idx, themes)
    if changed:
        new_theme = themes[new_theme_idx].lower()
        if new_theme != config.theme:
            config.set_theme(new_theme)
            STYLE.load(new_theme)
            apply_imgui_theme(STYLE.dark)

    imgui.separator()

    changed, state.backup_enabled = imgui.checkbox("Backup", state.backup_enabled)
    render_tooltip("Create a backup before applying any changes.")
    if changed:
        sync_config_from_settings()

    if state.backup_enabled:
        imgui.same_line()
        changed, state.use_git_backup = imgui.checkbox("Use Git", state.use_git_backup)
        render_tooltip("Use a hidden git branch for backups.")
        if changed:
            config.set_use_git_backup(state.use_git_backup)

    imgui.same_line()
    changed, state.verify_changes = imgui.checkbox("LLM Verify", state.verify_changes)
    render_tooltip("After applying changes, ask the LLM to verify if the file content matches the request.")
    if changed:
        sync_config_from_settings()

    changed, state.require_approval = imgui.checkbox("Require Approval", state.require_approval)
    render_tooltip("Pause and ask for confirmation before applying changes to disk.")
    if changed:
        sync_config_from_settings()

    imgui.same_line()
    changed, state.auto_review = imgui.checkbox("Auto-Review", state.auto_review)
    render_tooltip("Automatically open a diff report in browser after successful changes.")
    if changed:
        sync_config_from_settings()

    imgui.same_line()
    changed, state.block_on_fail = imgui.checkbox("Block on fail", state.block_on_fail)
    render_tooltip("Pause processing the task queue if a task fails.")
    if changed:
        sync_config_from_settings()

    imgui.text("Focus:")
    imgui.same_line()
    imgui.set_next_item_width(80)
    changed, state.focus_mode_idx = imgui.combo("##focus", state.focus_mode_idx, focus_modes)
    render_tooltip("Visual signal when generation completes. \nFlash: Flashes your screen.\nYank: Bring window to front and minimize other windows.")
    if changed:
        sync_config_from_settings()

    if state.focus_mode_idx != 0:
        imgui.same_line()
        focus_triggers = ["Task", "Queue"]
        imgui.set_next_item_width(70)
        changed, state.focus_trigger_idx = imgui.combo("##focustrigger", state.focus_trigger_idx, focus_triggers)
        render_tooltip("When to trigger focus: after every Task, or only when Queue is empty.")
        if changed:
            sync_config_from_settings()

    imgui.same_line()
    changed, state.persist_session = imgui.checkbox("Persist Session", state.persist_session)
    render_tooltip("Automatically load previous session state on boot.")
    if changed:
        sync_config_from_settings()

    imgui.separator()

    imgui.text("Tries:")
    imgui.same_line(70)
    imgui.set_next_item_width(50)
    changed, state.max_tries = imgui.input_text("##tries", state.max_tries)
    render_tooltip("Max retry attempts if the Model fails (e.g., bad diffs) or Validation fails.")
    if changed:
        sync_config_from_settings()

    imgui.same_line()
    imgui.text("Recurse:")
    imgui.same_line()
    imgui.set_next_item_width(50)
    changed, state.recursions = imgui.input_text("##recurse", state.recursions)
    render_tooltip("Reruns the prompt multiple times sequentially.")
    if changed:
        sync_config_from_settings()

    imgui.same_line()
    imgui.text("Timeout:")
    imgui.same_line()
    imgui.set_next_item_width(50)
    changed, state.timeout = imgui.input_text("##timeout", state.timeout)
    render_tooltip("Max seconds to wait for the validation command to complete.")
    if changed:
        sync_config_from_settings()

    imgui.text("Shard @:")
    imgui.same_line(70)
    imgui.set_next_item_width(60)
    changed, state.output_sharding_limit = imgui.input_text("##outshard", state.output_sharding_limit)
    render_tooltip("Token limit for output splitting (0 to disable).")
    if changed:
        sync_config_from_settings()

    imgui.same_line()
    imgui.text("Ratio:")
    imgui.same_line()
    imgui.set_next_item_width(40)
    changed, state.sharding_ratio = imgui.input_text("##shardratio", state.sharding_ratio)
    render_tooltip("Preemptively shard after a diff if above this % of shard limit.")
    if changed:
        sync_config_from_settings()

    imgui.same_line()
    imgui.text("Max Shards:")
    imgui.same_line()
    imgui.set_next_item_width(50)
    changed, state.max_shards = imgui.input_text("##maxshards", state.max_shards)
    render_tooltip("Maximum number of response chunks allowed.")
    if changed:
        sync_config_from_settings()

    imgui.separator()

    imgui.text("Validation Command:")

    changed_chk, state.validate_command_enabled = imgui.checkbox("##val_enabled", state.validate_command_enabled)
    render_tooltip("Enable/Disable the validation command.")
    if changed_chk:
        save_fileset()

    if not state.validate_command_enabled:
        imgui.begin_disabled()

    imgui.same_line()
    imgui.set_next_item_width(-1)
    changed, state.validation_cmd = imgui.input_text("##valcmd", state.validation_cmd)

    if not state.validate_command_enabled:
        imgui.end_disabled()

    render_tooltip("Shell command to run to verify changes (e.g. 'pytest', 'make test'). Saved per-project.")
    if changed:
        save_fileset()

    changed, state.validate_at_start = imgui.checkbox("Validate at start", state.validate_at_start)
    render_tooltip("Run validation command prior to generation to ensure clean state.")
    if changed:
        sync_config_from_settings()

    imgui.same_line()
    imgui.dummy((25.0, 1.0))
    imgui.same_line()
    imgui.text("On Fail:")
    imgui.same_line()
    imgui.set_next_item_width(120)
    val_behaviors = ["Correct", "Undo", "Retry", "Ignore"]
    changed, state.validation_failure_behavior_idx = imgui.combo("##valfail", state.validation_failure_behavior_idx, val_behaviors)
    render_tooltip("Action when the validation command fails.")
    if changed:
        sync_config_from_settings()

    imgui.text("Ambiguous diff:")
    imgui.same_line()
    imgui.set_next_item_width(100)
    changed, state.ambiguous_mode_idx = imgui.combo("##ambiguous", state.ambiguous_mode_idx, ambiguous_modes)
    if changed:
        modes = ["replace_all", "ignore", "fail"]
        if state.ambiguous_mode_idx < len(modes):
             config.set_default_ambiguous_mode(modes[state.ambiguous_mode_idx])
    render_tooltip("Strategy when a search block matches multiple locations in a file.")

    imgui.same_line()
    changed, state.add_new_files = imgui.checkbox("Add new files", state.add_new_files)
    render_tooltip("Add new files generated by the LLM to the context.")
    if changed:
        sync_config_from_settings()

    imgui.separator()
    imgui.text("Fuzzy Match:")

    imgui.same_line()
    imgui.text("Line Thresh:")
    imgui.same_line()
    imgui.set_next_item_width(40)
    changed1, state.diff_fuzzy_lines_threshold = imgui.input_text("##fuzz_t", state.diff_fuzzy_lines_threshold)
    render_tooltip("0.0-1.0: How similar a line must be to match")

    imgui.same_line()
    imgui.text("Bad Lines:")
    imgui.same_line()
    imgui.set_next_item_width(30)
    changed2, state.diff_fuzzy_max_bad_lines = imgui.input_text("##fuzz_b", state.diff_fuzzy_max_bad_lines)
    render_tooltip("Maximum number of fuzzy-matching lines allowed in a diff.")

    if changed1 or changed2:
        sync_config_from_settings()

    imgui.text("Dig Turns:")
    imgui.same_line()
    imgui.set_next_item_width(50)
    changed, state.filedig_max_turns = imgui.input_text("##fdturns", state.filedig_max_turns)
    render_tooltip("Max turns for Filedig agent.")
    if changed:
        sync_config_from_settings()

    imgui.same_line()
    changed, state.allow_rewrite = imgui.checkbox("Allow REWRITE", state.allow_rewrite)
    render_tooltip("Expose the REWRITE command to the LLM, allowing it to fully replace or delete files.")
    if changed:
        sync_config_from_settings()

    imgui.separator()

    # Cached check for git capability (throttled)
    if state.frame_count % 60 == 0:
        state.can_use_git = is_git_installed() and is_git_repo()

    btn_label = "Review" if state.can_use_git else "Review Latest Changes"

    avail_w = imgui.get_content_region_avail().x
    half_w = (avail_w - 8) / 2
    third_w = (avail_w - 16) / 3

    if imgui.button(btn_label, imgui.ImVec2(third_w, 0)):
        try:
            if state.can_use_git:
                open_diff_report(use_git=True)
                log_message("Opened git diff report in browser")
            else:
                backups = get_available_backups()
                if backups:
                    open_diff_report(backups[0]["session_id"])
                    log_message("Opened diff report for latest session")
                else:
                    log_message("No backups available to review")
        except Exception as e:
            log_message(f"Error opening diff report: {e}")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Generate and view a visual diff of the changes.")

    imgui.same_line()
    if imgui.button("Backups", imgui.ImVec2(third_w, 0)):
        state.show_backup_history = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("View backup history and restore files.")

    imgui.same_line()

    if imgui.button("Undo Latest", imgui.ImVec2(third_w, 0)):
        res = undo_last_changes()
        if "error" in res:
            log_message(f"Undo failed: {res['error']}")
        else:
            log_message(f"Undid changes to {len(res)} files.")
            refresh_project_files()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Rollback the most recent change. Can be repeated.")

    if imgui.button("Sessions", imgui.ImVec2(half_w, 0)):
        state.show_sessions_window = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("Open the Session Manager window.")

    imgui.same_line()

    if imgui.button("System Prompt", imgui.ImVec2(half_w, 0)):
        state.show_system_prompt_popup = True
        state.temp_system_prompt = config.extra_system_prompt
    if imgui.is_item_hovered():
        imgui.set_tooltip("Configure global custom instructions for the LLM.")