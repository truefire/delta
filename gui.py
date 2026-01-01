"""GUI implementation for Delta Tool."""
import math
import sys
import threading
import time
import queue
import json
import os
from pathlib import Path
from datetime import datetime

from imgui_bundle import imgui, hello_imgui, immapp, imgui_md

import core
from core import (
    process_request, get_file_stats, calculate_input_cost, estimate_tokens,
    open_diff_report, parse_plan, validate_files, init_git_repo, is_git_installed, is_git_repo,
    update_core_settings, CancelledError, backup_manager, get_available_backups,
    restore_git_backup, APP_DATA_DIR, config, AVAILABLE_MODELS, MAX_PROMPT_HISTORY
)
import application_state as state_module
from application_state import (
    state, init_app_state, create_session, get_active_session, close_session, unqueue_session,
    save_state, load_state, load_individual_session, delete_save, get_saves_list,
    load_prompt_history, save_prompt_history, change_working_directory, load_fileset,
    save_fileset, load_presets, save_presets, setup_logging, log_message,
    sync_settings_from_config, sync_config_from_settings,
    refresh_project_files, toggle_file_selection, toggle_folder_selection, to_relative,
    add_to_cwd_history, load_cwd_history, delete_save, SESSIONS_DIR
)
from widgets import ChatBubble, DiffViewer, render_file_tree
from styles import STYLE, apply_imgui_theme
from window_helpers import yank_window, flash_screens

def process_queue():
    """Process events from the GUI queue."""
    while not state.gui_queue.empty():
        try:
            event = state.gui_queue.get_nowait()
            handle_queue_event(event)
        except queue.Empty:
            break

def handle_queue_event(event: dict):
    """Handle a single queue event."""
    event_type = event.get("type")
    session_id = event.get("session_id")
    session = state.sessions.get(session_id) if session_id else get_active_session()

    if event_type == "text" and session:
        # Streaming text from LLM
        text = event.get("content", "")
        if session.current_bubble is None:
            # Auto-create bubble if missing (e.g. started new retry attempt)
            bubble = ChatBubble("assistant", len(session.bubbles))
            session.bubbles.append(bubble)
            session.current_bubble = bubble
            save_state("autosave")

        session.current_bubble.update(text)

    elif event_type == "flush" and session:
        if session.current_bubble:
            session.current_bubble.flush()

    elif event_type == "log_entry":        state.logs.append({
            "level": event.get("level"),
            "msg": event.get("message"),
            "time": event.get("timestamp")
        })

    elif event_type == "status":
        msg = event.get("message", "").rstrip()
        if msg:
            state.logs.append({
                "level": "INFO",
                "msg": msg,
                "time": time.time()
            })

    elif event_type == "start_response" and session:
        bubble = ChatBubble("assistant", len(session.bubbles))
        session.bubbles.append(bubble)
        session.current_bubble = bubble
        session.is_generating = True
        save_state("autosave")

    elif event_type == "end_response" and session:
        if session.current_bubble:
            session.current_bubble.flush()
        session.current_bubble = None
        session.is_generating = False

    elif event_type == "error":
        if session:
            session.failed = True
            bubble = ChatBubble("error", len(session.bubbles))
            bubble.update(event.get("message", "An error occurred"))
            session.bubbles.append(bubble)

    elif event_type == "diff_failure" and session:
        if session.bubbles and session.bubbles[-1].role == "assistant":
            session.bubbles.pop()
        
        session.current_bubble = None
        
        err_bubble = ChatBubble("error", len(session.bubbles))
        error_summary = event.get("error", "Unknown error")
        raw_content = event.get("raw_content", "")
        
        err_msg = f"Output parsing or application failed.\nError: {error_summary}\n\nThe assistant's response has been discarded from context for the retry."
        err_bubble.update(err_msg)
        err_bubble.set_error_details(error_summary, raw_content)
        
        session.bubbles.append(err_bubble)
    
    elif event_type == "validation_start" and session:
        if session.current_bubble:
            session.current_bubble.flush()
        vb = ChatBubble("system", len(session.bubbles))
        vb.update(f"Validating: `{event.get('command')}`...")
        session.bubbles.append(vb)
        session.current_bubble = vb

    elif event_type == "validation_success" and session:
        if session.current_bubble and session.current_bubble.role == "system":
            session.current_bubble.message.content = "Validation Passed."
            session.current_bubble._content_dirty = True
            session.current_bubble = None
        else:
            vb = ChatBubble("system", len(session.bubbles))
            vb.update("Validation Passed.")
            session.bubbles.append(vb)

    elif event_type == "validation_failure" and session:
        if session.current_bubble and session.current_bubble.role == "system":
            if session.bubbles and session.bubbles[-1] == session.current_bubble:
                session.bubbles.pop()
        
        if session.current_bubble:
            session.current_bubble.flush()
        session.current_bubble = None

        err_bubble = ChatBubble("error", len(session.bubbles))
        err_msg = f"Validation failed:\n{event.get('message', '')}"
        err_bubble.update(err_msg)
        session.bubbles.append(err_bubble)

    elif event_type == "success":
        if session:
            session.failed = False
            session.completed = True
            session.backup_id = event.get("backup_id")
            state.backup_list = None

    elif event_type == "failure":
        if session:
            session.failed = True

    elif event_type == "done":
        state.stats_dirty = True
        if session:
            session.is_generating = False
            session.is_queued = False

            if state.auto_review and session.backup_id:
                try:
                    open_diff_report(session.backup_id)
                except Exception as e:
                    log_message(f"Auto-review failed: {e}")
            
        if state.current_impl_sid == session_id:
            state.current_impl_sid = None

        save_state("autosave")

        if session and session.completed and session.is_planning:
            parse_and_distribute_plan(session)

        if config.focus_mode != "off":
            should_focus = False
            if config.focus_trigger == "task":
                should_focus = True
            elif config.focus_trigger == "queue":
                if not state.impl_queue or state.queue_blocked:
                    should_focus = True
            
            if should_focus:
                if config.focus_mode == "flash":
                    flash_screens()
                elif config.focus_mode == "yank":
                    yank_window()

        if state.impl_queue and state.current_impl_sid is None and not state.queue_blocked:
            next_sid = state.impl_queue.pop(0)
            start_generation(next_sid)

    elif event_type == "validation_failed":
        if state.block_on_fail:
            state.queue_blocked = True
        log_message(f"Validation failed: {event.get('message', '')}")

    elif event_type == "file_added":
        path = event.get("path")
        sid = event.get("session_id")
        if path and path.exists():
            rel_path = to_relative(path)
            state.selected_files.add(rel_path)
            state.file_checked[rel_path] = True
            state.stats_dirty = True
            save_fileset()
            
            session = state.sessions.get(sid) if sid else None
            if session:
                session.session_added_files.add(rel_path)
            
            refresh_project_files()

    elif event_type == "request_approval" and session:
        session.scroll_to_bottom = True
        if config.focus_mode == "flash":
            flash_screens()
        elif config.focus_mode == "yank":
            yank_window()

def ensure_user_bubble(session, text: str):
    """Ensure the latest message is a user bubble with the given text."""
    should_add = True
    if session.bubbles:
        last_b = session.bubbles[-1]
        if last_b.role == "user" and last_b.content == text:
            should_add = False

    if should_add:
        user_bubble = ChatBubble("user", len(session.bubbles))
        user_bubble.update(text)
        session.bubbles.append(user_bubble)

def start_generation(session_id: int):
    """Start generation for a session."""
    session = state.sessions.get(session_id)
    if not session:
        return

    if not session.is_ask_mode:
        state.current_impl_sid = session_id
    session.is_generating = True
    session.is_queued = False
    session.failed = False
    session.completed = False
    
    session.cancel_event.clear()

    ensure_user_bubble(session, session.last_prompt)
    checked_files = [f for f in state.selected_files if state.file_checked.get(f, True)]

    thread = threading.Thread(
        target=_generation_worker,
        args=(session_id, session.last_prompt, checked_files, session.cancel_event),
        daemon=True
    )
    thread.start()

def _generation_worker(session_id: int, prompt: str, files: list, cancel_event: threading.Event):
    """Worker thread for generation."""
    try:
        session = state.sessions.get(session_id)
        if not session:
            state.gui_queue.put({"type": "error", "session_id": session_id, "message": "Session not found"})
            return

        def output_func(msg: str, end: str = "\n", flush: bool = False):
            state.gui_queue.put({"type": "status", "message": msg.rstrip()})

        def stream_func(text: str, end: str = "", flush: bool = False):
            state.gui_queue.put({"type": "text", "session_id": session_id, "content": text})

        modes = ["replace_all", "ignore", "fail"]
        amb_mode = modes[state.ambiguous_mode_idx] if state.ambiguous_mode_idx < len(modes) else "replace_all"

        def on_diff_failure(error: str, raw_content: str):
            state.gui_queue.put({
                "type": "diff_failure", 
                "session_id": session_id, 
                "error": error, 
                "raw_content": raw_content
            })

        def on_validation_failure(msg: str):
            state.gui_queue.put({
                "type": "validation_failure",
                "session_id": session_id,
                "message": msg
            })

        def on_validation_start(cmd: str):
            state.gui_queue.put({
                "type": "validation_start",
                "session_id": session_id,
                "command": cmd
            })

        def on_validation_success():
            state.gui_queue.put({
                "type": "validation_success",
                "session_id": session_id
            })

        def on_file_added(path: Path):
            state.gui_queue.put({
                "type": "file_added",
                "path": path,
                "session_id": session_id
            })

        def confirmation_callback(diff_counts: dict, simulated_states: dict) -> bool:
            if state.require_approval:
                state.gui_queue.put({"type": "flush", "session_id": session_id})
                session.waiting_for_approval = True
                state.gui_queue.put({"type": "request_approval", "session_id": session_id})
                
                session.approval_event.clear()
                session.approval_event.wait()
                
                session.waiting_for_approval = False
                return session.approval_result
            return True

        file_strs = [str(f) for f in files]
        state.gui_queue.put({"type": "status", "message": f"Starting with {len(file_strs)} files..."})

        result = process_request(
            files=file_strs,
            prompt=prompt,
            history=session.history,
            output_func=output_func,
            stream_func=stream_func,
            cancel_event=cancel_event,
            validation_cmd=state.validation_cmd if (state.validation_cmd and state.validate_command_enabled) else "",
            validation_timeout=float(state.timeout) if state.timeout else 10.0,
            max_retries=int(state.max_tries),
            recursion_limit=int(state.recursions),
            ambiguous_mode=amb_mode,
            ask_mode=session.is_ask_mode,
            plan_mode=session.is_planning,
            allow_new_files=state.add_new_files,
            on_file_added=on_file_added,
            verify=state.verify_changes,
            validate_at_start=state.validate_at_start,
            on_diff_failure=on_diff_failure,
            on_validation_failure=on_validation_failure,
            on_validation_start=on_validation_start,
            on_validation_success=on_validation_success,
            confirmation_callback=confirmation_callback,
        )

        if result.get("success"):
            state.gui_queue.put({"type": "success", "session_id": session_id, "backup_id": result.get("backup_id")})
            state.gui_queue.put({"type": "status", "message": f"Completed: {result.get('message', 'Success')}"})
        else:
            error_msg = result.get("message", "Unknown error")
            state.gui_queue.put({"type": "failure", "session_id": session_id, "message": error_msg})
            state.gui_queue.put({"type": "status", "message": f"Failed: {error_msg}"})

    except CancelledError:
        state.gui_queue.put({"type": "status", "message": "Generation cancelled"})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        state.gui_queue.put({"type": "error", "session_id": session_id, "message": f"{e}\n{tb}"})
        state.gui_queue.put({"type": "status", "message": f"Error: {e}"})
    finally:
        state.gui_queue.put({"type": "end_response", "session_id": session_id})
        state.gui_queue.put({"type": "done", "session_id": session_id})

def _submit_common(session, prompt: str, is_planning: bool = False, ask_mode: bool = False):
    """Common submission logic for prompt and plan."""
    if prompt and (not state.prompt_history or state.prompt_history[-1] != prompt):
        state.prompt_history.append(prompt)
        if MAX_PROMPT_HISTORY > 0 and len(state.prompt_history) > MAX_PROMPT_HISTORY:
            state.prompt_history.pop(0)
        save_prompt_history()

    session.last_prompt = prompt
    session.input_text = ""
    session.is_ask_mode = ask_mode
    session.is_planning = is_planning
    state.prompt_history_idx = -1

    ensure_user_bubble(session, session.last_prompt)

    if session.is_ask_mode:
        start_generation(session.id)
    elif state.current_impl_sid is not None or state.queue_blocked or state.impl_queue:
        session.is_queued = True
        state.impl_queue.append(session.id)
        if state.current_impl_sid is None and not state.queue_blocked:
            next_sid = state.impl_queue.pop(0)
            start_generation(next_sid)
    else:
        start_generation(session.id)

def check_missing_files(session, prompt: str, is_planning: bool, ask_mode: bool) -> bool:
    checked_files = [f for f in state.selected_files if state.file_checked.get(f, True)]
    missing = [f for f in checked_files if not f.exists()]
    
    if missing:
        state.missing_files_list = missing
        state.show_missing_files_popup = True
        state.pending_prompt = prompt
        state.pending_is_planning = is_planning
        state.pending_ask_mode = ask_mode
        return False
    return True

def _resume_submission_after_missing_check():
    session = get_active_session()
    if not session: return
    
    prompt = state.pending_prompt
    is_planning = state.pending_is_planning
    ask_mode = state.pending_ask_mode
    
    if state.use_git_backup and state.backup_enabled and not ask_mode:
        if not is_git_installed():
            state.show_no_git_popup = True
            return
        if not is_git_repo():
            state.show_no_repo_popup = True
            return

    _submit_common(session, prompt, is_planning=is_planning, ask_mode=ask_mode)

def submit_prompt(ask_mode: bool = False):
    """Submit the current prompt."""
    session = get_active_session()
    if not session:
        return

    prompt = session.input_text.strip()
    if not prompt:
        return

    if not check_missing_files(session, prompt, is_planning=False, ask_mode=ask_mode):
        return

    if state.use_git_backup and state.backup_enabled and not ask_mode:
        if not is_git_installed():
            state.pending_prompt = prompt
            state.pending_is_planning = False
            state.show_no_git_popup = True
            return
        
        if not is_git_repo():
            state.pending_prompt = prompt
            state.pending_is_planning = False
            state.show_no_repo_popup = True
            return

    _submit_common(session, prompt, is_planning=False, ask_mode=ask_mode)

def submit_plan():
    """Submit a planning request."""
    session = get_active_session()
    if not session:
        return

    prompt = session.input_text.strip()
    if not prompt:
        return

    if not check_missing_files(session, prompt, is_planning=True, ask_mode=False):
        return

    if state.use_git_backup and state.backup_enabled:
        if not is_git_installed():
            state.pending_prompt = prompt
            state.pending_is_planning = True
            state.show_no_git_popup = True
            return
        
        if not is_git_repo():
            state.pending_prompt = prompt
            state.pending_is_planning = True
            state.show_no_repo_popup = True
            return

    _submit_common(session, prompt, is_planning=True, ask_mode=False)

def parse_and_distribute_plan(session):
    """Parse a completed plan and queue sub-tasks."""
    if not session.bubbles:
        return
    
    last_bubble = session.bubbles[-1]
    if last_bubble.role != "assistant":
        return
        
    text = last_bubble.content
    tasks = parse_plan(text)
    
    if not tasks:
        log_message("No valid plan blocks found in response.")
        return
        
    log_message(f"Plan parsed: {len(tasks)} tasks found. Queuing...")
    
    if session.group_id is None:
        session.group_id = state.next_group_id
        state.next_group_id += 1
    
    group_id = session.group_id
    
    new_sids = []
    for i, (title, prompt) in enumerate(tasks):
        new_sess = create_session()
        new_sess.group_id = group_id
        
        plan_overview = "Context: This is task {} of {} in the implementation plan.\n\nPlan Overview:\n".format(i+1, len(tasks))
        
        for j, (t_title, _) in enumerate(tasks):
            mark = "[x]" if j < i else "[ ]"
            indicator = " <--- CURRENT TASK" if j == i else ""
            plan_overview += f"{j+1}. {mark} {t_title}{indicator}\n"
            
        full_prompt = f"{plan_overview}\n\nTask: {title}\n\nInstructions: {prompt}"
        
        new_sess.input_text = ""
        new_sess.last_prompt = full_prompt
        new_sess.is_queued = True
        
        ensure_user_bubble(new_sess, full_prompt)
        new_sids.append(new_sess.id)

    state.impl_queue[0:0] = new_sids
    
    if state.impl_queue and state.current_impl_sid is None and not state.queue_blocked:
        next_sid = state.impl_queue.pop(0)
        start_generation(next_sid)

def cancel_generation(session_id: int | None = None):
    """Cancel current generation."""
    session = None
    if session_id is None:
        session = get_active_session()
    else:
        session = state.sessions.get(session_id)
        
    if session:
        session.cancel_event.set()
        if session.waiting_for_approval:
            # Unblock waiting thread so it can check cancel_event
            # We set result to False (Reject) to be safe
            session.approval_result = False
            session.approval_event.set()

def cancel_all_tasks():
    """Cancel all queued and active tasks."""
    for sid in list(state.impl_queue):
        unqueue_session(sid)
    
    state.queue_blocked = False
    
    for session in state.sessions.values():
        if session.is_generating:
            session.cancel_event.set()

def toggle_theme():
    """Toggle between light and dark theme."""
    new_theme = "light" if STYLE.dark else "dark"
    config.set_theme(new_theme)
    STYLE.load(new_theme)
    apply_imgui_theme(STYLE.dark)

def open_api_settings():
    """Open the API settings popup."""
    state.show_api_settings_popup = True
    state.api_settings_inputs = {
        "api_key": core.API_KEY,
        "base_url": core.API_BASE_URL,
        "models": json.dumps(core.AVAILABLE_MODELS, indent=2),
        "git_branch": config.git_backup_branch
    }
    state.api_settings_error = ""

def render_tooltip(text: str):
    """Render a tooltip."""
    if imgui.is_item_hovered():
        item_max = imgui.get_item_rect_max()
        imgui.set_next_window_pos(item_max, imgui.Cond_.always, imgui.ImVec2(1.0, 0.0))
        if imgui.begin_tooltip():
            imgui.push_text_wrap_pos(min(imgui.get_font_size() * 30, 400.0))
            imgui.text_wrapped(text)
            imgui.pop_text_wrap_pos()
            imgui.end_tooltip()

def render_api_settings_popup():
    """Render the API settings popup."""
    if state.show_api_settings_popup:
        imgui.open_popup("API Settings")
        
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
                
                state.show_api_settings_popup = False
                imgui.close_current_popup()
                log_message("API settings saved.")
                
            except Exception as e:
                state.api_settings_error = f"Error: {e}"
                
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(120, 0)):
            state.show_api_settings_popup = False
            imgui.close_current_popup()
            
        imgui.end_popup()

def render_missing_files_popup():
    if state.show_missing_files_popup:
        imgui.open_popup("Missing Files")
        state.show_missing_files_popup = False

    if imgui.begin_popup_modal("Missing Files", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text("The following selected files were not found:")
        imgui.spacing()
        
        imgui.begin_child("missing_list", imgui.ImVec2(400, 100), child_flags=imgui.ChildFlags_.borders)
        for f in state.missing_files_list:
            imgui.text(str(f))
        imgui.end_child()
            
        imgui.spacing()
        imgui.text("Remove them from context and continue?")
        imgui.separator()
        
        if imgui.button("Remove & Continue", imgui.ImVec2(160, 0)):
            for f in state.missing_files_list:
                state.selected_files.discard(f)
                if f in state.file_checked:
                    del state.file_checked[f]
            state.stats_dirty = True
            save_fileset()
            
            _resume_submission_after_missing_check()
            imgui.close_current_popup()
            
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(100, 0)):
            state.show_missing_files_popup = False
            imgui.close_current_popup()
            
        imgui.end_popup()

def render_git_check_popups():
    render_missing_files_popup()
    render_api_settings_popup()
    
    session = get_active_session()
    if not session: 
        return

    if state.show_no_git_popup:
        imgui.open_popup("Git Not Found")
        state.show_no_git_popup = False

    if imgui.begin_popup_modal("Git Not Found", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text("Git executable not found in PATH.")
        imgui.text("Git backup is enabled, but git is not available.")
        imgui.separator()
        
        if imgui.button("Use Normal Backups (File Copy)", imgui.ImVec2(-1, 0)):
            state.use_git_backup = False
            config.set_use_git_backup(False)
            _submit_common(session, state.pending_prompt, is_planning=state.pending_is_planning, ask_mode=False)
            imgui.close_current_popup()
            
        if imgui.button("Cancel Request", imgui.ImVec2(-1, 0)):
            imgui.close_current_popup()
            
        imgui.end_popup()

    if state.show_no_repo_popup:
        imgui.open_popup("Not a Git Repository")
        state.show_no_repo_popup = False

    if imgui.begin_popup_modal("Not a Git Repository", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text(f"Current directory is not a git repository:\n{Path.cwd()}")
        imgui.text("Git backup requirement cannot be met.")
        imgui.separator()

        if imgui.button("Initialize Repo (git init)", imgui.ImVec2(-1, 0)):
            ok, msg = init_git_repo()
            if ok:
                log_message("Initialized git repository.")
                _submit_common(session, state.pending_prompt, is_planning=state.pending_is_planning, ask_mode=False)
            else:
                log_message(f"Failed to init git: {msg}")
            imgui.close_current_popup()

        if imgui.button("Use Normal Backups Instead", imgui.ImVec2(-1, 0)):
            state.use_git_backup = False
            config.set_use_git_backup(False)
            _submit_common(session, state.pending_prompt, is_planning=state.pending_is_planning, ask_mode=False)
            imgui.close_current_popup()

        if imgui.button("Cancel Request", imgui.ImVec2(-1, 0)):
            imgui.close_current_popup()

        imgui.end_popup()

def _perform_exit(action: str):
    """Execute the exit actions."""
    if action == "restart":
        if state and config.persist_session:
            save_state("autosave")
        state.restart_requested = True
    
    hello_imgui.get_runner_params().app_shall_exit = True

def try_exit_app(action: str = "exit"):
    """Attempt to exit, showing confirmation if busy."""
    is_busy = any(s.is_generating or s.is_queued for s in state.sessions.values()) or bool(state.impl_queue)
    if is_busy:
        state.pending_exit_action = action
        state.show_exit_confirmation_popup = True
    else:
        _perform_exit(action)

def render_exit_confirmation_popup():
    if state.show_exit_confirmation_popup:
        imgui.open_popup("Confirm Exit")

    if imgui.begin_popup_modal("Confirm Exit", None, imgui.WindowFlags_.always_auto_resize)[0]:
        action_name = "Restart" if state.pending_exit_action == "restart" else "Exit"
        imgui.text("Tasks are currently running or queued.")
        imgui.text(f"Going to {action_name.lower()} will cancel all pending operations.")
        
        imgui.separator()
        imgui.spacing()

        if imgui.button(f"Cancel Tasks & {action_name}", imgui.ImVec2(180, 0)):
            cancel_all_tasks()
            _perform_exit(state.pending_exit_action)
            
            state.show_exit_confirmation_popup = False
            imgui.close_current_popup()

        imgui.same_line()
        
        if imgui.button("Keep Running", imgui.ImVec2(120, 0)):
            state.show_exit_confirmation_popup = False
            state.pending_exit_action = None
            imgui.close_current_popup()

        imgui.end_popup()

def render_close_tab_popup():
    if state.show_close_tab_popup:
        imgui.open_popup("Close Busy Tab?")

    if imgui.begin_popup_modal("Close Busy Tab?", None, imgui.WindowFlags_.always_auto_resize)[0]:
        sid = state.session_to_close_id
        session = state.sessions.get(sid)

        if not session:
            state.show_close_tab_popup = False
            imgui.close_current_popup()
            imgui.end_popup()
            return

        imgui.text(f"Tab #{sid} is active.")
        if session.is_generating:
            imgui.text_colored(STYLE.get_imvec4("queued"), "Status: Generating")
        elif session.is_queued:
            imgui.text_colored(STYLE.get_imvec4("queued"), "Status: Queued")
        
        imgui.spacing()
        imgui.text("Closing it will cancel this task.")
        if session.is_generating:
            imgui.text_colored(STYLE.get_imvec4("fg_dim"), "(Stops current generation)")
            
        imgui.separator()

        if imgui.button("Cancel Task & Close", imgui.ImVec2(160, 0)):
            if session.is_queued:
                unqueue_session(sid)
            if session.is_generating:
                cancel_generation(sid)
            
            close_session(sid)
            state.show_close_tab_popup = False
            state.session_to_close_id = None
            imgui.close_current_popup()

        imgui.same_line()

        if imgui.button("Keep Open", imgui.ImVec2(100, 0)):
            state.show_close_tab_popup = False
            state.session_to_close_id = None
            imgui.close_current_popup()

        imgui.end_popup()

def render_sessions_window():
    if not state.show_sessions_window:
        return

    imgui.set_next_window_size(imgui.ImVec2(800, 500), imgui.Cond_.first_use_ever)
    opened, state.show_sessions_window = imgui.begin("Sessions Manager", state.show_sessions_window)

    if opened:
        if imgui.begin_table("sessions_layout", 2, imgui.TableFlags_.resizable | imgui.TableFlags_.borders_inner_v):
            imgui.table_setup_column("List", imgui.TableColumnFlags_.width_stretch, 0.4)
            imgui.table_setup_column("Details", imgui.TableColumnFlags_.width_stretch, 0.6)
            
            imgui.table_next_row()
            imgui.table_next_column()
            
            imgui.text("Saved States")
            imgui.separator()
            imgui.spacing()

            imgui.set_next_item_width(120)
            _, state.new_save_name = imgui.input_text("##newsave", state.new_save_name)
            render_tooltip("Name for the manual save file.")
            imgui.same_line()
            if imgui.button("Save Current"):
                name = state.new_save_name.strip()
                if not name:
                    name = f"Save {datetime.now().strftime('%Y-%m-%d %H%M')}"
                save_state(name)
                state.new_save_name = ""
            if imgui.is_item_hovered():
                imgui.set_tooltip("Save the complete state of all open tabs to a JSON file.")
            
            saves = get_saves_list()
            
            imgui.begin_child("saves_list", imgui.ImVec2(0, 0), child_flags=imgui.ChildFlags_.borders)
            for save in saves:
                name = save["name"]
                is_selected = (state.selected_save_name == name)
                
                ts = datetime.fromtimestamp(save["mtime"]).strftime('%Y-%m-%d %H:%M')
                label = f"{name}  ({ts})"
                
                if imgui.selectable(label, is_selected)[0]:
                    state.selected_save_name = name
            imgui.end_child()
            
            imgui.table_next_column()
            
            imgui.text("Details")
            imgui.separator()
            
            if state.selected_save_name:
                name = state.selected_save_name
                filename = SESSIONS_DIR / f"{name}.json"
                
                if filename.exists():
                    try:
                        with open(filename, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        
                        sess_list = data.get("sessions", [])
                        imgui.text(f"Session Count: {len(sess_list)}")
                        
                        if imgui.button("Load Full State", imgui.ImVec2(120, 0)):
                            load_state(name)
                            state.show_sessions_window = False
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Restore this save, replacing all current open tabs.")
                            
                        imgui.same_line()
                        
                        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_cncl"))
                        if imgui.button("Delete Save", imgui.ImVec2(100, 0)):
                            delete_save(name)
                        imgui.pop_style_color()
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("Permanently delete this save file.")
                        
                        imgui.separator()
                        imgui.text("Tabs in this save:")
                        
                        imgui.begin_child("sess_preview", imgui.ImVec2(0, -5))
                        for i, s_data in enumerate(sess_list):
                            hist = s_data.get("history", [])
                            prompt = s_data.get("last_prompt", "")
                            if not prompt and len(hist) > 0:
                                for m in reversed(hist):
                                    if m.get("role") == "user":
                                        c = m.get("content", "")
                                        if isinstance(c, str): prompt = c
                                        break
                            
                            display_prompt = (prompt[:50] + "...") if len(prompt) > 50 else (prompt or "(Empty)")
                            display_prompt = display_prompt.replace("\n", " ")
                            
                            if imgui.small_button(f"Load##{i}"):
                                load_individual_session(name, i)
                            if imgui.is_item_hovered():
                                imgui.set_tooltip("Import just this tab into your current workspace.")
                                
                            imgui.same_line()
                            imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"#{i+1}:")
                            imgui.same_line()
                            imgui.text(display_prompt)
                            
                        imgui.end_child()
                        
                    except Exception as e:
                        imgui.text_colored(STYLE.get_imvec4("btn_cncl"), f"Error reading save: {e}")
                else:
                    imgui.text("File not found.")
            else:
                imgui.text_colored(STYLE.get_imvec4("fg_dim"), "Select a save to view details.")
                
            imgui.end_table()

    imgui.end()

def render_backup_history_window():
    if not state.show_backup_history:
        return

    imgui.set_next_window_size(imgui.ImVec2(500, 600), imgui.Cond_.first_use_ever)
    opened, state.show_backup_history = imgui.begin("Backup History", state.show_backup_history)

    if opened:
        if state.backup_list is None:
            state.backup_list = get_available_backups()

        if imgui.button("Refresh"):
            state.backup_list = get_available_backups()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Reload the list of backups.")

        imgui.same_line()
        if imgui.button("Clear Files"):
            imgui.open_popup("ConfirmClearAll")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Delete all file-based backup copies (frees up disk space).")

        if imgui.begin_popup_modal("ConfirmClearAll", None, imgui.WindowFlags_.always_auto_resize)[0]:
            imgui.text("Delete ALL file-based backups for this project?")
            imgui.text_colored(STYLE.get_imvec4("fg_dim"), "(Git history/backups are preserved)")
            imgui.separator()
            if imgui.button("Yes, Delete Files", imgui.ImVec2(120, 0)):
                backup_manager.clear_all_backups()
                state.backup_list = None
                log_message("All file backups deleted")
                imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                imgui.close_current_popup()
            imgui.end_popup()

        imgui.separator()

        backups = state.backup_list or []
        if not backups:
            imgui.text_colored(STYLE.get_imvec4("fg_dim"), "No backups available.")
        else:
            imgui.begin_child("backup_list")
            for i, b in enumerate(backups):
                sid = b["session_id"]
                ts = b["timestamp"]
                files = b["files"]
                source = b.get("source", "file")
                msg = b.get("message", "")

                imgui.push_id(f"bak_{i}")
                
                if source == "git":
                    imgui.push_style_color(imgui.Col_.text, STYLE.get_imvec4("badge_git"))
                    imgui.text("[GIT] ")
                else:
                    imgui.push_style_color(imgui.Col_.text, STYLE.get_imvec4("badge_file"))
                    imgui.text("[FILE]")
                imgui.pop_style_color()
                
                imgui.same_line()
                
                label = f"{ts}"
                if msg and source == "git":
                    label += f" - {msg[:30]}"
                
                if imgui.collapsing_header(f"{label}##{i}"):
                    imgui.indent()
                    
                    if imgui.button("Restore"):
                        imgui.open_popup("ConfirmRestore")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Rollback project files to before these changes.")

                    if imgui.begin_popup_modal("ConfirmRestore", None, imgui.WindowFlags_.always_auto_resize)[0]:
                        imgui.text(f"Restore state from {ts}?")
                        if source == "file":
                            imgui.text_colored(STYLE.get_imvec4("btn_cncl"), "This will undo this session AND all newer sessions.")
                        else:
                            imgui.text_colored(STYLE.get_imvec4("btn_cncl"), "This will reset workspace to this git commit.")
                        
                        imgui.separator()
                        
                        if imgui.button("Yes, Restore", imgui.ImVec2(120, 0)):
                            try:
                                if source == "git":
                                    res = restore_git_backup(sid)
                                else:
                                    res = backup_manager.rollback_to_session(sid)
                                    
                                log_message(f"Rolled back to {ts}")
                                for k, v in res.items():
                                    log_message(f"  {k}: {v}")
                                refresh_project_files()
                                state.backup_list = None
                            except Exception as e:
                                log_message(f"Error restoring: {e}")
                            imgui.close_current_popup()
                            
                        imgui.same_line()
                        if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                            imgui.close_current_popup()
                        imgui.end_popup()

                    imgui.same_line()
                    if imgui.button("Diff vs Current"):
                        try:
                            if source == "git":
                                open_diff_report(sid, diff_against_disk=True)
                            else:
                                sids_to_diff = [x["session_id"] for x in backups[:i+1] if x.get("source", "file") == "file"]
                                open_diff_report(sids_to_diff, diff_against_disk=True)
                        except Exception as e:
                            log_message(f"Error opening diff: {e}")
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("View changes between this backup and the current file system.")

                    if i + 1 < len(backups):
                        imgui.same_line()
                        if imgui.button("Diff vs Prev"):
                            try:
                                if source == "file":
                                    compare_to = backups[i-1]["session_id"] if i > 0 else None
                                    open_diff_report(sid, compare_session_id=compare_to)
                                else:
                                    open_diff_report(sid)
                            except Exception as e:
                                log_message(f"Error opening diff: {e}")
                        if imgui.is_item_hovered():
                            imgui.set_tooltip("View changes made in this specific session (vs previous state).")

                    if source == "file":
                        imgui.same_line()
                        if imgui.button("Delete"):
                            backup_manager.delete_session(sid)
                            state.backup_list = None
                            log_message(f"Deleted backup {ts}")

                    imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"Files: {len(files)}")
                    for f in files:
                        imgui.bullet_text(os.path.basename(f))
                    
                    imgui.unindent()
                imgui.pop_id()
            imgui.end_child()

    imgui.end()

def render_settings_panel():
    models = list(AVAILABLE_MODELS.keys())
    focus_modes = ["Off", "Flash", "Yank"]
    ambiguous_modes = ["Replace all", "Ignore", "Fail"]

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

    imgui.separator()

    imgui.text("Model:")
    imgui.same_line(80)
    imgui.set_next_item_width(-1)
    changed, state.model_idx = imgui.combo("##model", state.model_idx, models)
    render_tooltip("Select the LLM model to use.")
    if changed:
        sync_config_from_settings()

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
    changed, state.add_new_files = imgui.checkbox("Add new files", state.add_new_files)
    render_tooltip("Add new files generated by the LLM to the context.")
    if changed:
        sync_config_from_settings()

    imgui.same_line()
    changed, state.persist_session = imgui.checkbox("Persist Session", state.persist_session)
    render_tooltip("Automatically load previous session state on boot.")
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

    imgui.separator()

    # Cached check for git capability (throttled)
    if state.frame_count % 60 == 0:
        state.can_use_git = is_git_installed() and is_git_repo()

    btn_label = "Review Uncommitted" if state.can_use_git else "Review Latest Changes"

    if imgui.button(btn_label, imgui.ImVec2(-1, 0)):
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

    if imgui.button("Sessions", imgui.ImVec2(-1, 0)):
        state.show_sessions_window = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("Open the Session Manager window.")

    if imgui.button("System Prompt", imgui.ImVec2(-1, 0)):
        state.show_system_prompt_popup = True
        state.temp_system_prompt = config.extra_system_prompt
    if imgui.is_item_hovered():
        imgui.set_tooltip("Configure global custom instructions for the LLM.")

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

def _render_group_row(name: str):
    if name not in state.presets:
        return

    group_data = state.presets[name]
    group_files = group_data.get("files", [])
    
    imgui.push_id(name)
    
    if imgui.small_button("Set"):
        for f in state.selected_files:
            state.file_checked[f] = False
        
        for f in group_files:
            path = to_relative(Path(f))
            if path.exists():
                state.selected_files.add(path)
                state.file_checked[path] = True
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Replace selection with this group")
    
    imgui.same_line()
    if imgui.small_button("+"):
        for f in group_files:
            path = to_relative(Path(f))
            if path.exists():
                state.selected_files.add(path)
                state.file_checked[path] = True
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Add this group to selection")
        
    imgui.same_line()
    if imgui.small_button("-"):
        for f in group_files:
            path = to_relative(Path(f))
            if path in state.selected_files:
                state.file_checked[path] = False
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Remove this group from selection")
    
    imgui.same_line()
    imgui.text(f"{name} [{len(group_files)} files]")

    if imgui.begin_popup_context_item("group_ctx"):
        if imgui.selectable("Delete Group", False)[0]:
            del state.presets[name]
            save_presets()
        imgui.end_popup()
    imgui.pop_id()

def _render_group_manage_row(name: str):
    if name not in state.presets:
        return

    group_data = state.presets[name]
    group_files = group_data.get("files", [])
    
    imgui.push_id(f"manage_{name}")
    
    if imgui.small_button("Update"):
        files_to_save = [str(f) for f in state.selected_files if state.file_checked.get(f, True)]
        state.presets[name] = {"files": files_to_save}
        save_presets()
        log_message(f"Updated group '{name}' with {len(files_to_save)} files")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Overwrite this group with current selection")
    
    imgui.same_line()
    if imgui.small_button("Delete"):
        del state.presets[name]
        save_presets()
        log_message(f"Deleted group '{name}'")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Delete this group")
    
    imgui.same_line()
    imgui.text(f"{name} [{len(group_files)} files]")
    
    imgui.pop_id()

def render_files_panel():
    imgui.text("Groups")
    imgui.separator()

    if state.presets:
        for name in list(state.presets.keys()):
            _render_group_row(name)
    else:
        imgui.text_colored(STYLE.get_imvec4("fg_dim"), "No groups saved")

    imgui.dummy(imgui.ImVec2(0, 10))
    imgui.separator()

    # Rolling file existence check
    if state.cached_sorted_files:
        files_to_check = 5
        total_files = len(state.cached_sorted_files)
        
        if state.rolling_check_idx >= total_files:
            state.rolling_check_idx = 0
            
        start_idx = state.rolling_check_idx
        end_idx = min(start_idx + files_to_check, total_files)
        
        recalc_needed = False
        for i in range(start_idx, end_idx):
            f = state.cached_sorted_files[i]
            exists = f.exists()
            if state.file_exists_cache.get(f) != exists:
                state.file_exists_cache[f] = exists
                recalc_needed = True
        
        state.rolling_check_idx = end_idx if end_idx < total_files else 0
        
        if recalc_needed:
            state.stats_dirty = True

    # Optimizing stats calculation to run only when needed
    if state.stats_dirty:
        state.cached_sorted_files = sorted(state.selected_files, key=lambda p: str(p).lower())
        
        # Immediate check all on dirty (e.g. startup/refresh)
        for f in state.cached_sorted_files:
             if f not in state.file_exists_cache:
                 state.file_exists_cache[f] = f.exists()
        
        # Calculate totals
        checked = [f for f in state.cached_sorted_files 
                   if state.file_checked.get(f, True) and state.file_exists_cache.get(f, True)]
                   
        state.cached_total_lines = sum(get_file_stats(f)[0] for f in checked)
        state.cached_total_tokens = sum(get_file_stats(f)[1] for f in checked)
        
        model_list = list(AVAILABLE_MODELS.keys())
        model_name = model_list[state.model_idx] if state.model_idx < len(model_list) else ""
        _, state.cached_cost_str = calculate_input_cost(state.cached_total_tokens, model_name)
        
        state.stats_dirty = False
    
    sorted_files = state.cached_sorted_files
    
    checked_count = sum(1 for f in sorted_files if state.file_checked.get(f, True) and state.file_exists_cache.get(f, True))

    header_text = f"Selected Context ({checked_count}/{len(state.selected_files)} files)"
    imgui.text(header_text)
    if state.is_scanning:
        imgui.same_line()
        imgui.text_colored(STYLE.get_imvec4("queued"), "(Scanning...)")
    imgui.separator()

    imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"Lines: {state.cached_total_lines} | Tokens: ~{state.cached_total_tokens} | Cost: {state.cached_cost_str}")
    imgui.spacing()

    list_height = min(240, 22 * len(state.selected_files) + 10)
    imgui.begin_child("selected_files_list", imgui.ImVec2(0, list_height), child_flags=imgui.ChildFlags_.borders)

    table_flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit
    if imgui.begin_table("selected_files_table", 3, table_flags):
        imgui.table_setup_column("##chk", imgui.TableColumnFlags_.width_fixed, 24)
        imgui.table_setup_column("Tokens", imgui.TableColumnFlags_.width_fixed, 50)
        imgui.table_setup_column("File", imgui.TableColumnFlags_.width_stretch)

        clipper = imgui.ListClipper()
        clipper.begin(len(sorted_files))
        while clipper.step():
            for i in range(clipper.display_start, clipper.display_end):
                f = sorted_files[i]
                exists = state.file_exists_cache.get(f, False)
                if f not in state.file_checked:
                    state.file_checked[f] = True

                _, tokens, _ = get_file_stats(f)
                file_display = str(f.relative_to(Path.cwd())) if f.is_absolute() else str(f)

                imgui.table_next_row()

                imgui.table_next_column()
                imgui.push_id(str(f))
                is_checked = state.file_checked[f]
                changed, new_val = imgui.checkbox("##chk", is_checked)
                if changed:
                    state.file_checked[f] = new_val
                    state.stats_dirty = True
                    save_fileset()

                imgui.table_next_column()
                if exists:
                    imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"{tokens:>5}")
                else:
                    imgui.text_colored(STYLE.get_imvec4("btn_cncl"), " -- ")

                imgui.table_next_column()
                if not exists:
                    imgui.text_colored(STYLE.get_imvec4("btn_cncl"), f"{file_display} (Missing)")
                elif is_checked:
                    imgui.text(file_display)
                else:
                    imgui.text_colored(STYLE.get_imvec4("fg_dim"), file_display)

                if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.middle):
                    state.selected_files.discard(f)
                    state.file_checked.pop(f, None)
                    state.stats_dirty = True
                    save_fileset()

                if imgui.is_item_hovered():
                    imgui.set_tooltip("Middle-click to remove")

                imgui.pop_id()

        imgui.end_table()

    imgui.end_child()

    if imgui.button("Check All"):
        for f in state.selected_files:
            state.file_checked[f] = True
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Select all files in the list.")

    imgui.same_line()
    if imgui.button("Uncheck All"):
        for f in state.selected_files:
            state.file_checked[f] = False
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Deselect all files.")

    imgui.same_line()
    if imgui.button("Remove Unchecked"):
        unchecked = [f for f in state.selected_files if not state.file_checked.get(f, True)]
        for f in unchecked:
            state.selected_files.discard(f)
            state.file_checked.pop(f, None)
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Remove deselected files from this list.")

    imgui.separator()

    if imgui.button("Manage Context", imgui.ImVec2(-1, 0)):
        state.show_context_manager = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("Open the advanced file tree browser.")

    if state.show_create_group_popup:
        imgui.open_popup("Create Group")

    if imgui.begin_popup_modal("Create Group", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text("Enter group name:")
        imgui.set_next_item_width(200)
        _, state.new_group_name = imgui.input_text("##group_name", state.new_group_name)

        imgui.spacing()
        if imgui.button("Create", imgui.ImVec2(100, 0)):
            name = state.new_group_name.strip()
            if name:
                files_to_save = [str(f) for f in state.selected_files if state.file_checked.get(f, True)]
                state.presets[name] = {"files": files_to_save}
                save_presets()
                log_message(f"Created group '{name}' with {len(files_to_save)} files")
                state.show_create_group_popup = False
                imgui.close_current_popup()
            else:
                log_message("Group name cannot be empty")

        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(100, 0)):
            state.show_create_group_popup = False
            imgui.close_current_popup()

        imgui.end_popup()

def _draw_tab_icon(draw_list, cx: float, cy: float, status: str, badge: str = None):
    """Draw a colored status icon at the given center position."""
    colors = {
        "running": imgui.get_color_u32(imgui.ImVec4(0.13, 0.59, 0.95, 1.0)),      
        "running_ask": imgui.get_color_u32(imgui.ImVec4(0.61, 0.15, 0.69, 1.0)),  
        "running_plan": imgui.get_color_u32(imgui.ImVec4(0.00, 0.73, 0.83, 1.0)), 
        "queued": imgui.get_color_u32(imgui.ImVec4(1.0, 0.60, 0.0, 1.0)),         
        "done": imgui.get_color_u32(imgui.ImVec4(0.30, 0.69, 0.31, 1.0)),         
        "failed": imgui.get_color_u32(imgui.ImVec4(0.96, 0.26, 0.21, 1.0)),       
        "inactive": imgui.get_color_u32(imgui.ImVec4(0.62, 0.62, 0.62, 1.0)),     
        "debug": imgui.get_color_u32(imgui.ImVec4(0.96, 0.26, 0.21, 1.0)),        
    }
    colors["queued_plan"] = colors["queued"]
    colors["done_plan"] = colors["done"]
    colors["done_ask"] = colors["done"]
    color = colors.get(status, colors["inactive"])

    if status.startswith("running"):
        cy += math.sin(time.time() * 10.0) * 1.5

    if status == "running":
        draw_list.add_triangle_filled(
            imgui.ImVec2(cx - 4, cy - 5),
            imgui.ImVec2(cx - 4, cy + 5),
            imgui.ImVec2(cx + 5, cy),
            color
        )
    elif status in ("running_ask", "done_ask"):
        draw_list.add_circle(imgui.ImVec2(cx - 1, cy - 1), 4, color, 12, 2.0)
        draw_list.add_line(imgui.ImVec2(cx + 2, cy + 2), imgui.ImVec2(cx + 5, cy + 5), color, 2.0)
    elif status in ("running_plan", "done_plan", "queued_plan"):
        draw_list.add_line(imgui.ImVec2(cx - 3, cy - 3), imgui.ImVec2(cx + 3, cy - 3), color, 1.5)
        draw_list.add_line(imgui.ImVec2(cx - 3, cy),     imgui.ImVec2(cx + 3, cy),     color, 1.5)
        draw_list.add_line(imgui.ImVec2(cx - 3, cy + 3), imgui.ImVec2(cx + 3, cy + 3), color, 1.5)
    elif status == "queued":
        draw_list.add_circle(imgui.ImVec2(cx, cy), 5, color, 12, 1.5)
        draw_list.add_line(imgui.ImVec2(cx, cy), imgui.ImVec2(cx, cy - 3), color, 1.5)
        draw_list.add_line(imgui.ImVec2(cx, cy), imgui.ImVec2(cx + 2, cy), color, 1.5)
    elif status == "done":
        draw_list.add_polyline(
            [imgui.ImVec2(cx - 4, cy), imgui.ImVec2(cx - 1, cy + 3), imgui.ImVec2(cx + 5, cy - 4)],
            color, imgui.ImDrawFlags_.none, 2.5
        )
    elif status == "failed":
        draw_list.add_line(imgui.ImVec2(cx - 4, cy - 4), imgui.ImVec2(cx + 4, cy + 4), color, 2.5)
        draw_list.add_line(imgui.ImVec2(cx - 4, cy + 4), imgui.ImVec2(cx + 4, cy - 4), color, 2.5)
    elif status == "debug":
        draw_list.add_circle_filled(imgui.ImVec2(cx, cy), 3, color)
        draw_list.add_line(imgui.ImVec2(cx - 2, cy - 2), imgui.ImVec2(cx - 5, cy - 4), color, 1.5)
        draw_list.add_line(imgui.ImVec2(cx + 2, cy - 2), imgui.ImVec2(cx + 5, cy - 4), color, 1.5)
        draw_list.add_line(imgui.ImVec2(cx - 3, cy), imgui.ImVec2(cx - 6, cy), color, 1.5)
        draw_list.add_line(imgui.ImVec2(cx + 3, cy), imgui.ImVec2(cx + 6, cy), color, 1.5)
        draw_list.add_line(imgui.ImVec2(cx - 2, cy + 2), imgui.ImVec2(cx - 5, cy + 4), color, 1.5)
        draw_list.add_line(imgui.ImVec2(cx + 2, cy + 2), imgui.ImVec2(cx + 5, cy + 4), color, 1.5)
    else: 
        draw_list.add_circle_filled(imgui.ImVec2(cx, cy), 4, color, 12)

    if badge:
        badge_color = colors.get("queued", color)
        draw_list.add_text(imgui.ImVec2(cx + 7, cy - 6), badge_color, badge)

def perform_session_revert(session, index: int):
    """Revert session state to a specific bubble index."""
    if index < 0 or index >= len(session.bubbles):
        return

    if state.current_impl_sid is not None and state.current_impl_sid != session.id:
        log_message("Cannot revert this tab while another task is running.")
        return
        
    target_bubble = session.bubbles[index]
    role = target_bubble.role
    
    if session.is_generating or session.is_queued:
        cancel_generation(session.id)
        unqueue_session(session.id)
        
    prompt_to_resend = None
    
    if role == "user":
        session.history = session.history[:index]
        session.bubbles = session.bubbles[:index+1]
        prompt_to_resend = target_bubble.content
    elif role == "assistant":
        session.history = session.history[:index+1]
        session.bubbles = session.bubbles[:index+1]
        
    session.input_text = prompt_to_resend if prompt_to_resend else ""
    session.failed = False
    session.completed = (role == "assistant")
    session.current_bubble = None
    
    if role == "user" and prompt_to_resend:
        submit_prompt(ask_mode=session.is_ask_mode)

def render_chat_panel():
    tab_size = 28
    tab_spacing = 4

    draw_list = imgui.get_window_draw_list()

    group_map = {}
    for sid, sess in state.sessions.items():
        if sess.group_id is not None:
            if sess.group_id not in group_map:
                group_map[sess.group_id] = sid
            else:
                group_map[sess.group_id] = min(group_map[sess.group_id], sid)
    
    sorted_sessions = []
    for sid, sess in state.sessions.items():
        if sess.group_id is not None:
            start_id = group_map[sess.group_id]
            sorted_sessions.append(((start_id, sess.group_id), sess))
        else:
            sorted_sessions.append(((sid, None), sess))

    sorted_sessions.sort(key=lambda x: (x[0][0], x[0][1] if x[0][1] is not None else -1, x[1].id))

    sessions_to_close = []
    
    last_group_id = -999 
    first_tab = True
    
    for (sort_key, _), session in sorted_sessions:
        session_id = session.id
        group_id = session.group_id
        is_grouped = group_id is not None
        
        if not first_tab:
            if is_grouped and last_group_id == group_id:
                imgui.same_line(0, 0)
            else:
                imgui.same_line(0, tab_spacing)
                
        first_tab = False
        last_group_id = group_id

        if session.is_generating:
            if session.is_planning: status = "running_plan"
            elif session.is_ask_mode: status = "running_ask"
            else: status = "running"
        elif session.is_queued:
            if session.is_planning: status = "queued_plan"
            else: status = "queued"
        elif session.failed: status = "failed"
        elif session.completed:
            if session.is_planning: status = "done_plan"
            elif session.is_ask_mode: status = "done_ask"
            else: status = "done"
        elif session.is_debug: status = "debug"
        else: status = "inactive"

        badge = None
        if session.is_queued:
            try:
                badge = str(state.impl_queue.index(session_id) + 1)
            except ValueError:
                pass

        is_selected = (state.active_session_id == session_id)
        if is_selected:
            imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("bg_in"))
        else:
            imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("bg_cont"))

        imgui.push_style_color(imgui.Col_.button_hovered, STYLE.get_imvec4("sel_bg", 0.5))
        imgui.push_style_color(imgui.Col_.button_active, STYLE.get_imvec4("sel_bg"))

        imgui.push_id(session_id)
        
        if imgui.button(f"##{session_id}", imgui.ImVec2(tab_size, tab_size)):
            state.active_session_id = session_id

        if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.middle):
            if session.waiting_for_approval:
                # Treat waiting for approval like generating (busy)
                state.session_to_close_id = session_id
                state.show_close_tab_popup = True
            elif session.is_generating or session.is_queued:
                state.session_to_close_id = session_id
                state.show_close_tab_popup = True
            else:
                sessions_to_close.append(session_id)

        if imgui.begin_popup_context_item(f"tab_ctx_{session_id}"):
            if state.prompt_history:
                imgui.text_disabled("Prompt History")
                imgui.separator()
                for idx, hist_prompt in enumerate(reversed(state.prompt_history[-20:])):
                    display = hist_prompt[:60] + "..." if len(hist_prompt) > 60 else hist_prompt
                    display = display.replace("\n", " ")
                    if imgui.selectable(f"{display}##{idx}", False)[0]:
                        session.input_text = hist_prompt
            else:
                imgui.text_disabled("No prompt history")
            imgui.end_popup()

        btn_min = imgui.get_item_rect_min()
        btn_max = imgui.get_item_rect_max()
        cx = (btn_min.x + btn_max.x) / 2
        cy = (btn_min.y + btn_max.y) / 2

        _draw_tab_icon(draw_list, cx, cy, status, badge)
        
        if is_grouped:
            col = STYLE.get_u32("sel_bg")
            draw_list.add_rect_filled(
                imgui.ImVec2(btn_min.x, btn_max.y + 2),
                imgui.ImVec2(btn_max.x, btn_max.y + 4),
                col
            )

        imgui.pop_id()
        imgui.pop_style_color(3)

    imgui.same_line()

    imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_std"))
    imgui.push_style_color(imgui.Col_.button_hovered, STYLE.get_imvec4("btn_act"))
    imgui.push_style_color(imgui.Col_.button_active, STYLE.get_imvec4("sel_bg"))

    if imgui.button("+", imgui.ImVec2(tab_size, tab_size)):
        new_session = create_session()
        state.active_session_id = new_session.id
    if imgui.is_item_hovered():
        imgui.set_tooltip("Create a new tab")

    if imgui.begin_popup_context_item("new_tab_ctx"):
        if state.prompt_history:
            imgui.text_disabled("New Tab with Prompt")
            imgui.separator()
            for idx, hist_prompt in enumerate(reversed(state.prompt_history[-20:])):
                display = hist_prompt[:60] + "..." if len(hist_prompt) > 60 else hist_prompt
                display = display.replace("\n", " ")
                if imgui.selectable(f"{display}##{idx}", False)[0]:
                    new_session = create_session()
                    new_session.input_text = hist_prompt
                    state.active_session_id = new_session.id
        else:
            imgui.text_disabled("No prompt history")
        imgui.end_popup()

    imgui.pop_style_color(3)

    has_active = any(s.is_generating for s in state.sessions.values())
    if state.impl_queue or has_active:
        button_text = "Cancel All"
        text_size = imgui.calc_text_size(button_text)
        style = imgui.get_style()
        button_width = text_size.x + style.frame_padding.x * 2.0
        
        window_width = imgui.get_window_width()
        window_padding = style.window_padding.x
        
        cursor_x = imgui.get_cursor_pos_x()
        target_x = window_width - button_width - window_padding
        
        if target_x > cursor_x:
            imgui.same_line(target_x)
        else:
            imgui.same_line()
            
        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_cncl"))
        if imgui.button(button_text):
            cancel_all_tasks()
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Cancel all queued and running tasks")

    for sid in sessions_to_close:
        close_session(sid)

    imgui.separator()

    session = state.sessions.get(state.active_session_id)
    if session:
        render_chat_session(session)

def render_chat_session(session):
    avail = imgui.get_content_region_avail()
    
    # Fixed height overhead for Prompt label (25), Buttons (35), Context Text (35) plus Spacing
    bottom_overhead = 100
    if session.waiting_for_approval:
        bottom_overhead += 45

    # Clamp input height
    min_input = 60
    max_input = max(min_input, avail.y - 120)
    
    if state.chat_input_height < min_input: state.chat_input_height = min_input
    if state.chat_input_height > max_input: state.chat_input_height = max_input
    
    history_height = avail.y - state.chat_input_height - bottom_overhead
    if history_height < 50: history_height = 50

    imgui.begin_child("chat_history", imgui.ImVec2(0, history_height), child_flags=imgui.ChildFlags_.borders)

    revert_target_index = -1

    for i, bubble in enumerate(session.bubbles):
        is_last = (i == len(session.bubbles) - 1)
        is_loading = is_last and session.is_generating and bubble.role == "assistant" and not session.waiting_for_approval

        action = bubble.render(is_loading=is_loading)
        
        if action == "debug":
            if bubble.message.error_data:
                err_summary, raw_content = bubble.message.error_data
                debug_session = create_session()
                debug_session.is_debug = True
                state.active_session_id = debug_session.id
                
                debug_msg = f"## DEBUG INFO\n**Error:** {err_summary}\n\n**Raw Response:**\n\n{raw_content}"
                
                bub = ChatBubble("system", 0)
                bub.update("Viewing Debug Details for failed request.")
                debug_session.bubbles.append(bub)
                
                bub2 = ChatBubble("assistant", 1)
                bub2.update(debug_msg)
                debug_session.bubbles.append(bub2)

        elif action == "revert":
            revert_target_index = i
                
        imgui.spacing()

    if revert_target_index != -1:
        perform_session_revert(session, revert_target_index)

    if session.scroll_to_bottom:
        imgui.set_scroll_here_y(1.0)
        session.scroll_to_bottom = False

    imgui.end_child()

    if session.waiting_for_approval:
        imgui.push_style_color(imgui.Col_.child_bg, STYLE.get_imvec4("bg_in"))
        imgui.begin_child("approval_panel", imgui.ImVec2(0, 40), child_flags=imgui.ChildFlags_.borders)
        
        imgui.align_text_to_frame_padding()
        imgui.text("  Review Changes above and:")
        
        imgui.same_line()
        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_suc"))
        if imgui.button("APPROVE CHANGES", imgui.ImVec2(140, 0)):
            session.approval_result = True
            session.approval_event.set()
        imgui.pop_style_color()
        
        imgui.same_line()
        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_cncl"))
        if imgui.button("REJECT", imgui.ImVec2(100, 0)):
            session.approval_result = False
            session.approval_event.set()
        imgui.pop_style_color()
        
        imgui.end_child()
        imgui.pop_style_color()

    # Splitter
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0, 0, 0, 0))
    imgui.push_style_color(imgui.Col_.button_active, imgui.ImVec4(0, 0, 0, 0))
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0, 0, 0, 0))
    imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(0, 0, 0, 0))
    imgui.button("##h_splitter", imgui.ImVec2(-1, 5))
    if imgui.is_item_active():
        state.chat_input_height -= imgui.get_io().mouse_delta.y
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ns)
    imgui.pop_style_color(4)

    imgui.text("Prompt:")

    flags = imgui.InputTextFlags_.allow_tab_input | imgui.InputTextFlags_.word_wrap
    changed, session.input_text = imgui.input_text_multiline(
        "##prompt_input",
        session.input_text,
        imgui.ImVec2(-1, state.chat_input_height),
        flags
    )
    render_tooltip("Enter your request here. Press Ctrl+Enter to submit.")
    
    if changed:
        state.input_dirty = True
        state.last_input_time = time.time()

    if imgui.begin_popup_context_item("##prompt_ctx"):
        if imgui.menu_item("Copy", "", False, bool(session.input_text))[0]:
            imgui.set_clipboard_text(session.input_text)
        if imgui.menu_item("Paste", "", False)[0]:
            to_paste = imgui.get_clipboard_text()
            if to_paste:
                session.input_text += to_paste
        imgui.separator()
        if imgui.menu_item("Clear", "", False)[0]:
            session.input_text = ""
        imgui.end_popup()

    if imgui.is_item_active() or imgui.is_item_focused():
        if imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(imgui.Key.keypad_enter):
            if imgui.get_io().key_ctrl:
                submit_prompt(ask_mode=False)
                imgui.set_keyboard_focus_here(-1)
            elif imgui.get_io().key_shift and not changed:
                imgui.get_io().add_input_character(ord('\n'))

    is_busy = session.is_generating or session.is_queued
    is_cancelling = session.cancel_event.is_set()
    
    if is_busy:
        if is_cancelling:
            btn_text = "Cancelling..."
        elif session.is_queued:
            btn_text = "Queued"
        else:
            btn_text = "Generating..."

        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_war"))
        imgui.button(btn_text, imgui.ImVec2(100, 0))
        imgui.pop_style_color()

        if session.is_queued:
            imgui.same_line()
            imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_cncl"))
            if imgui.button("Unqueue", imgui.ImVec2(80, 0)):
                unqueue_session(session.id)
            imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip("Remove this task from the execution queue.")
    else:
        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_act"))
        if imgui.button("Run", imgui.ImVec2(80, 0)):
            submit_prompt(ask_mode=False)
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Modify files (Standard)")

        imgui.same_line()
        
        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_ask"))
        if imgui.button("Ask", imgui.ImVec2(80, 0)):
            submit_prompt(ask_mode=True)
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Ask questions without modifying files")

        imgui.same_line()

        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_run"))
        if imgui.button("Plan", imgui.ImVec2(80, 0)):
            submit_plan()
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Generate a plan and queue multiple tasks")

    if session.is_generating and not is_cancelling:
        imgui.same_line()
        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_cncl"))

        if imgui.button("Cancel", imgui.ImVec2(80, 0)):
            cancel_generation(session.id)
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Interrupt current generation.")

    if not session.is_generating and (session.backup_id or session.failed or session.completed):
        imgui.same_line()
        imgui.text_colored(STYLE.get_imvec4("bd"), "|")
        imgui.same_line()

        if imgui.button("Review"):
            try:
                if session.backup_id:
                    open_diff_report(session.backup_id)
                else:
                    log_message("No changes to review")
            except Exception as e:
                log_message(f"Review failed: {e}")
        if imgui.is_item_hovered():
            imgui.set_tooltip("View changes made in this session.")

    imgui.separator()

    checked_files = [f for f in state.selected_files if state.file_checked.get(f, True) and state.file_exists_cache.get(f, True)]
    context_tokens = sum(get_file_stats(f)[1] for f in checked_files)

    prompt_tokens = estimate_tokens(session.input_text)
    total_tokens = context_tokens + prompt_tokens

    model_list = list(AVAILABLE_MODELS.keys())
    model_name = model_list[state.model_idx] if state.model_idx < len(model_list) else ""
    _, price_str = calculate_input_cost(total_tokens, model_name)

    imgui.text_colored(STYLE.get_imvec4("fg_dim"),
        f"Context: ~{context_tokens} | Prompt: ~{prompt_tokens} | Total: ~{total_tokens} tokens | Est: {price_str}")

def render_logs_panel():
    if imgui.button("Clear"):
        state.logs.clear()
    
    imgui.same_line()
    _, state.show_debug_logs = imgui.checkbox("Debug", state.show_debug_logs)
    
    imgui.separator()

    imgui.begin_child("logs_content", imgui.ImVec2(0, 0))

    logs_list = list(state.logs)
    
    for entry in logs_list:
        level = entry.get("level", "INFO")
        msg = entry.get("msg", "")
        
        if level == "DEBUG" and not state.show_debug_logs:
            continue
            
        color = STYLE.get_imvec4("fg")
        if level == "ERROR" or level == "CRITICAL":
            color = STYLE.get_imvec4("btn_cncl")
        elif level == "WARNING":
            color = STYLE.get_imvec4("queued")
        elif level == "DEBUG":
            color = STYLE.get_imvec4("fg_dim")
        elif level == "INFO" and (msg.startswith(">") or "Success" in msg or "Completed" in msg):
             if "Success" in msg or "Completed" in msg:
                 color = STYLE.get_imvec4("txt_suc")
        
        ts = datetime.fromtimestamp(entry.get("time", time.time())).strftime("%H:%M:%S")
        
        imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"[{ts}]")
        imgui.same_line()
        imgui.text_colored(color, msg)

    if imgui.begin_popup_context_window():
        if imgui.menu_item("Clear Logs", "", False)[0]:
            state.logs.clear()
        if imgui.menu_item("Copy Logs", "", False)[0]:
            all_text = "\n".join([f"[{e.get('level')}] {e.get('msg')}" for e in state.logs])
            imgui.set_clipboard_text(all_text)
        imgui.end_popup()
    
    if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
        imgui.set_scroll_here_y(1.0)

    imgui.end_child()

def rebuild_view_tree():
    search = state.context_search_text.lower()
    
    if not search:
        # Pass a copy of file_paths to prevent modification during iteration by build_file_tree internals
        # build_file_tree creates new dict structure so raw list reading is fine.
        files_to_use = state.file_paths
        from application_state import build_file_tree
        root_tree = build_file_tree(files_to_use, Path.cwd())
    else:
        files_to_use = [f for f in state.file_paths if search in str(f).lower()]
        from application_state import build_file_tree
        root_tree = build_file_tree(files_to_use, Path.cwd())
        
    def annotate(node):
        n_lines = 0
        n_tokens = 0
        descendants = []
        
        if "_files" in node:
            for _, path in node["_files"]:
                descendants.append(path)
                l, t, _ = get_file_stats(path)
                n_lines += l
                n_tokens += t
        
        if "_children" in node:
            for key, child_node in node["_children"].items():
                if key.startswith("_"): continue
                cl, ct, c_desc = annotate(child_node)
                n_lines += cl
                n_tokens += ct
                descendants.extend(c_desc)
        
        node["_stats"] = (n_lines, n_tokens)
        node["_descendants"] = descendants
        return n_lines, n_tokens, descendants

    for _, node in root_tree.items():
        if isinstance(node, dict) and ("_files" in node or "_children" in node):
            annotate(node)
            
    if "_files" in root_tree:
        l, t, d = 0, 0, []
        for _, path in root_tree["_files"]:
            d.append(path)
            gl, gt, _ = get_file_stats(path)
            l += gl
            t += gt
        root_tree["_stats"] = (l, t)
        root_tree["_descendants"] = d
        
    state.view_tree = root_tree
    state.view_tree_dirty = False
    state.last_context_search = state.context_search_text

def render_context_manager():
    if not state.show_context_manager:
        return

    imgui.set_next_window_size(imgui.ImVec2(900, 600), imgui.Cond_.first_use_ever)
    opened, state.show_context_manager = imgui.begin("Manage Context", state.show_context_manager)

    if opened:
        if state.view_tree_dirty or state.context_search_text != state.last_context_search:
            rebuild_view_tree()

        imgui.text("Search:")
        imgui.same_line()
        imgui.set_next_item_width(300)
        changed, state.context_search_text = imgui.input_text("##search", state.context_search_text)
        render_tooltip("Filter files by name.")

        imgui.same_line()
        if imgui.button("Refresh"):
            refresh_project_files()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Rescan directory for new files.")

        imgui.same_line()
    
    if imgui.button("Clear"):
        state.selected_files.clear()
        state.file_checked.clear()
        state.file_exists_cache.clear()
        state.stats_dirty = True
        save_fileset()
    
    if imgui.is_item_hovered():
        imgui.set_tooltip("Deselect all files.")

    selected_files_list = [f for f in state.selected_files if f.exists()]
    total_lines = sum(get_file_stats(f)[0] for f in selected_files_list)
    total_tokens = sum(get_file_stats(f)[1] for f in selected_files_list)
    model_list = list(AVAILABLE_MODELS.keys())
    model_name = model_list[state.model_idx] if state.model_idx < len(model_list) else ""
    _, cost_str = calculate_input_cost(total_tokens, model_name)

    imgui.same_line()
    imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"  Selected: {len(selected_files_list)} files | {total_lines} lines | ~{total_tokens} tokens | {cost_str}")

    imgui.separator()

    imgui.columns(2, "ctx_cols")

    imgui.text("Groups")
    imgui.begin_child("groups_list", imgui.ImVec2(0, -30))
    for name in list(state.presets.keys()):
        _render_group_manage_row(name)
    imgui.end_child()

    if imgui.button("Create Group"):
        if state.selected_files:
            state.show_create_group_popup = True
            state.new_group_name = ""
        else:
            log_message("No files selected to create group")

    imgui.next_column()

    imgui.text("Files")

    if state.is_scanning:
        imgui.text_colored(STYLE.get_imvec4("queued"), "Scanning files in background... This may take a moment.")
        imgui.begin_child("files_list", imgui.ImVec2(0, 0))
        imgui.end_child()
        imgui.columns(1)
        imgui.end()
        return

    tree = state.view_tree

    imgui.begin_child("files_list", imgui.ImVec2(0, 0))

    table_flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.resizable | imgui.TableFlags_.scroll_y
    if imgui.begin_table("files_table", 3, table_flags):
        imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("Lines", imgui.TableColumnFlags_.width_fixed, 60)
        imgui.table_setup_column("Tokens", imgui.TableColumnFlags_.width_fixed, 70)
        imgui.table_headers_row()

        def render_tree_row(name, node, current_path, is_folder=False):
            if is_folder:
                full_path = current_path / name
                folder_key = str(full_path)
                is_open = state.folder_states.get(folder_key, False)

                folder_lines, folder_tokens = node.get("_stats", (0, 0))
                descendants = node.get("_descendants", [])
                
                if not descendants:
                    folder_selected = False
                else:
                    folder_selected = True
                    for f in descendants:
                        if f not in state.selected_files:
                            folder_selected = False
                            break

                imgui.table_next_row()
                imgui.table_next_column()

                # Checkbox + Folder Name in the same column
                changed, new_val = imgui.checkbox(f"##{folder_key}", folder_selected)
                if changed:
                    toggle_folder_selection(full_path, new_val)
                    if new_val:
                        # Auto-check files when folder is selected
                        for f in descendants:
                            state.file_checked[f] = True

                imgui.same_line()

                flags = imgui.TreeNodeFlags_.span_all_columns | imgui.TreeNodeFlags_.open_on_arrow
                if is_open:
                    flags |= imgui.TreeNodeFlags_.default_open
                node_open = imgui.tree_node_ex(f"{name}/", flags)
                state.folder_states[folder_key] = node_open

                imgui.table_next_column()
                imgui.text_colored(STYLE.get_imvec4("fg_dim"), str(folder_lines))

                imgui.table_next_column()
                imgui.text_colored(STYLE.get_imvec4("fg_dim"), str(folder_tokens))

                if node_open:
                    if "_children" in node:
                        sorted_children = sorted(node["_children"].items(), key=lambda x: x[0].lower())
                        for child_name, child_node in sorted_children:
                            render_tree_row(child_name, child_node, full_path, is_folder=True)
                    if "_files" in node:
                        sorted_files = sorted(node["_files"], key=lambda x: x[0].lower())
                        for file_name, file_path in sorted_files:
                            render_tree_row(file_name, {"_path": file_path}, current_path, is_folder=False)
                    imgui.tree_pop()
            else:
                file_path = node["_path"]
                is_selected = file_path in state.selected_files
                
                lines, tokens, _ = get_file_stats(file_path)

                imgui.table_next_row()
                imgui.table_next_column()

                # Checkbox + File Name in the same column
                imgui.push_id(str(file_path))
                changed, new_val = imgui.checkbox(f"##sel", is_selected)
                if changed:
                    toggle_file_selection(file_path, new_val)
                    if new_val:
                        state.file_checked[file_path] = True
                imgui.pop_id()

                imgui.same_line()
                
                label_color = None if is_selected else STYLE.get_imvec4("fg_dim")
                if label_color:
                    imgui.push_style_color(imgui.Col_.text, label_color)

                # Use Leaf node to ensure alignment with folder nodes (indentation)
                flags = imgui.TreeNodeFlags_.leaf | imgui.TreeNodeFlags_.no_tree_push_on_open | imgui.TreeNodeFlags_.span_all_columns
                imgui.tree_node_ex(name, flags)

                if label_color:
                    imgui.pop_style_color()

                imgui.table_next_column()
                imgui.text(str(lines))

                imgui.table_next_column()
                imgui.text(str(tokens))

        root_path = Path(".")
        sorted_root = sorted(tree.items(), key=lambda x: x[0].lower())
        
        for name, node in sorted_root:
            if name in ("_files", "_children", "_stats", "_descendants"):
                continue
            render_tree_row(name, node, root_path, is_folder=True)

        if "_files" in tree:
            root_files = tree.get("_files", [])
            for file_name, file_path in sorted(root_files, key=lambda x: x[0].lower()):
                render_tree_row(file_name, {"_path": file_path}, root_path, is_folder=False)

        imgui.end_table()

        imgui.end_child()

        imgui.columns(1)

    imgui.end()

def render_menu_bar():
    if imgui.begin_menu("File"):
        if imgui.menu_item("Sessions Manager", "Ctrl+O", False)[0]:
            state.show_sessions_window = True
        imgui.separator()
        if imgui.menu_item("Restart", "Ctrl+R", False)[0]:
            try_exit_app("restart")
        if imgui.menu_item("Exit", "Alt+F4", False)[0]:
            try_exit_app("exit")
        imgui.end_menu()

    if imgui.begin_menu("View"):
        runner_params = hello_imgui.get_runner_params()
        if imgui.menu_item("Restore Defaults", "", False)[0]:
            for window in runner_params.docking_params.dockable_windows:
                window.is_visible = True
            runner_params.docking_params.layout_reset = True

        imgui.separator()
        for window in runner_params.docking_params.dockable_windows:
            _, window.is_visible = imgui.menu_item(window.label, "", window.is_visible)
        imgui.end_menu()

    if imgui.begin_menu("Tools"):
        if imgui.menu_item("API Settings", "", False)[0]:
            open_api_settings()
        if imgui.menu_item("Context Manager", "", False)[0]:
            state.show_context_manager = True
        if imgui.menu_item("Backups", "", False)[0]:
            state.show_backup_history = True
        if imgui.menu_item("Toggle Theme", "", False)[0]:
            toggle_theme()
        imgui.end_menu()

    button_width = 20
    spacing = 4

    menu_bar_height = imgui.get_window_height()
    text_height = imgui.get_text_line_height()

    total_width = (button_width * 3) + (spacing * 2)
    avail_width = imgui.get_content_region_avail().x

    start_x = imgui.get_cursor_pos_x() + avail_width - total_width - 5

    imgui.push_style_var(imgui.StyleVar_.frame_padding, imgui.ImVec2(0, 0))
    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(0, 0, 0, 0))

    centering_y = (menu_bar_height - text_height) / 2

    # --- Button 1: Minimize ---
    imgui.same_line(start_x)
    imgui.set_cursor_pos_y(centering_y)
    # Note: passing button_width and text_height forces the size
    if imgui.button("-##min", imgui.ImVec2(button_width, text_height)):
        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                ctypes.windll.user32.ShowWindow(hwnd, 6)
            except:
                pass

    # --- Button 2: Maximize ---
    imgui.same_line(start_x + button_width, spacing)
    imgui.set_cursor_pos_y(centering_y)
    if imgui.button("##max", imgui.ImVec2(button_width, text_height)):
        if sys.platform == "win32":
            try:
                import ctypes
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                style_flags = ctypes.windll.user32.GetWindowLongW(hwnd, -16)
                if style_flags & 0x01000000:
                    ctypes.windll.user32.ShowWindow(hwnd, 9)
                else:
                    ctypes.windll.user32.ShowWindow(hwnd, 3)
            except:
                pass

    icon_col = imgui.get_color_u32(imgui.Col_.text)
    m_center = (imgui.get_item_rect_min() + imgui.get_item_rect_max()) / 2
    m_rad = 4

    # Rectangle for maximize cause our font doesn't have anything good
    imgui.get_window_draw_list().add_rect(
        imgui.ImVec2(m_center.x - m_rad, m_center.y - m_rad),
        imgui.ImVec2(m_center.x + m_rad, m_center.y + m_rad),
        icon_col,
        0.0,  # Rounding
        0,    # Flags (0 = defaults)
        1.0   # Thickness
    )

    # --- Button 3: Close ---
    imgui.same_line(start_x + button_width * 2 + spacing, spacing)
    imgui.set_cursor_pos_y(centering_y)
    imgui.push_style_color(imgui.Col_.button_hovered, imgui.ImVec4(0.9, 0.2, 0.2, 1.0))
    if imgui.button("X##close", imgui.ImVec2(button_width, text_height)):
        try_exit_app("exit")
    imgui.pop_style_color()

    imgui.pop_style_color()
    imgui.pop_style_var()

def main_gui():
    """Main GUI function called each frame."""
    state.frame_count += 1
    io = imgui.get_io()
    if io.key_ctrl:
        if imgui.is_key_pressed(imgui.Key.o):
            state.show_sessions_window = not state.show_sessions_window
        if imgui.is_key_pressed(imgui.Key.r):
            try_exit_app("restart")

    process_queue()

    is_generating = any(s.is_generating for s in state.sessions.values())
    hello_imgui.get_runner_params().fps_idling.enable_idling = not is_generating

    if state.input_dirty and time.time() - state.last_input_time > 2.0:
        save_state("autosave")
        state.input_dirty = False

    render_context_manager()
    render_sessions_window()
    render_backup_history_window()
    render_git_check_popups()
    render_close_tab_popup()
    render_exit_confirmation_popup()
    render_system_prompt_popup()

def create_docking_layout() -> tuple:
    splits = []

    s1 = hello_imgui.DockingSplit()
    s1.initial_dock = "MainDockSpace"
    s1.new_dock = "SidebarSpace"
    s1.direction = imgui.Dir_.right
    s1.ratio = 0.3
    splits.append(s1)

    s2 = hello_imgui.DockingSplit()
    s2.initial_dock = "MainDockSpace"
    s2.new_dock = "LogsSpace"
    s2.direction = imgui.Dir_.down
    s2.ratio = 0.2
    splits.append(s2)

    s3 = hello_imgui.DockingSplit()
    s3.initial_dock = "SidebarSpace"
    s3.new_dock = "FilesSpace"
    s3.direction = imgui.Dir_.down
    s3.ratio = 0.52
    splits.append(s3)

    windows = []

    chat_win = hello_imgui.DockableWindow()
    chat_win.label = "Chat"
    chat_win.dock_space_name = "MainDockSpace"
    chat_win.gui_function = render_chat_panel
    windows.append(chat_win)

    logs_win = hello_imgui.DockableWindow()
    logs_win.label = "Logs"
    logs_win.dock_space_name = "LogsSpace"
    logs_win.gui_function = render_logs_panel
    windows.append(logs_win)

    settings_win = hello_imgui.DockableWindow()
    settings_win.label = "Settings"
    settings_win.dock_space_name = "SidebarSpace"
    settings_win.gui_function = render_settings_panel
    windows.append(settings_win)

    files_win = hello_imgui.DockableWindow()
    files_win.label = "Files"
    files_win.dock_space_name = "FilesSpace"
    files_win.gui_function = render_files_panel
    windows.append(files_win)

    return splits, windows

def post_init():
    apply_imgui_theme(STYLE.dark)

def before_exit():
    if state and config.persist_session:
        save_state("autosave")

def run_gui():
    """Run the GUI application."""
    init_app_state()

    setup_logging()

    sync_settings_from_config()

    # Ensure theme is correctly applied on boot
    STYLE.load(config.theme)

    load_prompt_history()
    load_cwd_history()
    load_presets()
    refresh_project_files()
    load_fileset()

    add_to_cwd_history(str(Path.cwd()))

    if config.persist_session and (SESSIONS_DIR / "autosave.json").exists():
        load_state("autosave")

    if not state.sessions:
        initial_session = create_session()
        state.active_session_id = initial_session.id

    if not core.API_KEY:
        # Enforce dark theme on first boot for consistency
        if config.theme != "dark":
            config.set_theme("dark")
            STYLE.load("dark")

        open_api_settings()

    runner_params = hello_imgui.RunnerParams()
    runner_params.ini_filename = str(APP_DATA_DIR / "imgui.ini")
    runner_params.app_window_params.window_title = "Delta Tool"
    runner_params.app_window_params.window_geometry.size = (1200, 900)
    runner_params.app_window_params.restore_previous_geometry = True
    runner_params.app_window_params.borderless = True
    runner_params.app_window_params.borderless_movable = True
    runner_params.app_window_params.borderless_resizable = True
    runner_params.app_window_params.borderless_closable = False

    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )

    runner_params.imgui_window_params.enable_viewports = True
    runner_params.imgui_window_params.show_menu_bar = True
    runner_params.imgui_window_params.show_menu_view = False
    runner_params.imgui_window_params.show_status_bar = False

    # Set default theme based on config
    if config.theme == "light":
        runner_params.imgui_window_params.tweaked_theme.theme = hello_imgui.ImGuiTheme_.imgui_colors_light
    else:
        runner_params.imgui_window_params.tweaked_theme.theme = hello_imgui.ImGuiTheme_.imgui_colors_dark

    # Enable FPS Idling
    runner_params.fps_idling.enable_idling = True

    splits, windows = create_docking_layout()
    runner_params.docking_params.docking_splits = splits
    runner_params.docking_params.dockable_windows = windows

    runner_params.callbacks.post_init = post_init
    runner_params.callbacks.before_exit = before_exit
    runner_params.callbacks.show_gui = main_gui
    runner_params.callbacks.show_menus = render_menu_bar

    addons = immapp.AddOnsParams()
    addons.with_markdown = True

    immapp.run(runner_params, addons)

    if state.restart_requested:
        sys.exit(42)
