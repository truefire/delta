"""GUI implementation for Delta Tool."""
import math
import sys
import shutil
import subprocess
import threading
import time
import queue
import json
import os
import fnmatch
import stat
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
    add_to_cwd_history, load_cwd_history, SESSIONS_DIR,
    tree_lock, queue_scan_request, quicksave_session
)
from widgets import ChatBubble, DiffViewer, render_file_tree, DiffHunk, draw_status_icon
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
            err_bubble = ChatBubble("error", len(session.bubbles))
            err_bubble.update(event.get("message", "Task failed"))
            session.bubbles.append(err_bubble)

    elif event_type == "done":
        state.stats_dirty = True
        if session:
            session.request_end_time = time.time()
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

    elif event_type == "search_result":
        if event.get("query_id") == state.search_query_id:
            state.cached_flat_filtered_files = event.get("flat")
            state.cached_whitelist_dirs = event.get("dirs")
            state.cached_whitelist_files = event.get("files")
            state.is_searching = False
            state.view_tree_dirty = True

    elif event_type == "filedig_update" and session:
        # Show tool activity in UI
        msg = event.get("message", "")
        
        # Append to existing tool bubble if active, else create
        last_bubble = session.bubbles[-1] if session.bubbles else None
        
        if last_bubble and last_bubble.role == "tool":
            last_bubble.update(msg + "\n")
        else:
            if session.current_bubble and session.current_bubble.role == "assistant":
                session.current_bubble.flush()
                session.current_bubble = None
            
            tool_bubble = ChatBubble("tool", len(session.bubbles))
            tool_bubble.update(msg + "\n")
            session.bubbles.append(tool_bubble)
    
    elif event_type == "filedig_success":
        # Handle successful dig - spawn new session
        found_files = event.get("files", [])
        explanation = event.get("explanation", "")
        prompt = event.get("prompt", "")
        tool_calls = event.get("tool_calls", 0)
        
        # 0. Grouping logic: Ensure filedig session has a group, and new session joins it
        filedig_sess = state.sessions.get(session_id)
        target_group_id = None
        if filedig_sess:
            filedig_sess.completed = True
            filedig_sess.failed = False

            # Add completion bubble
            done_bubble = ChatBubble("assistant", len(filedig_sess.bubbles))
            done_bubble.update(f"Completed in {tool_calls} toolcall(s).")
            filedig_sess.bubbles.append(done_bubble)

            if filedig_sess.group_id is None:
                filedig_sess.group_id = state.next_group_id
                state.next_group_id += 1
            target_group_id = filedig_sess.group_id
            
        # 2. Prepare context files for Run session (do NOT add to global selection)
        valid_files = []
        for f in found_files:
            p = Path(f)
            # Try to resolve relative to CWD
            if not p.exists():
                p = Path.cwd() / f
            
            if p.exists():
                rel = to_relative(p)
                valid_files.append(rel)
        
        # 3. Create new RUN session
        new_sess = create_session()
        new_sess.group_id = target_group_id
        # Pass the files specifically to this session
        new_sess.forced_context_files = valid_files

        state.active_session_id = new_sess.id
        
        full_prompt = f"{prompt}\n\n(Context found via Agentic search:\n{explanation})"
        new_sess.input_text = ""
        new_sess.last_prompt = full_prompt
        
        _submit_common(new_sess, full_prompt, is_planning=False, ask_mode=False, save_to_history=False)

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
    
    session.execution_start_time = time.time()

    if not session.is_ask_mode:
        state.current_impl_sid = session_id
    session.is_generating = True
    session.is_queued = False
    session.failed = False
    session.completed = False
    
    session.cancel_event.clear()

    ensure_user_bubble(session, session.last_prompt)
    
    if session.forced_context_files is not None:
        checked_files = session.forced_context_files
    else:
        checked_files = [f for f in state.selected_files if state.file_checked.get(f, True)]
    
    # Snapshot actual files sent
    session.sent_files = [str(f) for f in checked_files]

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
            if not session.is_filedig:
                state.gui_queue.put({"type": "status", "message": msg.rstrip()})
            else:
                state.gui_queue.put({"type": "filedig_update", "session_id": session_id, "message": msg.rstrip()})

        def stream_func(text: str, end: str = "", flush: bool = False):
            state.gui_queue.put({"type": "text", "session_id": session_id, "content": text})

        if session.is_filedig:
            from core import run_filedig_agent
            result = run_filedig_agent(prompt, output_func, cancel_event)
            if result.get("success"):
                state.gui_queue.put({
                    "type": "filedig_success",
                    "session_id": session_id,
                    "files": result.get("files", []),
                    "explanation": result.get("explanation", ""),
                    "tool_calls": result.get("tool_calls", 0),
                    "prompt": prompt
                })
            else:
                state.gui_queue.put({"type": "failure", "session_id": session_id, "message": result.get("message")})
            
            state.gui_queue.put({"type": "done", "session_id": session_id})
            return

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

        def on_llm_start():
            state.gui_queue.put({
                "type": "start_response",
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
            on_llm_start=on_llm_start,
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

def _submit_common(session, prompt: str, is_planning: bool = False, ask_mode: bool = False, is_filedig: bool = False, save_to_history: bool = True):
    """Common submission logic for prompt and plan."""
    if save_to_history and prompt and (not state.prompt_history or state.prompt_history[-1] != prompt):
        state.prompt_history.append(prompt)
        if MAX_PROMPT_HISTORY > 0 and len(state.prompt_history) > MAX_PROMPT_HISTORY:
            state.prompt_history.pop(0)
        save_prompt_history()

    if state.queue_start_time == 0.0:
        state.queue_start_time = time.time()

    session.last_prompt = prompt
    session.input_text = ""
    session.request_start_time = time.time()
    session.execution_start_time = None
    session.request_end_time = 0.0
    session.is_ask_mode = ask_mode
    session.is_planning = is_planning
    session.is_filedig = is_filedig
    state.prompt_history_idx = -1

    ensure_user_bubble(session, session.last_prompt)

    if session.is_ask_mode or session.is_filedig:
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

def submit_filedig():
    """Submit a filedig request."""
    session = get_active_session()
    if not session: return
    
    prompt = session.input_text.strip()
    if not prompt: return
    
    # Filedig doesn't care about current files, it finds them.
    _submit_common(session, prompt, is_planning=False, ask_mode=False, is_filedig=True)

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
                
            except Exception as e:
                state.api_settings_error = f"Error: {e}"
                
        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(120, 0)):
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
                        if source == "git":
                            imgui.set_tooltip("Checkout this specific git commit state.")
                        else:
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

    imgui.same_line()
    if imgui.small_button("CMD"):
        core.open_terminal_in_os(Path.cwd())

    if imgui.is_item_hovered():
        imgui.set_tooltip("Open terminal in CWD.\nRight-click for File Explorer.")

    if imgui.begin_popup_context_item("cmd_explore_ctx"):
        if imgui.selectable("Explore", False)[0]:
            core.open_path_in_os(Path.cwd())
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

    imgui.same_line()
    imgui.text("Dig Turns:")
    imgui.same_line()
    imgui.set_next_item_width(50)
    changed, state.filedig_max_turns = imgui.input_text("##fdturns", state.filedig_max_turns)
    render_tooltip("Max turns for Filedig agent.")
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



def _perform_git_pull():
    state.update_in_progress = True
    state.update_status = "Pulling changes from remote..."
    
    def worker():
        try:
            askpass_script = core.create_askpass_wrapper()
            
            env = os.environ.copy()
            env["GIT_ASKPASS"] = askpass_script
            env["SSH_ASKPASS"] = askpass_script
            env["SSH_ASKPASS_REQUIRE"] = "force"
            env["DISPLAY"] = ":0" # Fake display to trick SSH into thinking it can askpass
            
            # Use subprocess directly to inject env
            tool_dir = core._TOOL_DIR
            proc = subprocess.run(
                ["git", "pull"],
                cwd=tool_dir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            success = (proc.returncode == 0)
            out = proc.stdout
            
            state.update_in_progress = False
            if success:
                if "Already up to date" in out:
                    state.update_status = "No new changes found."
                else:
                    state.update_status = "Update successful! Restart required."
                state.can_update = False
                state.update_func = None
            else:
                state.update_status = f"Git pull failed:\n{out}"
                
        except Exception as e:
            state.update_in_progress = False
            state.update_status = f"Error during update: {e}"
    
    threading.Thread(target=worker, daemon=True).start()

def _perform_uv_upgrade(tool_name):
    # Windows-specific handling: Spawn external script and exit
    if sys.platform == "win32":
        try:
            # 1. Create a temporary batch script in AppData
            ipc_dir = core.APP_DATA_DIR / "ipc"
            ipc_dir.mkdir(parents=True, exist_ok=True)
            updater_script = ipc_dir / "update_runner.bat"
            log_file = ipc_dir / "update_log.txt"
            result_file = ipc_dir / "update_result.txt"

            # Cleanup old files
            if log_file.exists(): log_file.unlink()
            if result_file.exists(): result_file.unlink()
            
            # The script waits for us to die, updates, restarts, self-deletes
            script_content = f"""@echo off
title Delta Updater
echo Waiting for Delta to close...
timeout /t 3 /nobreak >nul
echo Upgrading {tool_name}...
echo Running update... > "{log_file}"
uv tool upgrade {tool_name} >> "{log_file}" 2>&1
set ERR=%errorlevel%
echo %ERR% > "{result_file}"
if %ERR% neq 0 (
    echo Update failed with code %ERR%.
) else (
    echo Update complete.
)
echo Restarting Delta...
start delta
(goto) 2>nul & del "%~f0"
"""
            with open(updater_script, "w", encoding="utf-8") as f:
                f.write(script_content)

            # 2. Spawn the script detached from this process
            # CREATE_NEW_CONSOLE (0x10) ensures it has its own window
            subprocess.Popen(
                [str(updater_script)],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                shell=True
            )
            
            # 3. Request exit to release file lock
            hello_imgui.get_runner_params().app_shall_exit = True
            
        except Exception as e:
            state.update_status = f"Failed to launch updater: {e}"
        return

    state.update_in_progress = True
    state.update_status = f"Upgrading '{tool_name}' via uv..."
    
    def worker():
        success, out = core.run_command(f"uv tool upgrade {tool_name}")
        state.update_in_progress = False
        if success:
            if "Already up to date" in out:
                state.update_status = "No new changes found."
            else:
                state.update_status = "Update successful! Restart required."
            state.can_update = False
            state.update_func = None
        else:
            state.update_status = f"uv upgrade failed:\n{out}"
            
    threading.Thread(target=worker, daemon=True).start()

def check_update_result():
    """Check for results from a previous update attempt."""
    ipc_dir = core.APP_DATA_DIR / "ipc"
    result_file = ipc_dir / "update_result.txt"
    log_file = ipc_dir / "update_log.txt"
    
    if result_file.exists():
        try:
            content = result_file.read_text().strip()
            if not content: return
            
            exit_code = int(content)
            log_content = log_file.read_text("utf-8", errors="replace").strip() if log_file.exists() else ""
            
            # Clean up
            try: result_file.unlink() 
            except: pass
            try: log_file.unlink() 
            except: pass
            
            if exit_code == 0:
                state.update_status = "Update successful!\nDelta has been updated to the latest version."
            else:
                if len(log_content) > 1000:
                    log_content = log_content[:1000] + "\n...(truncated)"
                state.update_status = f"Update failed (Exit code {exit_code}).\n\nLog:\n{log_content}"
            
            state.show_update_popup = True
            state.update_in_progress = False
            state.can_update = False
            
        except Exception as e:
            log_message(f"Error checking update result: {e}")

def _check_updates_worker():
    tool_dir = core._TOOL_DIR
    
    # 1. Check Git
    if (tool_dir / ".git").exists():
        ok, out = core.run_command("git remote -v", cwd=tool_dir)
        if ok and "origin" in out:
            ok_f, _ = core.run_command("git fetch", cwd=tool_dir)
            ok_s, status_out = core.run_command("git status -uno", cwd=tool_dir)

            print(status_out)
            if "behind" in status_out:
                state.update_status = "Git repository detected.\nNew commits available."
                state.can_update = True
                state.update_func = _perform_git_pull
            elif "up to date" in status_out:
                state.update_status = "Git repository detected.\nAlready up to date."
                state.can_update = False
            elif "branch is ahead" in status_out:
                state.update_status = "Git repository detected.\nYou have local commits.\nWould you like to attempt an update?"
                state.can_update = True
                state.update_func = _perform_git_pull
            else:
                state.update_status = "Git repository detected.\nBranch status unclear."
                state.can_update = True
                state.update_func = _perform_git_pull
            return
    
    # 2. Check UV
    if shutil.which("uv"):
        ok, out = core.run_command("uv tool list")
        if ok:
            found_name = None
            for line in out.splitlines():
                if "delta" in line or "deltatool" in line:
                    found_name = line.split()[0]
                    break
            
            if found_name:
                warn = ""
                if sys.platform == "win32":
                    warn = "\n\nNOTE: The program must close to apply updates.\nA separate console will open, and Delta will restart automatically."

                state.update_status = f"Detected 'uv' installation ('{found_name}').{warn}"
                # uv upgrade handles already-updated state gracefully
                state.can_update = True
                state.update_func = lambda: _perform_uv_upgrade(found_name)
                return

    state.update_status = "Could not detect updatable installation.\n(Not a git repo, nor found in `uv tool list`)"
    state.can_update = False

def check_for_updates():
    state.show_update_popup = True
    state.update_status = "Checking installation method..."
    state.can_update = False
    state.update_func = None
    state.update_in_progress = False
    
    threading.Thread(target=_check_updates_worker, daemon=True).start()

def render_update_popup():
    if state.show_update_popup:
        imgui.open_popup("Check for Updates")
        state.show_update_popup = False
        
    if imgui.begin_popup_modal("Check for Updates", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text(state.update_status)
        imgui.separator()
        
        if state.update_in_progress:
            imgui.text_colored(STYLE.get_imvec4("queued"), "Working...")
        else:
            if "Update successful" in state.update_status:
                if imgui.button("Restart Now", imgui.ImVec2(120, 0)):
                    try_exit_app("restart")
                    imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Restart Later", imgui.ImVec2(120, 0)):
                    imgui.close_current_popup()
            else:
                if state.can_update and state.update_func:
                    if imgui.button("Update Now", imgui.ImVec2(120, 0)):
                        state.update_func()
                    imgui.same_line()
                
                if imgui.button("Close", imgui.ImVec2(120, 0)):
                    imgui.close_current_popup()
                
        imgui.end_popup()

def poll_auth_request():
    """Check for authentication requests from background processes."""
    if state.show_auth_popup or state.auth_request_data is not None:
        return # Already showing

    ipc_dir = APP_DATA_DIR / "ipc"
    if not ipc_dir.exists():
        return

    # Find .req files
    try:
        req_files = list(ipc_dir.glob("*.req"))
        if req_files:
            # Take the first one
            req_file = req_files[0]
            try:
                with open(req_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                state.auth_request_data = data
                state.auth_request_data["_file"] = req_file # Store path for cleanup/reply
                state.auth_input_text = ""
                state.show_auth_popup = True
            except Exception:
                # If corrupt, delete it so we don't loop
                try: req_file.unlink()
                except: pass
    except Exception:
        pass

def render_auth_popup():
    if state.show_auth_popup:
        imgui.open_popup("Authentication Required")
        state.show_auth_popup = False
    
    if imgui.begin_popup_modal("Authentication Required", None, imgui.WindowFlags_.always_auto_resize)[0]:
        is_submitted = state.auth_request_data and state.auth_request_data.get("_status") == "submitted"
        
        if is_submitted:
            imgui.text("Authentication submitted.")
            imgui.separator()
            
            if state.update_in_progress:
                imgui.text_colored(STYLE.get_imvec4("queued"), "Processing...")
                imgui.text_colored(STYLE.get_imvec4("fg_dim"), state.update_status)
            else:
                status_text = state.update_status if state.update_status else "Process completed."
                is_success = "success" in status_text.lower()
                
                if is_success:
                    imgui.text_colored(STYLE.get_imvec4("txt_suc"), status_text)
                elif "fail" in status_text.lower() or "error" in status_text.lower():
                     imgui.text_colored(STYLE.get_imvec4("btn_cncl"), status_text)
                else:
                     imgui.text(status_text)
                
                imgui.spacing()
                imgui.separator()
                
                if is_success and "restart" in status_text.lower():
                    if imgui.button("Restart Now", imgui.ImVec2(120, 0)):
                        state.auth_request_data = None
                        state.auth_input_text = ""
                        try_exit_app("restart")
                        imgui.close_current_popup()
                        
                    imgui.same_line()
                    if imgui.button("Restart Later", imgui.ImVec2(120, 0)):
                        state.auth_request_data = None
                        state.auth_input_text = ""
                        imgui.close_current_popup()
                else:
                    if imgui.button("Dismiss", imgui.ImVec2(120, 0)):
                        state.auth_request_data = None
                        state.auth_input_text = ""
                        imgui.close_current_popup()
        else:
            prompt = state.auth_request_data.get("prompt", "Password required:") if state.auth_request_data else "Password required:"
            imgui.text(prompt)
            imgui.spacing()
            
            imgui.set_next_item_width(300)
            confirm = False
            changed, state.auth_input_text = imgui.input_text("##auth_input", state.auth_input_text, imgui.InputTextFlags_.password | imgui.InputTextFlags_.enter_returns_true)
            if changed: confirm = True
            
            if not imgui.is_any_item_active():
                imgui.set_keyboard_focus_here(-1)

            imgui.separator()
            
            if imgui.button("OK", imgui.ImVec2(120, 0)) or confirm:
                # Write response
                try:
                    req_path = state.auth_request_data.get("_file")
                    if req_path:
                        # Construct response path matches req path stem
                        res_path = req_path.with_suffix(".res")
                        with open(res_path, "w", encoding="utf-8") as f:
                            json.dump({"password": state.auth_input_text}, f)
                        
                        req_path.unlink()
                        log_message("Authentication submitted.")
                        if state.auth_request_data:
                            state.auth_request_data["_status"] = "submitted"
                except Exception as e:
                    log_message(f"Auth response failed: {e}")
                    state.auth_request_data = None
                    state.auth_input_text = ""
                    imgui.close_current_popup()
            
            imgui.same_line()
            if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                # On cancel, maybe write empty or just delete req file?
                # If we delete req file, CLI waits until timeout then fails.
                # Faster to write empty json?
                try:
                    req_path = state.auth_request_data.get("_file")
                    if req_path:
                        req_path.unlink() # Just remove request, let CLI timeout
                except: pass
                
                state.auth_request_data = None
                state.auth_input_text = ""
                imgui.close_current_popup()
            
        imgui.end_popup()

def _render_group_row(name: str):
    if name not in state.presets or name == "__active":
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
    if name not in state.presets or name == "__active":
        return

    group_data = state.presets[name]
    group_files = group_data.get("files", [])
    
    imgui.push_id(f"manage_{name}")
    
    if imgui.small_button("Load"):
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
        imgui.set_tooltip("Load this group into selection")
    
    imgui.same_line()
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

def update_app_stats():
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
        
        # Update folder counts
        cwd = Path.cwd()
        state.folder_selection_counts.clear()
        for f_rel in state.selected_files:
            try:
                # We need to increment count for every parent folder up to root
                # Since we store keys as absolute path strings in visible_rows
                f_abs = (cwd / f_rel).resolve()
                parent = f_abs.parent
                while True:
                    # Don't go above cwd
                    if not str(parent).startswith(str(cwd)):
                        break
                    
                    k = str(parent)
                    state.folder_selection_counts[k] = state.folder_selection_counts.get(k, 0) + 1
                    
                    if parent == cwd:
                        break
                    parent = parent.parent
            except Exception:
                pass

        # Calculate totals
        checked = [f for f in state.cached_sorted_files 
                   if state.file_checked.get(f, True) and state.file_exists_cache.get(f, True)]
                   
        state.cached_total_lines = sum(get_file_stats(f)[0] for f in checked)
        state.cached_total_tokens = sum(get_file_stats(f)[1] for f in checked)
        
        model_list = list(AVAILABLE_MODELS.keys())
        model_name = model_list[state.model_idx] if state.model_idx < len(model_list) else ""
        _, state.cached_cost_str = calculate_input_cost(state.cached_total_tokens, model_name)
        
        state.stats_dirty = False

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

    update_app_stats()
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
        
    if role == "user":
        # Remove user bubble from history and put content in input for editing
        session.history = session.history[:index]
        session.bubbles = session.bubbles[:index]
        session.input_text = target_bubble.content
    elif role == "assistant":
        # Keep assistant bubble, clear input for new follow-up
        session.history = session.history[:index+1]
        session.bubbles = session.bubbles[:index+1]
        session.input_text = ""
        
    session.failed = False
    session.completed = (role == "assistant")
    session.current_bubble = None

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
            elif session.is_filedig: status = "running_filedig"
            else: status = "running"
        elif session.is_queued:
            if session.is_planning: status = "queued_plan"
            elif session.is_filedig: status = "queued_filedig"
            else: status = "queued"
        elif session.failed: status = "failed"
        elif session.completed:
            if session.is_planning: status = "done_plan"
            elif session.is_ask_mode: status = "done_ask"
            elif session.is_filedig: status = "done_filedig"
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

        draw_status_icon(draw_list, cx, cy, status, badge)
        
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

    # Buttons (System Prompt Toggle + Cancel All)
    style = imgui.get_style()
    window_padding = style.window_padding.x
    window_width = imgui.get_window_width()
    
    qs_btn_text = "Quicksave Tabs"
    qs_w = imgui.calc_text_size(qs_btn_text).x + style.frame_padding.x * 2.0

    sys_btn_text = "Hide System" if state.show_system_prompt else "Show System"
    sys_w = imgui.calc_text_size(sys_btn_text).x + style.frame_padding.x * 2.0
    
    cancel_btn_text = "Cancel All"
    cancel_w = 0
    if state.impl_queue or has_active:
        cancel_w = imgui.calc_text_size(cancel_btn_text).x + style.frame_padding.x * 2.0
        
    total_w = qs_w + style.item_spacing.x + sys_w
    if cancel_w > 0:
        total_w += cancel_w + style.item_spacing.x

    # Queue Timer calculation
    q_timer_text = ""
    
    if state.queue_start_time > 0.0:
        dur = time.time() - state.queue_start_time
        if dur > 0:
            q_timer_text = f"{dur:.1f}s"

    if q_timer_text:
        tw = imgui.calc_text_size(q_timer_text).x + style.frame_padding.x * 2
        total_w += tw + style.item_spacing.x

    cursor_x = imgui.get_cursor_pos_x()
    target_x = window_width - total_w - window_padding

    if target_x > cursor_x:
        imgui.same_line(target_x)
    else:
        imgui.same_line()

    if imgui.button(qs_btn_text):
        quicksave_session()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Instantly save all open sessions with a timestamp.")

    imgui.same_line()

    if imgui.button(sys_btn_text):
        state.show_system_prompt = not state.show_system_prompt
    if imgui.is_item_hovered():
        imgui.set_tooltip("Toggle visibility of the System Prompt at the top of the chat.")

    if q_timer_text:
        imgui.same_line()
        imgui.align_text_to_frame_padding()
        imgui.text_colored(STYLE.get_imvec4("fg_dim"), q_timer_text)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Total elapsed time since request submission")

    if cancel_w > 0:
        imgui.same_line()
        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_cncl"))
        if imgui.button(cancel_btn_text):
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
    bottom_overhead = 125
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

    if state.show_system_prompt and not session.is_filedig:
        # If the session has run before, show what was actually sent (snapshot).
        # Otherwise (fresh tab), show current selection (preview).
        if session.request_start_time > 0:
            checked_files = sorted(session.sent_files) if session.sent_files else []
        else:
            checked_files = sorted([str(f) for f in state.selected_files if state.file_checked.get(f, True)])
        
        # Calculate key with mtimes to ensure content freshness
        file_state_key = []
        for f in checked_files:
            try:
                mtime = Path(f).stat().st_mtime
            except Exception:
                mtime = 0
            file_state_key.append((f, mtime))

        current_key = (tuple(file_state_key), session.is_ask_mode, session.is_planning, config.extra_system_prompt)
        
        if state.cached_sys_key != current_key or state.cached_sys_bubble is None:
            sys_msg = core.build_system_message(
                checked_files, 
                ask_mode=session.is_ask_mode, 
                plan_mode=session.is_planning
            )
            state.cached_sys_bubble = ChatBubble("system", -1)
            state.cached_sys_bubble.update(sys_msg)
            
            # Create Viewers for each file
            for i, fpath in enumerate(checked_files):
                if core.is_image_file(fpath):
                    continue
                    
                try:
                    content = core.file_cache.get_or_read(fpath)
                    
                    # Manually create DiffViewer to show file content as "New File" (Green)
                    # We bypass string parsing to allow safe display of any content
                    dv = DiffViewer(content="", block_state={"collapsed": True}, viewer_id=f"preview_{i}", filename_hint=fpath)
                    
                    # Patch State
                    dv.state.filename = fpath
                    dv.state.is_creation = True
                    dv.state.suppress_new_label = True
                    dv.state.hunks = [DiffHunk(type="change", old="", new=content)]
                    dv.state.change_indices = [0]
                    
                    state.cached_sys_bubble.pre_viewers.append(dv)
                    
                except Exception as e:
                    # If read fails, just skip viewer
                    pass

            state.cached_sys_key = current_key
        
        state.cached_sys_bubble.render()
        
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

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

    if session.should_focus_input:
        if state.frame_count > 1:
            imgui.set_keyboard_focus_here()
            session.should_focus_input = False

    flags = imgui.InputTextFlags_.allow_tab_input | imgui.InputTextFlags_.word_wrap
    changed, session.input_text = imgui.input_text_multiline(
        f"##prompt_input_{session.id}",
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
            
        imgui.same_line()

        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_dig"))
        if imgui.button("Filedig", imgui.ImVec2(80, 0)):
            submit_filedig()
        imgui.pop_style_color()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Agentic search: Finds relevant files and then starts a Run")


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

    # Task Timer
    if session.execution_start_time is not None:
        t_now = time.time()
        t_end = t_now
        if session.request_end_time > 0:
            t_end = session.request_end_time
        
        t_dur = max(0.0, t_end - session.execution_start_time)
        
        if t_dur > 0 or session.is_generating:
            imgui.dummy(imgui.ImVec2(0, 2))
            run_str = f"Task: {t_dur:.1f}s"
            window_w = imgui.get_window_width()
            t_w = imgui.calc_text_size(run_str).x
            style = imgui.get_style()
            target_x = window_w - t_w - style.window_padding.x - 5
            imgui.set_cursor_pos_x(target_x)
            imgui.text_colored(STYLE.get_imvec4("fg_dim"), run_str)

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

def _search_worker(query_id: int, search_text: str):
    try:
        with tree_lock:
            file_paths = list(state.file_paths)

        search_lower = search_text.lower()
        is_glob = any(c in search_lower for c in "*?[]")
        
        if state.search_query_id != query_id: return

        # 1. Flat Filter
        if is_glob:
            filtered = [f for f in file_paths if fnmatch.fnmatch(str(f).lower(), search_lower)]
        else:
            filtered = [f for f in file_paths if search_lower in str(f).lower()]

        if state.search_query_id != query_id: return

        # 2. Tree Whitelists
        search_whitelist_dirs = set()
        search_whitelist_files = set()
        cwd = Path.cwd()
        
        for f_rel in filtered:
            if state.search_query_id != query_id: return
            
            f_abs = cwd / f_rel
            search_whitelist_files.add(f_abs)
            
            p = f_abs.parent
            while True:
                if p in search_whitelist_dirs:
                     break
                
                search_whitelist_dirs.add(p)
                if p == cwd or len(p.parts) <= len(cwd.parts):
                    break
                p = p.parent
                
        search_whitelist_dirs.add(cwd)
        
        if state.search_query_id != query_id: return

        state.gui_queue.put({
            "type": "search_result",
            "query_id": query_id,
            "flat": filtered,
            "dirs": search_whitelist_dirs,
            "files": search_whitelist_files
        })
    except Exception as e:
        log_message(f"Search thread error: {e}")

def trigger_context_search():
    state.last_context_search = state.context_search_text

    if not state.context_search_text:
        state.cached_flat_filtered_files = None
        state.cached_whitelist_dirs = None
        state.cached_whitelist_files = None
        state.is_searching = False
        state.view_tree_dirty = True
        return

    state.view_tree_dirty = False
    state.search_query_id += 1
    state.is_searching = True
    
    threading.Thread(
        target=_search_worker,
        args=(state.search_query_id, state.context_search_text),
        daemon=True
    ).start()

def _rebuild_context_rows():
    """Rebuild the flattened list of tree rows for the context manager."""
    visible_rows = []
    
    def count_files_recursive(node):
        count = len(node.get("_files", []))
        if "_children" in node:
            for child in node["_children"].values():
                count += count_files_recursive(child)
        return count

    search_whitelist_dirs = state.cached_whitelist_dirs
    search_whitelist_files = state.cached_whitelist_files

    def build_flat_tree(node_name, node, current_path, depth):
        full_path = current_path / node_name
        
        if search_whitelist_dirs is not None and full_path not in search_whitelist_dirs:
            return

        folder_key = str(full_path)
        
        if search_whitelist_dirs is not None:
            is_open = True
        else:
            is_open = state.folder_states.get(folder_key, False)
        
        # Lazy Load check (only if open)
        if is_open and not node.get("_scanned") and not node.get("_scanning"):
            node["_scanning"] = True
            queue_scan_request(full_path, priority=0)

        visible_rows.append({
            "type": "folder",
            "name": node_name,
            "key": folder_key,
            "is_open": is_open,
            "depth": depth,
            "scanning": node.get("_scanning", False),
            "path": full_path,
            "total_files": count_files_recursive(node)
        })
        
        if is_open:
            if node.get("_children"):
                for child_name, child_node in sorted(node["_children"].items(), key=lambda x: x[0].lower()):
                    build_flat_tree(child_name, child_node, full_path, depth + 1)
                    
            if node.get("_files"):
                for fname, fpath in sorted(node["_files"]):
                    if search_whitelist_files is not None and fpath not in search_whitelist_files:
                        continue
                    visible_rows.append({
                        "type": "file",
                        "name": fname,
                        "path": fpath,
                        "depth": depth + 1
                    })

    with tree_lock:
         root_node = state.file_tree
         
         if "_children" in root_node:
             for dname, dnode in sorted(root_node["_children"].items(), key=lambda x: x[0].lower()):
                 build_flat_tree(dname, dnode, Path.cwd(), 0)
         
         if "_files" in root_node:
             for fname, fpath in sorted(root_node["_files"]):
                 if search_whitelist_files is not None and fpath not in search_whitelist_files:
                     continue
                 visible_rows.append({
                     "type": "file",
                     "name": fname,
                     "path": fpath,
                     "depth": 0
                 })
    
    state.cached_context_rows = visible_rows
    state.view_tree_dirty = False

def render_context_manager():
    if not state.show_context_manager:
        return

    imgui.set_next_window_size(imgui.ImVec2(900, 600), imgui.Cond_.first_use_ever)
    opened, state.show_context_manager = imgui.begin("Manage Context", state.show_context_manager)

    if opened:
        if state.context_search_text != state.last_context_search:
            trigger_context_search()
        elif state.view_tree_dirty and state.context_search_text and state.is_scanning:
            trigger_context_search()

        imgui.text("Search:")
        imgui.same_line()
        imgui.set_next_item_width(300)
        changed, state.context_search_text = imgui.input_text("##search", state.context_search_text)
        render_tooltip("Filter files by name.")

        if state.is_searching:
            imgui.same_line()
            imgui.text_colored(STYLE.get_imvec4("queued"), "(Searching...)")

        imgui.same_line()
        _, state.context_flatten_search = imgui.checkbox("Flatten", state.context_flatten_search)
        render_tooltip("Show search results as a flat list.")

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

    update_app_stats()
    checked_count = sum(1 for f in state.cached_sorted_files if state.file_checked.get(f, True) and state.file_exists_cache.get(f, True))

    imgui.same_line()
    imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"  Selected: {checked_count} checked | {state.cached_total_lines} lines | ~{state.cached_total_tokens} tokens | {state.cached_cost_str}")

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

    # Search Mode
    use_flat_search = state.context_search_text and state.context_flatten_search

    if use_flat_search:
        imgui.begin_child("files_list", imgui.ImVec2(0, 0))
        table_flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.resizable | imgui.TableFlags_.scroll_y
        if imgui.begin_table("search_table", 3, table_flags):
            imgui.table_setup_column("Path", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column("Lines", imgui.TableColumnFlags_.width_fixed, 60)
            imgui.table_setup_column("Tokens", imgui.TableColumnFlags_.width_fixed, 70)
            imgui.table_headers_row()
            
            if state.cached_flat_filtered_files is not None:
                filtered = state.cached_flat_filtered_files
            else:
                filtered = state.file_paths
            
            clipper = imgui.ListClipper()
            clipper.begin(len(filtered))
            while clipper.step():
                for i in range(clipper.display_start, clipper.display_end):
                    f = filtered[i]
                    is_selected = f in state.selected_files
                    lines, tokens, _ = get_file_stats(f)
                    
                    imgui.table_next_row()
                    imgui.table_next_column()
                    
                    imgui.push_id(str(f))
                    changed, new_val = imgui.checkbox(f"##sel", is_selected)
                    imgui.pop_id()
                    if changed:
                        toggle_file_selection(f, new_val)
                        if new_val: state.file_checked[f] = True
                    
                    imgui.same_line()
                    imgui.text(str(f))
                    
                    imgui.table_next_column()
                    imgui.text(str(lines))
                    
                    imgui.table_next_column()
                    imgui.text(str(tokens))
            
            imgui.end_table()
        imgui.end_child()
        imgui.columns(1)
        imgui.end()
        return

    # Tree Mode Filters
    search_whitelist_dirs = state.cached_whitelist_dirs
    search_whitelist_files = state.cached_whitelist_files

    # Tree Mode
    imgui.begin_child("files_list", imgui.ImVec2(0, 0))

    if imgui.begin_table("files_table", 2, imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.resizable):
        imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch)
        imgui.table_setup_column("Status", imgui.TableColumnFlags_.width_fixed, 80)
        
        # Flatten tree into visible list
        if state.view_tree_dirty or state.cached_context_rows is None:
            _rebuild_context_rows()
            
        visible_rows = state.cached_context_rows

        io = imgui.get_io()
        # Apply Drag Selection
        if not io.key_shift and state.drag_start_idx is not None:
            if state.drag_end_idx is None:
                state.drag_end_idx = state.drag_start_idx

            start = min(state.drag_start_idx, state.drag_end_idx)
            end = max(state.drag_start_idx, state.drag_end_idx)
            target = state.drag_target_state
            
            # Batch updates to avoid frequent saves/stats updates
            modified = False
            
            for k in range(start, min(end + 1, len(visible_rows))):
                item = visible_rows[k]
                if item["type"] == "file":
                    p = to_relative(item["path"])
                    if target:
                        if p not in state.selected_files:
                            state.selected_files.add(p)
                            state.file_checked[p] = True
                            modified = True
                    else:
                        if p in state.selected_files:
                            state.selected_files.discard(p)
                            modified = True
                elif item["type"] == "folder":
                    toggle_folder_selection(item["path"], target) 
                    modified = True
            
            if modified:
                state.stats_dirty = True
                save_fileset()
            
            state.drag_start_idx = None
            state.drag_end_idx = None

        # Virtualize rendering
        row_height = imgui.get_frame_height()
        clipper = imgui.ListClipper()
        clipper.begin(len(visible_rows), row_height)
        
        io = imgui.get_io()
        hover_flags = imgui.HoveredFlags_.allow_when_blocked_by_active_item
        
        while clipper.step():
            for i in range(clipper.display_start, clipper.display_end):
                row = visible_rows[i]
                imgui.table_next_row(0, row_height)
                
                # Handle Drag Highlight
                if state.drag_start_idx is not None and state.drag_end_idx is not None:
                    low = min(state.drag_start_idx, state.drag_end_idx)
                    high = max(state.drag_start_idx, state.drag_end_idx)
                    if low <= i <= high:
                        col = STYLE.get_u32("diff_add", 100) if state.drag_target_state else STYLE.get_u32("diff_del", 100)
                        imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, col)

                imgui.table_next_column()

                # Pre-calculate selection state (for visuals and drag target logic)
                is_selected_row = False
                frel = None
                folder_prefix = None

                if row["type"] == "folder":
                    folder_prefix = str(row["path"])
                    for f in state.selected_files:
                        if str(f.resolve()).startswith(folder_prefix):
                            is_selected_row = True
                            break
                else:
                    frel = to_relative(row["path"])
                    is_selected_row = frel in state.selected_files

                # Row-wide hit detection using invisible selectable
                imgui.push_id(f"row_sel_{i}")
                sel_flags = imgui.SelectableFlags_.span_all_columns | imgui.SelectableFlags_.allow_overlap
                cursor_start = imgui.get_cursor_pos()
                clicked_row = imgui.selectable("##rowHit", False, flags=sel_flags, size=imgui.ImVec2(0, row_height))[0]
                
                if imgui.is_item_hovered(hover_flags) and io.key_shift:
                    if state.drag_start_idx is None:
                        state.drag_start_idx = i
                        state.drag_target_state = not is_selected_row
                    state.drag_end_idx = i
                
                imgui.set_cursor_pos(cursor_start)
                imgui.pop_id()
                
                # Manual indentation
                indent_w = float(row["depth"]) * 20.0
                if indent_w > 0:
                    imgui.indent(indent_w)
                
                if row["type"] == "folder":
                    imgui.push_id(row["key"])
                    
                    imgui.align_text_to_frame_padding()

                    # Mixed State Calculation
                    total_in_folder = row.get("total_files", 0)
                    sel_in_folder = state.folder_selection_counts.get(row["key"], 0)
                    
                    is_mixed = (sel_in_folder > 0) and (sel_in_folder < total_in_folder)
                    display_checked = is_selected_row or is_mixed

                    # We use 'display_checked' for the visual state (False allows standard empty box)
                    # If Mixed, we pass False to checkbox but draw custom rect, so user clicking it sends True -> Select All
                    # Actually passing False means click -> True. Passing True means click -> False.
                    # Behavior: 
                    #  Mixed -> Click -> Select All (True). So pass False.
                    #  All -> Click -> None (False). So pass True.
                    #  None -> Click -> All (True). So pass False.
                    
                    cb_val = True if (sel_in_folder == total_in_folder and total_in_folder > 0) else False

                    changed, val = imgui.checkbox("##f_chk", cb_val)

                    # Custom Mixed Indicator
                    if is_mixed:
                        # Draw a small box inside the checkbox frame
                        # We need previous item rect
                        min_p = imgui.get_item_rect_min()
                        max_p = imgui.get_item_rect_max()
                        
                        sz = max_p.x - min_p.x
                        pad = sz * 0.25
                        
                        dl = imgui.get_window_draw_list()
                        col = STYLE.get_u32("icon_chk", 150)
                        dl.add_rect_filled(
                            imgui.ImVec2(min_p.x + pad, min_p.y + pad),
                            imgui.ImVec2(max_p.x - pad, max_p.y - pad),
                            col,
                            2.0
                        )

                    if changed:
                         toggle_folder_selection(row["path"], val)
                         if val and folder_prefix:
                             for f in state.file_paths:
                                 if str(f).startswith(folder_prefix):
                                     rel = to_relative(f)
                                     state.file_checked[rel] = True
                    
                    # Prevent row click from firing if we just toggled the checkbox
                    # We can assume if changed=True, we shouldn't expand
                    if changed:
                        clicked_row = False

                    imgui.same_line()
                    
                    flags = 0
                    
                    if search_whitelist_dirs is not None:
                        imgui.set_next_item_open(True, imgui.Cond_.always)
                    else:
                        imgui.set_next_item_open(row["is_open"], imgui.Cond_.always)

                    is_node_open = imgui.tree_node_ex(f"{row['name']}/", flags)
                    
                    if search_whitelist_dirs is None and is_node_open != row["is_open"]:
                         state.folder_states[row["key"]] = is_node_open
                         state.view_tree_dirty = True

                    # Handle Row Click expansion for folders
                    if clicked_row and not io.key_shift and not io.key_ctrl:
                         # Use toggle logic on state directly
                         new_state = not row["is_open"]
                         state.folder_states[row["key"]] = new_state
                         state.view_tree_dirty = True
                    
                    if is_node_open:
                        imgui.tree_pop()
                        
                    imgui.pop_id()
                        
                    imgui.table_next_column()
                    if row["scanning"]:
                        imgui.text_colored(STYLE.get_imvec4("queued"), "Scanning...")

                else:
                    # File
                    fpath = row["path"]
                    msg = row["name"]
                    
                    imgui.push_id(str(fpath))
                    
                    changed, val = imgui.checkbox("", is_selected_row)
                    if changed:
                        toggle_file_selection(frel, not is_selected_row)
                        if not is_selected_row: state.file_checked[frel] = True
                        clicked_row = False # Prevent double toggle
                    
                    imgui.same_line()
                    imgui.text(msg)
                    
                    if clicked_row and not io.key_shift and not io.key_ctrl:
                        toggle_file_selection(frel, not is_selected_row)
                        if not is_selected_row: state.file_checked[frel] = True

                    imgui.pop_id()
                    
                    imgui.table_next_column()

                if indent_w > 0:
                    imgui.unindent(indent_w)

        imgui.end_table()

    imgui.end_child()

    imgui.columns(1)

    imgui.end()

def toggle_maximize_window():
    """Toggle window maximization state."""
    if sys.platform == "win32":
        try:
            import ctypes
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            style_flags = ctypes.windll.user32.GetWindowLongW(hwnd, -16)
            if style_flags & 0x01000000:
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            else:
                ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
        except:
            pass
    elif sys.platform == "linux":
        try:
            subprocess.run(['wmctrl', '-r', ':ACTIVE:', '-b', 'toggle,maximized_vert,maximized_horz'], check=False)
        except Exception:
            pass

def render_menu_bar():
    if imgui.begin_menu("File"):
        if imgui.menu_item("Sessions Manager", "Ctrl+O", False)[0]:
            state.show_sessions_window = True
        if imgui.menu_item("Quicksave", "Ctrl+S", False)[0]:
            quicksave_session()
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
        if imgui.menu_item("Check for Updates", "", False)[0]:
            check_for_updates()
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
        elif sys.platform == "linux":
            try:
                subprocess.run(['wmctrl', '-r', ':ACTIVE:', '-b', 'add,hidden'], check=False)
            except:
                pass

    # --- Button 2: Maximize ---
    imgui.same_line(start_x + button_width, spacing)
    imgui.set_cursor_pos_y(centering_y)
    if imgui.button("##max", imgui.ImVec2(button_width, text_height)):
        toggle_maximize_window()

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

    # Double-click empty space to toggle maximize
    if imgui.is_window_hovered() and not imgui.is_any_item_hovered() and imgui.is_mouse_double_clicked(imgui.MouseButton_.left):
        toggle_maximize_window()

def main_gui():
    """Main GUI function called each frame."""
    state.frame_count += 1
    
    # Poll auth (throttled)
    if state.frame_count % 30 == 0:
        poll_auth_request()

    io = imgui.get_io()
    if io.key_ctrl:
        if imgui.is_key_pressed(imgui.Key.o):
            state.show_sessions_window = not state.show_sessions_window
        if imgui.is_key_pressed(imgui.Key.s):
            quicksave_session()
        if imgui.is_key_pressed(imgui.Key.r):
            try_exit_app("restart")

    process_queue()

    is_generating = any(s.is_generating for s in state.sessions.values())
    is_busy = is_generating or bool(state.impl_queue)

    if not is_busy:
        state.queue_start_time = 0.0
    elif state.queue_start_time == 0.0:
        state.queue_start_time = time.time()

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
    render_update_popup()
    render_auth_popup()

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
    s3.ratio = 0.50
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

    check_update_result()

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
    runner_params.app_window_params.window_geometry.size = (1260, 945)
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
