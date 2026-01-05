"""Shared logic and queue handling for the GUI."""
import threading
import time
import queue
import json
from pathlib import Path
from datetime import datetime

from imgui_bundle import imgui

import core
from core import (
    process_request, parse_plan, validate_files, init_git_repo, is_git_installed, is_git_repo, get_file_stats
)
import application_state
from application_state import (
    state, create_session, get_active_session, close_session, unqueue_session,
    save_state, save_prompt_history, save_fileset,
    log_message, refresh_project_files, to_relative,
    state as app_state
)
from widgets import ChatBubble, draw_status_icon
from styles import STYLE
from window_helpers import yank_window, flash_screens


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
    # Imported here to avoid circular dependencies with other parts of the system if needed
    # (Though most imports are top-level)
    from core import config, open_diff_report

    event_type = event.get("type")
    session_id = event.get("session_id")
    session = state.sessions.get(session_id) if session_id else get_active_session()

    if event_type == "text" and session:
        # Streaming text from LLM
        text = event.get("content", "")
        if session.current_bubble is None:
            # Auto-create bubble if missing
            bubble = ChatBubble("assistant", len(session.bubbles))
            session.bubbles.append(bubble)
            session.current_bubble = bubble
            save_state("autosave")

        session.current_bubble.update(text)

    elif event_type == "flush" and session:
        if session.current_bubble:
            session.current_bubble.flush()

    elif event_type == "log_entry":
        state.logs.append({
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

        # Focus Logic
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

        # Queue Logic
        # Check if ANY implementation session is still generating
        tasks_running = False
        for sess in state.sessions.values():
            if sess.is_generating and not sess.is_ask_mode:
                tasks_running = True
                break

        if state.impl_queue and not tasks_running and not state.queue_blocked:
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

    elif event_type == "dig_update" and session:
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
    
    elif event_type == "dig_success":
        # Handle successful dig - spawn new session
        found_files = event.get("files", [])
        explanation = event.get("explanation", "")
        prompt = event.get("prompt", "")
        tool_calls = event.get("tool_calls", 0)
        
        # 0. Grouping logic: Ensure dig session has a group, and new session joins it
        dig_sess = state.sessions.get(session_id)
        target_group_id = None
        target_planning = False
        target_ask = False

        if dig_sess:
            dig_sess.completed = True
            dig_sess.failed = False

            # Add completion bubble
            done_bubble = ChatBubble("assistant", len(dig_sess.bubbles))
            done_bubble.update(f"Completed in {tool_calls} toolcall(s).")
            dig_sess.bubbles.append(done_bubble)

            if dig_sess.group_id is None:
                dig_sess.group_id = state.next_group_id
                state.next_group_id += 1
            target_group_id = dig_sess.group_id
            
            target_planning = dig_sess.is_planning
            target_ask = dig_sess.is_ask_mode
            
        # 2. Prepare context files for Run session (do NOT add to global selection)
        valid_files = []
        for f in found_files:
            p = Path(f)
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
        
        full_prompt = f"{prompt}\n\n(Context found via Digging:\n{explanation})"
        new_sess.input_text = ""
        new_sess.last_prompt = full_prompt
        
        _submit_common(new_sess, full_prompt, is_planning=target_planning, ask_mode=target_ask, save_to_history=False, high_priority=True)


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
    
    if session.is_no_context:
        checked_files = []
    elif session.forced_context_files is not None:
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
            if not session.is_dig:
                state.gui_queue.put({"type": "status", "message": msg.rstrip()})
            else:
                state.gui_queue.put({"type": "dig_update", "session_id": session_id, "message": msg.rstrip()})

        def stream_func(text: str, end: str = "", flush: bool = False):
            state.gui_queue.put({"type": "text", "session_id": session_id, "content": text})

        if session.is_dig:
            from core import run_dig_agent
            result = run_dig_agent(prompt, output_func, cancel_event, history=session.history)
            if result.get("success"):
                state.gui_queue.put({
                    "type": "dig_success",
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
                
                # Block until approval/rejection
                session.approval_event.clear()
                session.approval_event.wait()
                
                session.waiting_for_approval = False
                return session.approval_result
            return True

        file_strs = [str(f) for f in files]
        state.gui_queue.put({"type": "status", "message": f"Starting with {len(file_strs)} files..."})

        # Process Request
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
            no_context=session.is_no_context,
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

    except core.CancelledError:
        state.gui_queue.put({"type": "status", "message": "Generation cancelled"})
    except Exception as e:
        import traceback
        
        err_msg = str(e)
        if hasattr(e, "body") and e.body:
            err_msg += f"\n\nAPI Body: {e.body}"
            
        tb = traceback.format_exc()
        state.gui_queue.put({"type": "error", "session_id": session_id, "message": f"{err_msg}\n{tb}"})
        state.gui_queue.put({"type": "status", "message": f"Error: {e}"})
    finally:
        state.gui_queue.put({"type": "end_response", "session_id": session_id})
        state.gui_queue.put({"type": "done", "session_id": session_id})


def _submit_common(session, prompt: str, is_planning: bool = False, ask_mode: bool = False, is_dig: bool = False, save_to_history: bool = True, high_priority: bool = False, no_context: bool = False):
    """Common submission logic for prompt and plan."""
    from core import MAX_PROMPT_HISTORY

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
    session.is_dig = is_dig
    session.is_no_context = no_context
    state.prompt_history_idx = -1

    ensure_user_bubble(session, session.last_prompt)

    if session.is_ask_mode:
        start_generation(session.id)
    elif state.current_impl_sid is not None or state.queue_blocked or state.impl_queue:
        session.is_queued = True
        
        if high_priority:
            state.impl_queue.insert(0, session.id)
        else:
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


def submit_prompt(ask_mode: bool = False, no_context: bool = False):
    """Submit the current prompt."""
    session = get_active_session()
    if not session:
        return

    prompt = session.input_text.strip()
    if not prompt:
        return

    if not no_context and not check_missing_files(session, prompt, is_planning=False, ask_mode=ask_mode):
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

    _submit_common(session, prompt, is_planning=False, ask_mode=ask_mode, no_context=no_context)


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


def submit_dig(is_planning=False, ask_mode=False):
    """Submit a dig request."""
    session = get_active_session()
    if not session: return
    
    prompt = session.input_text.strip()
    if not prompt: return
    
    # Dig doesn't care about current files, it finds them.
    _submit_common(session, prompt, is_planning=is_planning, ask_mode=ask_mode, is_dig=True)


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

    # Insert into beginning of queue
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


# --- Update & Auth Utilities ---

def _perform_git_pull():
    state.update_in_progress = True
    state.update_status = "Pulling changes from remote..."
    
    def worker():
        try:
            askpass_script = core.create_askpass_wrapper()
            
            env = core.os.environ.copy()
            env["GIT_ASKPASS"] = askpass_script
            env["SSH_ASKPASS"] = askpass_script
            env["SSH_ASKPASS_REQUIRE"] = "force"
            env["DISPLAY"] = ":0" # Fake display to trick SSH
            
            # Use subprocess directly
            tool_dir = core._TOOL_DIR
            proc = core.subprocess.run(
                ["git", "pull"],
                cwd=tool_dir,
                env=env,
                stdout=core.subprocess.PIPE,
                stderr=core.subprocess.STDOUT,
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
    import sys
    import subprocess
    from imgui_bundle import hello_imgui

    # Windows-specific handling
    if sys.platform == "win32":
        try:
            ipc_dir = core.APP_DATA_DIR / "ipc"
            ipc_dir.mkdir(parents=True, exist_ok=True)
            updater_script = ipc_dir / "update_runner.bat"
            log_file = ipc_dir / "update_log.txt"
            result_file = ipc_dir / "update_result.txt"

            if log_file.exists(): log_file.unlink()
            if result_file.exists(): result_file.unlink()
            
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

            subprocess.Popen(
                [str(updater_script)],
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                shell=True
            )
            
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
    import sys
    import shutil
    tool_dir = core._TOOL_DIR
    
    # 1. Check Git
    if (tool_dir / ".git").exists():
        ok, out = core.run_command("git remote -v", cwd=tool_dir)
        if ok and "origin" in out:
            ok_f, _ = core.run_command("git fetch", cwd=tool_dir)
            ok_s, status_out = core.run_command("git status -uno", cwd=tool_dir)

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


def poll_auth_request():
    """Check for authentication requests from background processes."""
    if state.show_auth_popup or state.auth_request_data is not None:
        return # Already showing

    ipc_dir = core.APP_DATA_DIR / "ipc"
    if not ipc_dir.exists():
        return

    try:
        req_files = list(ipc_dir.glob("*.req"))
        if req_files:
            req_file = req_files[0]
            try:
                with open(req_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                state.auth_request_data = data
                state.auth_request_data["_file"] = req_file
                state.auth_input_text = ""
                state.show_auth_popup = True
            except Exception:
                try: req_file.unlink()
                except: pass
    except Exception:
        pass