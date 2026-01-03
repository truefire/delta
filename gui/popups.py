"""Common popup renderers."""
from pathlib import Path
import json
from imgui_bundle import imgui, hello_imgui

import core
from core import init_git_repo, is_git_installed, is_git_repo, config
from application_state import (
    state, close_session, unqueue_session, save_state, save_fileset,
    log_message
)
from styles import STYLE
from .common import _submit_common, _resume_submission_after_missing_check, cancel_all_tasks, cancel_generation


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

from application_state import get_active_session

def render_git_check_popups():
    render_missing_files_popup()
    
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
                try:
                    req_path = state.auth_request_data.get("_file")
                    if req_path:
                        req_path.unlink() 
                except: pass
                
                state.auth_request_data = None
                state.auth_input_text = ""
                imgui.close_current_popup()
            
        imgui.end_popup()