"""Sessions rendering logic."""
import os
import json
from datetime import datetime
from imgui_bundle import imgui

from core import get_available_backups, restore_git_backup, open_diff_report, backup_manager
from application_state import (
    state, save_state, load_state, load_individual_session, delete_save, get_saves_list, SESSIONS_DIR, log_message
)
from styles import STYLE
from .common import render_tooltip


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
            from application_state import refresh_project_files
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