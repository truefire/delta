"""Logs panel logic."""
import time
from datetime import datetime
from imgui_bundle import imgui
from application_state import state
from styles import STYLE


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