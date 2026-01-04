"""Menu bar logic."""
import sys
import subprocess
from imgui_bundle import imgui, hello_imgui

from application_state import state, quicksave_session
from .popups import try_exit_app
from .settings_panel import toggle_theme, open_api_settings
from .common import check_for_updates

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

    if imgui.begin_menu("Help"):
        if imgui.menu_item("Tutorial", "", False)[0]:
            from gui.tutorial import start_tutorial
            start_tutorial()
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