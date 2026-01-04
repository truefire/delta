"""Main GUI runner."""
import sys
import time
from pathlib import Path
from imgui_bundle import imgui, hello_imgui, immapp

import core
from core import APP_DATA_DIR, config
from application_state import (
    state, init_app_state, create_session, load_prompt_history, load_cwd_history, 
    load_presets, refresh_project_files, load_fileset, add_to_cwd_history, load_state, 
    sync_settings_from_config, save_state, SESSIONS_DIR
)
from styles import STYLE, apply_imgui_theme

# Panels
from .common import process_queue, poll_auth_request, check_update_result
from .files_panel import render_files_panel, render_context_manager
from .chat_panel import render_chat_panel
from .settings_panel import render_settings_panel, open_api_settings, render_system_prompt_popup, render_api_settings_popup
from .sessions_panel import render_sessions_window, render_backup_history_window
from .logs_panel import render_logs_panel
from .menu_bar import render_menu_bar
from .popups import (
    render_git_check_popups, render_close_tab_popup, render_exit_confirmation_popup, 
    render_update_popup, render_auth_popup
)
from .tutorial import render_tutorial

def create_docking_layout() -> tuple:
    splits = []

    s1 = hello_imgui.DockingSplit()
    s1.initial_dock = "MainDockSpace"
    s1.new_dock = "SidebarSpace"
    s1.direction = imgui.Dir.right
    s1.ratio = 0.3
    splits.append(s1)

    s2 = hello_imgui.DockingSplit()
    s2.initial_dock = "MainDockSpace"
    s2.new_dock = "LogsSpace"
    s2.direction = imgui.Dir.down
    s2.ratio = 0.2
    splits.append(s2)

    s3 = hello_imgui.DockingSplit()
    s3.initial_dock = "SidebarSpace"
    s3.new_dock = "FilesSpace"
    s3.direction = imgui.Dir.down
    s3.ratio = 0.48
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


def main_gui():
    """Main GUI function called each frame."""
    state.frame_count += 1

    # First run check
    if state.frame_count == 10:
        print("First run detected. Checking for API key...")
        if not core.API_KEY and "openrouter" in core.API_BASE_URL:
            print("No API key found. Please set one in the settings.")
            # Enforce dark theme on first boot for consistency
            if config.theme != "dark":
                config.set_theme("dark")
                STYLE.load("dark")
                apply_imgui_theme(STYLE.dark)
            open_api_settings()

    # Poll auth (throttled)
    if state.frame_count % 30 == 0:
        poll_auth_request()

    io = imgui.get_io()
    if io.key_ctrl:
        if imgui.is_key_pressed(imgui.Key.o):
            state.show_sessions_window = not state.show_sessions_window
        if imgui.is_key_pressed(imgui.Key.s):
            from .common import quicksave_session
            quicksave_session()
        if imgui.is_key_pressed(imgui.Key.r):
            from .popups import try_exit_app
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
    render_api_settings_popup()
    
    render_tutorial()


def run_gui():
    """Run the GUI application."""
    init_app_state()

    from application_state import setup_logging
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