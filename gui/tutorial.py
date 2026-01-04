"""Tutorial module for Delta Tool."""
from imgui_bundle import imgui
from styles import STYLE
from application_state import state
from core import config

class TutorialState:
    def __init__(self):
        self.is_active = False
        self.step = 0
        self.steps = [
            {
                "title": "Settings Panel",
                "text": "You can configure a lot of options here, most notably your model.",
                "area": "settings"
            },
            {
                "title": "Files & Context",
                "text": "This is where you manage the files the LLM can see. One of the first things you'll want to do is open up the Manage Context window and select some files.",
                "area": "files"
            },
            {
                "title": "Prompt & Execution",
                "text": "Most of your work will be done by typing a prompt here and clicking RUN or ASK",
                "area": "input_composite"
            },
            {
                "title": "Utilities",
                "text": "PLAN will break your request down into sub-tasks. FILEDIG will automatically discover context for you. These tools are less reliable than a vanilla RUN task, but they solve certain use cases.",
                "area": "utility_buttons"
            },
            {
                "title": "Sessions",
                "text": "You can switch between tabs here. Each tab is its own isolated conversation. Middle-click a tab to close it.",
                "area": "tabs"
            },
            {
                "title": "Context Menus",
                "text": "Many buttons have context menus with additional functionality when you right click them. Most importantly, tabs and the CHANGE CWD button will allow you to see your history on right click.",
                "area": "context_targets"
            },
            {
                "title": "Get Started",
                "text": "That's it for the tutorial. Remember: To get started, click Manage Context, select some files, type a prompt, press RUN.",
                "area": "center"
            }
        ]

_tutorial = TutorialState()

def start_tutorial():
    _tutorial.is_active = True
    _tutorial.step = 0

def stop_tutorial():
    _tutorial.is_active = False
    config.set_has_seen_tutorial(True)

def render_tutorial():
    # 1. Handle Offer Popup
    if state.show_tutorial_offer:
        imgui.open_popup("View Tutorial?")
        state.show_tutorial_offer = False

    if imgui.begin_popup_modal("View Tutorial?", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text("Welcome to Delta Tool!")
        imgui.text("Would you like to view a short tutorial (~1 min) on the interface?")
        imgui.separator()

        if imgui.button("Yes, show me", imgui.ImVec2(120, 0)):
            imgui.close_current_popup()
            start_tutorial()

        imgui.same_line()
        if imgui.button("No, thanks", imgui.ImVec2(120, 0)):
            config.set_has_seen_tutorial(True)
            imgui.close_current_popup()

        imgui.end_popup()

    # 2. Render Main Tutorial Overlay
    if not _tutorial.is_active:
        return

    # Full screen transparent window to block interactions with app
    viewport = imgui.get_main_viewport()
    viewport_pos = viewport.pos
    viewport_size = viewport.size

    imgui.set_next_window_pos(viewport_pos)
    imgui.set_next_window_size(viewport_size)

    flags = (imgui.WindowFlags_.no_decoration |
             imgui.WindowFlags_.no_move |
             imgui.WindowFlags_.no_resize |
             imgui.WindowFlags_.no_saved_settings |
             imgui.WindowFlags_.no_nav |
             imgui.WindowFlags_.no_bring_to_front_on_focus |
             imgui.WindowFlags_.no_background)

    if imgui.begin("TutorialOverlay", None, flags):
        draw_list = imgui.get_window_draw_list()
        fg_draw_list = imgui.get_foreground_draw_list()

        # Dimming
        draw_list.add_rect_filled(
            viewport_pos,
            imgui.ImVec2(viewport_pos.x + viewport_size.x, viewport_pos.y + viewport_size.y),
            imgui.get_color_u32(imgui.ImVec4(0, 0, 0, 0.45))
        )

        # Calculate Highlights based on standard layout ratios
        # Sidebar: 30% width (Left)
        #   Files: Top 48% height of Sidebar
        #   Settings: Bottom 52% of Sidebar (SidebarSpace)
        # Main: Remaining 70% width
        #   Logs: Bottom 20% height
        #   Chat: Remaining (Top 80% of Main)
        #     Tabs: Very top ~35px
        #     Input: Bottom ~100px of Chat

        w = viewport_size.x
        h = viewport_size.y
        x = viewport_pos.x
        y = viewport_pos.y

        # Layout Logic to match runner.py Splits
        # Main (Left 70%) | Sidebar (Right 30%)
        # Main -> Chat (Top 80%) / Logs (Bottom 20%)
        # Sidebar -> Settings (Top 52%) / Files (Bottom 48%)

        main_w = w * 0.7
        sidebar_w = w * 0.3
        
        c1_x = x
        c2_x = x + main_w
        
        chat_h = h * 0.8
        files_h = h * 0.48
        settings_h = h * 0.52

        pad = 10
        target_rects = []
        area = _tutorial.steps[_tutorial.step]["area"]

        if area == "settings":
            # Right Column Top
            target_rects.append((c2_x + pad, y + pad, c2_x + sidebar_w - pad, y + settings_h - pad))
        
        elif area == "files":
            # Right Column Bottom
            target_rects.append((c2_x + pad, y + settings_h + pad, c2_x + sidebar_w - pad, y + h - pad))
        
        elif area == "input_composite":
            # Bottom of Chat Area (Left Col)
            bottom = y + chat_h
            # Input (~100) + Buttons (~40)
            target_rects.append((c1_x + pad, bottom - 150, c1_x + main_w - pad, bottom - pad))
            
        elif area == "utility_buttons":
            # Button Row Only
            bottom = y + chat_h
            target_rects.append((c1_x + pad, bottom - 60, c1_x + main_w - pad, bottom - 20))
            
        elif area == "tabs":
            # Top of Chat Area (Left Col) - approx under title bar
            target_rects.append((c1_x + pad, y + 30, c1_x + main_w - pad, y + 70))
            
        elif area == "context_targets":
            # 1. Tabs
            target_rects.append((c1_x + pad, y + 30, c1_x + main_w - pad, y + 70))
            # 2. Settings Top (CWD)
            target_rects.append((c2_x + pad, y + 30, c2_x + sidebar_w - pad, y + 70))
            # 3. Buttons
            bottom = y + chat_h
            target_rects.append((c1_x + pad, bottom - 60, c1_x + main_w - pad, bottom - 20))

        # Red Box Color
        red_col = imgui.get_color_u32(imgui.ImVec4(1.0, 0.2, 0.2, 1.0))

        for (x1, y1, x2, y2) in target_rects:
            # Draw highlight box
            fg_draw_list.add_rect(
                imgui.ImVec2(x1, y1),
                imgui.ImVec2(x2, y2),
                red_col,
                5.0,
                0,
                4.0
            )

    imgui.end()

    # 3. Render Modal Dialog
    imgui.open_popup("TutorialStep")

    # Center the modal
    imgui.set_next_window_pos(viewport.get_center(), imgui.Cond_.always, imgui.ImVec2(0.5, 0.5))
    imgui.set_next_window_size(imgui.ImVec2(400, 0))

    if imgui.begin_popup_modal("TutorialStep", None,
                               imgui.WindowFlags_.no_resize | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_title_bar)[
        0]:
        step_data = _tutorial.steps[_tutorial.step]

        # Header
        imgui.text_colored(STYLE.get_imvec4("sel_bg"),
                           f"Step {_tutorial.step + 1} / {len(_tutorial.steps)}")
        imgui.same_line()
        imgui.text(f"| {step_data['title']}")
        imgui.separator()
        imgui.dummy(imgui.ImVec2(0, 10))

        # Body
        imgui.push_text_wrap_pos(0.0)
        imgui.text(step_data["text"])
        imgui.pop_text_wrap_pos()

        imgui.dummy(imgui.ImVec2(0, 20))
        imgui.separator()

        # Controls
        if _tutorial.step > 0:
            if imgui.button("Back", imgui.ImVec2(80, 0)):
                _tutorial.step -= 1
            imgui.same_line()

        is_last = (_tutorial.step == len(_tutorial.steps) - 1)
        btn_txt = "Finish" if is_last else "Next"

        if imgui.button(btn_txt, imgui.ImVec2(80, 0)):
            if is_last:
                imgui.close_current_popup()
                stop_tutorial()
            else:
                _tutorial.step += 1

        imgui.same_line()

        # Right align close
        avail = imgui.get_content_region_avail().x
        imgui.dummy(imgui.ImVec2(avail - 60, 0))
        imgui.same_line()

        if imgui.button("Close"):
            imgui.close_current_popup()
            stop_tutorial()

        imgui.end_popup()