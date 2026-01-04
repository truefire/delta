"""Chat rendering logic."""
import time
from pathlib import Path
from imgui_bundle import imgui

import core
from core import build_system_message, estimate_tokens, calculate_input_cost, open_diff_report, AVAILABLE_MODELS
from application_state import state, create_session, log_message, quicksave_session
from widgets import ChatBubble, DiffViewer, DiffHunk, draw_status_icon
from styles import STYLE
from .common import (
    submit_prompt, submit_plan, submit_filedig, cancel_generation, 
    ensure_user_bubble, unqueue_session, cancel_all_tasks,
    render_tooltip, start_generation
)


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

    style = imgui.get_style()
    window_padding = style.window_padding.x
    window_width = imgui.get_window_width()

    # Calculate right-side controls widths first to determine tab area
    qs_btn_text = "Quicksave Tabs"
    qs_w = imgui.calc_text_size(qs_btn_text).x + style.frame_padding.x * 2.0

    sys_btn_text = "Hide System" if state.show_system_prompt else "Show System"
    sys_w = imgui.calc_text_size(sys_btn_text).x + style.frame_padding.x * 2.0
    
    cancel_btn_text = "Cancel All"
    cancel_w = 0
    has_active = any(s.is_generating for s in state.sessions.values())
    if state.impl_queue or has_active:
        cancel_w = imgui.calc_text_size(cancel_btn_text).x + style.frame_padding.x * 2.0
        
    total_right_w = qs_w + style.item_spacing.x + sys_w
    if cancel_w > 0:
        total_right_w += cancel_w + style.item_spacing.x

    q_timer_text = ""
    if state.queue_start_time > 0.0:
        dur = time.time() - state.queue_start_time
        if dur > 0:
            q_timer_text = f"{dur:.1f}s"

    if q_timer_text:
        tw = imgui.calc_text_size(q_timer_text).x + style.frame_padding.x * 2
        total_right_w += tw + style.item_spacing.x

    total_right_w += 20 # Padding

    avail_w_for_tabs = window_width - total_right_w - window_padding
    if avail_w_for_tabs < 100: avail_w_for_tabs = 100

    # Custom Scrollbar Logic (Top of tabs)
    sb_height = style.scrollbar_size * 0.5
    
    if not hasattr(render_chat_panel, "scroll_state"):
        render_chat_panel.scroll_state = {"x": 0.0, "max": 0.0, "target": None, "last_active": -1, "destination": None}
    s_state = render_chat_panel.scroll_state

    # Ensure destination key exists if state persisted across reload
    if "destination" not in s_state: s_state["destination"] = None

    force_scroll_active = False
    if s_state.get("last_active") != state.active_session_id:
        force_scroll_active = True
        s_state["last_active"] = state.active_session_id

    # Draw Scrollbar if needed
    if s_state["max"] > 0:
        # Handle manual scrollbar target (snap)
        if s_state["target"] is not None:
            s_state["destination"] = s_state["target"]
            imgui.set_next_window_scroll(imgui.ImVec2(s_state["target"], 0))
            s_state["target"] = None
        # Handle smooth animation
        elif s_state["destination"] is not None:
            curr = s_state["x"]
            dest = s_state["destination"]
            
            # Clamp destination to max to prevent stuck scrolling loop if layout changed
            eff_dest = min(dest, s_state["max"])
            
            if abs(curr - eff_dest) < 1.0:
                imgui.set_next_window_scroll(imgui.ImVec2(eff_dest, 0))
                # Only clear destination if we reached the true target
                if abs(eff_dest - dest) < 1.0:
                    s_state["destination"] = None
            else:
                # Lerp
                new_x = curr + (eff_dest - curr) * 0.2
                if abs(new_x - eff_dest) < 1.0: new_x = eff_dest
                imgui.set_next_window_scroll(imgui.ImVec2(new_x, 0))

        p_min = imgui.get_cursor_screen_pos()
        imgui.invisible_button("##top_scrollbar", imgui.ImVec2(avail_w_for_tabs, sb_height))
        
        dl = imgui.get_window_draw_list()
        p_max = imgui.ImVec2(p_min.x + avail_w_for_tabs, p_min.y + sb_height)
        
        # Track
        dl.add_rect_filled(p_min, p_max, STYLE.get_u32("bg", 0.3), sb_height * 0.5)

        # Thumb
        window_w = avail_w_for_tabs
        content_w = window_w + s_state["max"]
        thumb_ratio = window_w / content_w
        thumb_size = max(20.0, window_w * thumb_ratio)
        
        scroll_ratio = s_state["x"] / s_state["max"]
        track_len = window_w - thumb_size
        thumb_start = p_min.x + (track_len * scroll_ratio)
        
        thumb_col_k = "scrollbar"
        if imgui.is_item_active():
            thumb_col_k = "sel_bg"
            delta = imgui.get_io().mouse_delta.x
            if delta != 0 and track_len > 0:
                s_state["target"] = s_state["x"] + (delta * (s_state["max"] / track_len))
                s_state["target"] = max(0.0, min(s_state["target"], s_state["max"]))
        elif imgui.is_item_hovered():
            thumb_col_k = "fg_dim"

        dl.add_rect_filled(
            imgui.ImVec2(thumb_start, p_min.y),
            imgui.ImVec2(thumb_start + thumb_size, p_max.y),
            STYLE.get_u32(thumb_col_k),
            sb_height * 0.5
        )
    else:
        render_chat_panel.scroll_state["target"] = None

    imgui.begin_child("TabRegion", imgui.ImVec2(avail_w_for_tabs, tab_size), child_flags=imgui.ChildFlags_.none, window_flags=imgui.WindowFlags_.horizontal_scrollbar | imgui.WindowFlags_.no_scrollbar)
    
    # Update state from child properties
    s_state["x"] = imgui.get_scroll_x()
    s_state["max"] = imgui.get_scroll_max_x()

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

        if is_selected and force_scroll_active:
            # Calculate target scroll to center this item
            item_min = imgui.get_item_rect_min()
            item_max = imgui.get_item_rect_max()
            
            # Window info (TabRegion)
            win_min = imgui.get_window_pos()
            win_w = imgui.get_window_width()
            
            c_item = (item_min.x + item_max.x) * 0.5
            c_win = win_min.x + (win_w * 0.5)
            
            # Offset needed to center item
            offset = c_item - c_win
            
            # Current scroll is s_state["x"]
            calc_dest = s_state["x"] + offset
            calc_dest = max(0.0, calc_dest)
            
            s_state["destination"] = calc_dest

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
            if session.is_queued:
                imgui.push_style_color(imgui.Col_.text, STYLE.get_imvec4("queued"))
                if imgui.menu_item("Start now (Beware Race Conditions)", "", False)[0]:
                    unqueue_session(session_id)
                    start_generation(session_id)
                imgui.pop_style_color()
                imgui.separator()

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

    # Draw gradients for scroll indicating
    if s_state["max"] > 0:
        p_min = imgui.get_window_pos()
        p_max = imgui.ImVec2(p_min.x + imgui.get_window_width(), p_min.y + imgui.get_window_height())
        
        # Use window background for fade color
        c = imgui.get_style_color_vec4(imgui.Col_.window_bg)
        col_opa = imgui.get_color_u32(c)
        col_tra = imgui.get_color_u32(imgui.ImVec4(c.x, c.y, c.z, 0.0))
        
        grad_w = 30.0
        dl = imgui.get_window_draw_list()
        
        if s_state["x"] > 0:
            dl.add_rect_filled_multi_color(
                p_min,
                imgui.ImVec2(p_min.x + grad_w, p_max.y),
                col_opa, col_tra, col_tra, col_opa
            )
            
        if s_state["x"] < s_state["max"]:
            dl.add_rect_filled_multi_color(
                imgui.ImVec2(p_max.x - grad_w, p_min.y),
                p_max,
                col_tra, col_opa, col_opa, col_tra
            )

    imgui.end_child()

    # Draw Right-Side Controls
    imgui.same_line()
    
    # Adjust cursor to ensure right-alignment
    target_x = window_width - total_right_w + 20 - window_padding
    if imgui.get_cursor_pos_x() < target_x:
        imgui.set_cursor_pos_x(target_x)

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

    # Defer closures
    from application_state import close_session
    for sid in sessions_to_close:
        close_session(sid)

    imgui.separator()

    session = state.sessions.get(state.active_session_id)
    if session:
        render_chat_session(session)


def render_chat_session(session):
    avail = imgui.get_content_region_avail()
    
    # Fixed height overhead
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
        if session.request_start_time > 0:
            checked_files = sorted(session.sent_files) if session.sent_files else []
        else:
            checked_files = sorted([str(f) for f in state.selected_files if state.file_checked.get(f, True)])
        
        # Calculate key
        file_state_key = []
        for f in checked_files:
            try:
                mtime = Path(f).stat().st_mtime
            except Exception:
                mtime = 0
            file_state_key.append((f, mtime))

        current_key = (tuple(file_state_key), session.is_ask_mode, session.is_planning, core.config.extra_system_prompt)
        
        if state.cached_sys_key != current_key or state.cached_sys_bubble is None:
            sys_msg = build_system_message(
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
                    
                    # Manually create DiffViewer
                    dv = DiffViewer(content="", block_state={"collapsed": True}, viewer_id=f"preview_{i}", filename_hint=fpath)
                    
                    # Patch State
                    dv.state.filename = fpath
                    dv.state.is_creation = True
                    dv.state.suppress_new_label = True
                    dv.state.hunks = [DiffHunk(type="change", old="", new=content)]
                    dv.state.change_indices = [0]
                    
                    state.cached_sys_bubble.pre_viewers.append(dv)
                    
                except Exception as e:
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
            imgui.set_tooltip("Agentic search for files")

        if imgui.begin_popup_context_item("filedig_ctx"):
            if imgui.menu_item("Filedig -> Plan", "", False)[0]:
                submit_filedig(is_planning=True)
            if imgui.menu_item("Filedig -> Ask", "", False)[0]:
                submit_filedig(ask_mode=True)
            imgui.end_popup()


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

    # Context calculations
    context_tokens = 0
    if not session.is_filedig:
        if session.request_start_time > 0 and not session.is_queued:
            # Show actual context used for this request
            checked_files = [Path(f) for f in session.sent_files]
        elif session.forced_context_files is not None:
            # Show context that will be used (e.g. from filedig)
            checked_files = [Path(f) for f in session.forced_context_files]
        else:
            # Show current selection
            checked_files = [f for f in state.selected_files if state.file_checked.get(f, True) and state.file_exists_cache.get(f, True)]

        context_tokens = sum(core.get_file_stats(f)[1] for f in checked_files)

    prompt_tokens = estimate_tokens(session.input_text)
    
    output_tokens = 0
    if session.bubbles:
        last_b = session.bubbles[-1]
        if last_b.role == "assistant":
            output_tokens = estimate_tokens(last_b.content)

    total_tokens = context_tokens + prompt_tokens + output_tokens

    model_list = list(AVAILABLE_MODELS.keys())
    model_name = model_list[state.model_idx] if state.model_idx < len(model_list) else ""
    _, price_str = calculate_input_cost(total_tokens, model_name)

    status_parts = []
    status_parts.append(f"Context: ~{context_tokens}")
    status_parts.append(f"Prompt: ~{prompt_tokens}")
    if output_tokens > 0:
        status_parts.append(f"Output: ~{output_tokens}")
    status_parts.append(f"Total: ~{total_tokens}")
    status_parts.append(f"Est: {price_str}")

    imgui.text_colored(STYLE.get_imvec4("fg_dim"), " | ".join(status_parts))

    # Task Timer (same line)
    if session.execution_start_time is not None:
        t_now = time.time()
        t_end = t_now
        if session.request_end_time > 0:
            t_end = session.request_end_time
        
        t_dur = max(0.0, t_end - session.execution_start_time)
        
        if t_dur > 0 or session.is_generating:
            run_str = f"Task: {t_dur:.1f}s"
            window_w = imgui.get_window_width()
            t_w = imgui.calc_text_size(run_str).x
            style = imgui.get_style()
            
            target_x = window_w - t_w - style.window_padding.x - 5
            # Ensure we don't draw over the status text
            if target_x > imgui.get_cursor_pos_x() + 10:
                imgui.same_line(target_x)
                imgui.text_colored(STYLE.get_imvec4("fg_dim"), run_str)