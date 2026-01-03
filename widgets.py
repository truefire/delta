"""Custom widgets and UI helpers for Delta Tool using imgui-bundle."""
import difflib
import math
import time
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Callable

from imgui_bundle import imgui, imgui_md

# Optional Dependencies
try:
    import versus
    from versus.interfaces import Myers
    _versus_available = True
except ImportError:
    _versus_available = False
    Myers = None

from core import build_tolerant_regex, parse_diffs
from styles import STYLE


def render_viewer_header(
    collapsed: bool,
    label_func: Callable[[], None],
    right_align_func: Callable[[], None] | None = None
) -> tuple[bool, bool]:
    """Render a standard viewer header with expand/collapse toggle.

    Returns:
        Tuple of (new_collapsed_state, toggled).
    """
    toggled = False
    header_height = 32
    imgui.push_style_color(imgui.Col_.child_bg, STYLE.get_imvec4("diff_head"))
    imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(4, 4))
    
    imgui.begin_child("header", imgui.ImVec2(0, header_height), child_flags=imgui.ChildFlags_.borders, window_flags=imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse)

    # Vertically center content
    content_height = 20
    padding = (header_height - content_height) / 2 - 4
    if padding > 0:
        imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + padding)

    # Expand/collapse arrow
    arrow = ">" if collapsed else "v"
    
    # We rely on the parent pushing an ID for the viewer
    if imgui.button(f"{arrow}##expand", imgui.ImVec2(24, 20)):
        collapsed = not collapsed
        toggled = True

    imgui.same_line()
    label_func()

    if right_align_func:
        right_align_func()

    imgui.end_child()
    imgui.pop_style_var()
    imgui.pop_style_color()

    return collapsed, toggled


@dataclass
class DiffHunk:
    """Represents a single diff hunk."""
    type: str  # "context", "change", "skipped"
    text: str = ""
    old: str = ""
    new: str = ""
    content: str = ""  # For skipped hunks, the hidden content
    start_line: int = 0


@dataclass
class DiffViewerState:
    """State for a DiffViewer widget."""
    content: str = ""
    filename: str = "Unknown"
    is_creation: bool = False
    suppress_new_label: bool = False
    hunks: list = field(default_factory=list)
    collapsed: bool = True
    reverted: set = field(default_factory=set)
    current_change_idx: int = 0
    change_indices: list = field(default_factory=list)
    left_scroll_y: float = 0.0
    right_scroll_y: float = 0.0
    id: int = 0


class DiffViewer:
    """Widget to display synchronized side-by-side diffs with navigation."""

    def __init__(self, content: str, block_state: dict, viewer_id: int = 0, filename_hint: str | None = None, language_hint: str | None = None):
        self.state = DiffViewerState(
            content=content,
            collapsed=block_state.get("collapsed", True),
            reverted=block_state.get("reverted", set()),
            id=viewer_id
        )
        self.block_state = block_state
        self.filename_hint = filename_hint
        self.language_hint = language_hint
        self._cached_height = None
        self._parse_content()

    def _parse_content(self):
        """Parse the diff content into hunks."""
        # Wrap content in dummy code block for parse_diffs logic
        prefix = f"{self.filename_hint}\n" if self.filename_hint else ""
        info = self.language_hint if self.language_hint else ""
        wrapped = f"{prefix}```{info}\n{self.state.content}\n```"
        parsed = parse_diffs(wrapped)

        if not parsed:
            return

        # Infer filename from first hunk (all hunks in block belong to one file)
        self.state.filename = parsed[0]["filename"]
        diff_hunks = [(d["original"], d["new"]) for d in parsed]

        # Try to match against actual file for full context
        full_content_found = False
        if self.state.filename != "Unknown":
            try:
                # parse_diffs already reconciles path
                real_path = Path.cwd() / self.state.filename

                if not real_path.exists():
                    # Check if creation (file missing and search block empty)
                    if diff_hunks and all(orig.strip() == "" for orig, _ in diff_hunks):
                        self.state.is_creation = True

                if real_path.exists() and real_path.is_file():
                    original_file_content = real_path.read_text("utf-8")
                    
                    if diff_hunks:
                        full_hunks = []
                        current_pos = 0
                        all_matched = True

                        for i, (search, replace) in enumerate(diff_hunks):
                            if search == "":
                                all_matched = False
                                break
                            pattern = re.compile(build_tolerant_regex(search))
                            match = pattern.search(original_file_content, current_pos)
                            if match:
                                start, end = match.span()
                                if start > current_pos:
                                    ctx = original_file_content[current_pos:start]
                                    full_hunks.extend(self._compress_context(ctx, i == 0, False))
                                full_hunks.extend(self._refine_diff(original_file_content[start:end], replace))
                                current_pos = end
                            else:
                                all_matched = False
                                break

                        if all_matched:
                            if current_pos < len(original_file_content):
                                ctx = original_file_content[current_pos:]
                                full_hunks.extend(self._compress_context(ctx, False, True))
                            self.state.hunks = full_hunks
                            full_content_found = True
            except Exception:
                pass

        if not full_content_found:
            for original, new in diff_hunks:
                self.state.hunks.extend(self._refine_diff(original, new))

        # Build change indices
        self.state.change_indices = [i for i, h in enumerate(self.state.hunks) if h.type == "change"]
        self._cached_height = None

    def _compress_context(self, text: str, is_first: bool, is_last: bool) -> list:
        """Compress large context sections."""
        lines = text.splitlines(keepends=True)
        limit, keep = 12, 4

        if len(lines) <= limit:
            return [DiffHunk(type="context", text=text)]

        res = []
        if is_first:
            skipped = "".join(lines[:-keep])
            res.append(DiffHunk(type="skipped", text=f"... ({len(lines)-keep} lines skipped) ...\n", content=skipped))
            res.append(DiffHunk(type="context", text="".join(lines[-keep:])))
        elif is_last:
            res.append(DiffHunk(type="context", text="".join(lines[:keep])))
            skipped = "".join(lines[keep:])
            res.append(DiffHunk(type="skipped", text=f"... ({len(lines)-keep} lines skipped) ...\n", content=skipped))
        else:
            mid_skip = len(lines) - 2 * keep
            res.append(DiffHunk(type="context", text="".join(lines[:keep])))
            skipped = "".join(lines[keep:-keep])
            res.append(DiffHunk(type="skipped", text=f"... ({mid_skip} lines skipped) ...\n", content=skipped))
            res.append(DiffHunk(type="context", text="".join(lines[-keep:])))
        return res

    def _refine_diff(self, old_text: str, new_text: str) -> list:
        """Refine diff between old and new text."""
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)
        hunks = []

        if _versus_available:
            try:
                hunks = self._refine_diff_versus(old_lines, new_lines)
            except Exception:
                hunks = []

        if not hunks:
            hunks = self._refine_diff_difflib(old_lines, new_lines)
        return hunks

    def _refine_diff_versus(self, old_lines, new_lines) -> list:
        """Use Myers diff algorithm."""
        hunks = []
        diffs = Myers.diff(old_lines, new_lines)
        current_op = None
        old_buf, new_buf = [], []

        for op, line in diffs:
            if op == 'equal':
                if current_op == 'diff':
                    hunks.append(DiffHunk(type="change", old="".join(old_buf), new="".join(new_buf)))
                    old_buf, new_buf = [], []
                current_op = 'equal'
                old_buf.append(line)
            else:
                if current_op == 'equal':
                    hunks.append(DiffHunk(type="context", text="".join(old_buf)))
                    old_buf = []
                current_op = 'diff'
                if op == 'delete':
                    old_buf.append(line)
                elif op == 'insert':
                    new_buf.append(line)

        if current_op == 'equal':
            hunks.append(DiffHunk(type="context", text="".join(old_buf)))
        elif current_op == 'diff':
            hunks.append(DiffHunk(type="change", old="".join(old_buf), new="".join(new_buf)))
        return hunks

    def _refine_diff_difflib(self, old_lines, new_lines) -> list:
        """Fallback to difflib."""
        hunks = []
        matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                hunks.append(DiffHunk(type="context", text="".join(old_lines[i1:i2])))
            else:
                hunks.append(DiffHunk(type="change", old="".join(old_lines[i1:i2]), new="".join(new_lines[j1:j2])))
        return hunks

    def update_content(self, new_content: str):
        """Update the diff content."""
        if self.state.content == new_content:
            return
        self.state.content = new_content
        self.state.hunks = []
        self._cached_height = None
        self._parse_content()

    def render(self) -> bool:
        """Render the diff viewer. Returns True if clicked to expand/collapse."""
        imgui.push_id(f"diff_{self.state.id}")

        def render_label():
            if self.state.is_creation:
                label = f"{self.state.filename} (New File)" if not self.state.suppress_new_label else self.state.filename
                imgui.text_colored(STYLE.get_imvec4("txt_suc"), label)
            else:
                imgui.text(f"{self.state.filename}")

        def render_right():
            # Apply button
            imgui.same_line(imgui.get_window_width() - 190)
            if imgui.button(f"Reapply##apply_{self.state.id}", imgui.ImVec2(0, 20)):
                try:
                    import core
                    import application_state
                    
                    # Reconstruct wrappable content
                    prefix = f"{self.filename_hint}\n" if self.filename_hint else ""
                    info = self.language_hint if self.language_hint else ""
                    wrapped = f"{prefix}```{info}\n{self.state.content}\n```"
                    
                    core.apply_diffs(wrapped)
                    application_state.log_message(f"Manually applied diff to {self.state.filename}")
                    
                    # Refresh
                    core.file_cache.invalidate(self.state.filename)
                    if hasattr(application_state, 'state'):
                         application_state.state.stats_dirty = True
                    
                except Exception as e:
                    import application_state
                    application_state.log_message(f"Failed to apply diff: {e}")

            if imgui.is_item_hovered():
                imgui.set_tooltip("Attempt to apply this diff again")

            # Navigation controls on the right
            num_changes = len(self.state.change_indices)
            if num_changes > 0:
                imgui.same_line(imgui.get_window_width() - 120)
                if self.state.collapsed:
                    imgui.text(f"{num_changes} change{'s' if num_changes != 1 else ''}")
                else:
                    imgui.text(f"{self.state.current_change_idx + 1}/{num_changes}")
                    imgui.same_line()
                    if imgui.button(f"^##prev_{self.state.id}", imgui.ImVec2(20, 20)):
                        self._prev_change()
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Previous change")
                    imgui.same_line()
                    if imgui.button(f"v##next_{self.state.id}", imgui.ImVec2(20, 20)):
                        self._next_change()
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Next change")

        new_collapsed, changed = render_viewer_header(
            self.state.collapsed,
            render_label,
            render_right
        )
        
        if changed:
            self.state.collapsed = new_collapsed
            self.block_state["collapsed"] = self.state.collapsed

        # Content area (if expanded)
        if not self.state.collapsed:
            avail = imgui.get_content_region_avail()
            content_height = min(400, max(100, self._calc_content_height()))

            imgui.push_style_color(imgui.Col_.child_bg, STYLE.get_imvec4("diff_txt"))
            imgui.begin_child("diff_content", imgui.ImVec2(0, content_height), child_flags=imgui.ChildFlags_.borders)

            if self.state.is_creation:
                # Single column for new files
                self._render_pane(is_left=False)
            else:
                # Two columns for side-by-side
                imgui.columns(2, "diff_cols", borders=True)

                # Left pane (old)
                self._render_pane(is_left=True)

                imgui.next_column()

                # Right pane (new)
                self._render_pane(is_left=False)

                imgui.columns(1)
            imgui.end_child()
            imgui.pop_style_color()

        imgui.pop_id()
        return changed

    def _render_pane(self, is_left: bool):
        """Render one side of the diff."""
        imgui.push_id("left" if is_left else "right")

        for i, hunk in enumerate(self.state.hunks):
            imgui.push_id(i)

            if hunk.type == "context":
                # Render context line by line to keep columns in sync
                lines = hunk.text.rstrip('\n').split('\n')
                for line in lines:
                    imgui.text_colored(STYLE.get_imvec4("fg_dim"), line)

            elif hunk.type == "skipped":
                imgui.push_style_color(imgui.Col_.text, STYLE.get_imvec4("fg_dim"))
                if imgui.selectable(f"{hunk.text.rstrip()}##skip_{i}", False)[0]:
                    # Expand skipped section
                    self.state.hunks[i] = DiffHunk(type="context", text=hunk.content)
                imgui.pop_style_color()

            elif hunk.type == "change":
                text = hunk.old if is_left else hunk.new
                other_text = hunk.new if is_left else hunk.old

                # Get line counts for both sides to sync
                lines = text.rstrip('\n').split('\n') if text else []
                other_lines = other_text.rstrip('\n').split('\n') if other_text else []
                max_lines = max(len(lines), len(other_lines), 1)

                # Pad to match line count
                while len(lines) < max_lines:
                    lines.append("")

                draw_list = imgui.get_window_draw_list()
                line_height = imgui.get_text_line_height()
                bg_key = "diff_del" if is_left else "diff_add"

                for line in lines:
                    pos = imgui.get_cursor_screen_pos()
                    draw_list.add_rect_filled(
                        imgui.ImVec2(pos.x - 2, pos.y),
                        imgui.ImVec2(pos.x + imgui.get_content_region_avail().x, pos.y + line_height),
                        STYLE.get_u32(bg_key)
                    )
                    imgui.text(line if line else " ")

            imgui.pop_id()

        imgui.pop_id()

    def _calc_content_height(self) -> float:
        """Calculate appropriate content height."""
        if self._cached_height is not None:
            return self._cached_height

        line_count = 0
        for hunk in self.state.hunks:
            if hunk.type == "context":
                line_count += hunk.text.count('\n') + 1
            elif hunk.type == "skipped":
                line_count += 1
            elif hunk.type == "change":
                line_count += max(hunk.old.count('\n'), hunk.new.count('\n')) + 1
        
        self._cached_height = line_count * imgui.get_text_line_height() + 20
        return self._cached_height

    def _prev_change(self):
        """Navigate to previous change."""
        if not self.state.change_indices:
            return
        self.state.current_change_idx = (self.state.current_change_idx - 1) % len(self.state.change_indices)

    def _next_change(self):
        """Navigate to next change."""
        if not self.state.change_indices:
            return
        self.state.current_change_idx = (self.state.current_change_idx + 1) % len(self.state.change_indices)


@dataclass
class PlanViewerState:
    """State for a PlanViewer widget."""
    title: str = "Unknown Plan"
    prompt: str = ""
    collapsed: bool = True
    id: int = 0


class PlanViewer:
    """Widget to display an implementation plan item."""

    def __init__(self, content: str, block_state: dict, viewer_id: int):
        self.state = PlanViewerState(
            title="Unknown Plan",
            prompt="",
            collapsed=block_state.get("collapsed", True),
            id=viewer_id
        )
        self.block_state = block_state
        self._parse_content(content)

    def _parse_content(self, content: str):
        """Parse the plan content into title and prompt."""
        lines = content.split('\n')
        title = "Untitled"
        prompt_lines = []
        
        mode = "header"
        for line in lines:
            if line.startswith("Title:"):
                title = line[6:].strip()
                mode = "prompt_search"
            elif line.startswith("Prompt:") and (mode == "prompt_search" or mode == "header"):
                prompt_content = line[7:].strip()
                if prompt_content:
                    prompt_lines.append(prompt_content)
                mode = "prompt"
            elif mode == "prompt":
                prompt_lines.append(line)
        
        self.state.title = title
        self.state.prompt = "\n".join(prompt_lines).strip()

    def update_content(self, new_content: str):
        """Update the plan content."""
        self._parse_content(new_content)

    def render(self):
        """Render the plan viewer."""
        imgui.push_id(f"plan_{self.state.id}")

        def render_label():
            # Draw "PLAN" badge
            imgui.text_colored(STYLE.get_imvec4("btn_run"), "[PLAN]")
            imgui.same_line()
            imgui.text(self.state.title)

        new_collapsed, changed = render_viewer_header(self.state.collapsed, render_label)
        
        if changed:
            self.state.collapsed = new_collapsed
            self.block_state["collapsed"] = self.state.collapsed

        # Content area
        if not self.state.collapsed:
            imgui.push_style_color(imgui.Col_.child_bg, STYLE.get_imvec4("bg_cont"))
            
            imgui.begin_child("plan_content", imgui.ImVec2(0, 0), child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y)
            
            imgui.begin_group()
            imgui.push_text_wrap_pos(0.0)
            imgui_md.render(self.state.prompt)
            imgui.pop_text_wrap_pos()
            imgui.end_group()
            
            imgui.end_child()
            imgui.pop_style_color()

        imgui.pop_id()


@dataclass
class ChatMessage:
    """Represents a single chat message."""
    role: str  # "user", "assistant", "system", "error"
    content: str = ""
    diff_viewers: dict = field(default_factory=dict)
    block_states: dict = field(default_factory=dict)
    plan_viewers: dict = field(default_factory=dict)
    plan_states: dict = field(default_factory=dict)
    anchors: list = field(default_factory=list)
    error_data: tuple[str, str] | None = None  # (error_summary, raw_content)


class ChatBubble:
    """Renders a chat message bubble with markdown and embedded diffs."""

    def __init__(self, role: str, message_id: int = 0):
        self.message = ChatMessage(role=role)
        self.message_id = message_id
        self.pre_viewers = []
        self._line_buffer = ""
        self._in_code_block = False
        self._current_block_content = []
        self._current_block_counter = 0
        
        self._in_plan_block = False
        self._current_plan_content = []
        self._current_plan_counter = 0
        self._show_raw = False
        self._filename_candidate = None
        self._current_block_info = None
        self._cached_segments = None
        self._content_dirty = True

    @property
    def role(self) -> str:
        return self.message.role

    @property
    def content(self) -> str:
        return self.message.content

    @property
    def anchors(self) -> list:
        return self.message.anchors

    @property
    def diff_viewers(self) -> dict:
        return self.message.diff_viewers

    def update(self, text: str):
        """Stream text into the bubble."""
        self.message.content += text
        self._content_dirty = True
        self._process_stream(text)

    def flush(self):
        """Flush any remaining buffered content."""
        if self._line_buffer:
            self._process_line(self._line_buffer)
            self._line_buffer = ""
            self._content_dirty = True

    def _process_stream(self, text: str):
        """Process streamed text line by line."""
        self._line_buffer += text
        while '\n' in self._line_buffer:
            line, rest = self._line_buffer.split('\n', 1)
            self._line_buffer = rest
            self._process_line(line)

    def _process_line(self, line: str):
        """Process a single line."""
        stripped = line.strip()

        # Handle Plan Blocks
        if stripped == '<<<<<<< PLAN' and not self._in_code_block:
            self._in_plan_block = True
            self._current_plan_content = []
            return

        if stripped == '>>>>>>> END' and self._in_plan_block:
            # End plan block
            plan_content = "\n".join(self._current_plan_content)
            pv_id = self._current_plan_counter
            
            if pv_id not in self.message.plan_states:
                self.message.plan_states[pv_id] = {"collapsed": False}
                
            pv = PlanViewer(plan_content, self.message.plan_states[pv_id], pv_id)
            self.message.plan_viewers[pv_id] = pv
            
            self._in_plan_block = False
            self._current_plan_content = []
            self._current_plan_counter += 1
            return
            
        if self._in_plan_block:
            self._current_plan_content.append(line)
            # Update existing plan viewer if it exists
            if self._current_plan_counter in self.message.plan_viewers:
                 plan_content = "\n".join(self._current_plan_content)
                 self.message.plan_viewers[self._current_plan_counter].update_content(plan_content)
            return

        # Handle Code Blocks
        if stripped.startswith('```'):
            if self._in_code_block:
                # End code block - check if it's a diff
                block_content = "\n".join(self._current_block_content)
                if "<<<<<<< SEARCH" in block_content:
                    # Create a DiffViewer
                    dv_id = self._current_block_counter
                    if dv_id not in self.message.block_states:
                        self.message.block_states[dv_id] = {"collapsed": True, "reverted": set()}
                    dv = DiffViewer(block_content, self.message.block_states[dv_id], dv_id, self._filename_candidate, self._current_block_info)
                    self.message.diff_viewers[dv_id] = dv

                    # Add anchor
                    self.message.anchors.append({
                        "type": "diff",
                        "label": f"diff: {dv.state.filename}"
                    })

                self._in_code_block = False
                self._current_block_content = []
                self._current_block_counter += 1
                self._filename_candidate = None
                self._current_block_info = None
            else:
                self._in_code_block = True
                self._current_block_content = []
                self._current_block_info = stripped[3:].strip()
        elif self._in_code_block:
            self._current_block_content.append(line)
            # Update existing diff viewer if it exists
            if self._current_block_counter in self.message.diff_viewers:
                block_content = "\n".join(self._current_block_content)
                self.message.diff_viewers[self._current_block_counter].update_content(block_content)
        elif stripped:
            self._filename_candidate = line

    def set_error_details(self, summary: str, raw_content: str):
        """Attach error details to this bubble."""
        self.message.error_data = (summary, raw_content)

    def render(self, is_loading: bool = False) -> str | None:
        """Render the chat bubble. Returns action string ('debug', 'revert') or None."""
        action = None

        # Background color based on role
        bg_colors = {
            "user": "msg_u",
            "assistant": "msg_a",
            "system": "msg_a",
            "error": "msg_e"
        }
        bg_key = bg_colors.get(self.message.role, "msg_a")

        if self.message.role == "system" and "Validation Passed" in self.message.content:
            bg_key = "msg_s"

        imgui.push_style_color(imgui.Col_.child_bg, STYLE.get_imvec4(bg_key))
        imgui.begin_child(
            f"bubble_{self.message_id}",
            imgui.ImVec2(0, 0),
            child_flags=imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.borders
        )

        # Header
        header = self._get_header_text()
        imgui.text_colored(STYLE.get_imvec4("fg"), header)

        # Buttons on the right
        imgui.same_line()
        
        # Calculate widths
        md_label = "Markdown" if not self._show_raw else "Raw"
        md_w = imgui.calc_text_size(md_label).x + 20
        revert_label = "<<"
        revert_w = imgui.calc_text_size(revert_label).x + 20
        spacing = 8
        
        total_btn_w = md_w
        show_revert = (self.message.role != "error")
        if show_revert:
            total_btn_w += revert_w + spacing
            
        avail_w = imgui.get_content_region_avail().x
        current_x = imgui.get_cursor_pos_x()
        
        if avail_w > total_btn_w:
            imgui.set_cursor_pos_x(current_x + avail_w - total_btn_w)
            
        if show_revert:
            imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_small"))
            popup_id = f"Confirm Revert?###ConfirmRevert_{self.message_id}"
            if imgui.small_button(f"{revert_label}##rev_{self.message_id}"):
                imgui.open_popup(popup_id)
            imgui.pop_style_color()

            if imgui.is_item_hovered():
                imgui.set_tooltip("Revert session to this point (undoing subsequent messages)")
            
            imgui.same_line(0, spacing)

            center = imgui.get_main_viewport().get_center()
            imgui.set_next_window_pos(center, imgui.Cond_.appearing, imgui.ImVec2(0.5, 0.5))
            if imgui.begin_popup_modal(popup_id, None, imgui.WindowFlags_.always_auto_resize)[0]:
                imgui.text("Are you sure you want to revert the session to this message?")
                imgui.text_colored(STYLE.get_imvec4("btn_cncl"), "All subsequent messages and history will be lost.")
                imgui.separator()
                
                if imgui.button("Yes, Revert", imgui.ImVec2(120, 0)):
                    action = "revert"
                    imgui.close_current_popup()
                    
                imgui.same_line()
                if imgui.button("Cancel", imgui.ImVec2(120, 0)):
                    imgui.close_current_popup()
                    
                imgui.end_popup()

        imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_small"))
        if imgui.small_button(f"{md_label}##{self.message_id}"):
            self._show_raw = not self._show_raw
        imgui.pop_style_color()

        imgui.separator()

        # Content - render markdown and embedded diffs
        if self._show_raw:
            # Use InputTextMultiline to allow selection and copying
            avail_width = imgui.get_content_region_avail().x
            display_content = self.message.content.rstrip()

            # Calculate height needed for content including wrapping
            # Reduce wrap width slightly to account for InputText internal padding
            text_size = imgui.calc_text_size(display_content, wrap_width=avail_width - 25.0)
            style = imgui.get_style()
            height = text_size.y + style.frame_padding.y * 4 + 20.0
            
            # Render with transparent background to preserve bubble styling
            imgui.push_style_color(imgui.Col_.frame_bg, imgui.ImVec4(0, 0, 0, 0))
            imgui.push_style_var(imgui.StyleVar_.frame_border_size, 0.0)
            imgui.input_text_multiline(
                f"##raw_{self.message_id}",
                display_content,
                imgui.ImVec2(-1, height),
                imgui.InputTextFlags_.read_only | imgui.InputTextFlags_.word_wrap
            )
            imgui.pop_style_var()
            imgui.pop_style_color()

            if imgui.begin_popup_context_item(f"ctx_raw_{self.message_id}"):
                if imgui.menu_item("Copy All", "", False)[0]:
                    imgui.set_clipboard_text(self.message.content)
                imgui.end_popup()
        else:
            for v in self.pre_viewers:
                v.render()
                imgui.spacing()

            self._render_content()

        # Render Loading Spinner
        if is_loading:
            self._render_loading_indicator()

        # Render error details button if data is present
        if self.message.error_data:
            imgui.spacing()
            imgui.separator()
            imgui.push_style_color(imgui.Col_.button, STYLE.get_imvec4("btn_war"))
            if imgui.button(f"View Debug Details##err_{self.message_id}"):
                action = "debug"
            imgui.pop_style_color()
            imgui.same_line()
            imgui.text_colored(STYLE.get_imvec4("fg_dim"), "(Opens new tab)")

        imgui.end_child()
        imgui.pop_style_color()
        
        return action

    def _render_loading_indicator(self):
        """Render a spinner indicator at the bottom of the bubble."""
        imgui.spacing()
        imgui.separator()
        imgui.spacing()

        # Canvas drawing parameters
        radius = 7.0
        thickness = 2.0
        
        pos = imgui.get_cursor_screen_pos()
        center = imgui.ImVec2(pos.x + radius + 4, pos.y + radius + 1)
        
        draw_list = imgui.get_window_draw_list()
        
        # Background ring (dimmed)
        color_bg = STYLE.get_u32("fg_dim", 50) 
        draw_list.add_circle(center, radius, color_bg, 20, thickness)
        
        # Rotating Arc (time-based)
        time = imgui.get_time()
        start_angle = time * 6.0
        end_angle = start_angle + (math.pi / 1.5)
        
        color_fg = STYLE.get_u32("fg_dim")
        draw_list.path_clear()
        draw_list.path_arc_to(center, radius, start_angle, end_angle, 20)
        draw_list.path_stroke(color_fg, 0, thickness)
        
        # Reserve space for the graphic
        imgui.dummy(imgui.ImVec2(radius * 2 + 10, radius * 2 + 2))
        imgui.same_line()
        
        # Animated text
        dots = int(time * 2) % 4
        imgui.align_text_to_frame_padding()
        imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"Generating{'.' * dots}")

    def _get_header_text(self) -> str:
        """Get the header text for the bubble."""
        headers = {
            "user": "User",
            "assistant": "Assistant",
            "system": "System",
            "error": "System Alert"
        }
        return headers.get(self.message.role, self.message.role.capitalize())

    def _render_content(self):
        """Render the message content with markdown."""
        # Parse content into segments (text vs diff blocks)
        if self._content_dirty or self._cached_segments is None:
            self._cached_segments = self._parse_content_segments()
            self._content_dirty = False

        segments = self._cached_segments

        for idx, segment in enumerate(segments):
            if segment["type"] == "text":
                # Render markdown text with context menu
                imgui.begin_group()
                # Force hard line breaks for single newlines
                text_content = segment["content"].replace("\n", "  \n")

                # Tables render weird, but we don't have any control over the internals of imgui_md afaik
                imgui_md.render(text_content)
                imgui.end_group()

                if imgui.begin_popup_context_item(f"ctx_txt_{self.message_id}_{idx}"):
                    if imgui.menu_item("Copy text", "", False)[0]:
                        imgui.set_clipboard_text(segment["content"])
                    imgui.end_popup()

            elif segment["type"] == "diff":
                # Render diff viewer
                dv_id = segment["dv_id"]
                if dv_id in self.message.diff_viewers:
                    self.message.diff_viewers[dv_id].render()

            elif segment["type"] == "plan":
                # Render plan viewer
                pv_id = segment["pv_id"]
                if pv_id in self.message.plan_viewers:
                    self.message.plan_viewers[pv_id].render()

            elif segment["type"] == "code":
                # Render code block without diff
                imgui.push_style_color(imgui.Col_.child_bg, STYLE.get_imvec4("code_bg"))
                imgui.begin_child(f"code_{segment['id']}", imgui.ImVec2(0, 0),
                                 child_flags=imgui.ChildFlags_.auto_resize_y)
                imgui.push_style_color(imgui.Col_.text, STYLE.get_imvec4("code_fg"))
                imgui.text(segment["content"])
                imgui.pop_style_color()
                imgui.end_child()
                imgui.pop_style_color()

                if imgui.begin_popup_context_item(f"ctx_code_{self.message_id}_{segment['id']}"):
                    if imgui.menu_item("Copy code", "", False)[0]:
                        imgui.set_clipboard_text(segment["content"])
                    imgui.end_popup()

    def _parse_content_segments(self) -> list:
        """Parse content into renderable segments."""
        segments = []
        lines = self.message.content.split('\n')
        current_text = []
        
        in_code = False
        code_content = []
        code_id = 0
        dv_idx = 0
        
        in_plan = False
        plan_content = []
        pv_idx = 0

        for line in lines:
            stripped = line.strip()
            
            # PLAN handling
            if stripped == '<<<<<<< PLAN' and not in_code:
                # Flush text
                if current_text:
                    segments.append({"type": "text", "content": '\n'.join(current_text)})
                    current_text = []
                in_plan = True
                continue

            if stripped == '>>>>>>> END' and in_plan:
                # End plan
                segments.append({"type": "plan", "pv_id": pv_idx})
                pv_idx += 1
                in_plan = False
                plan_content = [] # Reset, though captured by index
                continue
                
            if in_plan:
                plan_content.append(line)
                continue

            # CODE handling
            if stripped.startswith('```'):
                if in_code:
                    # End code block
                    block_content = '\n'.join(code_content)
                    if "<<<<<<< SEARCH" in block_content and dv_idx in self.message.diff_viewers:
                        # Flush text before diff
                        if current_text:
                            segments.append({"type": "text", "content": '\n'.join(current_text)})
                            current_text = []
                        segments.append({"type": "diff", "dv_id": dv_idx})
                        dv_idx += 1
                    else:
                        # Regular code block
                        if current_text:
                            segments.append({"type": "text", "content": '\n'.join(current_text)})
                            current_text = []
                        segments.append({"type": "code", "content": block_content, "id": code_id})
                        code_id += 1
                    in_code = False
                    code_content = []
                else:
                    # Start code block
                    if current_text:
                        segments.append({"type": "text", "content": '\n'.join(current_text)})
                        current_text = []
                    in_code = True
            elif in_code:
                code_content.append(line)
            else:
                current_text.append(line)

        # Flush remaining text
        if current_text:
            segments.append({"type": "text", "content": '\n'.join(current_text)})

        return segments


def render_file_tree(
    files: list,
    selected: set,
    folder_states: dict,
    on_file_toggle: Callable,
    on_folder_toggle: Callable,
    root_path: Path = None
) -> None:
    """Render a file tree with checkboxes.

    Args:
        files: List of file paths
        selected: Set of selected file paths
        folder_states: Dict mapping folder paths to expansion state
        on_file_toggle: Callback(path, selected) when file checkbox toggled
        on_folder_toggle: Callback(folder_path, selected) when folder toggled
        root_path: Root path to display relative to
    """
    if root_path is None:
        root_path = Path.cwd()

    # Build tree structure
    tree = {}
    for file_path in files:
        try:
            rel = file_path.relative_to(root_path)
            parts = rel.parts
        except ValueError:
            parts = file_path.parts

        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        # Mark file as leaf with None
        current[parts[-1]] = None

    def render_node(name: str, node: dict | None, current_path: Path):
        """Recursively render a tree node."""
        full_path = current_path / name

        if node is None:
            # It's a file
            is_selected = full_path in selected
            changed, new_val = imgui.checkbox(f"##{full_path}", is_selected)
            if changed:
                on_file_toggle(full_path, new_val)
            imgui.same_line()
            imgui.text(name)
        else:
            # It's a folder
            folder_key = str(full_path)
            is_open = folder_states.get(folder_key, False)

            # Folder checkbox (select all in folder)
            folder_selected = all(
                f in selected
                for f in files
                if str(f).startswith(str(full_path) + ("/" if "/" in str(full_path) else "\\"))
            )
            changed, new_val = imgui.checkbox(f"##{folder_key}", folder_selected)
            if changed:
                on_folder_toggle(full_path, new_val)

            imgui.same_line()

            # Tree node for folder
            flags = imgui.TreeNodeFlags_.open_on_arrow
            if is_open:
                flags |= imgui.TreeNodeFlags_.default_open

            node_open = imgui.tree_node_ex(name, flags)
            folder_states[folder_key] = node_open

            if node_open:
                # Sort: folders first, then files
                sorted_items = sorted(node.items(), key=lambda x: (x[1] is None, x[0].lower()))
                for child_name, child_node in sorted_items:
                    render_node(child_name, child_node, full_path)
                imgui.tree_pop()

    # Render root level
    sorted_root = sorted(tree.items(), key=lambda x: (x[1] is None, x[0].lower()))
    for name, node in sorted_root:
        render_node(name, node, root_path)


def draw_status_icon(draw_list, cx: float, cy: float, status: str, badge: str = None):
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
