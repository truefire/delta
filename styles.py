"""UI Styles and Theme management for imgui-bundle."""
from imgui_bundle import imgui
from core import config


def hex_to_imvec4(hex_color: str, alpha: float = 1.0) -> imgui.ImVec4:
    """Convert hex color string to ImVec4 (0-1 range)."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return imgui.ImVec4(r, g, b, alpha)


def hex_to_u32(hex_color: str, alpha: int = 255) -> int:
    """Convert hex color string to ImU32 for draw list operations."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return imgui.get_color_u32(imgui.ImVec4(r/255.0, g/255.0, b/255.0, alpha/255.0))


class AppStyle:
    def __init__(self, mode="light"):
        self.c = {}
        self.dark = False
        self.load(mode)

    def load(self, mode):
        self.dark = mode == "dark"
        # Define palette: (light_val, dark_val)
        palette = {
            "bg": ("#f0f0f0", "#2b2b2b"), "bg_cont": ("#ffffff", "#333333"),
            "fg": ("#000000", "#e0e0e0"), "fg_dim": ("#555555", "#aaaaaa"),
            "bg_in": ("#ffffff", "#1e1e1e"), "fg_in": ("#000000", "#ffffff"),
            "bd": ("#888888", "#555555"),
            "diff_bg": ("#e0e0e0", "#333333"), "diff_head": ("#dcdcdc", "#3c3c3c"),
            "diff_txt": ("#fbfbfb", "#1e1e1e"), "diff_cv": ("#f0f0f0", "#252526"),
            "diff_del": ("#ffecec", "#4b1818"), "diff_add": ("#eaffea", "#1e3a1e"),
            "diff_emp": ("#f0f0f0", "#2b2b2b"), "diff_nav": ("#e0e0e0", "#3c3c3c"),
            "btn_act": ("#e1f5fe", "#00695c"), "btn_ask": ("#f3e5f5", "#7b1fa2"), "btn_cncl": ("#ffebee", "#c62828"),
            "btn_sec": ("#e0f2f1", "#455a64"), "btn_std": ("#f0f0f0", "#424242"), "btn_fg": ("#000000", "#ffffff"),
            "btn_suc": ("#e8f5e9", "#2e7d32"), "btn_run": ("#b2dfdb", "#0277bd"),
            "btn_roll": ("#ffebee", "#c62828"), "btn_skip": ("#fffde7", "#9e8c00"), "btn_req": ("#fff3e0", "#c95800"),
            "btn_dig": ("#fff8e1", "#ff8f00"),
            "btn_war": ("#fff9c4", "#c46200"), "btn_small": ("#f5f5f5", "#424242"),
            "queued": ("#ff9800", "#ff9800"),
            "badge_git": ("#e0f7fa", "#006064"), "badge_file": ("#e3f2fd", "#0d47a1"),
            "txt_suc": ("#2e7d32", "#4CAF50"), "msg_u": ("#e3f2fd", "#263238"),
            "msg_a": ("#f5f5f5", "#37474f"), "msg_e": ("#fff0f0", "#3b2020"),
            "msg_s": ("#f1f8e9", "#1e3a1e"),
            "code_bg": ("#2d2d2d", "#121212"), "code_fg": ("#f8f8f2", "#e0e0e0"),
            "tt_bg": ("#ffffe0", "#424242"), "tt_fg": ("#000000", "#ffffff"),
            "hl_text": ("#0066cc", "#64b5f6"), "link_err": ("#d32f2f", "#e57373"),
            "sel_bg": ("#0078d7", "#0050a0"), "icon_bg": ("#ffffff", "#333333"),
            "icon_bd": ("#888888", "#aaaaaa"), "icon_chk": ("#000000", "#ffffff"),
            "scrollbar": ("#dcdcdc", "#424242")
        }
        self.c = {k: v[1] if self.dark else v[0] for k, v in palette.items()}

    def __getitem__(self, k):
        return self.c.get(k, "#ff00ff")

    def get_imvec4(self, k, alpha: float = 1.0) -> imgui.ImVec4:
        """Get color as ImVec4 for imgui styling."""
        return hex_to_imvec4(self[k], alpha)

    def get_u32(self, k, alpha: int = 255) -> int:
        """Get color as ImU32 for draw list operations."""
        return hex_to_u32(self[k], alpha)


STYLE = AppStyle(config.theme)


def apply_imgui_theme(dark: bool) -> None:
    """Apply dark or light theme to imgui."""
    style = imgui.get_style()

    if dark:
        imgui.style_colors_dark(style)
    else:
        imgui.style_colors_light(style)

    # Helper to set color using the correct API
    def set_color(col_enum, color):
        style.set_color_(col_enum, color)

    # Window
    set_color(imgui.Col_.window_bg, STYLE.get_imvec4("bg"))
    set_color(imgui.Col_.child_bg, STYLE.get_imvec4("bg_cont", 0.0))
    set_color(imgui.Col_.popup_bg, STYLE.get_imvec4("bg_cont", 0.95))

    # Text
    set_color(imgui.Col_.text, STYLE.get_imvec4("fg"))
    set_color(imgui.Col_.text_disabled, STYLE.get_imvec4("fg_dim"))

    # Borders
    set_color(imgui.Col_.border, STYLE.get_imvec4("bd", 0.5))
    set_color(imgui.Col_.border_shadow, STYLE.get_imvec4("bd", 0.0))

    # Frame backgrounds (inputs, etc.)
    set_color(imgui.Col_.frame_bg, STYLE.get_imvec4("bg_in", 0.54))
    set_color(imgui.Col_.frame_bg_hovered, STYLE.get_imvec4("sel_bg", 0.4))
    set_color(imgui.Col_.frame_bg_active, STYLE.get_imvec4("sel_bg", 0.67))

    # Title bar
    set_color(imgui.Col_.title_bg, STYLE.get_imvec4("bg"))
    set_color(imgui.Col_.title_bg_active, STYLE.get_imvec4("sel_bg"))
    set_color(imgui.Col_.title_bg_collapsed, STYLE.get_imvec4("bg", 0.51))

    # Buttons
    set_color(imgui.Col_.button, STYLE.get_imvec4("btn_std"))
    set_color(imgui.Col_.button_hovered, STYLE.get_imvec4("btn_act"))
    set_color(imgui.Col_.button_active, STYLE.get_imvec4("sel_bg"))

    # Headers (collapsing headers, etc.)
    set_color(imgui.Col_.header, STYLE.get_imvec4("diff_head"))
    set_color(imgui.Col_.header_hovered, STYLE.get_imvec4("sel_bg", 0.8))
    set_color(imgui.Col_.header_active, STYLE.get_imvec4("sel_bg"))

    # Tabs
    set_color(imgui.Col_.tab, STYLE.get_imvec4("btn_std"))
    set_color(imgui.Col_.tab_hovered, STYLE.get_imvec4("sel_bg", 0.8))
    set_color(imgui.Col_.tab_selected, STYLE.get_imvec4("sel_bg"))
    set_color(imgui.Col_.tab_dimmed, STYLE.get_imvec4("bg", 0.8))
    set_color(imgui.Col_.tab_dimmed_selected, STYLE.get_imvec4("sel_bg", 0.6))

    # Checkmark
    set_color(imgui.Col_.check_mark, STYLE.get_imvec4("icon_chk"))

    # Scrollbar
    set_color(imgui.Col_.scrollbar_bg, STYLE.get_imvec4("bg", 0.2))
    set_color(imgui.Col_.scrollbar_grab, STYLE.get_imvec4("scrollbar"))
    set_color(imgui.Col_.scrollbar_grab_hovered, STYLE.get_imvec4("fg_dim"))
    set_color(imgui.Col_.scrollbar_grab_active, STYLE.get_imvec4("fg"))

    # Separator
    set_color(imgui.Col_.separator, STYLE.get_imvec4("bd"))
    set_color(imgui.Col_.separator_hovered, STYLE.get_imvec4("sel_bg", 0.78))
    set_color(imgui.Col_.separator_active, STYLE.get_imvec4("sel_bg"))

    # Style tweaks
    style.window_rounding = 4.0
    style.frame_rounding = 3.0
    style.scrollbar_rounding = 3.0
    style.grab_rounding = 3.0
    style.tab_rounding = 4.0
    style.window_border_size = 1.0
    style.frame_border_size = 0.0
    style.popup_border_size = 1.0
