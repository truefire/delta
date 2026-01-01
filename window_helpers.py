"""Window manipulation helpers for Delta Tool.

Note: These helpers are platform-specific. The imgui-bundle version
has limited window manipulation compared to the tkinter version,
as hello_imgui handles most window management internally.
"""
import sys
import os
import subprocess
import multiprocessing


def _get_process_hwnd():
    """Get the window handle for the current process."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        user32 = ctypes.windll.user32
        my_pid = os.getpid()
        result = []

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        def enum_proc(hwnd, lParam):
            pid = ctypes.c_ulong()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == my_pid and user32.IsWindowVisible(hwnd):
                # If we find a visible window with text, assume it's the one
                if user32.GetWindowTextLengthW(hwnd) > 0:
                    result.append(hwnd)
                    return False
            return True

        user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
        return result[0] if result else None
    except Exception:
        return None


def yank_window() -> None:
    """Bring the application window to focus and minimize others."""
    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            
            # 1. Bring our window to foreground
            hwnd_target = _get_process_hwnd() or user32.FindWindowW(None, "Delta Tool")
            if hwnd_target:
                if user32.IsIconic(hwnd_target):
                    user32.ShowWindow(hwnd_target, 9)  # SW_RESTORE
                user32.SetForegroundWindow(hwnd_target)

            # 2. Minimize others
            my_pid = os.getpid()
            WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

            def enum_proc(hwnd, lParam):
                if user32.IsWindowVisible(hwnd):
                    pid = ctypes.c_ulong()
                    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

                    if pid.value != my_pid and hwnd != hwnd_target:
                        if hwnd == user32.GetShellWindow():
                            return True
                        if user32.GetWindowTextLengthW(hwnd) > 0:
                            user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
                return True

            user32.EnumWindows(WNDENUMPROC(enum_proc), 0)
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            import subprocess
            cmd = ['osascript', '-e',
                   'tell application "System Events" to set visible of '
                   '(every process whose visible is true and frontmost is false) to false']
            subprocess.run(cmd, check=False)
        except Exception:
            pass
    else:
        try:
            import subprocess
            subprocess.run(['wmctrl', '-k', 'on'], check=False)
        except Exception:
            pass


def _flash_worker():
    """Worker function to run the flash overlay in a separate process."""
    try:
        import tkinter as tk
    except ImportError:
        return

    def get_monitors(root):
        monitors = []
        if sys.platform == "win32":
            try:
                import ctypes
                user32 = ctypes.windll.user32
                class RECT(ctypes.Structure):
                    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), 
                                ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
                def cb(hMonitor, hdcMonitor, lprcMonitor, dwData):
                    r = lprcMonitor.contents
                    monitors.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
                    return 1
                MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(RECT), ctypes.c_void_p)
                user32.EnumDisplayMonitors(None, None, MonitorEnumProc(cb), None)
            except Exception: pass
        
        if not monitors:
            monitors.append((0, 0, root.winfo_screenwidth(), root.winfo_screenheight()))
        return monitors

    try:
        root = tk.Tk()
        root.withdraw()
        
        monitors = get_monitors(root)
        windows = []
        
        for x, y, w, h in monitors:
            win = tk.Toplevel(root)
            win.overrideredirect(True)
            win.attributes('-topmost', True)
            win.attributes('-alpha', 0.0)
            win.configure(bg='#FF8C00')
            win.geometry(f"{w}x{h}+{x}+{y}")
            windows.append(win)
        
        def animate(alpha, step):
            new_alpha = alpha + step
            if new_alpha >= 0.25: step = -0.02
            if new_alpha <= 0:
                root.destroy()
                return
            for win in windows:
                win.attributes('-alpha', new_alpha)
            root.after(16, lambda: animate(new_alpha, step))
            
        root.after(10, lambda: animate(0.0, 0.04))
        root.mainloop()
    except Exception:
        pass


def flash_screens() -> None:
    """Flash the screen orange and application window icon."""
    # 1. Overlay Flash (Orange Screen)
    # Run in a separate process to avoid blocking main GUI thread or conflicts
    try:
        p = multiprocessing.Process(target=_flash_worker)
        p.daemon = True
        p.start()
    except Exception:
        pass

    # 2. Taskbar Flash (Windows)
    if sys.platform == "win32":
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = _get_process_hwnd() or user32.FindWindowW(None, "Delta Tool")
            if hwnd:
                class FLASHWINFO(ctypes.Structure):
                    _fields_ = [("cbSize", ctypes.c_uint),
                                ("hwnd", ctypes.c_void_p),
                                ("dwFlags", ctypes.c_uint),
                                ("uCount", ctypes.c_uint),
                                ("dwTimeout", ctypes.c_uint)]
                
                FLASHW_ALL = 3
                info = FLASHWINFO()
                info.cbSize = ctypes.sizeof(FLASHWINFO)
                info.hwnd = hwnd
                info.dwFlags = FLASHW_ALL
                info.uCount = 5
                info.dwTimeout = 0
                
                user32.FlashWindowEx(ctypes.byref(info))
        except Exception:
            pass
