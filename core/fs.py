"""File system operations and caching."""
import base64
import logging
import mimetypes
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import APP_DATA_DIR, load_json_file, save_json_file, DEFAULT_HIDDEN, estimate_tokens, _TOOL_DIR

logger = logging.getLogger(__name__)

class FileCache:
    """Cache for file contents to avoid repeated disk reads."""
    
    def __init__(self):
        self._cache: dict[str, tuple[str, float]] = {}  # path -> (content, mtime)
    
    def get(self, filepath: str) -> str | None:
        if filepath not in self._cache:
            return None
        cached_content, cached_mtime = self._cache[filepath]
        try:
            current_mtime = Path(filepath).stat().st_mtime
            if current_mtime == cached_mtime:
                return cached_content
        except Exception:
            pass
        del self._cache[filepath]
        return None
    
    def set(self, filepath: str, content: str) -> None:
        try:
            mtime = Path(filepath).stat().st_mtime
            self._cache[filepath] = (content, mtime)
        except Exception:
            pass
    
    def invalidate(self, filepath: str) -> None:
        self._cache.pop(filepath, None)
        try:
            abs_key = str(Path(filepath).resolve())
            if abs_key in _stats_cache:
                del _stats_cache[abs_key]
        except Exception:
            pass
    
    def clear(self) -> None:
        self._cache.clear()
    
    def get_or_read(self, filepath: str) -> str:
        cached = self.get(filepath)
        if cached is not None:
            return cached
        try:
            content = Path(filepath).read_text(encoding="utf-8", errors="replace")
            self.set(filepath, content)
            return content
        except Exception as e:
            raise IOError(f"Failed to read {filepath}: {e}") from e

file_cache = FileCache()
_stats_cache: dict[str, tuple[int, int, str]] = {}

def clear_stats_cache() -> None:
    _stats_cache.clear()

def is_image_file(path: Path | str) -> bool:
    path_str = str(path).lower()
    if path_str.endswith(".svg"):
        return False
    if path_str.endswith(".webp") or path_str.endswith(".jpg") or path_str.endswith(".jpeg") or path_str.endswith(".png"):
        return True
    guess, _ = mimetypes.guess_type(str(path))
    return guess is not None and guess.startswith("image/")

def is_binary_file(path: Path | str) -> bool:
    p = Path(path)
    BINARY_EXTENSIONS = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.exe', '.bin', '.obj', '.o',
        '.a', '.lib', '.iso', '.tar', '.zip', '.7z', '.gz', '.rar', '.pdf',
        '.sqlite', '.db', '.class', '.jar', '.war', '.ear', '.parquet', '.ds_store'
    }
    if p.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(p, "rb") as f:
            chunk = f.read(1024)
            if b"\0" in chunk:
                return True
    except Exception:
        pass
    return False

def get_display_path(path: Path | str, cwd: Path | None = None) -> str:
    if cwd is None:
        cwd = Path.cwd()
    p = Path(path)
    try:
        return str(p.relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")

def encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_mime_type(image_path: str | Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type:
        return mime_type
    if Path(image_path).suffix.lower() == ".webp":
        return "image/webp"
    return "application/octet-stream"

def open_path_in_os(path: Path | str):
    p = str(Path(path).resolve())
    if sys.platform == "win32":
        os.startfile(p)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def open_terminal_in_os(path: Path | str):
    p = Path(path).resolve()
    if sys.platform == "win32":
        subprocess.Popen("start cmd", shell=True, cwd=p)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-a", "Terminal", "."], cwd=p)
    else:
        for term in ["x-terminal-emulator", "gnome-terminal", "konsole", "xterm"]:
            if shutil.which(term):
                try:
                    subprocess.Popen([term], cwd=p)
                    return
                except Exception:
                    pass

def scan_directory(path: Path, include_hidden: bool = False) -> tuple[list[str], list[str]]:
    files = []
    dirs = []
    try:
        with os.scandir(str(path)) as it:
            for entry in it:
                if not include_hidden:
                    if entry.name.startswith('.') or entry.name in DEFAULT_HIDDEN:
                        continue
                if entry.is_file():
                    files.append(entry.name)
                elif entry.is_dir():
                    dirs.append(entry.name)
    except OSError:
        pass
        
    files.sort(key=lambda s: s.lower())
    dirs.sort(key=lambda s: s.lower())
    return files, dirs

def load_cwd_data(filepath: Path | str) -> Any:
    data = load_json_file(filepath, {})
    if isinstance(data, dict):
        return data.get(str(Path.cwd()))
    return None

def save_cwd_data(filepath: Path | str, value: Any, indent: int = 2) -> None:
    data = load_json_file(filepath, {})
    if not isinstance(data, dict):
        data = {}
    data[str(Path.cwd())] = value
    save_json_file(filepath, data, indent)

def ensure_temp_dir() -> Path:
    temp_dir = APP_DATA_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def run_command(cmd: str, timeout: float = 10.0, cwd: Path | None = None) -> tuple[bool, str]:
    if cwd is None:
        cwd = Path.cwd()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        return result.returncode == 0, result.stdout or ""
    except subprocess.TimeoutExpired:
        return False, "(timeout reached)"
    except Exception as e:
        return False, f"Error running command: {e}"

def is_path_within_cwd(file_path: Path, cwd: Path) -> bool:
    try:
        f = file_path.resolve()
        c = cwd.resolve()
        return f.is_relative_to(c)
    except ValueError:
        return False

def _validate_paths(filenames: Sequence[str | Path]) -> tuple[list[str], list[str]]:
    try:
        cwd = Path.cwd().resolve(strict=True)
    except FileNotFoundError:
        return [], ["Error: Invalid CWD"]

    validated, errors = [], []
    for f in filenames:
        try:
            p = Path(f)
            if not p.exists() or not p.is_file():
                errors.append(f"Invalid file: '{f}'")
                continue
            abs_path = p.resolve(strict=True)
            if is_path_within_cwd(abs_path, cwd):
                validated.append(str(abs_path))
            else:
                errors.append(f"Outside CWD: '{f}'")
        except Exception as e:
            errors.append(f"Error checking '{f}': {e}")
    return validated, errors

def validate_files(filenames: Sequence[str | Path]) -> tuple[list[str], str | None]:
    validated, errors = _validate_paths(filenames)
    return validated, "\n".join(errors) if errors else None

def get_file_stats(path: Path | str) -> tuple[int, int, str]:
    p = Path(path)
    try:
        key = str(p)
        if key in _stats_cache:
            return _stats_cache[key]
    except Exception:
        key = str(p)

    if not p.is_file():
        return 0, 0, ""
    
    try:
        size = p.stat().st_size
        if size > 10 * 1024 * 1024:
            res = (0, 0, f"{p.name} (>10MB)")
            _stats_cache[key] = res
            return res
    except Exception:
        pass

    if is_image_file(p):
        res = (0, 1000, f"{p.name} (IMG)")
        _stats_cache[key] = res
        return res
        
    if is_binary_file(p):
        res = (0, 0, f"{p.name} (BIN)")
        _stats_cache[key] = res
        return res
        
    try:
        content = file_cache.get_or_read(str(p))
        lines = len(content.splitlines())
        tokens = estimate_tokens(content)
        res = (lines, tokens, f"{p.name} ({lines}|{tokens})")
        _stats_cache[key] = res
        return res
    except Exception:
        return 0, 0, f"{p.name} (?)"

def create_askpass_wrapper() -> str:
    ipc_dir = APP_DATA_DIR / "ipc"
    ipc_dir.mkdir(parents=True, exist_ok=True)
    
    wrapper_path = ipc_dir / ("askpass.bat" if sys.platform == "win32" else "askpass.sh")
    
    delta_cmd = sys.argv[0]
    full_cmd = ""
    
    if delta_cmd.endswith(".py"):
        py_exe = sys.executable
        full_cmd = f'"{py_exe}" "{delta_cmd}" askpass'
    else:
        full_cmd = f'"{delta_cmd}" askpass'

    if sys.platform == "win32":
        content = f'@echo off\n{full_cmd} %*\n'
    else:
        content = f'#!/bin/sh\n{full_cmd} "$@"\n'

    with open(wrapper_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    if sys.platform != "win32":
        st = os.stat(wrapper_path)
        os.chmod(wrapper_path, st.st_mode | stat.S_IEXEC)
        
    return str(wrapper_path)