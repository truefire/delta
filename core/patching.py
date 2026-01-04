"""Diff parsing and application logic."""
import re
import difflib
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Any

from pattern import code_block_pattern, search_block_pattern, plan_block_pattern, rewrite_block_pattern
from .config import config
from .fs import is_image_file
from .backups import backup_manager, GitShadowHandler

logger = logging.getLogger(__name__)

class CancelledError(Exception): pass

def _get_line_ending(text: str) -> str | None:
    if text.endswith("\r\n"):
        return "\r\n"
    elif text.endswith("\n"):
        return "\n"
    return None

def _split_line_content_and_ending(line: str) -> tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    elif line.endswith("\n"):
        return line[:-1], "\n"
    return line, ""

def build_tolerant_regex(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines(keepends=True)
    pattern_parts = []
    overall_ending = _get_line_ending(text)

    for i, line_str in enumerate(lines):
        is_last_line = i == len(lines) - 1
        content, line_ending = _split_line_content_and_ending(line_str)

        if content.strip() == "":
            content_regex = r"\s*"
        else:
            content_regex = r"[ \t\f\v]*" + re.escape(content.lstrip())

        ending_regex = ""
        if line_ending:
            if is_last_line and overall_ending == line_ending:
                ending_regex = r"(?:" + re.escape(line_ending) + r")?"
            else:
                ending_regex = re.escape(line_ending)

        pattern_parts.append(content_regex + ending_regex)

    return "".join(pattern_parts)

def reconcile_path(filename_str: str) -> str:
    for prefix in ["file:", "filename:"]:
        if filename_str.strip().lower().startswith(prefix):
            filename_str = filename_str.strip()[len(prefix):].strip()
            
    clean_filename = filename_str.strip().strip('"\'`').lstrip("/\\")
    cwd = Path.cwd()

    if (cwd / clean_filename).exists():
        return clean_filename

    parts = clean_filename.replace("\\", "/").split("/")
    
    if len(parts) > 1 and parts[0] == cwd.name:
        candidate = "/".join(parts[1:])
        if (cwd / candidate).exists():
            return candidate
        return candidate

    if len(parts) > 1:
        basename = parts[-1]
        if (cwd / basename).exists():
            return basename

    return clean_filename

def _is_valid_filename(text: str) -> bool:
    if not text: return False
    if text.strip().startswith("<<<<<<<"): return False

    clean = reconcile_path(text)
    if not clean: return False

    try:
        p = Path.cwd() / clean
        if p.exists() and p.is_file():
            return True
    except Exception:
        pass

    if " " in clean: return False
    if clean.endswith(":"): return False
    if clean.lower() in {"bash", "python", "javascript", "typescript", "html", "css", "json", "yaml", "xml", "diff", "sh", "zsh"}:
        return False

    return True

def parse_diffs(diff_string: str) -> list[dict]:
    parsed_diffs = []

    for block_match in code_block_pattern.finditer(diff_string):
        before_block = block_match.group(1) 
        info_string = block_match.group(2)
        content = block_match.group(3)

        if not content: continue

        lines = content.split('\n', 1)
        first_line_inside = lines[0].strip()
        rest_of_inside = lines[1] if len(lines) > 1 else ""

        candidates = []
        candidates.append({"text": first_line_inside, "content": rest_of_inside})
        
        if before_block:
            candidates.append({
                "text": before_block.strip().split("\n")[-1].strip(),
                "content": content
            })
            
        if info_string:
            parts = info_string.strip().split()
            if parts:
                candidates.append({"text": parts[-1], "content": content})

        filename = None
        diff_content = content
        
        for cand in candidates:
            if _is_valid_filename(cand["text"]):
                filename = reconcile_path(cand["text"])
                diff_content = cand["content"]
                break
        
        if not filename: continue

        for hunk_match in search_block_pattern.finditer(diff_content):
            original = hunk_match.group(1)
            parsed_diffs.append({
                "filename": filename,
                "original": original if original and original.strip() else "",
                "new": hunk_match.group(2),
                "type": "search"
            })
        
        if config.allow_rewrite:
            for hunk_match in rewrite_block_pattern.finditer(diff_content):
                parsed_diffs.append({
                    "filename": filename,
                    "original": "",
                    "new": hunk_match.group(1),
                    "type": "rewrite"
                })

    return parsed_diffs

def parse_plan(text: str) -> list[tuple[str, str]]:
    tasks = []
    for match in plan_block_pattern.finditer(text):
        title = match.group(1).strip()
        prompt = match.group(2).strip()
        if title and prompt:
            tasks.append((title, prompt))
    return tasks

def _find_best_fuzzy_match(content: str, search_block: str, line_threshold: float, max_bad_lines: int) -> tuple[int, int] | None:
    search_lines = search_block.splitlines()
    if not search_lines: return None
        
    n_search = len(search_lines)
    content_lines = content.splitlines(keepends=True)
    n_content = len(content_lines)
    
    if n_content < n_search: return None

    line_offsets = [0]
    for line in content_lines:
        line_offsets.append(line_offsets[-1] + len(line))
    
    search_lines_stripped = [s.strip() for s in search_lines]
    content_lines_stripped = [c.strip() for c in content_lines]
    
    best_bad_count = max_bad_lines + 1
    best_total_score = -1.0
    best_window_start = -1
    
    check_first_line = (max_bad_lines == 0)

    for i in range(n_content - n_search + 1):
        if check_first_line:
             if abs(len(content_lines_stripped[i]) - len(search_lines_stripped[0])) > max(len(search_lines_stripped[0]), 5) * 0.5:
                 continue

        current_bad_lines = 0
        current_total_score = 0.0
        possible = True
        
        for j in range(n_search):
            s_line = search_lines_stripped[j]
            c_line = content_lines_stripped[i+j]
            
            if s_line == c_line:
                score = 1.0
            else:
                score = difflib.SequenceMatcher(None, c_line, s_line).ratio()
            
            current_total_score += score
            
            if score < line_threshold:
                possible = False
                break

            if score < 1.0:
                current_bad_lines += 1
            
            if current_bad_lines > max_bad_lines:
                possible = False
                break
        
        if possible:
            is_better = False
            if best_window_start == -1: is_better = True
            elif current_bad_lines < best_bad_count: is_better = True
            elif current_bad_lines == best_bad_count and current_total_score > best_total_score: is_better = True
            
            if is_better:
                best_bad_count = current_bad_lines
                best_total_score = current_total_score
                best_window_start = i

    if best_window_start != -1:
        start_line = best_window_start
        end_line = start_line + n_search
        return line_offsets[start_line], line_offsets[end_line]
        
    return None

def _load_file_state(filename: str, file_states: dict, simulated_states: dict) -> None:
    if filename in file_states: return

    p = Path.cwd() / filename
    try:
        if p.is_file():
            if is_image_file(p):
                raise ValueError(f"{filename}: Cannot apply text diffs to image file.")
            content = p.read_text(encoding="utf-8")
            file_states[filename] = (True, content)
            simulated_states[filename] = content
        elif p.exists():
            raise ValueError(f"{filename}: Path exists but is not a regular file.")
        else:
            if is_image_file(filename):
                raise ValueError(f"{filename}: Cannot generate new image file via text diffs.")
            file_states[filename] = (False, None)
            simulated_states[filename] = ""
    except ValueError:
        raise
    except Exception as e:
        raise IOError(f"Error accessing file '{filename}' during validation: {e}") from e

def _apply_single_diff(diff_info: dict, file_states: dict, simulated_states: dict, diff_counts: defaultdict, ambiguous_mode: str = "replace_all") -> None:
    filename = diff_info["filename"]
    original = diff_info["original"]
    new = diff_info["new"]
    diff_type = diff_info.get("type", "search")

    current_content = simulated_states[filename]
    is_existing_file, _ = file_states[filename]
    
    if diff_type == "rewrite":
        if not new:
             simulated_states[filename] = None # Marker for deletion
        else:
             simulated_states[filename] = new
        diff_counts[filename] += 1
        return

    if current_content is None:
        raise ValueError(f"{filename}: Cannot apply search/replace after a delete rewrite.")

    if not is_existing_file and diff_counts[filename] == 0:
        if original == "":
            simulated_states[filename] = new
            diff_counts[filename] += 1
            return
        raise ValueError(f"{filename}: File not found and original text not blank.")

    if original == "":
        raise ValueError(f"{filename}: Original text cannot be blank for an existing file.")

    pattern = re.compile(build_tolerant_regex(original))
    matches = list(pattern.finditer(current_content))
    match_count = len(matches)

    if match_count == 0 and config.diff_fuzzy_max_bad_lines > 0:
        fuzzy_match = _find_best_fuzzy_match(
            current_content, 
            original, 
            config.diff_fuzzy_lines_threshold, 
            config.diff_fuzzy_max_bad_lines
        )
        if fuzzy_match:
            start, end = fuzzy_match
            simulated_states[filename] = current_content[:start] + new + current_content[end:]
            diff_counts[filename] += 1
            return

    if match_count == 0:
        raise ValueError(f"Original text not found in file: {filename}")
    
    if match_count == 1:
        start, end = matches[0].span()
        simulated_states[filename] = current_content[:start] + new + current_content[end:]
    else:
        if ambiguous_mode == "fail":
            raise ValueError(f"{filename}: Original text ambiguous ({match_count} matches) in file.")
        elif ambiguous_mode == "replace_all":
            running_content = current_content
            for m in reversed(matches):
                start, end = m.span()
                running_content = running_content[:start] + new + running_content[end:]
            simulated_states[filename] = running_content
        elif ambiguous_mode == "ignore":
            pass

    diff_counts[filename] += 1

def apply_diffs(diff_string: str, create_backup: bool = True, ambiguous_mode: str = "replace_all", original_prompt: str | None = None, confirmation_callback: Callable[[dict[str, int], dict[str, str]], bool] | None = None) -> tuple[dict[str, str], str | None]:
    parsed_diffs = parse_diffs(diff_string)
    logger.debug(f"Diff parsing complete. Found {len(parsed_diffs)} blocks.")

    if not parsed_diffs:
        error_msg = "No diffs found in response."
        raise ValueError(error_msg)

    file_states: dict[str, tuple[bool, str | None]] = {}
    simulated_states: dict[str, str] = {}
    diff_counts: defaultdict[str, int] = defaultdict(int)

    for diff_info in parsed_diffs:
        filename = diff_info["filename"]
        _load_file_state(filename, file_states, simulated_states)
        _apply_single_diff(diff_info, file_states, simulated_states, diff_counts, ambiguous_mode=ambiguous_mode)

    if confirmation_callback:
        if not confirmation_callback(diff_counts, simulated_states):
            raise CancelledError("Cancelled by confirmation callback")

    session_id = None
    git_backup_active = config.use_git_backup and config.backup_enabled
    
    files_to_modify = []
    
    if git_backup_active:
        for filename in simulated_states:
            if filename in diff_counts:
                files_to_modify.append(filename)
        
        git_handler = GitShadowHandler()
        if git_handler.is_available():
            snap_msg = "Snapshot: Pre-modification context for request"
            if original_prompt:
                snap_msg += f"\n\nRequest: {original_prompt}"

            git_handler.commit_files(files_to_modify, snap_msg)
        else:
            git_backup_active = False

    if config.backup_enabled and not git_backup_active:
        session_id = backup_manager.start_session()

    applied: dict[str, str] = {}
    for filename, final_content in simulated_states.items():
        if filename not in diff_counts: continue

        p = Path.cwd() / filename
        is_existing_file, _ = file_states[filename]

        try:
            if is_existing_file:
                _, original_content = file_states[filename]
                if final_content == original_content:
                    continue

            if config.backup_enabled and not git_backup_active:
                if is_existing_file:
                    backup_manager.backup_file(p)
                else:
                    backup_manager.register_created_file(p)
            
            if final_content is not None:
                if config.backup_enabled and not git_backup_active:
                    if is_existing_file:
                        backup_manager.backup_file(p)
                    else:
                        backup_manager.register_created_file(p)

                if not is_existing_file:
                    p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(final_content, encoding="utf-8")
                
                if not is_existing_file:
                    applied[filename] = "Created"
                else:
                    applied[filename] = f"{diff_counts[filename]} diff(s)"
            else:
                if config.backup_enabled and not git_backup_active and is_existing_file:
                    backup_manager.backup_file(p)
                    
                if p.exists():
                    p.unlink()
                    applied[filename] = "Deleted"

        except Exception as e:
            raise IOError(f"Error writing file '{p}' during application phase: {e}\n\nRaw response: {diff_string}") from e

    if git_backup_active and applied:
        msg = f"Delta: Applied changes to {len(applied)} files"
        if original_prompt:
            msg += f"\n\nRequest: {original_prompt}"

        commit_id = git_handler.commit_files(files_to_modify, msg)
        session_id = commit_id or "GIT_LATEST"

    return applied, session_id