"""Main orchestration workflows."""
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

from .config import config, DEFAULT_HIDDEN
from .fs import run_command, file_cache, is_path_within_cwd, is_image_file, is_binary_file, get_display_path, validate_files
from .llm import generate, _create_openai_client, OutputFunc, CancelledError, GenerationError
from .patching import apply_diffs
from .backups import undo_last_changes

logger = logging.getLogger(__name__)

DIG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List files and subdirectories in a specific directory (non-recursive). Use this to explore the project structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path to listing (e.g. '.' for root, 'src/')"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_snippet",
            "description": "Read a snippet of a file to understand its contents. Useful for checking imports or class definitions without reading the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the file"},
                    "start_line": {"type": "integer", "description": "Start line number (1-based)", "default": 1},
                    "end_line": {"type": "integer", "description": "End line number (1-based), max 100 lines at a time", "default": 50}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Search for a text string or regex pattern in the codebase (recursive text search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Check for this string/pattern"},
                    "path": {"type": "string", "description": "Root path to start search (default '.')"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_findings",
            "description": "Call this when you have found the files necessary to complete the user's request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}, "description": "List of relevant file paths found"},
                    "explanation": {"type": "string", "description": "Brief explanation of why these files were selected"}
                },
                "required": ["files"]
            }
        }
    }
]

def _run_validation(validation_cmd: str, validation_timeout: float) -> tuple[bool, str]:
    logger.info(f"Validating: {validation_cmd}")
    ok, output = run_command(validation_cmd, validation_timeout)
    if ok:
        logger.info("Validation passed.")
        return True, ""
    error_msg = output if output else "Unknown validation error"
    logger.error(f"Validation failed:\n{error_msg}")
    return False, error_msg

def _execute_attempt(
    attempt: int, max_retries: int, validated_files: list[str], current_prompt: str, original_prompt: str,
    history: list[dict], output_func: OutputFunc, stream_func: OutputFunc | None, cancel_event: threading.Event | None,
    validation_cmd: str, validation_timeout: float, verify: bool, ambiguous_mode: str, allow_new_files: bool,
    on_file_added: Callable[[Path], None] | None, on_diff_failure: Callable[[str, str], None] | None,
    on_validation_failure: Callable[[str], None] | None, on_validation_start: Callable[[str], None] | None,
    on_validation_success: Callable[[], None] | None, confirmation_callback: Callable[[dict[str, int], dict[str, str]], bool] | None,
    on_stream_start: Callable[[], None] | None = None,
) -> tuple[bool, str | None, str | None, str | None]:
    logger.info(f"--- Attempt {attempt}/{max_retries} ---")

    try:
        result_diff = generate(
            validated_files, current_prompt, output_func_override=output_func,
            raw_stream_output_func=stream_func, conversation_history=history,
            cancel_event=cancel_event, on_stream_start=on_stream_start
        )
    except CancelledError: raise
    except Exception as e:
        return False, None, f"Generation failed: {e}", None

    if cancel_event and cancel_event.is_set(): raise CancelledError("Cancelled")

    backup_id = None
    try:
        diffs, session_backup_id = apply_diffs(
            result_diff, ambiguous_mode=ambiguous_mode, original_prompt=original_prompt,
            confirmation_callback=confirmation_callback
        )
        if session_backup_id:
            backup_id = session_backup_id

        for fname in diffs:
            fpath = Path.cwd() / fname
            file_cache.invalidate(str(fpath))
            resolved = fpath.resolve()
            if allow_new_files and str(resolved) not in validated_files:
                if on_file_added: on_file_added(resolved)
                validated_files.append(str(resolved))

        logger.info(f"Applied diffs to {len(diffs)} file(s).")
        for f, status in diffs.items(): logger.info(f"> {f}: {status}")

    except CancelledError: raise
    except Exception as e:
        logger.error(f"Diff application failed: {e}")
        if on_diff_failure: on_diff_failure(str(e), result_diff)
        if history and len(history) >= 2:
            history.pop()
            history.pop()
        return False, backup_id, f"Diff application failed: {e}", None

    if validation_cmd:
        if on_validation_start: on_validation_start(validation_cmd)
        valid, error_msg = _run_validation(validation_cmd, validation_timeout)
        if not valid:
            if on_validation_failure: on_validation_failure(f"[{config.validation_failure_behavior.upper()}] {error_msg}")
            
            behavior = config.validation_failure_behavior
            if behavior == "ignore":
                logger.info("Validation failed but behavior is 'ignore'. Proceeding as success.")
                return True, backup_id, None, None
            elif behavior == "undo":
                logger.info("Validation failed. Undoing changes and aborting.")
                undo_last_changes()
                raise GenerationError(f"Validation failed (Undo): {error_msg}")
            elif behavior == "retry":
                logger.info("Validation failed. Undoing changes and retrying.")
                undo_last_changes()
                return False, None, f"Validation failed (Retry): {error_msg}", None
            else:
                return False, backup_id, f"Validation error: {error_msg}", f"Fix this error:\n\n{error_msg}"
        
        if on_validation_success: on_validation_success()

    if verify:
        verification_prompt = (
            f"I have applied changes to satisfy this request: '{original_prompt}'\n\n"
            "Here are the current file contents. Does this successfully satisfy the original request? "
            "Output 'YES' if satisfied, or a brief list of remaining issues if not."
        )
        logger.info("Verifying changes with LLM...")
        try:
            verify_response = generate(
                validated_files, verification_prompt, output_func_override=output_func,
                ask_mode=True, cancel_event=cancel_event
            )
            first_words = verify_response[:50].upper()
            if "YES" not in first_words.split() and "YES." not in first_words.split():
                critique = verify_response
                logger.error(f"Verification Failed.\nCritique: {critique[:300]}...")
                if on_validation_failure: on_validation_failure(f"Verification failed: {critique[:100]}...")
                return False, backup_id, f"Verification failed: {critique}", f"The previous attempt failed verification: {critique}. Please fix."
            logger.info("Verification Passed.")
        except CancelledError: raise
        except Exception as e:
            logger.error(f"Error during verification: {e}")
            return False, backup_id, f"Verification process error: {e}", f"Verification process error: {e}"

    return True, backup_id, None, None

def run_dig_agent(
    prompt: str, output_func: OutputFunc, cancel_event: threading.Event | None = None, history: list[dict] | None = None,
) -> dict:
    def _tool_ls(args):
        p = Path(args.get("path", ".")).resolve()
        try:
            if not is_path_within_cwd(p, Path.cwd()): return "Error: Path outside current working directory."
            if not p.exists(): return "Error: Path does not exist."
            if not p.is_dir(): return "Error: Path is not a directory."
            items = []
            with os.scandir(str(p)) as it:
                for entry in it:
                    if entry.name.startswith(".") or entry.name in DEFAULT_HIDDEN: continue
                    kind = "DIR" if entry.is_dir() else "FILE"
                    items.append(f"{kind}: {entry.name}")
            return "\n".join(sorted(items))
        except Exception as e: return f"Error listing directory: {e}"

    def _tool_read(args):
        p = Path(args.get("path", "")).resolve()
        start = max(1, args.get("start_line", 1))
        end = args.get("end_line", start + 50)
        if end - start > 200: end = start + 200
        try:
            if not is_path_within_cwd(p, Path.cwd()): return "Error: Path outside CWD."
            if not p.exists(): return "Error: File not found."
            if not p.is_file(): return "Error: Not a file."
            if is_image_file(p) or is_binary_file(p): return "Error: Cannot read binary/image files."
            lines = p.read_text("utf-8", errors="replace").splitlines()
            total_lines = len(lines)
            if start > total_lines: return f"Error: File only has {total_lines} lines."
            snippet = "\n".join(lines[start-1:end])
            return f"--- {p.name} ({start}-{min(end, total_lines)} of {total_lines}) ---\n{snippet}"
        except Exception as e: return f"Error reading file: {e}"

    def _tool_search(args):
        query = args.get("query", "")
        root = Path(args.get("path", ".")).resolve()
        if not query: return "Error: Empty query."
        results = []
        limit = 20
        count = 0
        try:
            for r, d, f in os.walk(str(root)):
                d[:] = [dn for dn in d if dn not in DEFAULT_HIDDEN and not dn.startswith(".")]
                for file in f:
                    if file in DEFAULT_HIDDEN or file.startswith("."): continue
                    fpath = Path(r) / file
                    if is_binary_file(fpath) or is_image_file(fpath): continue
                    try:
                        content = fpath.read_text("utf-8", errors="replace")
                        if query in content:
                            lines = content.splitlines()
                            for i, line in enumerate(lines):
                                if query in line:
                                    rel_path = get_display_path(fpath)
                                    results.append(f"{rel_path}:{i+1}: {line.strip()[:60]}")
                                    count += 1
                                    if count >= limit: break
                    except Exception: pass
                    if count >= limit: break
                if count >= limit: break
            if not results: return "No matches found."
            return "\n".join(results)
        except Exception as e: return f"Error searching: {e}"

    root_listing = _tool_ls({"path": "."})
    system_msg = f"You are 'Dig', an autonomous file exploration agent.\nYour goal is to find relevant files.\n\nCurrent Root:\n{root_listing}\n\nProcess:\n1. Analyze request.\n2. Use ls/search/read.\n3. Call 'submit_findings'."
    
    if history is not None:
        messages = history
        if not messages:
            messages.append({"role": "system", "content": system_msg})
            messages.append({"role": "user", "content": f"Find files relevant to this request: {prompt}"})
    else:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": f"Find files relevant to this request: {prompt}"}]

    client = _create_openai_client()
    max_turns = config.dig_max_turns
    total_tool_calls = 0

    logger.info("Starting Dig Loop")

    for turn in range(max_turns):
        if cancel_event and cancel_event.is_set(): raise CancelledError("Cancelled by user")
        try:
            response = client.chat.completions.create(model=config.model, messages=messages, tools=DIG_TOOLS, tool_choice="auto")
            msg = response.choices[0].message
            try: msg_dict = msg.model_dump(exclude_none=True)
            except AttributeError: msg_dict = msg.dict(exclude_none=True)
            messages.append(msg_dict)

            if msg.tool_calls:
                total_tool_calls += len(msg.tool_calls)
                for tool_call in msg.tool_calls:
                    fname = tool_call.function.name
                    args_str = tool_call.function.arguments
                    try: args = json.loads(args_str)
                    except: args = {}
                    
                    readable_args = " ".join([f"{k}='{v}'" for k,v in args.items()])
                    output_func(f"> Running: {fname} {readable_args}")
                    start_time = time.time()
                    
                    if fname == "list_directory": result_content = _tool_ls(args)
                    elif fname == "read_file_snippet": result_content = _tool_read(args)
                    elif fname == "search_codebase": result_content = _tool_search(args)
                    elif fname == "submit_findings":
                        return {"success": True, "files": args.get("files", []), "explanation": args.get("explanation", ""), "tool_calls": total_tool_calls}
                    else: result_content = "Error: Unknown tool."

                    runtime = time.time() - start_time
                    output_func(f"> {fname} finished in {runtime:.2f}s.\nOutput:\n{result_content}")
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": fname, "content": result_content})
            else:
                if msg.content: output_func(msg.content)
        except Exception as e:
            logger.error(f"Dig error: {e}")
            output_func(f"Error: {e}")
            break
            
    return {"success": False, "message": "Max turns reached without submission."}

def process_request(
    files: list[str], prompt: str, history: list[dict], output_func: OutputFunc, stream_func: OutputFunc | None = None,
    cancel_event: threading.Event | None = None, validation_cmd: str = "", validation_timeout: float = 10.0,
    max_retries: int = 2, recursion_limit: int = 0, ambiguous_mode: str = "replace_all", ask_mode: bool = False,
    plan_mode: bool = False, allow_new_files: bool = True, on_file_added: Callable[[Path], None] | None = None,
    on_diff_failure: Callable[[str, str], None] | None = None, on_validation_failure: Callable[[str], None] | None = None,
    verify: bool = False, validate_at_start: bool = False, on_validation_start: Callable[[str], None] | None = None,
    on_validation_success: Callable[[], None] | None = None, confirmation_callback: Callable[[dict[str, int], dict[str, str]], bool] | None = None,
    on_llm_start: Callable[[], None] | None = None,
) -> dict:
    def is_cancelled(): return cancel_event and cancel_event.is_set()
    def make_result(success: bool, backup_id: str | None, message: str) -> dict: return {"success": success, "backup_id": backup_id, "message": message}

    validated_files, err = validate_files(files)
    if err:
        logger.error(f"File validation failed: {err}")
        return make_result(False, None, f"File validation failed: {err}")

    if validate_at_start and validation_cmd and not (ask_mode or plan_mode):
        output_func("Running pre-validation...")
        if on_validation_start: on_validation_start(validation_cmd)
        valid, error_msg = _run_validation(validation_cmd, validation_timeout)
        if not valid:
            if on_validation_failure: on_validation_failure(f"Pre-validation failed: {error_msg}")
            prompt = f"The existing code is failing validation:\n\n{error_msg}\n\nTask: {prompt}\n\nPlease fix the existing errors and complete the task."
        else:
            if on_validation_success: on_validation_success()

    if ask_mode or plan_mode:
        try:
            if plan_mode: logger.info("--- Planning Mode ---")
            else: logger.info("--- Ask Mode ---")
            generate(validated_files, prompt, output_func_override=output_func, raw_stream_output_func=stream_func,
                    conversation_history=history, ask_mode=True, plan_mode=plan_mode, cancel_event=cancel_event, on_stream_start=on_llm_start)
            return make_result(True, None, "Planning complete." if plan_mode else "Ask mode complete.")
        except CancelledError as e: return make_result(False, None, str(e) or "Cancelled.")
        except Exception as e: return make_result(False, None, str(e))

    backup_id = None
    num_iterations = recursion_limit + 1
    last_error_msg = "Failed after retries."

    for iteration in range(num_iterations):
        current_prompt = prompt
        if is_cancelled(): return make_result(False, backup_id, "Cancelled.")
        if num_iterations > 1: logger.info(f"=== Iteration {iteration + 1}/{num_iterations} ===")
        iteration_succeeded = False

        for attempt in range(1, max_retries + 1):
            if is_cancelled(): return make_result(False, backup_id, "Cancelled.")
            try:
                success, sess_backup_id, user_fail_msg, llm_fail_prompt = _execute_attempt(
                    attempt, max_retries, validated_files, current_prompt, prompt, history, output_func, stream_func,
                    cancel_event, validation_cmd, validation_timeout, verify, ambiguous_mode, allow_new_files, on_file_added,
                    on_diff_failure, on_validation_failure, on_validation_start, on_validation_success, confirmation_callback,
                    on_stream_start=on_llm_start
                )
                if sess_backup_id: backup_id = sess_backup_id
                if success:
                    iteration_succeeded = True
                    break
                if user_fail_msg: last_error_msg = user_fail_msg
                if attempt < max_retries:
                    logger.warning("Retrying...")
                    if llm_fail_prompt: current_prompt = llm_fail_prompt
                    continue
                logger.warning("Max retries reached.")
            except CancelledError as e: return make_result(False, backup_id, str(e) or "Cancelled.")
            except Exception as e: return make_result(False, backup_id, f"Execution failed: {e}")

        if not iteration_succeeded: return make_result(False, backup_id, last_error_msg)
        if num_iterations > 1 and iteration < num_iterations - 1: validated_files, _ = validate_files(validated_files)

    return make_result(True, backup_id, "Task complete.")