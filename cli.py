"""CLI implementation for Delta Tool."""
import argparse
import sys
import os
import subprocess
import threading
import glob as globlib
from pathlib import Path
import logging

import core
from core import process_request, open_diff_report, undo_last_changes, get_available_backups, get_file_stats, config, APP_DATA_DIR
from application_state import (
    state, init_app_state, load_fileset, save_fileset, to_relative,
    setup_logging, load_presets, save_presets
)

def run_cli():
    """Run in CLI mode with subcommands."""
    init_app_state()

    # CLI uses standard logging to stderr, plus core logging setup
    setup_logging(enable_gui=False)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    parser = argparse.ArgumentParser(
        description="Delta Tool - LLM-powered file modification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  run       Apply changes to files
  ask       Ask questions without modifying files
  add       Add files to saved state
  remove    Remove files from saved state
  clear     Clear all saved files
  state     Show current saved state
  undo      Undo last changes
  backups   List available backups
  review    Review changes from sessions

Examples:
  delta run "Add error handling" main.py utils.py
  delta ask "How does this work?" auth.py
  delta add src/*.py
  delta review --git
"""
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # run command
    run_parser = subparsers.add_parser("run", help="Apply changes to files")
    run_parser.add_argument("prompt", help="The prompt to send to the LLM")
    run_parser.add_argument("files", nargs="*", help="Files to include in context")
    run_parser.add_argument("-m", "--model", help="Model to use")
    run_parser.add_argument("-t", "--tries", type=int, default=2, help="Max retry attempts")
    run_parser.add_argument("-r", "--recurse", type=int, default=0, help="Recursion iterations")
    run_parser.add_argument("-v", "--validate", help="Validation command")
    run_parser.add_argument("--timeout", type=float, default=10, help="Validation timeout (seconds)")
    run_parser.add_argument("--no-backup", action="store_true", help="Disable automatic backups")
    run_parser.add_argument("--ambiguous-mode", choices=["replace_all", "ignore", "fail"], help="How to handle ambiguous matches")
    run_parser.add_argument("--verify", action="store_true", help="Verify changes with LLM after application")
    run_parser.add_argument("--dry-run", action="store_true", help="Perform a dry run (simulate changes and exit)")
    run_parser.add_argument("--review", action="store_true", help="Open visual diff report after completion")
    run_parser.add_argument("-V", "--verbosity", choices=["verbose", "low", "diff", "silent"], default="low", help="Output level")
    run_parser.add_argument("-p", "--preset", help="Use files from a specific preset")

    # plan command
    plan_parser = subparsers.add_parser("plan", help="Create an implementation plan")
    plan_parser.add_argument("prompt", help="The prompt to send to the LLM")
    plan_parser.add_argument("files", nargs="*", help="Files to include in context")
    plan_parser.add_argument("-m", "--model", help="Model to use")
    plan_parser.add_argument("-p", "--preset", help="Use files from a specific preset")

    # ask command
    ask_parser = subparsers.add_parser("ask", help="Ask questions without modifying files")
    ask_parser.add_argument("prompt", help="The question to ask")
    ask_parser.add_argument("files", nargs="*", help="Files to include in context")
    ask_parser.add_argument("-m", "--model", help="Model to use")
    ask_parser.add_argument("-p", "--preset", help="Use files from a specific preset")

    # add command
    add_parser = subparsers.add_parser("add", help="Add files to saved state")
    add_parser.add_argument("patterns", nargs="+", help="File patterns to add")
    add_parser.add_argument("-p", "--preset", help="Target a specific preset instead of current context")

    # remove command
    remove_parser = subparsers.add_parser("remove", help="Remove files from saved state")
    remove_parser.add_argument("patterns", nargs="+", help="File patterns to remove")
    remove_parser.add_argument("-p", "--preset", help="Target a specific preset instead of current context")

    # clear command
    subparsers.add_parser("clear", help="Clear all saved files")

    # state command
    state_parser = subparsers.add_parser("state", help="Show current saved state")
    state_parser.add_argument("-p", "--preset", help="Show state of a specific preset")

    # undo command
    subparsers.add_parser("undo", help="Undo last changes")

    # backups command
    subparsers.add_parser("backups", help="List available backups")

    # review command
    review_parser = subparsers.add_parser("review", help="Review changes")
    review_parser.add_argument("range", nargs="?", help="Session range (e.g., 0..3)")
    review_parser.add_argument("--git", action="store_true", help="Review uncommitted git changes")

    # config command
    config_parser = subparsers.add_parser("config", help="Manage configuration")
    config_parser.add_argument("--system-prompt", help="Set the extra system prompt")
    config_parser.add_argument("--path", action="store_true", help="Print the AppData folder path")
    config_parser.add_argument("--open", action="store_true", help="Open the AppData folder")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Load saved file state
    load_fileset()

    # Create callback functions for CLI output
    def output_func(msg: str, end: str = "\n", flush: bool = False):
        print(msg, end=end, file=sys.stderr, flush=flush)

    def stream_func(text: str, end: str = "", flush: bool = False):
        print(text, end=end, flush=True)

    # Handle commands
    if args.command in ("run", "ask", "plan"):
        ask_mode = args.command == "ask"
        plan_mode = args.command == "plan"

        # Collect files
        files = []
        explicit_source = False

        if hasattr(args, 'files') and args.files:
            explicit_source = True
            for pattern in args.files:
                matched = list(globlib.glob(pattern, recursive=True))
                if matched:
                    for m in matched:
                        path = Path(m)
                        if path.is_file() and path.exists():
                            files.append(path)
                else:
                    path = Path(pattern)
                    if path.exists() and path.is_file():
                        files.append(path)
                    else:
                        print(f"Warning: No files found for pattern: {pattern}", file=sys.stderr)
        
        if hasattr(args, 'preset') and args.preset:
            explicit_source = True
            load_presets()
            if args.preset in state.presets:
                preset_files = state.presets[args.preset].get("files", [])
                for f in preset_files:
                    path = Path(f)
                    if not any(existing.resolve() == path.resolve() for existing in files):
                        files.append(path)
                print(f"Loaded {len(preset_files)} files from preset '{args.preset}'", file=sys.stderr)
            else:
                print(f"Error: Preset '{args.preset}' not found", file=sys.stderr)
                sys.exit(1)

        if not explicit_source:
            # Use saved state
            files = list(state.selected_files)
            if files:
                print(f"Using {len(files)} files from saved state", file=sys.stderr)

        if not files and not ask_mode and not plan_mode:
            print("Error: No files specified and no saved state", file=sys.stderr)
            sys.exit(1)

        # Set model if specified
        if hasattr(args, 'model') and args.model:
            config.model = args.model

        file_strs = [str(f) for f in files]

        check_callback = None
        if hasattr(args, 'dry_run') and args.dry_run:
            def dry_run_cb(counts, states):
                print("\n[Dry Run] Analysis complete. Proposed changes:")
                changes_found = False
                for fname, cnt in counts.items():
                    if cnt > 0:
                        is_new = not Path(fname).exists()
                        status = "Created" if is_new else f"{cnt} modification(s)"
                        print(f"  {fname}: {status}")
                        changes_found = True
                if not changes_found:
                    print("  No changes detected.")
                return False # Abort execution

            check_callback = dry_run_cb

        try:
            result = process_request(
                files=file_strs,
                prompt=args.prompt,
                history=[],
                output_func=output_func,
                stream_func=stream_func,
                cancel_event=threading.Event(),
                ask_mode=ask_mode,
                plan_mode=plan_mode,
                max_retries=getattr(args, 'tries', 2),
                recursion_limit=getattr(args, 'recurse', 0),
                validation_cmd=getattr(args, 'validate', "") or "",
                validation_timeout=getattr(args, 'timeout', 60.0),
                verify=getattr(args, 'verify', False),
                validate_at_start=config.validate_at_start,
                confirmation_callback=check_callback,
            )

            if result.get("success"):
                if not plan_mode:
                    print("\n\nSuccess!")
                    if hasattr(args, 'review') and args.review and result.get("backup_id"):
                        open_diff_report(result.get("backup_id"))
            else:
                msg = result.get('message', 'Unknown error')
                if check_callback and msg == "Cancelled by confirmation callback":
                    print("\nDry run completed (no changes applied).")
                else:
                    print(f"\n\nFailed: {msg}", file=sys.stderr)
                    sys.exit(1)

        except KeyboardInterrupt:
            print("\nCancelled", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            print(f"\nError: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "add":
        target_set = state.selected_files
        is_preset = False
        
        if args.preset:
            load_presets()
            if args.preset not in state.presets:
                state.presets[args.preset] = {"files": []}
                print(f"Created new preset: {args.preset}")
            
            # Use a temporary set to handle logic
            current_files = set(Path(f) for f in state.presets[args.preset].get("files", []))
            target_set = current_files
            is_preset = True

        for pattern in args.patterns:
            matched = list(globlib.glob(pattern, recursive=True))
            if matched:
                for m in matched:
                    path = to_relative(Path(m))
                    if path.is_file():
                        target_set.add(path)
                        print(f"Added: {path}")
            else:
                path = to_relative(Path(pattern))
                if path.exists() and path.is_file():
                    target_set.add(path)
                    print(f"Added: {path}")
                else:
                    print(f"Warning: No files found for: {pattern}", file=sys.stderr)
        
        if is_preset:
            state.presets[args.preset]["files"] = [str(f) for f in target_set]
            save_presets()
            print(f"\nTotal: {len(target_set)} files in preset '{args.preset}'")
        else:
            save_fileset()
            print(f"\nTotal: {len(state.selected_files)} files in saved state")

    elif args.command == "remove":
        target_set = state.selected_files
        is_preset = False
        
        if args.preset:
            load_presets()
            if args.preset not in state.presets:
                print(f"Error: Preset '{args.preset}' not found", file=sys.stderr)
                sys.exit(1)
            
            # Use a temporary set
            current_files = set(Path(f) for f in state.presets[args.preset].get("files", []))
            target_set = current_files
            is_preset = True

        for pattern in args.patterns:
            matched = list(globlib.glob(pattern, recursive=True))
            if matched:
                for m in matched:
                    path = to_relative(Path(m))
                    if path in target_set:
                        target_set.discard(path)
                        print(f"Removed: {path}")
            else:
                path = to_relative(Path(pattern))
                if path in target_set:
                    target_set.discard(path)
                    print(f"Removed: {path}")
        
        if is_preset:
            state.presets[args.preset]["files"] = [str(f) for f in target_set]
            save_presets()
            print(f"\nTotal: {len(target_set)} files in preset '{args.preset}'")
        else:
            save_fileset()
            print(f"\nTotal: {len(state.selected_files)} files in saved state")

    elif args.command == "clear":
        state.selected_files.clear()
        save_fileset()
        print("Cleared all saved files")

    elif args.command == "state":
        target_files = state.selected_files
        title = "Saved state"
        
        if args.preset:
            load_presets()
            if args.preset not in state.presets:
                print(f"Preset '{args.preset}' not found")
                return
            target_files = {Path(f) for f in state.presets[args.preset].get("files", [])}
            title = f"Preset '{args.preset}'"

        if target_files:
            print(f"{title} ({len(target_files)} files):")
            for f in sorted(target_files, key=lambda p: str(p).lower()):
                lines, tokens, _ = get_file_stats(f)
                print(f"  {f} ({lines} lines, ~{tokens} tokens)")
        else:
            print(f"No files in {title.lower()}")

    elif args.command == "undo":
        backups = get_available_backups()
        if backups:
            latest = backups[0]["session_id"]
            try:
                undo_last_changes()
                print(f"Undone changes from: {latest}")
            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("No backups available")

    elif args.command == "backups":
        backups = get_available_backups()
        if backups:
            print(f"Available backups ({len(backups)}):")
            for i, b in enumerate(backups[:20]):
                print(f"  {i}: {b}")
            if len(backups) > 20:
                print(f"  ... and {len(backups) - 20} more")
        else:
            print("No backups available")

    elif args.command == "review":
        try:
            if args.git:
                open_diff_report(use_git=True)
            elif args.range:
                # Parse range like "0..3"
                if ".." in args.range:
                    start, end = args.range.split("..", 1)
                    backups = get_available_backups()
                    start_idx = int(start) if start else 0
                    end_idx = int(end) if end else len(backups)
                    session_ids = [b["session_id"] for b in backups[start_idx:end_idx]]
                    if session_ids:
                        open_diff_report(session_ids)
                    else:
                        print("No backups in that range")
                else:
                    # Single session
                    backups = get_available_backups()
                    idx = int(args.range)
                    if 0 <= idx < len(backups):
                        open_diff_report(backups[idx]["session_id"])
                    else:
                        print(f"Invalid backup index: {idx}")
            else:
                # Review latest
                backups = get_available_backups()
                if backups:
                    open_diff_report(backups[0]["session_id"])
                else:
                    print("No backups available to review")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "config":
        if args.system_prompt:
            config.set_extra_system_prompt(args.system_prompt)
            print("Updated system prompt.")
        
        if args.path:
            print(str(APP_DATA_DIR))
            
        if args.open:
            path = str(APP_DATA_DIR)
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            print(f"Opened: {path}")
