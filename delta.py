"""Main entry point for the delta tool - a GUI/CLI harness for LLM-powered file modification."""

import os
import sys
import subprocess
import logging

from gui import run_gui
from cli import run_cli

RESTART_EXIT_CODE = 42

def _real_main():
    if len(sys.argv) == 1:
        run_gui()
    else:
        run_cli()

def main():
    """Main entry point."""
    # Check for child process flag
    if os.environ.get("DELTA_CHILD_PROCESS"):
        _real_main()
        return

    # Supervisor Loop
    while True:
        try:
            # Prepare environment
            env = os.environ.copy()
            env["DELTA_CHILD_PROCESS"] = "1"

            # Construct command
            cmd = sys.argv
            # If running as a raw python script, execution via interpreter is safest
            if cmd[0].strip().lower().endswith(".py"):
                cmd = [sys.executable] + cmd

            # Spawn subprocess
            exit_code = subprocess.call(cmd, env=env)

            # Check for restart request
            if exit_code != RESTART_EXIT_CODE:
                sys.exit(exit_code)

            # If we are here, restart was requested. Loop continues.
            logging.log(logging.INFO, "[Delta] Restarting...")

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            sys.exit(130)
        except Exception as e:
            logging.exception(f"Error in Delta supervisor")
            sys.exit(1)

if __name__ == "__main__":
    main()
