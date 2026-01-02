<img src="https://github.com/user-attachments/assets/65364d23-209a-49e5-ad96-fd9ab6516382" height="300px">

# Overview

Hi! Delta is a minimal LLM-powered code editor. It is not an IDE, nor is it an agent. It is a tool that you point at some files, tell it to make changes, and it makes them.

This project was born from a frustration with existing LLM-powered code editors constantly getting stuck in tool-calls, unable to navigate a project's structure, performing poorly because they bloated their context with irrelevant files, etc. Delta can only do two things: Make new files or edit existing ones. This is all you need 99% of the time.

Delta is designed for experienced developers who find the act of pointing the LLM in the right direction and evaluating its output to be trivial. Delta will not do everything for you, but it will do most of the work for you -- and will waste a lot less of your time than the alternatives -- if you set it on the right track.

See [this page](ARCHITECTURE.md) for details on implementation.


# Installation

Install uv if you don't already have it:
```bash
# Linux / Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install Delta:
```bash
# Install Delta
uv tool install git+https://github.com/truefire/delta-private
```

# Contributing

For users who want to contribute to the project, clone the repo and run `uv tool install` with the `--editable` flag:
```bash
git clone git@github.com:truefire/delta-private.git
cd delta-private
uv tool install . --editable
```

See [this page](CONTRIBUTING.md) for more details.

# Usage

Delta should always be run from within your project folder, as it will have permission to write to any file within the directory from which it is run (including subdirectories). Delta was built to be a "second monitor" GUI alongside your primary editor/IDE, but it can also be used from the command line if that fits your use case better.

## GUI

Run `delta` without arguments to open the GUI:

```bash
delta
```

The GUI provides:
- **File Selection**: Browse and select files to include in context
- **Multi-turn Chat**: Have conversations with the LLM about your code
- **Diff Viewer**: View and review changes before they're applied
- **Session Management**: Save and load conversation sessions
- **Validation**: Run commands after changes to verify correctness
- **Planning Mode**: Break down complex tasks into a sequence of operations
- **Backup/Undo**: Automatic backups with one-click rollback

## Command Line

Delta CLI uses subcommands:

```bash
# Apply changes to files
delta run "Add error handling to the main function" main.py utils.py

# With validation
delta run "Fix the failing tests" -v "pytest" --timeout 30

# With retries and recursion
delta run "Implement the feature" -t 3 -r 2 --validate "npm test"

# Plan and multi-step implementation
delta plan "Refactor the database layer"

# Ask questions without modifying files
delta ask "Explain how the authentication works" auth.py

# Manage saved file selection
delta add src/*.py           # Add files to saved state
delta remove tests/*.py      # Remove files
delta clear                  # Clear all saved files
delta state                  # Show current saved state

# Use Presets (groups of files)
delta add src/*.rs -p core   # Add files to 'core' preset
delta run "Fix bug" -p core  # Run using files from 'core' preset
delta state -p core          # Show state of 'core' preset

# Pass no files to use saved file-list
delta ask "I forget which file has the render config."

# Backup management
delta undo                   # Undo last changes
delta backups                # List available backups

# Review changes
delta review                 # Review changes from the latest session
delta review 0..3            # Review changes over the last 3 sessions
delta review --git           # Review uncommitted git changes

# Configuration
delta config --settings      # Open settings.json in default text editor
delta config --open          # Open AppData folder
```

### CLI Options (run command)

| Option | Description                                               |
|--------|-----------------------------------------------------------|
| `-m, --model` | Model to use (from settings)                              |
| `-t, --tries` | Max retry attempts (default: 2)                           |
| `-r, --recurse` | Recursion iterations (default: 0)                         |
| `-v, --validate` | Validation command to run after changes                   |
| `--timeout` | Validation timeout in seconds (default: 10)               |
| `--no-backup` | Disable automatic backups                                 |
| `--ambiguous-mode` | How to handle ambiguous matches (replace_all/ignore/fail) |
| `--verify` | Ask LLM to verify changes satisfy the prompt              |
| `--review` | Open visual diff report upon successful completion        |
| `-V, --verbosity` | Output level (verbose/low/diff/silent)                    |
| `-p, --preset` | Use files from a specific preset                          |
| `--dry-run` | Perform a dry run (exit without applying changes)         |

## LLM Providers

Delta uses the standard OpenAI client library, meaning it is compatible with any OpenAI-compatible API. This includes:

- **OpenRouter**
- **OpenAI**
- **Anthropic** (via adapter or OpenRouter)
- **Local LLMs** (via Ollama, LM Studio, vLLM, etc.)

To configure the `api_base_url`, `api_key`, and custom model definitions:
- **GUI**: Tools -> API Settings
- **CLI**: `delta config --settings`

## Why not use <other tool> instead?

Other tools such as Claude Code work pretty good these days, especially for "unsupervised" work -- use them if they fit your use case better.

When I want to remain in the loop, I find delta allows me to code faster, cheaper, and with less frustration and mistakes compared to those.