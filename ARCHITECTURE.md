# Technical Documentation: Delta Tool

## 1. System Overview

Delta Tool is a hybrid CLI/GUI application designed to facilitate LLM-powered file modifications. It acts as a bridge between a local file system and Large Language Models (LLMs), managing context, executing file patches, and providing safety mechanisms like backups and validation loops.

### Key Capabilities
*   **Context Management**: Intelligently bundling file contents for the LLM.
*   **Diff Application**: Robust parsing and application of `SEARCH`/`REPLACE` blocks.
*   **Planning Mode**: Decomposing complex requests into sequential sub-tasks.
*   **Session Management**: Multi-turn conversations with history preservation.
*   **Safety**: Automatic backups (File or Git-based), undo functionality, and validation command execution.

---

## 2. Architecture

The codebase is modularized to separate core logic from the user interface.

### 2.1 Component Structure

*   **`delta.py`**: The application entry point. It creates the main `DeltaToolApp` class, manages the GUI event loop, and dispatches CLI commands.
*   **`core.py`**: The business logic layer. Contains functions for:
    *   LLM API communication.
    *   File I/O and Caching.
    *   Diff parsing and application logic.
    *   Backup management (`BackupManager`).
    *   Cost/Token estimation.
*   **`widgets.py`**: Custom ImGui widgets, including the `DiffViewer`, `PlanViewer`, and `ChatBubble` (Markdown renderer).
*   **`styles.py`**: Centralized theming (Light/Dark mode) and OS-specific window integrations (Windows DWM).

### 2.2 Threading Model

To prevent UI freezing during network requests or heavy file IO:

1.  **Main Thread**: Handles the Tkinter event loop and UI updates.
2.  **Worker Threads**:
    *   **Generation**: LLM requests (`generate`) and file modification loops (`process_request`) run in a daemon thread.
    *   **Stats Calculation**: Token counting and line counting for the file tree run in a separate thread to handle large projects without stutter.
3.  **Communication**: A thread-safe `queue.Queue` (`gui_queue`) marshals events (e.g., streaming text, status updates) from workers back to the Main Thread for rendering.

---

## 3. Core Systems

### 3.1 The Modification Engine (`core.py`)

Delta Tool relies on a **Search/Replace** block format rather than unified diffs. This reduces hallucinations common in LLMs when calculating line numbers.

1.  **Parsing**: The system extracts `filename`, `SEARCH` block, and `REPLACE` block using regex patterns defined in `pattern.py`.
2.  **Robust Matching Strategies**: The tool implements a multi-stage fallback mechanism to locate search blocks, addressing the common failure mode where LLMs hallucinate whitespace or slightly alter context lines.
    *   **Tolerant Regex**: `build_tolerant_regex` constructs patterns that normalize whitespace (tabs vs spaces, varying indentation), allowing matches even if the LLM's output has slight formatting deviations.
    *   **Fuzzy Matching Algorithm**: If exact and regex matching fail, `_find_best_fuzzy_match` employs a line-by-line similarity scoring algorithm (using `difflib.SequenceMatcher`). It scans the file window-by-window, tolerating a configurable threshold of "bad lines" (matches < 1.0 score) while ensuring the overall block structure is statistically probable. This allows the tool to successfully apply patches even when the LLM makes minor transcription errors in the search block.
3.  **Ambiguity Handling**: If a `SEARCH` block matches multiple locations in a file, the `ambiguous_mode` setting determines behavior:
    *   `replace_all`: Applies the change to all matches (safe for global renames).
    *   `fail`: Aborts to prevent accidental overwrites.
    *   `ignore`: Skips the modification.

### 3.2 LLM Integration

*   **Provider**: Defaults to OpenRouter API (OpenAI client compatible).
*   **Sharding**: The `generate` function implements a "sharding" loop. If the LLM hits a max token limit `finish_reason="length"`, the system automatically appends the partial result to the history and prompts the model to "continue exactly where you left off," stitching the outputs together seamlessly.
*   **Context Optimization**: File contents are cached (`FileCache`). When building a prompt, text files are read and estimated for token cost. Image files are Base64 encoded for multimodal models.

### 3.3 Backup System

Safety is paramount. The system supports two backup strategies:

**1. File-System Backups (`BackupManager`)**:
*   **Sessions**: Every modification request creates a unique "Session ID".
*   **Manifests**: Tracks which files were modified and which were *created* (to allow deletion on rollback).
*   **Storage**: Copies original files to a hidden `.backup` folder in the system AppData directory.
*   **Rollback**: Supports atomic undoing of a complete session, restoring modified files and deleting created ones.

**2. Git Shadow Backups (`GitShadowHandler`)**:
*   **Shadow Indexing**: By manipulating the `GIT_INDEX_FILE` environment variable, Delta creates a parallel git history in a hidden branch (`delta-backup`) without ever locking or polluting the user's primary git staging area (`.git/index`).
*   **Context Snapshots**: Before applying changes, `GitShadowHandler` snapshots the *current* working directory state (including untracked files) to this shadow branch. This provides a guaranteed restoration point even if the user's workspace is dirty.
*   **Differential Analysis**: Since the backup history exists as standard git commits, Delta leverages `git diff-tree` to generate perfectly accurate, syntax-highlighted review reports comparing the "Before" and "After" states of the modification.

---

## 4. UI Implementation Details

The GUI is built using **Immediate Mode GUI (ImGui)** via `imgui_bundle`. This choice was deliberate:
*   **Tooling Standard**: ImGui is the industry standard for internal tools in game development and graphics engineering.
*   **Performance**: Immediate mode eliminates state synchronization bugs common in retained-mode UIs (React, DOM) when dealing with high-frequency updates like streaming LLM tokens.
*   **Responsiveness**: The UI redraws every frame based on the current application state data, ensuring that the visual representation is always consistent with the underlying data model.

### 4.1 Chat Architecture

The chat interface renders a list of **`ChatBubble`** objects within an ImGui scrolling child region.
*   **Markdown Rendering**: `ChatBubble` uses **`imgui_md`** (from `imgui_bundle`) to render Markdown text, including headers and code blocks with syntax highlighting capabilities.
*   **Interactive Blocks**: Code blocks containing diffs or implementation plans are parsed into interactive widgets (`DiffViewer`, `PlanViewer`) embedded directly in the chat stream.
*   **Styles**: Bubbles are styled using ImGui child windows with background colors indicating the role (User/Assistant/System).

### 4.2 Diff Viewer

The **`DiffViewer`** (`widgets.py`) is an immediate-mode widget rendered using standard ImGui calls and draw lists.
*   **Algorithms**: Uses the Myers diff algorithm (via the `versus` library) for minimal diffs, falling back to `difflib`.
*   **Hunk Parsing**: Parses file content and diff response into "Hunks" representing context, changes, or skipped sections.
*   **Visuals**: Uses `imgui.get_window_draw_list()` to draw colored background rectangles for additions (Green) and deletions (Red) behind the text.

### 4.3 Focus Modes

Located in `window_helpers.py`, these are productivity features:
*   **Flash**: Flashes the screen orange upon task completion. Uses a subprocess to spawn a transparent overlay window.
*   **Yank**: Minimizes all other windows and centers Delta Tool under the mouse cursor. This requires OS-specific implementations:
    *   **Windows**: Uses `ctypes` to enumerate windows and `SetWindowPos`.
    *   **macOS**: Uses AppleScript (`osascript`).
    *   **Linux**: Uses `wmctrl`.

---

## 5. Persistence and Settings

Data is stored in the OS-specific Application Data directory (`%APPDATA%` on Windows, `~/.config` on Linux, `~/Library/Application Support` on macOS).

*   `settings.json`: Global configuration (API keys, themes, model costs).
*   `filesets.json`: Persists the list of selected files per working directory.
*   `selection_presets.json`: Stores user-defined "Groups" of files for quick context switching.
*   `prompt_history.json`: Stores last 50 prompts.
*   `autosave.json`: Disasters recovery for the current prompt draft.

---

## 6. Key Design Decisions

1.  **Block-Based Diffs vs Unified Diffs**: 
    *   *Decision*: Use `<<<<<<< SEARCH` blocks.
    *   *Reason*: LLMs are terrible at line arithmetic. Unified diffs (lines +/-) fail often because the model creates context lines that don't match exact line numbers. Search blocks rely on string matching, which LLMs are good at generating.

2.  **Separate Context Window**:
    *   *Decision*: Move complex tree management to a separate window (`ContextManagerWindow`).
    *   *Reason*: Kept the main UI clean for the "Chat" capability while allowing power users to manage massive file trees (with thousands of files) in a dedicated space without cluttering the sidebar.

3.  **Local History storage**:
    *   *Decision*: Store session history JSONs locally.
    *   *Reason*: Allows "Loading" a previous conversation to resume work later without needing a backend database. Each session file is self-contained.
