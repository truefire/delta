# Technical Documentation: Delta Tool

<img src="https://github.com/user-attachments/assets/e537e399-33b0-481a-ba52-78dcd6bf20e8">

## 1. System Overview

Delta Tool is a hybrid CLI/GUI application designed to facilitate LLM-powered file modifications. It acts as a bridge between a local file system and Large Language Models (LLMs), managing context, executing file patches, and providing safety mechanisms like backups and validation loops.

### Key Capabilities
*   **Context Management**: Provides user-facing tools to manage LLM file context.
*   **Diff Application**: Robust parsing and application of LLM generated diff blocks.
*   **Session Management**: Multi-turn conversations with history preservation.
*   **Safety**: Automatic backups (File or Git-based), undo functionality, and validation command execution.
*   **QOL Focus**: Productivity features include focus modes for background task notifications, project navigation utilities, git integration, and visual diff functionality.
*   **Optional Agentic Tools**: Plan mode allows you to break down complex tasks into subtasks, and Dig mode allows you to discover context agentically.
---

## 2. Architecture

The codebase is modularized to separate core logic from the user interface.

### 2.1 Component Structure

*   **`delta.py`**: The application entry point. It acts as a supervisor process, checking for restart signals and dispatching execution to `gui.py` or `cli.py`.
*   **`gui.py`**: The main GUI application logic. It configures the ImGui runner (`hello_imgui`), manages the window layout, and handles the render loop callbacks.
*   **`cli.py`**: The command-line interface implementation for headless operation.
*   **`application_state.py`**: Holds the global singleton `AppState` and `ChatSession` definitions. This separates data persistence and runtime state from the UI rendering logic.
*   **`core.py`**: The business logic layer. Contains functions for:
    *   LLM API communication.
    *   File I/O and Caching.
    *   Diff parsing and application logic.
    *   Backup management (`BackupManager`, `GitShadowHandler`).
    *   Cost/Token estimation.
    *   File discovery (`run_filedig_agent`).
*   **`widgets.py`**: Custom ImGui widgets, including the `DiffViewer`, `PlanViewer`, and `ChatBubble` (Markdown renderer).
*   **`styles.py`**: Centralized theming (Light/Dark mode) and color definitions.
*   **`window_helpers.py`**: OS-specific window manipulation (flashing, yanking focus).

### 2.2 Threading Model

To prevent UI freezing during network requests or heavy file IO:

1.  **Main Thread**: Handles the ImGui render loop (via `hello_imgui`) and processes the event queue.
2.  **Worker Threads**:
    *   **Generation**: LLM requests (`generate`) and file modification loops (`process_request`) run in a daemon thread.
    *   **File Scanning**: Directory traversal and tree construction (`_folder_scan_worker`) happen in a background thread to prevent UI locking with large projects.
    *   **Search**: Context filtering runs in a separate thread to support instant keystroke response even with massive file trees.
3.  **Communication**: A thread-safe `queue.Queue` (`gui_queue`) marshals events (e.g., streaming text, status updates, search results) from workers back to the Main Thread for rendering.

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

*   **Provider**: Built on the OpenAI client standard, compatible with OpenAI, Anthropic, OpenRouter, and local inference servers (Ollama/vLLM). Defaults to OpenRouter for broad model access.
*   **Sharding**: The system implements a "sharding" loop to get around output limits:
    *   **Hard Limits**: Handles standard `finish_reason="length"` by prompting the model to continue.
    *   **Speculative Sharding**: Pro-actively monitors generation length. If approaching a limit, it looks for "clean breaks" (end of code blocks) to pause generation early. This prevents splitting a block across shards, which typically improves consistency.
*   **Context Optimization**: When building a prompt, text files are read and estimated for token cost. Image files are Base64 encoded for multimodal models. File contents are cached (`FileCache`).

### 3.3 Backup System

Delta makes backups (unless disabled) every time the LLM performs file modifications. The tool supports two backup strategies:

**1. File-System Backups (`BackupManager`)**:
*   **Sessions**: Every modification request creates a unique "Session ID".
*   **Manifests**: Tracks which files were modified and which were created (to allow deletion on rollback).
*   **Storage**: Copies original files to a hidden `.backup` folder in the system AppData directory.
*   **Rollback**: Supports atomic undoing of a complete session, restoring modified files and deleting created ones.

**2. Git Shadow Backups (`GitShadowHandler`)**:
*   **Shadow Indexing**: By manipulating the `GIT_INDEX_FILE` environment variable, Delta creates a parallel git history in a hidden branch (`delta-backup`) without ever locking or polluting the user's primary git staging area (`.git/index`).
*   **Context Snapshots**: Before applying changes, `GitShadowHandler` snapshots the *current* working directory state (including untracked files) to this shadow branch. This provides a guaranteed restoration point even if the user's workspace is dirty.
*   **Differential Analysis**: Since the backup history exists as standard git commits, Delta leverages `git diff-tree` to generate perfectly accurate, syntax-highlighted review reports comparing the "Before" and "After" states of the modification.
*   **Git Tooling Support**: Because these backups are stored via git, they are compatible with any other git tooling a developer Smay wish to use.
---

## 4. UI Implementation Details

The GUI is built using **Immediate Mode GUI (ImGui)** via `imgui_bundle`. This choice was deliberate:
*   **Tooling Standard**: ImGui is the industry standard for internal tools in game development and graphics engineering.
*   **Performance**: Immediate mode eliminates state synchronization bugs common in retained-mode UIs (React, DOM) when dealing with high-frequency updates like streaming LLM tokens.
*   **Responsiveness**: The UI redraws every frame based on the current application state data, ensuring that the visual representation is always consistent with the underlying data model.

### 4.1 Chat Architecture

The chat interface renders a list of **`ChatBubble`** objects within an imgui scrolling child region.
*   **Markdown Rendering**: `ChatBubble` uses **`imgui_md`** (from `imgui_bundle`) to render Markdown text, including headers and code blocks with syntax highlighting capabilities.
*   **Interactive Blocks**: Code blocks containing diffs or implementation plans are parsed into interactive widgets (`DiffViewer`, `PlanViewer`) embedded directly in the chat stream.
*   **Styles**: Bubbles are styled using imgui child windows with background colors indicating the role (User/Assistant/System).

### 4.2 Diff Viewer

The **`DiffViewer`** (`widgets.py`) is an immediate-mode widget rendered using standard ImGui calls and draw lists.
*   **Algorithms**: Uses the Myers diff algorithm (via the `versus` library) for minimal diffs, falling back to `difflib`.
*   **Hunk Parsing**: Parses file content and diff response into "Hunks" representing context, changes, or skipped sections.
*   **Visuals**: Uses `imgui.get_window_draw_list()` to draw colored background rectangles for additions (Green) and deletions (Red) behind the text.

### 4.3 Focus Modes

Located in `window_helpers.py`, these are productivity features:
*   **Flash**: Flashes the screen orange upon task completion. Uses a subprocess to spawn a transparent overlay window.
*   **Yank**: Minimizes all other windows and centers Delta Tool in the active monitor (based on mouse position). This requires OS-specific implementations:
    *   **Windows**: Uses `ctypes` to enumerate windows and `SetWindowPos`.
    *   **macOS**: Uses AppleScript (`osascript`).
    *   **Linux**: Uses `wmctrl`.

---

## 5. Persistence and Settings

Data is stored in the OS-specific Application Data directory (`%APPDATA%/deltatool` on Windows, `~/.config/deltatool` on Linux, `~/Library/Application Support/deltatool` on macOS).

*   **`settings.json`**: Global configuration (API keys, themes, model definitions, cost estimates).
*   **`filesets.json`**: Persists the list of selected files per working directory.
*   **`selection_presets.json`**: Stores user-defined "Groups" of files per project for quick context switching.
*   **`prompt_history.json`**: Stores a record of user prompts.
*   **`cwd_history.json`**: Stores a list of recently visited project paths.
*   **`imgui.ini`**: Persists window layout, docking positions, and sizes (managed by ImGui).
*   **`sessions/`**: Directory containing JSON serializations of chat sessions.
    *   `autosave.json`: Crash recovery/continuity state.
    *   User-or-timestamp-named save files.
*   **`backups/`**: Storage for file-system based backups (`.bak` files and manifests) when Git backups are disabled.
*   **`logs/`**: Application log files.
*   **`delta_git_index`**: A standalone git index file used by `GitShadowHandler` to manage the shadow backup branch without interfering with the user's workspace index.

---

## 6. Agentic Capabilities

Delta includes optional workflows that leverage light agentic behaviors.

### 6.1 Plan Mode
When "Plan" is selected, the System Prompt is adjusted to request a decomposition of the task rather than code edits.
*   **Parsing**: The LLM outputs `<<<<<<< PLAN` blocks containing a `Title` and a specific technical `Prompt`.
*   **Execution**: The GUI parses these blocks and populates the `impl_queue` in `AppState`. New chat sessions are spawned for each step, linked by a `group_id`.
*   **Orchestration**: Sessions run sequentially. If a step fails validation (if enabled), the queue halts to allow user intervention.

### 6.2 Dig (Discovery Agent)
Dig is a tool-use loop designed to modify the *context* rather than the code.
*   **Tool Loop**: Located in `core.run_dig_agent`, it enters a loop where the LLM can call defined tools: `list_directory`, `read_file_snippet`, and `search_codebase`.
*   **State**: The agent maintains an internal message history separate from the main chat session until it succeeds.
*   **Handoff**: Upon calling the `submit_findings` tool, the agent terminates. The file paths discovered are passed to a new standard generation session, which then executes the original user prompt using the discovered context.

---

## 7. Validation and Verification

Delta implements a multi-stage validation pipeline designed to catch and correct (or abort) errors.

1.  **Diff Validation (Core)**:
    *   Before any file is touched, the validity of the diff block is checked (does the file exist? is the search block unique?).
    *   Ambiguity resolution (`ambiguous_mode`) prevents patching files where the location is uncertain.
2.  **External Validation (`subprocess`)**:
    *   Users can supply a shell command (e.g., `pytest`, `cargo check`).
    *   This command runs *after* changes are applied. If it fails, the system captures `stdout/stderr`.
    *   Recursion loop: If validation fails, the output is fed back to the LLM in a new turn with instructions to fix the error.
3.  **LLM Verification**:
    *   An optional "Verify" step (`verify_changes` flag) triggers a second LLM call.
    *   The model is shown the file state *after* modification and asked if it satisfies the original request.
    *   This catches "lazy" edits where the model deleted code instead of fixing it, missed a requirement, or stubbed something it was supposed to implement.

---

## 8. LLM Accomodation

Several design decisions were made to accommodate idiosyncracies of LLM generated code.

*   **Find-and-Replace diff style**: LLMs are notoriously bad at counting line numbers. Instead of unified diffs, Delta uses a Search/Replace block to apply changes to files.
*   **Fuzzy parsing where possible**: We normalize leading whitespace and line endings, as well as allowing for a limited amount of fuzzy matches in line content, to apply diffs where the search term is correctibly differnt from actual file contents.
*   **Lenient filepath resolution**: LLMs often hallucinate or incorrectly labels paths (e.g. `project/src/main.py` when the file is just `src/main.py`, or `filename: main.py`). Delta heuristically resolves these to valid files in the context.

---

## 9. Testing

The codebase uses `pytest` for unit and integration testing.

*   **Mocking**: The extensive use of file I/O and external API calls necessitates heavy mocking for unit tests. Key components like `generate` and `process_request` are designed to accept dependency injection or be patchable.
*   **Diff Scenarios**: A dedicated suite of tests covers edge cases in `delta.core.apply_diffs`, ensuring that fuzzy matching, ambiguity handling, and new file creation behave as expected.
*   **Report_coverage.py**: Provides a convenient LLM-readable test coverage report to facilitate the addition of LLM generated tests.