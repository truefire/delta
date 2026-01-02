# Contributing to Delta

Contributions are welcome. There's no formal submission process, but use common sense before submitting a PR.

## Prerequisites

- **Python**: 3.10 or higher.
- **Package Manager**: [uv](https://github.com/astral-sh/uv) (recommended) or pip.
- **Git**: Required for the shadow backup functionality.

## Setting Up the Development Environment

1. **Clone the repository**
   ```bash
   git clone https://github.com/truefire/delta.git
   cd delta
   ```

2. **Install in Editable Mode**
   I recommend using `uv` to install the tool in editable mode. This exposes the `delta` command globally while linking it to your local source code.
   ```bash
   uv tool install . --editable
   # Or with pip
   pip install -e .
   ```

## Architecture and Philosophy

Before diving into the code, please read the [architecture doc](ARCHITECTURE.md).
Delta follows specific design patterns that differ from typical web-based AI tools.

- We use immediate mode rendering for the GUI. If you're more familiar with typical retained mode UI systems common in webdev, this will likely be unfamiliar to you.
- Delta is designed as a human-in-the-loop tool. While some amount of agentic or unsupervised behavior isn't strictly forbidden, we prefer to be involved in the process so we can keep things in check. Keep this in mind when deciding on new features to submit.
- To accommodate self-modification, we prefer the project to be built out of "mid-sized" files. Large files with little modularity bloat  the context, while collections of many small files force us to make an excessive number of diffs to accomplish a task.

## Vibe Coding

Using delta to make changes to delta is encouraged -- much of the tool was built this way, and it's a good method of dogfooding.
That said, PRs should still be reviewed by a human before submission.

As a general rule, when using delta, it's recommended to review each change and fix/reject poor or misguided implementations to avoid a buildup of slop code. AI are not immune to tech debt.

## Testing

Tests are done using pytest. New tests are welcome, both for improved coverage of existing systems, as well as alongside new contributions.
Please ensure all tests pass before submitting a PR.

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. If you've added code that should be tested, ensure existing functionality is not broken and existing tests pass.
3. While no official style guide is enforced, try to stay within a reasonable distance of the style conventions used across the codebase. 
4. Submit the Pull Request.

## Reporting Bugs

Use the GitHub Issues tab to report bugs. Please include:
- The OS you are running (Windows/Linux/macOS).
- The version of Delta.
- Steps to reproduce the error.
- Relevant logs.
- If your issue involves an LLM request:
  - The LLM model name
  - The LLM provider
  - The entire conversation history
  - Any relevant log files (found in the logs folder of `delta config --open`)
  - If possible, your session file (found in the sessions folder of `delta config --open`)

## License

By contributing to Delta, you agree that your contributions will be licensed under the project's [License](LICENSE).