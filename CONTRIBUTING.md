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

## Architecture and Design

Before diving into the code, please read the [ARCHITECTURE.md](ARCHITECTURE.md).
Delta follows specific design patterns that differ from typical web-based AI tools:

- [TODO]

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
  - If possible, your session file (found in `delta config --open` -> sessions)

## License

By contributing to Delta, you agree that your contributions will be licensed under the project's [License](LICENSE).