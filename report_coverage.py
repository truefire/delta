import ast
import os
import sys
import subprocess
import argparse

try:
    import coverage
except ImportError:
    print("Error: 'coverage' library is missing. Please install it (pip install coverage pytest-cov).", file=sys.stderr)
    sys.exit(1)

def get_function_bounds(filename):
    """
    Parses a python file and returns a list of (function_name, start_line, end_line).
    Handles top-level functions and methods within classes.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filename)
    except Exception:
        return []

    bounds = []

    # Maintain a stack to track class distinct names if desired, 
    # but for simplicity and robustness we will join names or just use the function name.
    # Here we perform a traversal to find all Defs.
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Python 3.8+ adds end_lineno. 
            if hasattr(node, "end_lineno") and node.end_lineno:
                bounds.append((node.name, node.lineno, node.end_lineno))
    
    return bounds

def main():
    parser = argparse.ArgumentParser(description="Run pytest and report function-level coverage.")
    parser.add_argument("--threshold", type=int, default=100, help="Coverage threshold percentage (default 100).")
    # Allow passing extra args to pytest
    args, pytest_args = parser.parse_known_args()

    # 1. Run pytest internally
    # We redirect stdout/stderr to stderr so the clean report appears on stdout at the end
    cmd = [sys.executable, "-m", "pytest", "--cov=."] + pytest_args
    print(f"Running: {' '.join(cmd)}", file=sys.stderr)
    
    try:
        # We allow check=False because tests might fail, but we still want the report
        subprocess.run(cmd, check=False, stdout=sys.stderr, stderr=sys.stderr)
    except Exception as e:
        print(f"Failed to run pytest: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Analyze Coverage
    cov = coverage.Coverage()
    try:
        cov.load()
    except Exception:
        print("Error: Could not load .coverage data.", file=sys.stderr)
        sys.exit(1)

    results = []
    cwd = os.getcwd()

    for filename in cov.get_data().measured_files():
        # Filter for files within current directory only
        if not filename.startswith(cwd):
            continue
            
        relative_filename = os.path.relpath(filename, cwd)
        
        # Get executable lines (statements) and missing lines from coverage analysis
        try:
            # analysis2 returns (filename, statements, excluded, missing, missing_formatted)
            # statements and missing are lists of line numbers
            _, statements, _, missing, _ = cov.analysis2(filename)
            executable_lines = set(statements)
            missing_lines = set(missing)
            covered_lines = executable_lines - missing_lines
        except Exception:
            # Skip files that can't be analyzed (e.g. deleted since run)
            continue

        # Map mapped lines to functions
        funcs = get_function_bounds(filename)
        
        for func_name, start, end in funcs:
            # Find executable statements that fall within this function's range
            func_executable = {l for l in executable_lines if start <= l <= end}
            
            if not func_executable:
                continue # No executable code in this function (only comments/docstrings)

            func_covered = {l for l in covered_lines if l in func_executable}
            
            total_stmts = len(func_executable)
            covered_stmts = len(func_covered)
            percent = (covered_stmts / total_stmts) * 100

            if percent < args.threshold:
                # Format: filename:functionname: percentage%
                display_name = f"{relative_filename}:{func_name}"
                results.append((display_name, percent))

    # 3. Sort and Output
    # Sort by percentage (lowest first)
    results.sort(key=lambda x: x[1])

    for name, pct in results:
        print(f"{name}: {int(pct)}%")

if __name__ == "__main__":
    main()