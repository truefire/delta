import pytest
from pathlib import Path
from core import parse_diffs, apply_diffs, build_tolerant_regex, run_command
from core.patching import _find_best_fuzzy_match
from core.workflow import _execute_attempt
import sys
from tests.conftest import stub_to_diff, wrap_in_code_block

class TestDiffParsing:
    def test_parse_simple_diff(self):
        stub = """
example.py
[SEARCH]
def hello():
    print("Hi")
[REPLACE]
def hello():
    print("Hello World")
[END]
"""
        full_text = wrap_in_code_block(stub_to_diff(stub))
        diffs = parse_diffs(full_text)
        
        assert len(diffs) == 1
        assert diffs[0]["filename"] == "example.py"
        assert 'print("Hi")' in diffs[0]["original"]
        assert 'print("Hello World")' in diffs[0]["new"]

    def test_parse_multiple_files(self):
        stub1 = """
file1.py
[SEARCH]
foo
[REPLACE]
bar
[END]
"""
        stub2 = """
file2.py
[SEARCH]
baz
[REPLACE]
qux
[END]
"""
        # Wrapping each manually to simulate LLM distinct blocks or single block logic
        full_text = wrap_in_code_block(stub_to_diff(stub1)) + "\n" + wrap_in_code_block(stub_to_diff(stub2))
        diffs = parse_diffs(full_text)
        assert len(diffs) == 2
        assert diffs[0]["filename"] == "file1.py"
        assert diffs[1]["filename"] == "file2.py"

    def test_tolerant_regex_leading_whitespaces(self):
        original = "    def test( ):\n    pass"
        regex = build_tolerant_regex(original)
        
        # Should match different spacing
        target = "\tdef test( ):\n    pass"
        import re
        assert re.match(regex, target), "Regex should match tabs/spaces variations"

    def test_parse_triple_backticks_in_diff(self):
        stub = """
markdown_doc.md
[SEARCH]
Here is some code:
```python
print("old")
```
[REPLACE]
Here is some code:
```python
print("new")
```
[END]
"""
        diff_text = wrap_in_code_block(stub_to_diff(stub))
        
        diffs = parse_diffs(diff_text)
        assert len(diffs) == 1
        assert diffs[0]["filename"] == "markdown_doc.md"
        assert '```python' in diffs[0]["original"]
        assert 'print("old")' in diffs[0]["original"]
        assert 'print("new")' in diffs[0]["new"]

class TestDiffApplication:
    def test_exact_application(self, temp_cwd):
        f = temp_cwd / "test.txt"
        f.write_text("Line 1\nLine 2\nLine 3\n")
        
        stub = """
test.txt
[SEARCH]
Line 2
[REPLACE]
Line Two
[END]
"""
        diff_text = wrap_in_code_block(stub_to_diff(stub))
        apply_diffs(diff_text, create_backup=False)
        
        assert f.read_text() == "Line 1\nLine Two\nLine 3\n"

    def test_file_creation(self, temp_cwd):
        stub = """
new_file.py
[SEARCH]
[REPLACE]
print("Created")
[END]
"""
        diff_text = wrap_in_code_block(stub_to_diff(stub))
        apply_diffs(diff_text, create_backup=False)
        
        assert (temp_cwd / "new_file.py").exists()
        assert (temp_cwd / "new_file.py").read_text() == 'print("Created")'

    def test_fail_on_missing_original(self, temp_cwd):
        f = temp_cwd / "test.txt"
        f.write_text("Hello")
        
        stub = """
test.txt
[SEARCH]
Goodbye
[REPLACE]
World
[END]
"""
        diff_text = wrap_in_code_block(stub_to_diff(stub))
        
        with pytest.raises(ValueError) as exc:
            apply_diffs(diff_text, create_backup=False)
        assert "Original text not found" in str(exc.value)

class TestFuzzyMatching:
    def test_fuzzy_match_whitespace_diff(self):
        content = "def foo():\n    return 1\n"
        search = "def foo():\n  return 1\n" # 2 spaces vs 4
        
        # Should find match despite whitespace if regex logic holds, 
        # checking specifically fuzzy logic if regex failed fallback
        match = _find_best_fuzzy_match(content, search, 0.8, 0)
        assert match is not None
        
    def test_fuzzy_match_typo(self):
        content = "The quick brown fox jumps over the lazy dog"
        search =  "The quick brawn fox jumps over the lazy dog" # Typo
        
        match = _find_best_fuzzy_match(content, search, 0.9, 1)
        assert match is not None
        start, end = match
        assert content[start:end] == "The quick brown fox jumps over the lazy dog"

    def test_fuzzy_match_indentation_mismatch(self):
        content = "    def my_func():\n        return True"
        search = "  def my_func():\n    return True" # 2 spaces
        
        match = _find_best_fuzzy_match(content, search, 0.8, 1)
        assert match is not None

    def test_fuzzy_match_variable_rename(self):
        content = "def process(data):\n    return data + 1"
        search = "def process(items):\n    return items + 1"
        
        # This checks tolerance for small content changes within structure
        match = _find_best_fuzzy_match(content, search, 0.7, 2)
        assert match is not None

from core import reconcile_path

class TestPathReconciliation:
    def test_reconcile_exact(self, temp_cwd):
        (temp_cwd / "main.py").touch()
        assert reconcile_path("main.py") == "main.py"
        
    def test_reconcile_nested(self, temp_cwd):
        (temp_cwd / "src").mkdir()
        (temp_cwd / "src/utils.py").touch()
        assert reconcile_path("src/utils.py") == "src/utils.py"

    def test_reconcile_remove_cwd_name_prefix(self, temp_cwd):
        # If cwd is .../project and path is project/main.py -> main.py
        cwd_name = temp_cwd.name
        (temp_cwd / "main.py").touch()
        
        input_path = f"{cwd_name}/main.py"
        assert reconcile_path(input_path) == "main.py"
        
    def test_reconcile_basename_fallback(self, temp_cwd):
        # If path is hallucinated/absolute/whatever but basename exists in root
        (temp_cwd / "config.json").touch()
        assert reconcile_path("/opt/hallucinated/path/config.json") == "config.json"

class TestAdvancedDiffParsing:
    def test_parse_filename_in_code_header(self):
        # ```python main.py
        diff = stub_to_diff("""
[SEARCH]
a
[REPLACE]
b
[END]
""")
        text = f"```python main.py\n{diff}\n```"
        parsed = parse_diffs(text)
        assert len(parsed) == 1
        assert parsed[0]["filename"] == "main.py"

    def test_parse_filename_before_block(self):
        # File: main.py
        # ```
        diff = stub_to_diff("""
[SEARCH]
a
[REPLACE]
b
[END]
""")
        text = f"File: main.py\n```\n{diff}\n```"
        parsed = parse_diffs(text)
        assert len(parsed) == 1
        assert parsed[0]["filename"] == "main.py"

    def test_multiple_hunks_one_file(self):
        diff = stub_to_diff("""
main.py
[SEARCH]
one
[REPLACE]
1
[END]
text between blocks
[SEARCH]
two
[REPLACE]
2
[END]
""")
        text = wrap_in_code_block(diff)
        parsed = parse_diffs(text)
        assert len(parsed) == 2
        assert parsed[0]["filename"] == "main.py"
        assert parsed[1]["filename"] == "main.py"
        assert parsed[0]["original"].strip() == "one"
        assert parsed[1]["original"].strip() == "two"

    def test_chatty_response(self):
        diff = stub_to_diff("""
main.py
[SEARCH]
old
[REPLACE]
new
[END]
""")
        text = f"Here is the change:\n\n{wrap_in_code_block(diff)}\n\nHope that helps!"
        parsed = parse_diffs(text)
        assert len(parsed) == 1
        assert parsed[0]["filename"] == "main.py"

class TestAmbiguousApplication:
    def test_replace_all(self, temp_cwd):
        f = temp_cwd / "ambig.py"
        f.write_text("print(1)\nprint(1)\nprint(1)")
        
        diff = stub_to_diff("""
ambig.py
[SEARCH]
print(1)
[REPLACE]
print(2)
[END]
""")
        text = wrap_in_code_block(diff)
        
        # Default ambiguous_mode="replace_all" passed via kwarg
        apply_diffs(text, ambiguous_mode="replace_all", create_backup=False)
        assert f.read_text() == "print(2)\nprint(2)\nprint(2)"

    def test_fail_mode(self, temp_cwd):
        f = temp_cwd / "ambig.py"
        f.write_text("v=1\nv=1\n")
        
        diff = stub_to_diff("""
ambig.py
[SEARCH]
v=1
[REPLACE]
v=2
[END]
""")
        text = wrap_in_code_block(diff)
        
        with pytest.raises(ValueError) as exc:
            apply_diffs(text, ambiguous_mode="fail", create_backup=False)
        assert "ambiguous" in str(exc.value)

    def test_ignore_mode(self, temp_cwd):
        # Should do nothing if ambiguous
        content = "v=1\nv=1"
        f = temp_cwd / "ambig.py"
        f.write_text(content)
        
        diff = stub_to_diff("""
ambig.py
[SEARCH]
v=1
[REPLACE]
v=2
[END]
""")
        text = wrap_in_code_block(diff)
        
        applied, _ = apply_diffs(text, ambiguous_mode="ignore", create_backup=False)
        assert f.read_text() == content

    def test_ambiguous_application_context(self, temp_cwd):
        f = temp_cwd / "repeats.py"
        content = """def func():
    return 1

def func():
    return 1

def func():
    return 1
"""
        f.write_text(content)
        
        diff = stub_to_diff("""
repeats.py
[SEARCH]
def func():
    return 1
[REPLACE]
def func():
    return 2
[END]
""")
        text = wrap_in_code_block(diff)
        
        # Test Fail
        with pytest.raises(ValueError) as exc:
            apply_diffs(text, ambiguous_mode="fail", create_backup=False)
        assert "ambiguous" in str(exc.value)

        # Test Replace All
        apply_diffs(text, ambiguous_mode="replace_all", create_backup=False)
        expected = """def func():
    return 2

def func():
    return 2

def func():
    return 2
"""
        assert f.read_text() == expected

from unittest.mock import MagicMock, patch
from core import is_image_file, validate_files, generate, process_request, AVAILABLE_MODELS
from core.llm import _calculate_cost

def test_is_image_file():
    assert is_image_file("image.png")
    assert is_image_file("photo.jpg")
    assert is_image_file("icon.ico")
    assert is_image_file("stuff.webp")
    assert not is_image_file("script.py")
    assert not is_image_file("README.md")
    assert not is_image_file("vector.svg")

def test_validate_files(temp_cwd):
    (temp_cwd / "good.py").touch()
    (temp_cwd / "bad.py").touch()
    
    # Good case
    valid, err = validate_files(["good.py"])
    assert len(valid) == 1
    assert err is None
    
    # Missing file
    valid, err = validate_files(["missing.py"])
    assert "Invalid file" in err
    
def test_cost_calculation():
    # Mock models
    AVAILABLE_MODELS["test-model"] = {
        "input": 1.0, # $1 per 1M
        "output": 2.0 # $2 per 1M
    }
    
    cost, text = _calculate_cost(1_000_000, 0, "test-model")
    assert cost == 1.0
    
    cost, text = _calculate_cost(0, 500_000, "test-model")
    assert cost == 1.0 # 0.5 * 2

@patch("core._create_openai_client")
def test_generate_simple(mock_create_client):
    # Mock response
    mock_client = MagicMock()
    mock_stream = MagicMock()
    
    # Mock chunk structure
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta.content = "result"
    chunk.choices[0].finish_reason = "stop"
    chunk.usage = None
    
    mock_stream.__iter__.return_value = [chunk]
    mock_client.chat.completions.create.return_value = mock_stream
    mock_create_client.return_value = mock_client
    
    res = generate(["dummy.txt"], "prompt")
    assert res == "result"

@patch("core._create_openai_client")
def test_generate_sharding(mock_create_client):
    """Test that it loops when finish_reason='length'."""
    mock_client = MagicMock()
    
    # First call stream (limited)
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Part 1..."
    chunk1.choices[0].finish_reason = "length" # Triggers shard loop
    chunk1.usage = None

    # Second call stream (finish)
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "Part 2"
    chunk2.choices[0].finish_reason = "stop"
    chunk2.usage = None
    
    # We need to simulate multiple calls to create(). The first returns iter([chunk1]), second iter([chunk2])
    stream1 = MagicMock()
    stream1.__iter__.return_value = [chunk1]
    
    stream2 = MagicMock()
    stream2.__iter__.return_value = [chunk2]
    
    mock_client.chat.completions.create.side_effect = [stream1, stream2]
    mock_create_client.return_value = mock_client
    
    from core import config
    config.max_shards = 2
    # Ensure our logic for speculative sharding (char count) doesn't false-trigger, 
    # relying on finish_reason explicitly here.
    
    res = generate(["dummy.txt"], "prompt")
    assert res == "Part 1...Part 2"
    assert mock_client.chat.completions.create.call_count == 2

@patch("core.generate")
@patch("core.apply_diffs")
def test_process_request_flow(mock_apply, mock_generate, temp_cwd):
    # Setup
    (temp_cwd / "main.py").touch()
    mock_generate.return_value = "diff content"
    mock_apply.return_value = ({"main.py": 1}, "backup_id")
    
    res = process_request(
        files=["main.py"],
        prompt="Fix it",
        history=[],
        output_func=lambda x: None
    )
    
    assert res["success"] is True
    assert res["backup_id"] == "backup_id"
    mock_generate.assert_called()
    mock_apply.assert_called()

@patch("core.generate")
def test_process_request_validation_fail(mock_generate, temp_cwd):
    # Test file validation fail
    res = process_request(
        files=["nonexistent.py"],
        prompt="Fix it",
        history=[],
        output_func=lambda x: None
    )
    assert res["success"] is False
    assert "validation failed" in res["message"]
    mock_generate.assert_not_called()

@patch("core.generate")
@patch("core.apply_diffs")
@patch("core._run_validation")
def test_process_request_with_validation_cmd(mock_run_val, mock_apply, mock_gen, temp_cwd):
    (temp_cwd / "main.py").touch()
    mock_gen.return_value = "diffs"
    mock_apply.return_value = ({}, "id")
    
    # 1. Validation fails
    mock_run_val.return_value = (False, "Bad code")
    
    res = process_request(
        files=["main.py"], prompt="fix", history=[], output_func=lambda x: None,
        validation_cmd="pytest", max_retries=1
    )
    # Should fail if max_retries reached with failures
    assert res["success"] is False
    assert "Validation error" in res["message"]
    
    # 2. Validation succeeds
    mock_run_val.return_value = (True, "")
    res = process_request(
        files=["main.py"], prompt="fix", history=[], output_func=lambda x: None,
        validation_cmd="pytest"
    )
    assert res["success"] is True

def test_get_file_stats(temp_cwd):
    from core import get_file_stats
    f = temp_cwd / "test.py"
    f.write_text("a\nb\nc")
    
    lines, tokens, disp = get_file_stats(f)
    assert lines == 3
    assert tokens > 0
    assert "test.py" in disp

def test_save_load_cwd_data(temp_cwd):
    from core import save_cwd_data, load_cwd_data
    f = temp_cwd / "data.json"
    
    save_cwd_data(f, {"k": "v"})
    loaded = load_cwd_data(f)
    assert loaded == {"k": "v"}

def test_undo_last_changes_delegation():
    from core import undo_last_changes
    from unittest.mock import patch
    
    with patch("core.backup_manager") as mock_bm:
        mock_bm.get_sessions.return_value = ["s1"]
        mock_bm.undo_session.return_value = {"res": "ok"}
        
        # Ensure config doesn't force git
        with patch("core.config.use_git_backup", False):
            res = undo_last_changes()
            mock_bm.undo_session.assert_called_with("s1")
            assert res == {"res": "ok"}

def test_undo_last_changes_git_mode():
    from core import undo_last_changes
    from unittest.mock import patch, MagicMock
    
    with patch("core.GitShadowHandler") as MockGit, \
         patch("core.config") as mock_conf:
        
        mock_conf.use_git_backup = True
        mock_conf.backup_enabled = True
        
        git = MockGit.return_value
        git.is_available.return_value = True
        git.branch = "delta-backup"
        git.get_commit_files.return_value = ["a.py"]
        
        res = undo_last_changes()
        
        git.get_commit_files.assert_called_with("delta-backup")
        git.restore_files.assert_called_with(["a.py"], "~1")
        git.commit_files.assert_called()
        assert "a.py" in res

def test_get_available_backups_merged():
    from core import get_available_backups
    from unittest.mock import patch
    
    with patch("core.backup_manager") as mock_bm, \
         patch("core.GitShadowHandler") as MockGit:
        
        # File backup (OLDER)
        mock_bm.get_sessions.return_value = ["projectHash_20230101_120000_000"]
        mock_bm.get_session_files.return_value = []
        
        # Git backup (NEWER)
        git = MockGit.return_value
        git.is_available.return_value = True
        git.get_history.return_value = [{
            "session_id": "sha1", 
            "timestamp": "2023-01-02 12:00:00",
            "message": "gitmsg",
            "source": "git"
        }]
        git.get_commit_files.return_value = []
        
        backups = get_available_backups()
        assert len(backups) == 2
        # Should be sorted via timestamp descending
        assert backups[0]["source"] == "git"
        assert backups[1]["source"] == "file"

class TestExecuteAttempt:
    @pytest.fixture
    def mock_deps(self):
        with patch("core.generate") as mock_gen, \
             patch("core.apply_diffs") as mock_apply, \
             patch("core._run_validation") as mock_val, \
             patch("core.file_cache") as mock_cache:
            yield {
                "generate": mock_gen,
                "apply_diffs": mock_apply,
                "validate": mock_val,
                "cache": mock_cache
            }

    def test_execute_attempt_success(self, mock_deps):
        mock_deps["generate"].return_value = "diff"
        mock_deps["apply_diffs"].return_value = ({"f": 1}, "bak_id")
        
        success, bak_id, msg, prompt = _execute_attempt(
            attempt=1, max_retries=3, validated_files=[], 
            current_prompt="p", original_prompt="p", history=[],
            output_func=MagicMock(), stream_func=None, cancel_event=None,
            validation_cmd="", validation_timeout=0, verify=False,
            ambiguous_mode="replace_all", allow_new_files=True,
            on_file_added=None, on_diff_failure=None, on_validation_failure=None
        )
        
        assert success is True
        assert bak_id == "bak_id"
        assert msg is None

    def test_execute_attempt_validation_fail(self, mock_deps):
        mock_deps["generate"].return_value = "diff"
        mock_deps["apply_diffs"].return_value = ({"f": 1}, "bak_id")
        mock_deps["validate"].return_value = (False, "Error bad code")
        
        success, bak_id, msg, prompt = _execute_attempt(
            attempt=1, max_retries=3, validated_files=[], 
            current_prompt="p", original_prompt="p", history=[],
            output_func=MagicMock(), stream_func=None, cancel_event=None,
            validation_cmd="pytest", validation_timeout=10, verify=False,
            ambiguous_mode="replace_all", allow_new_files=True,
            on_file_added=None, on_diff_failure=None, on_validation_failure=None
        )
        
        assert success is False
        assert "Validation error" in msg
        assert "Fix this error" in prompt

    def test_execute_attempt_diff_fail(self, mock_deps):
        mock_deps["generate"].return_value = "bad diff"
        mock_deps["apply_diffs"].side_effect = ValueError("Parse error")
        
        success, bak_id, msg, prompt = _execute_attempt(
            attempt=1, max_retries=3, validated_files=[], 
            current_prompt="p", original_prompt="p", history=[],
            output_func=MagicMock(), stream_func=None, cancel_event=None,
            validation_cmd="", validation_timeout=0, verify=False,
            ambiguous_mode="replace_all", allow_new_files=True,
            on_file_added=None, on_diff_failure=MagicMock(), on_validation_failure=None
        )
        
        assert success is False
        assert "Diff application failed" in msg

    def test_execute_verify_fail(self, mock_deps):
        mock_deps["generate"].side_effect = ["diff", "NO. It is broken."] # First generate, then verify response
        mock_deps["apply_diffs"].return_value = ({"f": 1}, "bak_id")
        
        success, bak_id, msg, prompt = _execute_attempt(
            attempt=1, max_retries=3, validated_files=[], 
            current_prompt="p", original_prompt="p", history=[], 
            output_func=MagicMock(), stream_func=None, cancel_event=None,
            validation_cmd="", validation_timeout=0, verify=True, # Verify Enabled
            ambiguous_mode="replace_all", allow_new_files=True,
            on_file_added=None, on_diff_failure=None, on_validation_failure=MagicMock()
        )
        
        assert success is False
        assert "Verification failed" in msg

class TestRunCommand:
    def test_run_command_success(self):
        ok, out = run_command('echo "hello"')
        assert ok
        assert "hello" in out

    def test_run_command_timeout(self):
        # Platform agnostic sleep
        import sys
        cmd_str = f'{sys.executable} -c "import time; time.sleep(2)"'
        
        # We need a small timeout for the test
        ok, out = run_command(cmd_str, timeout=0.1)
        assert not ok
        assert "timeout" in out

class TestProcessRequestModes:
    @pytest.fixture
    def mock_gen_apply(self):
        with patch("core.generate") as mk_gen, \
             patch("core.apply_diffs") as mk_apply, \
             patch("core._run_validation") as mk_val:
            yield mk_gen, mk_apply, mk_val

    def test_plan_mode(self, temp_cwd, mock_gen_apply):
        mk_gen, _, _ = mock_gen_apply
        mk_gen.return_value = "Plan text"
        
        res = process_request(
            files=[], prompt="Plan X", history=[], 
            output_func=MagicMock(), plan_mode=True
        )
        
        assert res["success"] is True
        assert "Planning complete" in res["message"]
        # Verify generate called with plan_mode=True
        args = mk_gen.call_args
        assert args.kwargs.get("plan_mode") is True

    def test_ask_mode(self, temp_cwd, mock_gen_apply):
        mk_gen, _, _ = mock_gen_apply
        mk_gen.return_value = "Answer"
        
        res = process_request(
            files=[], prompt="Q", history=[], 
            output_func=MagicMock(), ask_mode=True
        )
        
        assert res["success"] is True
        assert "Ask mode complete" in res["message"]
        # Verify generate called with ask_mode=True
        args = mk_gen.call_args
        assert args.kwargs.get("ask_mode") is True

    def test_process_recursion(self, temp_cwd, mock_gen_apply):
        mk_gen, mk_app, mk_val = mock_gen_apply
        (temp_cwd / "f.txt").touch()
        
        # Setup mock for 2 passes: 
        # Pass 1: generate -> diff (success) -> validation (fail)
        # Pass 2: generate -> diff (success) -> validation (success)
        
        mk_gen.side_effect = ["Diff1", "Diff2"]
        # apply_diffs returns (stats, backup_id)
        mk_app.return_value = ({"f.txt": 1}, "bak")
        
        res = process_request(
            files=["f.txt"], prompt="Fix", history=[], 
            output_func=MagicMock(),
            max_retries=1, # 1 try per iteration
            recursion_limit=1 # 2 total iterations
        )
        
        # Should call generate twice
        assert mk_gen.call_count == 2
        assert res["success"] is True

    def test_process_request_pre_validation(self, temp_cwd, mock_gen_apply):
        mk_gen, mk_app, mk_val = mock_gen_apply
        (temp_cwd / "f.txt").touch()

        # fail on the first call, but succeed on the second call (post-generation validation).
        mk_val.side_effect = [(False, "Pre-check failed"), (True, "")]

        # Mock generation success
        mk_gen.return_value = "diff"
        mk_app.return_value = ({"f.txt": 1}, "bak")

        res = process_request(
            files=["f.txt"], prompt="Fix", history=[],
            output_func=MagicMock(),
            validation_cmd="check",
            validate_at_start=True
        )

        # Now this will pass
        assert res["success"] is True
        mk_gen.assert_called()

        args = mk_gen.call_args[0]
        assert "Pre-check failed" in args[1]

    def test_update_core_settings(self, mock_settings):
        # mock_settings fixture (conftest) patches core._settings
        from core import update_core_settings, API_KEY, AVAILABLE_MODELS, config
        
        new_models = {"new-model": {"input": 1.0, "output": 2.0}}
        
        # Need to patch save_settings to avoid file IO error in tests
        with patch("core._save_settings"):
            update_core_settings("new_key", "http://new.url", new_models, "new-branch")
            
            # Check globals updated
            from core import API_KEY, API_BASE_URL, AVAILABLE_MODELS
            assert API_KEY == "new_key"
            assert API_BASE_URL == "http://new.url"
            assert "new-model" in AVAILABLE_MODELS
            assert config.git_backup_branch == "new-branch"

    def test_validation_infinite_loop_prevention(self, temp_cwd, mock_gen_apply):
        mk_gen, mk_apply, mk_val = mock_gen_apply
        (temp_cwd / "f.txt").touch()

        # Mock generate to always return valid diff
        mk_gen.return_value = "diff"
        # Mock apply to succeed
        mk_apply.return_value = ({"f.txt": 1}, "bak")
        # Mock validation to always fail
        mk_val.return_value = (False, "Error")

        res = process_request(
            files=["f.txt"], 
            prompt="Fix", 
            history=[], 
            output_func=MagicMock(),
            validation_cmd="pytest",
            recursion_limit=0,
            max_retries=4
        )

        assert mk_gen.call_count == 4
        assert res["success"] is False
