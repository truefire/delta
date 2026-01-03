import pytest
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
import core

@pytest.fixture
def mock_openai(monkeypatch):
    mock_client = MagicMock()
    mock_create_client = MagicMock(return_value=mock_client)
    monkeypatch.setattr("core._create_openai_client", mock_create_client)
    return mock_client

def create_tool_call_response(tool_name, args):
    """Helper to create a mock OpenAI response with a tool call."""
    import json
    
    mock_func = MagicMock()
    mock_func.name = tool_name
    mock_func.arguments = json.dumps(args)

    mock_msg = MagicMock()
    mock_msg.tool_calls = [
        MagicMock(
            id="call_123",
            function=mock_func
        )
    ]
    mock_msg.content = None
    
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    return mock_response

def create_text_response(content):
    mock_msg = MagicMock()
    mock_msg.tool_calls = None
    mock_msg.content = content
    
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    
    return mock_response

def test_filedig_success_flow(temp_cwd, mock_openai):
    # Setup filesystem
    (temp_cwd / "target.py").write_text("target content")
    (temp_cwd / "ignore.py").touch()
    
    # Mock LLM sequence
    # 1. list_directory
    # 2. submit_findings
    
    mock_openai.chat.completions.create.side_effect = [
        create_tool_call_response("list_directory", {"path": "."}),
        create_tool_call_response("submit_findings", {"files": ["target.py"], "explanation": "found it"})
    ]
    
    output_log = []
    def output_func(msg):
        output_log.append(msg)
        
    result = core.run_filedig_agent("Find target", output_func)
    
    assert result["success"] is True
    assert result["files"] == ["target.py"]
    assert "found it" in result["explanation"]
    
    # Verify list_directory output was processed
    # We can check if the tool output was fed back to LLM in the next call
    second_call_args = mock_openai.chat.completions.create.call_args_list[1]
    messages = second_call_args.kwargs['messages']
    
    # Message history: [System, User, Assistant(ToolCall), Tool(Result), Assistant(Submit)]
    # Note: 'messages' is passed by reference, so it contains the final state.
    
    # Filter for tool messages (dicts) vs mock assistant messages
    tool_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) >= 1
    tool_msg = tool_msgs[0]
    
    assert tool_msg["role"] == "tool"
    assert "target.py" in tool_msg["content"]
    assert "ignore.py" in tool_msg["content"]

def test_filedig_search_tool(temp_cwd, mock_openai):
    # Setup
    (temp_cwd / "src").mkdir()
    (temp_cwd / "src/main.py").write_text("def my_func(): pass")
    (temp_cwd / "src/util.py").write_text("print('hello')")
    
    # Mock LLM: search -> submit
    mock_openai.chat.completions.create.side_effect = [
        create_tool_call_response("search_codebase", {"query": "my_func"}),
        create_tool_call_response("submit_findings", {"files": ["src/main.py"]})
    ]
    
    result = core.run_filedig_agent("Find func", lambda x: None)
    
    assert result["success"] is True
    assert "src/main.py" in result["files"]
    
    # Check tool output
    second_call = mock_openai.chat.completions.create.call_args_list[1]
    messages = second_call.kwargs['messages']
    tool_msg = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"][0]
    
    assert "src/main.py" in tool_msg["content"]
    assert "def my_func():" in tool_msg["content"]

def test_filedig_read_tool(temp_cwd, mock_openai):
    # Setup
    f = temp_cwd / "long.py"
    lines = [f"Line {i}" for i in range(100)]
    f.write_text("\n".join(lines))
    
    # Mock LLM: read lines 10-20 -> submit
    mock_openai.chat.completions.create.side_effect = [
        create_tool_call_response("read_file_snippet", {"path": "long.py", "start_line": 10, "end_line": 20}),
        create_tool_call_response("submit_findings", {"files": []})
    ]
    
    core.run_filedig_agent("read it", lambda x: None)
    
    second_call = mock_openai.chat.completions.create.call_args_list[1]
    messages = second_call.kwargs['messages']
    tool_msg = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"][0]
    
    assert "Line 9" in tool_msg["content"] # 0-indexed check or content check
    # start_line 10 is 1-based, so index 9. "Line 9".
    assert "Line 19" in tool_msg["content"] 
    # Check bounds logic? The test content "Line 0" to "Line 99".
    # Snippet lines[9:20] -> Line 9 ... Line 19.

def test_filedig_path_security(temp_cwd, mock_openai):
    # Try to read outside CWD
    mock_openai.chat.completions.create.side_effect = [
        create_tool_call_response("read_file_snippet", {"path": "../secret.txt"}),
        create_tool_call_response("submit_findings", {"files": []})
    ]
    
    core.run_filedig_agent("hack", lambda x: None)
    
    second_call = mock_openai.chat.completions.create.call_args_list[1]
    messages = second_call.kwargs['messages']
    tool_msg = [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"][0]
    
    assert "Error" in tool_msg["content"]
    assert "outside" in tool_msg["content"]

def test_filedig_max_turns(temp_cwd, mock_openai):
    # Set max turns low
    from core import config
    orig = config.filedig_max_turns
    config.filedig_max_turns = 2
    
    try:
        # Mock LLM just chatting (no tool calls or tools that don't terminate)
        mock_openai.chat.completions.create.side_effect = [
            create_text_response("I am thinking..."),
            create_text_response("Still thinking..."),
            create_tool_call_response("submit_findings", {"files": []}) # Should not be reached
        ]
        
        result = core.run_filedig_agent("hello", lambda x: None)
        
        assert result["success"] is False
        assert "Max turns" in result["message"]
        assert mock_openai.chat.completions.create.call_count == 2
        
    finally:
        config.filedig_max_turns = orig

def test_filedig_cancel(temp_cwd, mock_openai):
    import threading
    cancel_event = threading.Event()
    cancel_event.set()
    
    from core import CancelledError
    
    with pytest.raises(CancelledError):
        core.run_filedig_agent("go", lambda x: None, cancel_event=cancel_event)