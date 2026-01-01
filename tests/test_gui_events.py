import sys
from unittest.mock import MagicMock

# Mock imgui_bundle before importing gui
# This ensures tests run even if GUI dependencies are missing
sys.modules["imgui_bundle"] = MagicMock()
sys.modules["imgui_bundle.imgui"] = MagicMock()
sys.modules["imgui_bundle.hello_imgui"] = MagicMock()
sys.modules["imgui_bundle.immapp"] = MagicMock()
sys.modules["imgui_bundle.imgui_md"] = MagicMock()
try:
    sys.modules["imgui_bundle.imgui_color_text_edit"] = MagicMock()
except Exception: pass

import pytest
from unittest.mock import patch
from application_state import state, init_app_state, ChatSession
import gui

@pytest.fixture
def mock_gui_state():
    init_app_state()
    session = ChatSession(id=1)
    state.sessions[1] = session
    state.active_session_id = 1
    yield session

def test_handle_text_event(mock_gui_state):
    # Simulate partial text stream
    event = {"type": "text", "session_id": 1, "content": "Hello"}
    gui.handle_queue_event(event)

    sess = state.sessions[1]
    assert len(sess.bubbles) == 1
    assert sess.bubbles[0].content == "Hello"
    assert sess.current_bubble is sess.bubbles[0]

    # Second chunk
    event2 = {"type": "text", "session_id": 1, "content": " World"}
    gui.handle_queue_event(event2)
    assert sess.bubbles[0].content == "Hello World"

def test_handle_diff_failure(mock_gui_state):
    event = {
        "type": "diff_failure", 
        "session_id": 1, 
        "error": "Bad syntax", 
        "raw_content": "raw"
    }
    # Pre-populate with an assistant bubble that failed
    from widgets import ChatBubble
    mock_gui_state.bubbles.append(ChatBubble("assistant"))

    gui.handle_queue_event(event)

    # Should remove the assistant bubble and add error bubble
    assert len(mock_gui_state.bubbles) == 1
    assert mock_gui_state.bubbles[0].role == "error"
    assert "parsing" in mock_gui_state.bubbles[0].content
    assert mock_gui_state.bubbles[0].message.error_data == ("Bad syntax", "raw")

def test_handle_done_chaining(mock_gui_state):
    # Setup queue
    state.impl_queue = [2, 3]
    state.current_impl_sid = 1

    # Mock start generation to verify it picks up next
    with patch("gui.start_generation") as mock_start:
        gui.handle_queue_event({"type": "done", "session_id": 1})

        assert state.current_impl_sid is None
        assert state.impl_queue == [3] # Popped 2
        mock_start.assert_called_with(2)

def test_parse_and_distribute_plan(mock_gui_state):
    # Setup a session with a Plan result
    sess = state.sessions[1]
    from widgets import ChatBubble
    b = ChatBubble("assistant")
    # Using the Plan block format
    b.message.content = "<<<<<<< PLAN\nTitle: Subtask A\nPrompt: Do A\n>>>>>>> END"
    sess.bubbles.append(b)
    
    with patch("gui.create_session") as mock_create_sess, \
         patch("gui.ensure_user_bubble") as mock_ensure, \
         patch("gui.start_generation") as mock_start: # Prevent actual start
        
        new_sess = ChatSession(id=2)
        mock_create_sess.return_value = new_sess
        
        gui.parse_and_distribute_plan(sess)
        
        # Verify new session created
        assert new_sess.group_id is not None
        assert new_sess.is_queued is True
        
        # It should have auto-started immediately because the queue was idle
        mock_start.assert_called_with(2)
        assert len(state.impl_queue) == 0