
from widgets import DiffViewer, PlanViewer
from tests.conftest import stub_to_diff

def test_diff_viewer_parse_content():
    # DiffViewer logic uses parse_diffs which parses code blocks.
    # DiffViewer wraps content in a code block inside _parse_content.
    # But usually content coming to it is the inside of the block if coming from ChatBubble.
    
    # We pass "raw" stubs without markers (markers added by stub_to_diff)
    
    stub = stub_to_diff("""
[SEARCH]
old line
[REPLACE]
new line
[END]
""")
    
    dv = DiffViewer(stub, {}, viewer_id=1, filename_hint="test.py")
    
    assert dv.state.filename == "test.py"
    assert len(dv.state.hunks) > 0
    
    changes = [h for h in dv.state.hunks if h.type == "change"]
    assert len(changes) == 1
    assert "old line" in changes[0].old
    assert "new line" in changes[0].new

def test_diff_viewer_parse_filename_from_lang_hint():
    stub = stub_to_diff("""
[SEARCH]
old
[REPLACE]
new
[END]
""")
    dv = DiffViewer(stub, {}, viewer_id=1, filename_hint=None, language_hint="test.py")
    assert dv.state.filename == "test.py"

def test_plan_viewer_parse():
    content = """Title: My Plan
Prompt: Execute Order 66
"""
    pv = PlanViewer(content, {}, viewer_id=1)
    
    assert pv.state.title == "My Plan"
    assert pv.state.prompt == "Execute Order 66"

def test_compress_context():
    from widgets import DiffViewer
    viewer = DiffViewer("", {}, 0)
    
    # Create 20 lines
    lines = "\n".join([f"Line {i}" for i in range(20)])
    
    # Test middle context (not first, not last)
    hunks = viewer._compress_context(lines, is_first=False, is_last=False)
    
    # Default limit=12, keep=4. 20 > 12.
    # Should yield: context(4), skipped(12), context(4) -> 3 hunks
    assert len(hunks) == 3
    assert hunks[0].type == "context"
    assert len(hunks[0].text.splitlines()) == 4
    assert hunks[1].type == "skipped"
    assert hunks[2].type == "context"

def test_refine_diff_difflib():
    from widgets import DiffViewer
    viewer = DiffViewer("", {}, 0)
    
    old = "A\nB\nC\n"
    new = "A\nX\nC\n"
    
    # Force _refine_diff_difflib by calling directly
    hunks = viewer._refine_diff_difflib(old.splitlines(keepends=True), new.splitlines(keepends=True))
    
    # Expect: Context(A), Change(B->X), Context(C)
    assert len(hunks) == 3
    assert hunks[0].type == "context"
    assert hunks[1].type == "change"
    assert hunks[1].old == "B\n"
    assert hunks[1].new == "X\n"

class TestChatBubbleParsing:
    def test_parse_segments_pure_text(self):
        from widgets import ChatBubble
        b = ChatBubble("user")
        b.message.content = "Hello world"
        segs = b._parse_content_segments()
        assert len(segs) == 1
        assert segs[0]["type"] == "text"
        assert segs[0]["content"] == "Hello world"
    
    def test_parse_segments_mixed(self):
        # Stub format for diff (code block)
        # text -> code -> text
        content = "Here is a code block:\n```python\nprint(1)\n```\nAnd text after."
        from widgets import ChatBubble
        b = ChatBubble("assistant")
        b.message.content = content
        segs = b._parse_content_segments()
        
        assert len(segs) == 3
        assert segs[0]["type"] == "text"
        assert "Here is" in segs[0]["content"]
        assert segs[1]["type"] == "code"
        assert "print(1)" in segs[1]["content"]
        assert segs[2]["type"] == "text"
        assert "after" in segs[2]["content"]

    def test_parse_segments_plan(self):
        # text -> plan -> text
        content = "Start\n<<<<<<< PLAN\nTitle: T\nPrompt: P\n>>>>>>> END\nFinish"
        from widgets import ChatBubble
        b = ChatBubble("assistant")
        b.message.content = content
        segs = b._parse_content_segments()
        
        assert len(segs) == 3
        assert segs[0]["type"] == "text"
        assert "Start" in segs[0]["content"]
        assert segs[1]["type"] == "plan"
        assert segs[2]["type"] == "text"
        assert "Finish" in segs[2]["content"]