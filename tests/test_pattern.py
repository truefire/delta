import re
from pattern import search_block_pattern, plan_block_pattern
from tests.conftest import stub_to_diff

def test_search_block_pattern_basic():
    text = stub_to_diff("""
[SEARCH]
old
[REPLACE]
new
[END]
""")
    match = search_block_pattern.search(text)
    assert match
    assert match.group(1).strip() == "old"
    assert match.group(2).strip() == "new"

def test_search_block_pattern_empty_search():
    # New file creation case
    text = stub_to_diff("""
[SEARCH]
[REPLACE]
new content
[END]
""")
    match = search_block_pattern.search(text)
    assert match
    assert not match.group(1) # Search group matches empty
    assert match.group(2).strip() == "new content"

def test_plan_block_pattern():
    # Stub format for plan:
    # <<<<<<< PLAN
    # Title: ...
    # Prompt: ...
    # >>>>>>> END
    
    text = """<<<<<<< PLAN
Title: Refactor database
Prompt: Create schema.sql
>>>>>>> END"""
    
    match = plan_block_pattern.search(text)
    assert match
    assert match.group(1).strip() == "Refactor database"
    assert match.group(2).strip() == "Create schema.sql"

def test_search_block_with_varying_newlines():
    # Test tolerance for CRLF vs LF
    stub = stub_to_diff("""
[SEARCH]
line1
line2
[REPLACE]
new
[END]
""")
    # Replace LF with CRLF
    crlf_stub = stub.replace("\n", "\r\n")
    
    from pattern import search_block_pattern
    match = search_block_pattern.search(crlf_stub)
    assert match
    assert "line1" in match.group(1)
    
def test_plan_block_parsing_conftest_stub():
    # Verify the stub helper works for the new PLAN format
    from tests.conftest import stub_to_diff
    stub = """
[PLAN]
Title: T1
Prompt: P1
[PLAN_END]
"""
    diff_fmt = stub_to_diff(stub)
    assert "<<<<<<< PLAN" in diff_fmt
    assert ">>>>>>> END" in diff_fmt
