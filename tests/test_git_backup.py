import pytest
import shutil
import subprocess
from pathlib import Path
from core import GitShadowHandler, is_git_installed

@pytest.fixture
def git_repo(temp_cwd):
    if not is_git_installed():
        pytest.skip("Git not installed")
    
    subprocess.run(["git", "init"], cwd=temp_cwd, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=temp_cwd)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=temp_cwd)
    
    # Commit initial file to HEAD
    f = temp_cwd / "main.py"
    f.write_text("print('init')")
    subprocess.run(["git", "add", "."], cwd=temp_cwd)
    subprocess.run(["git", "commit", "-m", "Initial"], cwd=temp_cwd)
    return temp_cwd

def test_shadow_branch_creation(git_repo):
    handler = GitShadowHandler(branch_name="delta-shadow-test")
    
    # Modify file in workspace (dirty state)
    f = git_repo / "main.py"
    f.write_text("print('modified')")
    
    # Create snapshot commit
    assert handler.commit_files(["main.py"], "Snapshot")
    
    # Check if branch exists
    res = subprocess.run(
        ["git", "rev-parse", "--verify", "delta-shadow-test"], 
        cwd=git_repo, capture_output=True, text=True
    )
    assert res.returncode == 0

    # Ensure HEAD wasn't moved
    res_head = subprocess.run(
        ["git", "branch", "--show-current"], 
        cwd=git_repo, capture_output=True, text=True
    )
    assert res_head.stdout.strip() != "delta-shadow-test"

def test_read_history(git_repo):
    handler = GitShadowHandler(branch_name="delta-shadow-test")
    handler.commit_files(["main.py"], "Commit 1")

    (git_repo / "main.py").write_text("print('change')")
    handler.commit_files(["main.py"], "Commit 2")
    
    history = handler.get_history()
    assert len(history) == 2
    assert history[0]["message"] == "Commit 2"
    assert history[1]["message"] == "Commit 1"

def test_restore_snapshot(git_repo):
    handler = GitShadowHandler(branch_name="delta-shadow-test")
    f = git_repo / "main.py"
    
    # State 1
    f.write_text("print('v1')")
    handler.commit_files(["main.py"], "V1")
    hash_v1 = handler.get_history()[0]["session_id"]
    
    # State 2
    f.write_text("print('v2')")
    handler.commit_files(["main.py"], "V2")
    
    assert f.read_text() == "print('v2')"
    
    # Restore V1
    handler.restore_snapshot(hash_v1)
    
    assert f.read_text() == "print('v1')"

def test_restore_on_dirty_wd(git_repo):
    handler = GitShadowHandler(branch_name="delta-shadow-test")
    f = git_repo / "main.py"
    
    # Commit V1 to shadow
    f.write_text("v1")
    handler.commit_files(["main.py"], "V1")
    hash_v1 = handler.get_history()[0]["session_id"]
    
    # Make WD dirty (untracked change because index uses shadow env)
    # Writing to file makes it dirty in WD vs HEAD (which is following shadow)
    f.write_text("v1_dirty")
    
    # Attempt restore
    handler.restore_snapshot(hash_v1)
    
    assert f.read_text() == "v1"

def test_get_git_changes(git_repo):
    from core import _get_git_changes
    
    # 1. Modify tracked file
    (git_repo / "main.py").write_text("print('mod')")
    
    # 2. Add untracked file
    (git_repo / "new.py").write_text("print('new')")
    
    changes = _get_git_changes()
    
    # Should see main.py mod
    change_main = next((c for c in changes if c[0] == "main.py"), None)
    assert change_main
    assert "print('mod')" in "".join(change_main[2])
    
    # Should see new.py un-tracked
    change_new = next((c for c in changes if "new.py" in c[0]), None)
    assert change_new
    assert "(untracked)" in change_new[0]
    assert "print('new')" in "".join(change_new[2])

def test_restore_git_backup_wrapper(git_repo):
    from core import GitShadowHandler, restore_git_backup
    h = GitShadowHandler()
    h.commit_files(["main.py"], "init")
    sid = h.get_history()[0]["session_id"]
    
    res = restore_git_backup(sid)
    assert "git" in res
    assert "Restored snapshot" in res["git"]

def test_shadow_with_staged_and_unstaged_changes(git_repo):
    handler = GitShadowHandler(branch_name="delta-shadow-complex")
    
    # 1. Create file A, commit it.
    f_a = git_repo / "a.txt"
    f_a.write_text("A_v1")
    subprocess.run(["git", "add", "a.txt"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "Add A"], cwd=git_repo, check=True)

    # 2. Create file B, commit it.
    f_b = git_repo / "b.txt"
    f_b.write_text("B_v1")
    subprocess.run(["git", "add", "b.txt"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-m", "Add B"], cwd=git_repo, check=True)

    # 3. Modify A (staged).
    f_a.write_text("A_staged")
    subprocess.run(["git", "add", "a.txt"], cwd=git_repo, check=True)

    # 4. Modify B (unstaged).
    f_b.write_text("B_unstaged")

    # 5. Create untracked file C.
    f_c = git_repo / "c.txt"
    f_c.write_text("C_untracked")

    # 6. Run GitShadowHandler.commit_files on just B (simulating modification scope)
    # Note: verify that this doesn't accidentally commit A to the user's repo or clear staging
    commit_hash = handler.commit_files(["b.txt"], "Delta modification")
    assert commit_hash

    # Assert WD state remains correct
    
    # A should still be staged with "A_staged"
    status_a = subprocess.check_output(["git", "diff", "--cached", "a.txt"], cwd=git_repo, text=True)
    assert "A_staged" in status_a
    # A should NOT be modified relative to index (unstaged changes empty)
    diff_a = subprocess.check_output(["git", "diff", "a.txt"], cwd=git_repo, text=True)
    assert diff_a.strip() == ""

    # B should still be modified in WD (unstaged) relative to HEAD/Index
    # Since B_v1 is in HEAD, and B_unstaged is on disk.
    diff_b = subprocess.check_output(["git", "diff", "b.txt"], cwd=git_repo, text=True)
    assert "B_unstaged" in diff_b

    # C should exist and be untracked
    assert f_c.exists()
    status_c = subprocess.check_output(["git", "status", "--porcelain", "c.txt"], cwd=git_repo, text=True)
    assert "??" in status_c

    # Assert shadow commit exists and contains snapshot of B
    # The shadow commit should reflect the WD state (A_staged, B_unstaged, C_untracked)
    
    # Check content of B in shadow commit
    ok, content_b = handler._run(["show", f"{commit_hash}:b.txt"])
    assert ok
    assert content_b == "B_unstaged"

    # Check content of A in shadow commit
    ok, content_a = handler._run(["show", f"{commit_hash}:a.txt"])
    assert ok
    assert content_a == "A_staged"

    # Check content of C in shadow commit (should be tracked in shadow)
    ok, content_c = handler._run(["show", f"{commit_hash}:c.txt"])
    assert ok
    assert content_c == "C_untracked"
