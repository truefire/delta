import pytest
import shutil
from pathlib import Path
from core.backups import BackupManager

@pytest.fixture
def backup_mgr(temp_cwd):
    bm = BackupManager(backup_dir=str(temp_cwd / ".backups"))
    return bm

def test_backup_and_restore(backup_mgr, temp_cwd):
    f = temp_cwd / "data.txt"
    f.write_text("Version 1")
    
    # 1. Start Session
    sid = backup_mgr.start_session()
    
    # 2. Backup file
    backup_path = backup_mgr.backup_file(f)
    assert backup_path.exists()
    
    # 3. Modify File
    f.write_text("Version 2")
    assert f.read_text() == "Version 2"
    
    # 4. Undo
    res = backup_mgr.undo_session(sid)
    assert str(f) in res
    assert f.read_text() == "Version 1"

def test_created_file_cleanup(backup_mgr, temp_cwd):
    # 1. Start Session
    sid = backup_mgr.start_session()
    
    # 2. Register creation of new file
    new_f = temp_cwd / "created.txt"
    backup_mgr.register_created_file(new_f)
    new_f.write_text("I am new")
    
    assert new_f.exists()
    
    # 3. Undo
    res = backup_mgr.undo_session(sid)
    assert str(new_f) in res
    assert not new_f.exists()

def test_cleanup_old_backups(backup_mgr, temp_cwd):
    # Create 5 sessions
    sids = []
    for i in range(5):
        sids.append(backup_mgr.start_session())
        
        # Ensure timestamps differ slightly (BackupManager uses datetime in ID)
        import time
        time.sleep(0.01)
        
    assert len(backup_mgr.get_sessions()) == 5
    
    # Keep only 2
    deleted = backup_mgr.cleanup_old_backups(keep_sessions=2)
    assert deleted == 3
    
    remaining = backup_mgr.get_sessions()
    assert len(remaining) == 2
    # Should keep the newest (last in list returned by start_session logic, but get_sessions sorts reverse)
    # get_sessions returns sorted reverse (newest first).
    
    # Remaining should be sids[-1] and sids[-2] (the newest ones)
    assert sids[-1] in remaining
    assert sids[-2] in remaining

def test_clear_all_backups(backup_mgr):
    backup_mgr.start_session()
    backup_mgr.start_session()
    
    assert len(backup_mgr.get_sessions()) == 2
    
    backup_mgr.clear_all_backups()
    assert len(backup_mgr.get_sessions()) == 0

def test_rollback_to_session(backup_mgr, temp_cwd):
    # Setup history:
    # Session 1: f1 created
    f1 = temp_cwd / "f1.txt"
    f1.write_text("v1")
    s1 = backup_mgr.start_session()
    backup_mgr.register_created_file(f1)
    
    # Session 2: f1 mod, f2 created
    f2 = temp_cwd / "f2.txt"
    f2.write_text("v2")
    s2 = backup_mgr.start_session()
    backup_mgr.backup_file(f1) # Backing up v1
    backup_mgr.register_created_file(f2)
    f1.write_text("v2")
    
    # Session 3: f2 mod
    s3 = backup_mgr.start_session()
    backup_mgr.backup_file(f2) # Backing up v2
    f2.write_text("v3")
    
    # Rollback to Session 1 "state"
    # Docstring: "undoes the target session and all sessions that occurred after it."
    # So if we target S2, S2 and S3 are undone. State becomes end of S1.
    
    res = backup_mgr.rollback_to_session(s2)
    
    # S3 undone: f2 restored to v2
    # S2 undone: f2 deleted (was created), f1 restored to v1
    
    assert not f2.exists()
    assert f1.read_text() == "v1"
    
    # Sessions S2 and S3 should be gone
    current_sessions = backup_mgr.get_sessions()
    assert s1 in current_sessions
    assert s2 not in current_sessions
    assert s3 not in current_sessions
