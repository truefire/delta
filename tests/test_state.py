import pytest
from pathlib import Path
from application_state import (
    ChatSession, build_file_tree, AppState, state, to_relative,
    save_state, load_state, toggle_file_selection, toggle_folder_selection,
    sync_settings_from_config, sync_config_from_settings,
    save_fileset, load_fileset, get_saves_list, load_individual_session,
    change_working_directory, load_presets, save_presets
)
import core

def test_chat_session_serialization():
    sess = ChatSession(id=1)
    sess.last_prompt = "Fix the bug"
    sess.history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    sess.is_ask_mode = True
    sess.session_added_files = {Path("foo.txt"), Path("bar.py")}
    
    data = sess.to_dict()
    
    assert data["last_prompt"] == "Fix the bug"
    assert data["is_ask_mode"] is True
    assert "foo.txt" in data["session_added_files"]
    
    sess2 = ChatSession(id=2)
    sess2.from_dict(data)
    
    assert sess2.last_prompt == sess.last_prompt
    assert sess2.history == sess.history
    assert sess2.is_ask_mode == sess.is_ask_mode
    assert Path("foo.txt") in sess2.session_added_files

def test_file_tree_building():
    # Setup some paths
    root = Path("/project")
    paths = [
        root / "main.py",
        root / "utils.py",
        root / "src/algo.py",
        root / "src/core/base.py",
        root / "tests/test.py"
    ]
    
    tree = build_file_tree(paths, root)
    
    # Structure match:
    assert "_files" in tree
    files = [x[0] for x in tree["_files"]] # (name, path)
    assert "main.py" in files
    assert "utils.py" in files
    
    # Root folders are direct keys
    assert "src" in tree
    # But src node has structure
    assert "algo.py" in [x[0] for x in tree["src"]["_files"]]
    
    src_node = tree["src"]
    assert "core" in src_node["_children"]
    core_node = src_node["_children"]["core"]
    assert "base.py" in [x[0] for x in core_node["_files"]]

def test_app_state_reset():
    state.active_session_id = 999
    
    from application_state import init_app_state
    init_app_state()
    
    assert state.active_session_id is None
    assert state.sessions == {}

def test_to_relative(temp_cwd):
    # Setup
    p = temp_cwd / "subdir" / "file.txt"
    assert to_relative(p) == Path("subdir/file.txt")
    
    # Test path outside CWD (if possible on this OS)
    outside = Path("/tmp/other_random_path_delta_test")
    # relative_to throws ValueError if not relative
    try:
         outside.relative_to(temp_cwd)
    except ValueError:
         assert to_relative(outside) == outside

def test_recover_lost_prompt():
    sess = ChatSession(id=1)
    sess.last_prompt = "Old prompt"
    sess.input_text = ""
    sess.history = [{"role": "user", "content": "Different prompt"}]
    
    # Cases where last prompt differs from history
    sess.recover_lost_prompt()
    assert sess.input_text == "Old prompt"
    
    # Case where input already exists
    sess.input_text = "Draft"
    sess.recover_lost_prompt()
    assert sess.input_text == "Draft"

def test_persistence(temp_cwd):
    # Change persistent paths variables to temp dir
    import application_state
    original_session_dir = application_state.SESSIONS_DIR
    application_state.SESSIONS_DIR = temp_cwd / "sessions"
    application_state.SESSIONS_DIR.mkdir()

    try:
        # 1. Setup state
        state.sessions = {
            1: ChatSession(id=1, last_prompt="Saved prompt", input_text="draft")
        }
        state.active_session_id = 1
        
        save_state("mysave")
        
        # 2. Clear state
        state.sessions = {}
        state.active_session_id = None
        
        # 3. Load state
        load_state("mysave")
        
        assert 1 in state.sessions
        assert state.sessions[1].last_prompt == "Saved prompt"
        assert state.sessions[1].input_text == "draft"
        assert state.active_session_id == 1
    
    finally:
        application_state.SESSIONS_DIR = original_session_dir

def test_selection_toggles(temp_cwd):
    # We need to rely on the global state object logic for paths
    state.selected_files = set()
    # Mock SCAN results roughly
    state.file_paths = [Path("a.py"), Path("b.py"), Path("sub/c.py")]
    
    toggle_file_selection(Path("a.py"), True)
    assert Path("a.py") in state.selected_files
    
    toggle_file_selection(Path("a.py"), False)
    assert Path("a.py") not in state.selected_files
    
    # Folder toggle logic iterates state.file_paths
    toggle_folder_selection(Path("sub"), True)
    assert Path("sub/c.py") in state.selected_files
    
    toggle_folder_selection(Path("sub"), False)
    assert Path("sub/c.py") not in state.selected_files

def test_config_sync():
    # Setup core config
    core.config.model = "model_A"
    
    # Mock AVAILABLE_MODELS keys since index lookup depends on it
    original_models = core.AVAILABLE_MODELS.copy()
    core.AVAILABLE_MODELS.clear()
    core.AVAILABLE_MODELS.update({"model_A": {}, "model_B": {}})
    
    try:
        # Config -> UI State
        sync_settings_from_config()
        assert state.model_idx == 0
        
        # UI State -> Config
        state.model_idx = 1
        state.max_tries = "5"
        state.recursions = "3"
        state.timeout = "99.5"
        
        sync_config_from_settings()
        assert core.config.model == "model_B"
        assert core.config.default_tries == 5
        assert core.config.default_recurse == 3
        assert core.config.default_timeout == 99.5
    finally:
        core.AVAILABLE_MODELS.clear()
        core.AVAILABLE_MODELS.update(original_models)
        
def test_fileset_persistence(temp_cwd):
    # Hack global paths
    import application_state
    orig_path = application_state.PRESETS_PATH
    application_state.PRESETS_PATH = str(temp_cwd / "presets_test.json")
    
    state.selected_files = {Path("a.py")}
    state.file_checked = {Path("a.py"): True}
    state.validation_cmd = "pytest"
    
    save_fileset()
    
    # Clear
    state.selected_files = set()
    state.validation_cmd = ""
    state.presets = {}
    
    # Load
    load_presets()
    load_fileset()
    
    # Relative path handling is used in load_fileset. 
    # Since we are in temp_cwd, 'a.py' is relative to CWD.
    assert Path("a.py") in state.selected_files
    assert state.validation_cmd == "pytest"
    assert state.file_checked.get(Path("a.py")) is True
    
    application_state.PRESETS_PATH = orig_path

def test_change_working_directory(temp_cwd):
    subdir = temp_cwd / "subdir"
    subdir.mkdir()
    
    state.active_session_id = 12345
    
    change_working_directory(str(subdir))
    
    assert Path.cwd() == subdir
    assert state.active_session_id != 12345
    assert state.sessions # Auto-created session

def test_get_saves_list(temp_cwd):
    import application_state
    orig = application_state.SESSIONS_DIR
    try:
        application_state.SESSIONS_DIR = temp_cwd / "saves"
        application_state.SESSIONS_DIR.mkdir()
        
        (application_state.SESSIONS_DIR / "save1.json").touch()
        import time
        time.sleep(0.1)
        (application_state.SESSIONS_DIR / "save2.json").touch()
        
        saves = get_saves_list()
        assert len(saves) == 2
        assert saves[0]["name"] == "save2" # Newest
    finally:
        application_state.SESSIONS_DIR = orig

def test_load_individual_session(temp_cwd):
    import application_state
    import json
    orig = application_state.SESSIONS_DIR
    try:
        application_state.SESSIONS_DIR = temp_cwd / "saves"
        application_state.SESSIONS_DIR.mkdir()
        
        data = {
            "sessions": [
                {"__id__": 1, "last_prompt": "sess1"},
                {"__id__": 2, "last_prompt": "sess2"}
            ]
        }
        with open(application_state.SESSIONS_DIR / "multi.json", "w") as f:
            json.dump(data, f)
        
        state.sessions = {}
        load_individual_session("multi", 1) # Load 2nd session
        
        assert len(state.sessions) == 1
        s = next(iter(state.sessions.values()))
        assert s.last_prompt == "sess2"
    finally:
        application_state.SESSIONS_DIR = orig

def test_close_and_unqueue_session():
    from application_state import create_session, close_session, unqueue_session, state, init_app_state
    
    # Clean global state
    init_app_state()
    
    # Setup
    s1 = create_session()
    s2 = create_session()
    state.active_session_id = s2.id
    
    # Unqueue
    s1.is_queued = True
    state.impl_queue.append(s1.id)
    unqueue_session(s1.id)
    assert s1.id not in state.impl_queue
    assert s1.is_queued is False

    # Close active
    close_session(s2.id)
    assert s2.id not in state.sessions
    assert state.active_session_id == s1.id
    
    # Close last
    close_session(s1.id)
    assert state.active_session_id is None

def test_delete_save(temp_cwd):
    import application_state
    orig = application_state.SESSIONS_DIR
    application_state.SESSIONS_DIR = temp_cwd
    
    (temp_cwd / "mysave.json").touch()
    
    from application_state import delete_save
    delete_save("mysave")
    
    assert not (temp_cwd / "mysave.json").exists()
    
    application_state.SESSIONS_DIR = orig