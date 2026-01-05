"""Backup management and git integration."""
import os
import sys
import shutil
import json
import hashlib
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import config, APP_DATA_DIR, BACKUP_DIR
from .fs import run_command

# IO Checks & Caching
_io_cache: dict[str, Any] = {
    "git_installed": None,
    "git_repo_path": None,
    "git_is_repo": False,
    "last_check": 0.0
}
IO_CACHE_INTERVAL = 10.0

def io_throttled_update() -> None:
    t = time.time()
    if t - _io_cache["last_check"] > IO_CACHE_INTERVAL:
        _io_cache["git_installed"] = shutil.which("git") is not None
        
        cwd = str(Path.cwd())
        _io_cache["git_repo_path"] = cwd
        
        is_repo = False
        if _io_cache["git_installed"]:
            try:
                subprocess.run(
                    ["git", "rev-parse", "--is-inside-work-tree"],
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL,
                    cwd=cwd,
                    check=True
                )
                is_repo = True
            except Exception:
                is_repo = False
        
        _io_cache["git_is_repo"] = is_repo
        _io_cache["last_check"] = t

def force_io_cache_refresh() -> None:
    _io_cache["last_check"] = 0.0
    io_throttled_update()

def is_git_installed() -> bool:
    io_throttled_update()
    if _io_cache["git_installed"] is None:
        force_io_cache_refresh()
    return _io_cache["git_installed"]

def is_git_repo(path: Path | None = None) -> bool:
    p = path or Path.cwd()
    if path is None or str(path) == str(Path.cwd()):
        io_throttled_update()
        if _io_cache["git_repo_path"] == str(Path.cwd()):
            return _io_cache["git_is_repo"]
            
    if is_git_installed():
        try:
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=p,
                check=True
            )
            return True
        except Exception:
            return False
    return False

def init_git_repo(path: Path | None = None) -> tuple[bool, str]:
    p = path or Path.cwd()
    success, msg = run_command("git init", cwd=p)
    if success:
        force_io_cache_refresh()
    return success, msg

class GitShadowHandler:
    def __init__(self, branch_name=None):
        self.branch = branch_name or config.git_backup_branch
        self.env = os.environ.copy()
        self.env["GIT_INDEX_FILE"] = str(APP_DATA_DIR / "delta_git_index")

    def _run(self, args: list[str]) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                ["git"] + args, 
                env=self.env, 
                cwd=Path.cwd(), 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            output = result.stdout.strip() if result.stdout else result.stderr.strip()
            return result.returncode == 0, output
        except Exception as e:
            return False, str(e)

    def is_available(self) -> bool:
        return is_git_installed() and is_git_repo()

    def ensure_init(self) -> bool:
        if not self.is_available():
            return False
        ok, _ = self._run(["rev-parse", "--verify", self.branch])
        return True

    def commit_files(self, file_paths: list[str], message: str) -> str | None:
        if not self.is_available():
            return None

        self._run(["read-tree", "HEAD"])
        self._run(["add", "-A"])
        
        ok, tree_oid = self._run(["write-tree"])
        if not ok: return None
        
        ok, parent_oid = self._run(["rev-parse", self.branch])
        
        if ok and parent_oid:
            ok_ptree, parent_tree = self._run(["rev-parse", f"{self.branch}^{{tree}}"])
            if ok_ptree and parent_tree == tree_oid:
                return parent_oid

        commit_args = ["commit-tree", tree_oid, "-m", message]
        if ok and parent_oid:
            commit_args.extend(["-p", parent_oid])
            
        ok, commit_oid = self._run(commit_args)
        if not ok: return None
        
        self._run(["update-ref", f"refs/heads/{self.branch}", commit_oid])
        return commit_oid

    def restore_files(self, file_paths: list[str], revision: str = "HEAD") -> None:
        if not self.is_available(): return
        
        target = f"{self.branch}{revision}" if revision != "HEAD" else self.branch
        if revision.startswith("~") or revision.startswith("^"):
             target = f"{self.branch}{revision}"
             
        args = ["checkout", target, "--"] + file_paths
        self._run(args)

    def restore_snapshot(self, commit_hash: str) -> bool:
        if not self.is_available(): return False
        ok, _ = self._run(["checkout", commit_hash, "--", "."])
        return ok

    def get_history(self, limit: int = 20) -> list[dict]:
        if not self.is_available(): return []
        
        args = ["log", self.branch, f"-n{limit}", "--pretty=format:%H|%ad|%s", "--date=iso"]
        ok, out = self._run(args)
        if not ok or not out: return []
        
        history = []
        for line in out.splitlines():
            try:
                parts = line.split("|", 2)
                if len(parts) == 3:
                    history.append({
                        "session_id": parts[0],
                        "timestamp": parts[1],
                        "message": parts[2],
                        "files": [],
                        "source": "git"
                    })
            except Exception: pass
        return history
        
    def get_commit_files(self, commit_hash: str) -> list[str]:
        ok, out = self._run(["diff-tree", "--no-commit-id", "--name-only", "-r", commit_hash])
        if ok and out:
            return out.splitlines()
        return []

class BackupManager:
    def __init__(self, backup_dir: str = BACKUP_DIR):
        self.backup_dir = Path(backup_dir)
        self._current_session: str | None = None
        self._project_dir: Path = Path.cwd()
        self._active_manifest: dict = {"created": [], "modified": []}
    
    def _ensure_backup_dir(self) -> None:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        gitignore = self.backup_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")
    
    def _get_project_hash(self) -> str:
        return hashlib.md5(str(Path.cwd()).encode()).hexdigest()[:8]

    def _get_manifest_path(self, session_id: str) -> Path:
        return self.backup_dir / f"{session_id}_manifest.json"

    def _save_manifest(self, session_id: str) -> None:
        try:
            path = self._get_manifest_path(session_id)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._active_manifest, f)
        except Exception:
            pass

    def _load_manifest(self, session_id: str) -> dict:
        try:
            path = self._get_manifest_path(session_id)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {"created": [], "modified": []}

    def start_session(self) -> str:
        project_hash = self._get_project_hash()
        self._project_dir = Path.cwd()
        self._current_session = f"{project_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self._active_manifest = {"created": [], "modified": []}
        self._ensure_backup_dir()
        self._save_manifest(self._current_session)
        return self._current_session
    
    def backup_file(self, filepath: Path) -> Path | None:
        if not config.backup_enabled or not filepath.exists():
            return None
        
        session_id = self._current_session
        if not session_id:
            session_id = self.start_session()
        
        try:
            rel_path = filepath.relative_to(self._project_dir)
        except ValueError:
            rel_path = Path(str(filepath).replace(":", ""))

        safe_name = str(rel_path).replace(os.sep, "__").replace("/", "__")
        backup_path = self.backup_dir / f"{session_id}_{safe_name}.bak"
        
        try:
            shutil.copy2(filepath, backup_path)
            if str(rel_path) not in self._active_manifest["modified"]:
                self._active_manifest["modified"].append(str(rel_path))
                self._save_manifest(session_id)
            return backup_path
        except Exception:
            return None

    def register_created_file(self, filepath: Path) -> None:
        if not config.backup_enabled: return
        session_id = self._current_session
        if not session_id:
            session_id = self.start_session()
            
        try:
            rel_path = filepath.relative_to(self._project_dir)
            if str(rel_path) not in self._active_manifest["created"]:
                self._active_manifest["created"].append(str(rel_path))
                self._save_manifest(session_id)
        except Exception:
            pass
    
    def get_sessions(self) -> list[str]:
        if not self.backup_dir.exists():
            return []
        
        project_hash = self._get_project_hash()
        sessions = set()
        
        try:
            with os.scandir(str(self.backup_dir)) as it:
                for entry in it:
                    if entry.name.startswith(project_hash):
                        parts = entry.name.split("_")
                        if len(parts) >= 4:
                            sessions.add(f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}")
        except OSError:
            pass
        
        return sorted(sessions, reverse=True)
    
    def get_session_files(self, session_id: str) -> list[tuple[Path, Path]]:
        if not self.backup_dir.exists():
            return []
        
        files = []
        try:
            prefix = f"{session_id}_"
            with os.scandir(str(self.backup_dir)) as it:
                for entry in it:
                    if entry.name.startswith(prefix) and entry.name.endswith(".bak") and entry.is_file():
                        name_without_session = entry.name[len(prefix):-4]
                        original_rel = name_without_session.replace("__", os.sep)
                        original_path = Path.cwd() / original_rel
                        files.append((Path(entry.path), original_path))
        except OSError:
            pass
        return files

    def get_all_sessions_with_files(self) -> list[dict]:
        """Get all sessions and their files in a single pass."""
        if not self.backup_dir.exists():
            return []

        project_hash = self._get_project_hash()
        sessions = {} 

        try:
            with os.scandir(str(self.backup_dir)) as it:
                for entry in it:
                    if not entry.name.startswith(project_hash):
                        continue
                    
                    parts = entry.name.split("_")
                    if len(parts) < 4: continue
                    
                    sid = "_".join(parts[:4])
                    
                    if sid not in sessions:
                        # Attempt to extract timestamp for sorting: YYYYMMDD_HHMMSS
                        sort_key = f"{parts[1]}_{parts[2]}" if len(parts) >= 3 else sid
                        sessions[sid] = {
                            "session_id": sid,
                            "files": [],
                            "sort_key": sort_key
                        }
                    
                    if entry.name.endswith(".bak") and entry.is_file():
                        prefix_len = len(sid) + 1
                        if len(entry.name) > prefix_len:
                            safe_name = entry.name[prefix_len:-4]
                            original_rel = safe_name.replace("__", os.sep)
                            original_path = Path.cwd() / original_rel
                            sessions[sid]["files"].append((Path(entry.path), original_path))
                            
        except OSError:
            return []
            
        results = list(sessions.values())
        results.sort(key=lambda x: x["sort_key"], reverse=True)
        return results
    
    def undo_session(self, session_id: str) -> dict[str, str]:
        results = {}
        manifest = self._load_manifest(session_id)
        
        def _clean_empty_dirs(path, root):
            try:
                if path == root: return
                if path.exists() and path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
                    _clean_empty_dirs(path.parent, root)
            except Exception: pass

        for rel_path in manifest.get("created", []):
            full_path = Path.cwd() / rel_path
            try:
                if full_path.exists():
                    full_path.unlink()
                    _clean_empty_dirs(full_path.parent, Path.cwd())
                    results[str(full_path)] = "deleted (was created)"
                else:
                    results[str(full_path)] = "already missing"
            except Exception as e:
                results[str(full_path)] = f"error deleting: {e}"

        session_files = self.get_session_files(session_id)
        for backup_path, original_path in session_files:
            try:
                if backup_path.exists():
                    original_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, original_path)
                    results[str(original_path)] = "restored"
                else:
                    results[str(original_path)] = "backup not found"
            except Exception as e:
                results[str(original_path)] = f"error restoring: {e}"
        
        return results
    
    def delete_session(self, session_id: str) -> int:
        count = 0
        for backup_path, _ in self.get_session_files(session_id):
            try:
                backup_path.unlink()
                count += 1
            except Exception:
                pass
        
        try:
            manifest_path = self._get_manifest_path(session_id)
            if manifest_path.exists():
                manifest_path.unlink()
        except Exception:
            pass
            
        return count
    
    def rollback_to_session(self, target_session_id: str) -> dict[str, str]:
        sessions = self.get_sessions()
        try:
            target_idx = sessions.index(target_session_id)
        except ValueError:
            return {"error": f"Session {target_session_id} not found"}
        
        sessions_to_undo = sessions[:target_idx+1]
        results = {}
        
        for session_id in sessions_to_undo:
            undo_res = self.undo_session(session_id)
            for file_path, status in undo_res.items():
                results[file_path] = status
            self.delete_session(session_id)
            
        return results

    def cleanup_old_backups(self, keep_sessions: int = 10) -> int:
        sessions = self.get_sessions()
        deleted = 0
        for session_id in sessions[keep_sessions:]:
            self.delete_session(session_id)
            deleted += 1
        return deleted
    
    def clear_all_backups(self) -> int:
        sessions = self.get_sessions()
        deleted = 0
        for session_id in sessions:
            self.delete_session(session_id)
            deleted += 1
        return deleted

backup_manager = BackupManager()

def undo_last_changes() -> dict[str, str]:
    if config.use_git_backup and config.backup_enabled:
        git = GitShadowHandler()
        if git.is_available():
            files = git.get_commit_files(git.branch)
            if not files:
                return {"error": "No changes found in git history to undo"}
            git.restore_files(files, "~1")
            git.commit_files(files, "Delta: Reverted last change")
            return {f: "Restored via Git" for f in files}

    sessions = backup_manager.get_sessions()
    if not sessions:
        return {"error": "No backup sessions found"}
    
    return backup_manager.undo_session(sessions[0])

def iter_backup_items():
    """Yield available backup items."""
    # Efficient single-pass fetch for file backups
    for item in backup_manager.get_all_sessions_with_files():
        session_id = item["session_id"]
        files = item["files"]
        
        parts = session_id.split("_")
        try:
            if len(parts) >= 3:
                dt = datetime.strptime(f"{parts[1]}_{parts[2]}", "%Y%m%d_%H%M%S")
                timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp = session_id
        except ValueError:
            timestamp = session_id
            
        yield {
            "session_id": session_id,
            "timestamp": timestamp,
            "sort_key": item["sort_key"],
            "files": [str(orig) for _, orig in files],
            "source": "file",
            "message": "File Backup"
        }

    git = GitShadowHandler()
    if git.is_available():
        history = git.get_history()
        for item in history:
            item["sort_key"] = item["timestamp"]
            git_files = git.get_commit_files(item["session_id"])
            item["files"] = git_files
            yield item

def get_available_backups() -> list[dict]:
    all_backups = list(iter_backup_items())
    all_backups.sort(key=lambda x: str(x.get("sort_key", "")), reverse=True)
    return all_backups

def restore_git_backup(commit_hash: str) -> dict[str, str]:
    git = GitShadowHandler()
    if git.restore_snapshot(commit_hash):
        return {"git": f"Restored snapshot {commit_hash[:8]}"}
    return {"error": "Failed to restore git snapshot"}