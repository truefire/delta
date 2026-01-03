"""Reporting and analysis tools."""
import difflib
import logging
import os
import tempfile
import webbrowser
from pathlib import Path

from .backups import GitShadowHandler, backup_manager

logger = logging.getLogger(__name__)

def _get_git_changes() -> list[tuple[str, list[str], list[str]]]:
    import subprocess
    changes = []
    
    def git_cmd(args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True, cwd=Path.cwd())
        except subprocess.CalledProcessError:
            return ""

    status_out = git_cmd(["git", "diff", "HEAD", "--name-status"])
    
    for line in status_out.splitlines():
        if not line.strip(): continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2: continue
        status, rel_path = parts
        
        orig_lines = []
        new_lines = []
        
        if status != "A":
            try:
                content = git_cmd(["git", "show", f"HEAD:{rel_path}"])
                orig_lines = content.splitlines(keepends=True)
            except Exception: pass
            
        if status != "D":
            p = Path.cwd() / rel_path
            if p.exists():
                new_lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                
        changes.append((rel_path, orig_lines, new_lines))
        
    untracked_out = git_cmd(["git", "ls-files", "--others", "--exclude-standard"])
    for rel_path in untracked_out.splitlines():
        if not rel_path.strip(): continue
        
        orig_lines = []
        p = Path.cwd() / rel_path
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    if b'\0' in f.read(1024): continue
                new_lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                changes.append((f"{rel_path} (untracked)", orig_lines, new_lines))
            except Exception: pass
            
    return changes

def open_diff_report(session_ids: str | list[str] | None = None, use_git: bool = False, compare_session_id: str | None = None, diff_against_disk: bool = False) -> None:
    try:
        full_diff_text = ""
        project_root = Path.cwd()
        title_extra = "Git Local"
        change_sets = [] 

        if use_git:
            change_sets = _get_git_changes()
        elif session_ids:
            is_git_mode = False
            first_id = session_ids[0] if isinstance(session_ids, list) else session_ids
            
            if "_" not in first_id:
                is_git_mode = True
            elif first_id == "GIT_LATEST":
                is_git_mode = True
            
            if is_git_mode:
                git = GitShadowHandler()
                
                if session_ids == "GIT_LATEST" or session_ids == ["GIT_LATEST"]:
                    title_extra = "Latest Session"
                    changed_files = git.get_commit_files(git.branch)
                    for rel_path in changed_files:
                        ok, content = git._run(["show", f"{git.branch}~1:{rel_path}"])
                        orig_lines = content.splitlines(keepends=True) if ok else []

                        if diff_against_disk:
                            p = project_root / rel_path
                            if p.exists():
                                new_lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                            else:
                                new_lines = []
                        else:
                            ok_new, content_new = git._run(["show", f"{git.branch}:{rel_path}"])
                            new_lines = content_new.splitlines(keepends=True) if ok_new else []

                        change_sets.append((rel_path, orig_lines, new_lines))
                else:
                    if isinstance(session_ids, str): session_ids = [session_ids]
                    title_extra = session_ids[0][:8] if len(session_ids) == 1 else f"Range ({len(session_ids)} commits)"

                    for sid in session_ids:
                        files = git.get_commit_files(sid)
                        for rel_path in files:
                            if any(c[0] == rel_path for c in change_sets): continue
                            
                            ok, content = git._run(["show", f"{sid}~1:{rel_path}"])
                            orig_lines = content.splitlines(keepends=True) if ok else []
                            
                            if diff_against_disk:
                                p = project_root / rel_path
                                if p.exists():
                                    new_lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                                else:
                                    new_lines = []
                            else:
                                ok_new, content_new = git._run(["show", f"{sid}:{rel_path}"])
                                new_lines = content_new.splitlines(keepends=True) if ok_new else []
                                
                            change_sets.append((rel_path, orig_lines, new_lines))
            else:
                if not session_ids: return
                if isinstance(session_ids, str):
                    session_ids = [session_ids]
                title_extra = session_ids[0] if len(session_ids) == 1 else f"Range ({len(session_ids)} sessions)"

                files_map: dict[str, Path] = {}
                created_files: set[str] = set()
                all_known = backup_manager.get_sessions()
                
                def age_key(sid):
                    try: return all_known.index(sid)
                    except ValueError: return -1
                    
                sorted_sids = sorted(session_ids, key=age_key, reverse=True)
                
                for sid in sorted_sids:
                    for backup_path, original_path in backup_manager.get_session_files(sid):
                        try:
                            rel = str(original_path.relative_to(project_root)).replace("\\", "/")
                            if rel not in files_map and rel not in created_files:
                                files_map[rel] = backup_path
                        except ValueError: pass

                    manifest = backup_manager._load_manifest(sid)
                    for rel in manifest.get("created", []):
                        rel = rel.replace("\\", "/")
                        if rel not in files_map:
                            created_files.add(rel)

                all_rels = set(files_map.keys()) | created_files
                for rel_path in sorted(list(all_rels)):
                    full_path = project_root / rel_path
                    if rel_path in files_map:
                        valid_bak = files_map[rel_path]
                        if valid_bak.exists():
                            orig_lines = valid_bak.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                        else:
                            orig_lines = []
                    else:
                        orig_lines = []

                    new_lines = []
                    content_found = False

                    if compare_session_id:
                        try:
                            start_idx = all_known.index(compare_session_id)
                            for i in range(start_idx, -1, -1):
                                scan_sid = all_known[i]
                                scan_files = backup_manager.get_session_files(scan_sid)
                                for bak_path, orig_path in scan_files:
                                    try:
                                        sc_rel = str(orig_path.relative_to(project_root)).replace("\\", "/")
                                        if sc_rel == rel_path:
                                            if bak_path.exists():
                                                new_lines = bak_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                                            content_found = True
                                            break
                                    except ValueError: pass
                                if content_found: break

                                scan_manifest = backup_manager._load_manifest(scan_sid)
                                if any(c.replace("\\", "/") == rel_path for c in scan_manifest.get("created", [])):
                                    new_lines = []
                                    content_found = True
                                    break
                        except ValueError: pass
                    
                    if not content_found:
                        if full_path.exists():
                            new_lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
                        else:
                            new_lines = []

                    change_sets.append((rel_path, orig_lines, new_lines))

        for display_path, orig_lines, new_lines in change_sets:
            try:
                diff = difflib.unified_diff(
                    orig_lines, 
                    new_lines, 
                    fromfile=f"a/{display_path}", 
                    tofile=f"b/{display_path}",
                    n=3
                )
                full_diff_text += "".join(diff)
            except Exception as e:
                logger.error(f"Error diffing {display_path}: {e}")

        if not full_diff_text.strip():
            full_diff_text = "No textual changes detected."

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <title>Delta Diff - {title_extra}</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css" />
            <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/diff2html/bundles/css/diff2html.min.css" />
            <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/diff2html/bundles/js/diff2html-ui.min.js"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background: #f4f4f4; }}
                .d2h-file-header {{ background-color: #f0f0f0 !important; }}
                h2 {{ color: #333; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <h2>Review: {title_extra}</h2>
            <div id="diff_target"></div>
            <script>
                document.addEventListener('DOMContentLoaded', function () {{
                    var diffString = `{full_diff_text.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$').replace('</', '<\\/')}`;
                    var targetElement = document.getElementById('diff_target');
                    var configuration = {{
                        drawFileList: true,
                        fileListToggle: false,
                        fileListStartVisible: false,
                        fileContentToggle: true,
                        matching: 'lines',
                        outputFormat: 'side-by-side',
                        synchronisedScroll: true,
                        highlight: true,
                        renderNothingWhenEmpty: false,
                    }};
                    var diff2htmlUi = new Diff2HtmlUI(targetElement, diffString, configuration);
                    diff2htmlUi.draw();
                    diff2htmlUi.highlightCode();
                }});
            </script>
        </body>
        </html>
        """

        fd, path = tempfile.mkstemp(suffix=".html", prefix=f"delta_review_")
        with os.fdopen(fd, 'w', encoding="utf-8") as f:
            f.write(html_content)
        
        webbrowser.open(f"file://{path}")

    except Exception as e:
        logger.error(f"Failed to open diff report: {e}")