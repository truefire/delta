"""Files panel logic."""
import fnmatch
import threading
from pathlib import Path
from imgui_bundle import imgui

import core
from core import get_file_stats, calculate_input_cost, AVAILABLE_MODELS
from application_state import (
    state, save_fileset, save_presets, log_message,
    toggle_file_selection, toggle_folder_selection, refresh_project_files,
    to_relative, tree_lock, queue_scan_request
)
from styles import STYLE
from .common import render_tooltip


def _render_group_row(name: str):
    if name not in state.presets or name == "__active":
        return

    group_data = state.presets[name]
    group_files = group_data.get("files", [])
    
    imgui.push_id(name)
    
    if imgui.small_button("Set"):
        for f in state.selected_files:
            state.file_checked[f] = False
        
        for f in group_files:
            path = to_relative(Path(f))
            if path.exists():
                state.selected_files.add(path)
                state.file_checked[path] = True
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Replace selection with this group")
    
    imgui.same_line()
    if imgui.small_button("+"):
        for f in group_files:
            path = to_relative(Path(f))
            if path.exists():
                state.selected_files.add(path)
                state.file_checked[path] = True
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Add this group to selection")
        
    imgui.same_line()
    if imgui.small_button("-"):
        for f in group_files:
            path = to_relative(Path(f))
            if path in state.selected_files:
                state.file_checked[path] = False
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Remove this group from selection")
    
    imgui.same_line()
    imgui.text(f"{name} [{len(group_files)} files]")

    if imgui.begin_popup_context_item("group_ctx"):
        if imgui.selectable("Delete Group", False)[0]:
            del state.presets[name]
            save_presets()
        imgui.end_popup()
    imgui.pop_id()


def _render_group_manage_row(name: str):
    if name not in state.presets or name == "__active":
        return

    group_data = state.presets[name]
    group_files = group_data.get("files", [])
    
    imgui.push_id(f"manage_{name}")
    
    if imgui.small_button("Load"):
        for f in state.selected_files:
            state.file_checked[f] = False
        
        for f in group_files:
            path = to_relative(Path(f))
            if path.exists():
                state.selected_files.add(path)
                state.file_checked[path] = True
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Load this group into selection")
    
    imgui.same_line()
    if imgui.small_button("Update"):
        files_to_save = [str(f) for f in state.selected_files if state.file_checked.get(f, True)]
        state.presets[name] = {"files": files_to_save}
        save_presets()
        log_message(f"Updated group '{name}' with {len(files_to_save)} files")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Overwrite this group with current selection")
    
    imgui.same_line()
    if imgui.small_button("Delete"):
        del state.presets[name]
        save_presets()
        log_message(f"Deleted group '{name}'")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Delete this group")
    
    imgui.same_line()
    imgui.text(f"{name} [{len(group_files)} files]")
    
    imgui.pop_id()


def update_app_stats():
    # Rolling file existence check
    if state.cached_sorted_files:
        files_to_check = 5
        total_files = len(state.cached_sorted_files)
        
        if state.rolling_check_idx >= total_files:
            state.rolling_check_idx = 0
            
        start_idx = state.rolling_check_idx
        end_idx = min(start_idx + files_to_check, total_files)
        
        recalc_needed = False
        for i in range(start_idx, end_idx):
            f = state.cached_sorted_files[i]
            exists = f.exists()
            if state.file_exists_cache.get(f) != exists:
                state.file_exists_cache[f] = exists
                recalc_needed = True
        
        state.rolling_check_idx = end_idx if end_idx < total_files else 0
        
        if recalc_needed:
            state.stats_dirty = True

    if state.stats_dirty:
        state.cached_sorted_files = sorted(state.selected_files, key=lambda p: str(p).lower())
        
        for f in state.cached_sorted_files:
             if f not in state.file_exists_cache:
                 state.file_exists_cache[f] = f.exists()
        
        cwd = Path.cwd()
        state.folder_selection_counts.clear()
        for f_rel in state.selected_files:
            try:
                f_abs = (cwd / f_rel).resolve()
                parent = f_abs.parent
                while True:
                    if not str(parent).startswith(str(cwd)):
                        break
                    
                    k = str(parent)
                    state.folder_selection_counts[k] = state.folder_selection_counts.get(k, 0) + 1
                    
                    if parent == cwd:
                        break
                    parent = parent.parent
            except Exception:
                pass

        checked = [f for f in state.cached_sorted_files 
                   if state.file_checked.get(f, True) and state.file_exists_cache.get(f, True)]
                   
        state.cached_total_lines = sum(get_file_stats(f)[0] for f in checked)
        state.cached_total_tokens = sum(get_file_stats(f)[1] for f in checked)
        
        model_list = list(AVAILABLE_MODELS.keys())
        model_name = model_list[state.model_idx] if state.model_idx < len(model_list) else ""
        _, state.cached_cost_str = calculate_input_cost(state.cached_total_tokens, model_name)
        
        state.stats_dirty = False


def render_files_panel():
    imgui.text("Groups")
    imgui.separator()

    if state.presets:
        for name in list(state.presets.keys()):
            _render_group_row(name)
    else:
        imgui.text_colored(STYLE.get_imvec4("fg_dim"), "No groups saved")

    imgui.dummy(imgui.ImVec2(0, 10))
    imgui.separator()

    update_app_stats()
    sorted_files = state.cached_sorted_files

    checked_count = sum(1 for f in sorted_files if state.file_checked.get(f, True) and state.file_exists_cache.get(f, True))

    header_text = f"Selected Context ({checked_count}/{len(state.selected_files)} files)"
    imgui.text(header_text)
    if state.is_scanning:
        imgui.same_line()
        imgui.text_colored(STYLE.get_imvec4("queued"), "(Scanning...)")
    imgui.separator()

    imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"Lines: {state.cached_total_lines} | Tokens: ~{state.cached_total_tokens} | Cost: {state.cached_cost_str}")
    imgui.spacing()

    list_height = min(240, 22 * len(state.selected_files) + 10)
    imgui.begin_child("selected_files_list", imgui.ImVec2(0, list_height), child_flags=imgui.ChildFlags_.borders)

    table_flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.sizing_fixed_fit
    if imgui.begin_table("selected_files_table", 3, table_flags):
        imgui.table_setup_column("##chk", imgui.TableColumnFlags_.width_fixed, 24)
        imgui.table_setup_column("Tokens", imgui.TableColumnFlags_.width_fixed, 50)
        imgui.table_setup_column("File", imgui.TableColumnFlags_.width_stretch)

        row_height = imgui.get_frame_height()
        clipper = imgui.ListClipper()
        clipper.begin(len(sorted_files), row_height)
        while clipper.step():
            for i in range(clipper.display_start, clipper.display_end):
                f = sorted_files[i]
                exists = state.file_exists_cache.get(f, False)
                if f not in state.file_checked:
                    state.file_checked[f] = True

                _, tokens, _ = get_file_stats(f)
                file_display = str(f.relative_to(Path.cwd())) if f.is_absolute() else str(f)

                imgui.table_next_row(0, row_height)

                imgui.table_next_column()
                imgui.push_id(str(f))
                is_checked = state.file_checked[f]
                changed, new_val = imgui.checkbox("##chk", is_checked)
                if changed:
                    state.file_checked[f] = new_val
                    state.stats_dirty = True
                    save_fileset()

                imgui.table_next_column()
                if exists:
                    imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"{tokens:>5}")
                else:
                    imgui.text_colored(STYLE.get_imvec4("btn_cncl"), " -- ")

                imgui.table_next_column()
                if not exists:
                    imgui.text_colored(STYLE.get_imvec4("btn_cncl"), f"{file_display} (Missing)")
                elif is_checked:
                    imgui.text(file_display)
                else:
                    imgui.text_colored(STYLE.get_imvec4("fg_dim"), file_display)

                if imgui.is_item_hovered() and imgui.is_mouse_clicked(imgui.MouseButton_.middle):
                    state.selected_files.discard(f)
                    state.file_checked.pop(f, None)
                    state.stats_dirty = True
                    save_fileset()

                if imgui.is_item_hovered():
                    imgui.set_tooltip("Middle-click to remove")

                imgui.pop_id()

        imgui.end_table()

    imgui.dummy(imgui.ImVec2(0, 30))
    imgui.end_child()

    if imgui.button("Check All"):
        for f in state.selected_files:
            state.file_checked[f] = True
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Select all files in the list.")

    imgui.same_line()
    if imgui.button("Uncheck All"):
        for f in state.selected_files:
            state.file_checked[f] = False
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Deselect all files.")

    imgui.same_line()
    if imgui.button("Remove Unchecked"):
        unchecked = [f for f in state.selected_files if not state.file_checked.get(f, True)]
        for f in unchecked:
            state.selected_files.discard(f)
            state.file_checked.pop(f, None)
        state.stats_dirty = True
        save_fileset()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Remove deselected files from this list.")

    imgui.separator()

    if imgui.button("Manage Context", imgui.ImVec2(-1, 0)):
        state.show_context_manager = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("Open the advanced file tree browser.")

    if state.show_create_group_popup:
        imgui.open_popup("Create Group")

    if imgui.begin_popup_modal("Create Group", None, imgui.WindowFlags_.always_auto_resize)[0]:
        imgui.text("Enter group name:")
        imgui.set_next_item_width(200)
        _, state.new_group_name = imgui.input_text("##group_name", state.new_group_name)

        imgui.spacing()
        if imgui.button("Create", imgui.ImVec2(100, 0)):
            name = state.new_group_name.strip()
            if name:
                files_to_save = [str(f) for f in state.selected_files if state.file_checked.get(f, True)]
                state.presets[name] = {"files": files_to_save}
                save_presets()
                log_message(f"Created group '{name}' with {len(files_to_save)} files")
                state.show_create_group_popup = False
                imgui.close_current_popup()
            else:
                log_message("Group name cannot be empty")

        imgui.same_line()
        if imgui.button("Cancel", imgui.ImVec2(100, 0)):
            state.show_create_group_popup = False
            imgui.close_current_popup()

        imgui.end_popup()

def _search_worker(query_id: int, search_text: str):
    try:
        with tree_lock:
            file_paths = list(state.file_paths)

        search_lower = search_text.lower()
        is_glob = any(c in search_lower for c in "*?[]")
        
        if state.search_query_id != query_id: return

        # 1. Flat Filter
        if is_glob:
            filtered = [f for f in file_paths if fnmatch.fnmatch(str(f).lower(), search_lower)]
        else:
            filtered = [f for f in file_paths if search_lower in str(f).lower()]

        if state.search_query_id != query_id: return

        # 2. Tree Whitelists
        search_whitelist_dirs = set()
        search_whitelist_files = set()
        cwd = Path.cwd()
        
        for f_rel in filtered:
            if state.search_query_id != query_id: return
            f_abs = cwd / f_rel
            search_whitelist_files.add(f_abs)
            
            p = f_abs.parent
            while True:
                if p in search_whitelist_dirs:
                     break
                search_whitelist_dirs.add(p)
                if p == cwd or len(p.parts) <= len(cwd.parts):
                    break
                p = p.parent
                
        search_whitelist_dirs.add(cwd)
        if state.search_query_id != query_id: return

        state.gui_queue.put({
            "type": "search_result",
            "query_id": query_id,
            "flat": filtered,
            "dirs": search_whitelist_dirs,
            "files": search_whitelist_files
        })
    except Exception as e:
        log_message(f"Search thread error: {e}")


def trigger_context_search():
    state.last_context_search = state.context_search_text

    if not state.context_search_text:
        state.cached_flat_filtered_files = None
        state.cached_whitelist_dirs = None
        state.cached_whitelist_files = None
        state.is_searching = False
        state.view_tree_dirty = True
        return

    state.view_tree_dirty = False
    state.search_query_id += 1
    state.is_searching = True
    
    threading.Thread(
        target=_search_worker,
        args=(state.search_query_id, state.context_search_text),
        daemon=True
    ).start()


def _rebuild_context_rows():
    """Rebuild the flattened list of tree rows for the context manager."""
    visible_rows = []
    
    def count_files_recursive(node):
        count = len(node.get("_files", []))
        if "_children" in node:
            for child in node["_children"].values():
                count += count_files_recursive(child)
        return count

    search_whitelist_dirs = state.cached_whitelist_dirs
    search_whitelist_files = state.cached_whitelist_files

    def build_flat_tree(node_name, node, current_path, depth):
        full_path = current_path / node_name
        
        if search_whitelist_dirs is not None and full_path not in search_whitelist_dirs:
            return

        folder_key = str(full_path)
        
        if search_whitelist_dirs is not None:
            is_open = True
        else:
            is_open = state.folder_states.get(folder_key, False)
        
        if is_open and not node.get("_scanned") and not node.get("_scanning"):
            node["_scanning"] = True
            queue_scan_request(full_path, priority=0)

        visible_rows.append({
            "type": "folder",
            "name": node_name,
            "key": folder_key,
            "is_open": is_open,
            "depth": depth,
            "scanning": node.get("_scanning", False),
            "path": full_path,
            "total_files": count_files_recursive(node)
        })
        
        if is_open:
            if node.get("_children"):
                for child_name, child_node in sorted(node["_children"].items(), key=lambda x: x[0].lower()):
                    build_flat_tree(child_name, child_node, full_path, depth + 1)
                    
            if node.get("_files"):
                for fname, fpath in sorted(node["_files"]):
                    if search_whitelist_files is not None and fpath not in search_whitelist_files:
                        continue
                    visible_rows.append({
                        "type": "file",
                        "name": fname,
                        "path": fpath,
                        "depth": depth + 1
                    })

    with tree_lock:
         root_node = state.file_tree
         
         if "_children" in root_node:
             for dname, dnode in sorted(root_node["_children"].items(), key=lambda x: x[0].lower()):
                 build_flat_tree(dname, dnode, Path.cwd(), 0)
         
         if "_files" in root_node:
             for fname, fpath in sorted(root_node["_files"]):
                 if search_whitelist_files is not None and fpath not in search_whitelist_files:
                     continue
                 visible_rows.append({
                     "type": "file",
                     "name": fname,
                     "path": fpath,
                     "depth": 0
                 })
    
    state.cached_context_rows = visible_rows
    state.view_tree_dirty = False


def render_context_manager():
    if not state.show_context_manager:
        return

    imgui.set_next_window_size(imgui.ImVec2(900, 600), imgui.Cond_.first_use_ever)
    opened, state.show_context_manager = imgui.begin("Manage Context", state.show_context_manager)

    if opened:
        if state.context_search_text != state.last_context_search:
            trigger_context_search()
        elif state.view_tree_dirty and state.context_search_text and state.is_scanning:
            trigger_context_search()

        imgui.text("Search:")
        imgui.same_line()
        imgui.set_next_item_width(300)
        changed, state.context_search_text = imgui.input_text("##search", state.context_search_text)
        render_tooltip("Filter files by name.")

        if state.is_searching:
            imgui.same_line()
            imgui.text_colored(STYLE.get_imvec4("queued"), "(Searching...)")

        imgui.same_line()
        _, state.context_flatten_search = imgui.checkbox("Flatten", state.context_flatten_search)
        render_tooltip("Show search results as a flat list.")

        imgui.same_line()
        if imgui.button("Refresh"):
            refresh_project_files()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Rescan directory for new files.")

        imgui.same_line()
        if imgui.button("Clear"):
            state.selected_files.clear()
            state.file_checked.clear()
            state.file_exists_cache.clear()
            state.stats_dirty = True
            save_fileset()
        
        if imgui.is_item_hovered():
            imgui.set_tooltip("Deselect all files.")

        update_app_stats()
        checked_count = sum(1 for f in state.cached_sorted_files if state.file_checked.get(f, True) and state.file_exists_cache.get(f, True))

        imgui.same_line()
        imgui.text_colored(STYLE.get_imvec4("fg_dim"), f"  Selected: {checked_count} checked | {state.cached_total_lines} lines | ~{state.cached_total_tokens} tokens | {state.cached_cost_str}")

        imgui.separator()

        imgui.columns(2, "ctx_cols")

        imgui.text("Groups")
        imgui.begin_child("groups_list", imgui.ImVec2(0, -30))
        for name in list(state.presets.keys()):
            _render_group_manage_row(name)
        imgui.end_child()

        if imgui.button("Create Group"):
            if state.selected_files:
                state.show_create_group_popup = True
                state.new_group_name = ""
            else:
                log_message("No files selected to create group")

        imgui.next_column()

        imgui.text("Files")

        # Search Mode
        use_flat_search = state.context_search_text and state.context_flatten_search

        if use_flat_search:
            imgui.begin_child("files_list", imgui.ImVec2(0, 0))
            table_flags = imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.resizable | imgui.TableFlags_.scroll_y
            if imgui.begin_table("search_table", 3, table_flags):
                imgui.table_setup_column("Path", imgui.TableColumnFlags_.width_stretch)
                imgui.table_setup_column("Lines", imgui.TableColumnFlags_.width_fixed, 60)
                imgui.table_setup_column("Tokens", imgui.TableColumnFlags_.width_fixed, 70)
                imgui.table_headers_row()
                
                if state.cached_flat_filtered_files is not None:
                    filtered = state.cached_flat_filtered_files
                else:
                    filtered = state.file_paths
                
                clipper = imgui.ListClipper()
                clipper.begin(len(filtered))
                while clipper.step():
                    for i in range(clipper.display_start, clipper.display_end):
                        f = filtered[i]
                        is_selected = f in state.selected_files
                        lines, tokens, _ = get_file_stats(f)
                        
                        imgui.table_next_row()
                        imgui.table_next_column()
                        
                        imgui.push_id(str(f))
                        changed, new_val = imgui.checkbox(f"##sel", is_selected)
                        imgui.pop_id()
                        if changed:
                            toggle_file_selection(f, new_val)
                            if new_val: state.file_checked[f] = True
                        
                        imgui.same_line()
                        imgui.text(str(f))
                        
                        imgui.table_next_column()
                        imgui.text(str(lines))
                        
                        imgui.table_next_column()
                        imgui.text(str(tokens))
                
                imgui.end_table()
            imgui.end_child()
            imgui.columns(1)
            imgui.end()
            return

        # Tree Mode
        search_whitelist_dirs = state.cached_whitelist_dirs
        search_whitelist_files = state.cached_whitelist_files

        imgui.begin_child("files_list", imgui.ImVec2(0, 0))

        if imgui.begin_table("files_table", 2, imgui.TableFlags_.row_bg | imgui.TableFlags_.borders_inner_v | imgui.TableFlags_.resizable):
            imgui.table_setup_column("Name", imgui.TableColumnFlags_.width_stretch)
            imgui.table_setup_column("Status", imgui.TableColumnFlags_.width_fixed, 80)
            
            # Flatten tree into visible list
            if state.view_tree_dirty or state.cached_context_rows is None:
                _rebuild_context_rows()
                
            visible_rows = state.cached_context_rows

            io = imgui.get_io()
            # Apply Drag Selection
            if not io.key_shift and state.drag_start_idx is not None:
                if state.drag_end_idx is None:
                    state.drag_end_idx = state.drag_start_idx

                start = min(state.drag_start_idx, state.drag_end_idx)
                end = max(state.drag_start_idx, state.drag_end_idx)
                target = state.drag_target_state
                
                modified = False
                for k in range(start, min(end + 1, len(visible_rows))):
                    item = visible_rows[k]
                    if item["type"] == "file":
                        p = to_relative(item["path"])
                        if target:
                            if p not in state.selected_files:
                                state.selected_files.add(p)
                                state.file_checked[p] = True
                                modified = True
                        else:
                            if p in state.selected_files:
                                state.selected_files.discard(p)
                                modified = True
                    elif item["type"] == "folder":
                        toggle_folder_selection(item["path"], target) 
                        modified = True
                
                if modified:
                    state.stats_dirty = True
                    save_fileset()
                
                state.drag_start_idx = None
                state.drag_end_idx = None

            row_height = imgui.get_frame_height()
            clipper = imgui.ListClipper()
            clipper.begin(len(visible_rows), row_height)
            
            io = imgui.get_io()
            hover_flags = imgui.HoveredFlags_.allow_when_blocked_by_active_item
            
            while clipper.step():
                for i in range(clipper.display_start, clipper.display_end):
                    row = visible_rows[i]
                    imgui.table_next_row(0, row_height)
                    
                    # Handle Drag Highlight
                    if state.drag_start_idx is not None and state.drag_end_idx is not None:
                        low = min(state.drag_start_idx, state.drag_end_idx)
                        high = max(state.drag_start_idx, state.drag_end_idx)
                        if low <= i <= high:
                            col = STYLE.get_u32("diff_add", 100) if state.drag_target_state else STYLE.get_u32("diff_del", 100)
                            imgui.table_set_bg_color(imgui.TableBgTarget_.row_bg0, col)

                    imgui.table_next_column()

                    is_selected_row = False
                    frel = None
                    folder_prefix = None

                    if row["type"] == "folder":
                        folder_prefix = str(row["path"])
                        for f in state.selected_files:
                            if str(f.resolve()).startswith(folder_prefix):
                                is_selected_row = True
                                break
                    else:
                        frel = to_relative(row["path"])
                        is_selected_row = frel in state.selected_files

                    imgui.push_id(f"row_sel_{i}")
                    sel_flags = imgui.SelectableFlags_.span_all_columns | imgui.SelectableFlags_.allow_overlap
                    cursor_start = imgui.get_cursor_pos()
                    clicked_row = imgui.selectable("##rowHit", False, flags=sel_flags, size=imgui.ImVec2(0, row_height))[0]
                    
                    if imgui.is_item_hovered(hover_flags) and io.key_shift:
                        if state.drag_start_idx is None:
                            state.drag_start_idx = i
                            state.drag_target_state = not is_selected_row
                        state.drag_end_idx = i
                    
                    imgui.set_cursor_pos(cursor_start)
                    imgui.pop_id()
                    
                    indent_w = float(row["depth"]) * 20.0
                    if indent_w > 0:
                        imgui.indent(indent_w)
                    
                    if row["type"] == "folder":
                        imgui.push_id(row["key"])
                        imgui.align_text_to_frame_padding()

                        total_in_folder = row.get("total_files", 0)
                        sel_in_folder = state.folder_selection_counts.get(row["key"], 0)
                        
                        is_mixed = (sel_in_folder > 0) and (sel_in_folder < total_in_folder)
                        
                        cb_val = True if (sel_in_folder == total_in_folder and total_in_folder > 0) else False
                        changed, val = imgui.checkbox("##f_chk", cb_val)

                        if is_mixed:
                            min_p = imgui.get_item_rect_min()
                            max_p = imgui.get_item_rect_max()
                            sz = max_p.x - min_p.x
                            pad = sz * 0.25
                            dl = imgui.get_window_draw_list()
                            col = STYLE.get_u32("icon_chk", 150)
                            dl.add_rect_filled(
                                imgui.ImVec2(min_p.x + pad, min_p.y + pad),
                                imgui.ImVec2(max_p.x - pad, max_p.y - pad),
                                col, 2.0
                            )

                        if changed:
                             toggle_folder_selection(row["path"], val)
                             if val and folder_prefix:
                                 for f in state.file_paths:
                                     if str(f).startswith(folder_prefix):
                                         rel = to_relative(f)
                                         state.file_checked[rel] = True
                        
                        if changed:
                            clicked_row = False

                        imgui.same_line()
                        
                        flags = 0
                        if search_whitelist_dirs is not None:
                            imgui.set_next_item_open(True, imgui.Cond_.always)
                        else:
                            imgui.set_next_item_open(row["is_open"], imgui.Cond_.always)

                        is_node_open = imgui.tree_node_ex(f"{row['name']}/", flags)
                        
                        if search_whitelist_dirs is None and is_node_open != row["is_open"]:
                             state.folder_states[row["key"]] = is_node_open
                             state.view_tree_dirty = True

                        if clicked_row and not io.key_shift and not io.key_ctrl:
                             new_state = not row["is_open"]
                             state.folder_states[row["key"]] = new_state
                             state.view_tree_dirty = True
                        
                        if is_node_open:
                            imgui.tree_pop()
                            
                        imgui.pop_id()
                        imgui.table_next_column()
                        if row["scanning"]:
                            imgui.text_colored(STYLE.get_imvec4("queued"), "Scanning...")

                    else:
                        imgui.push_id(str(row["path"]))
                        changed, val = imgui.checkbox("", is_selected_row)
                        if changed:
                            toggle_file_selection(frel, not is_selected_row)
                            if not is_selected_row: state.file_checked[frel] = True
                            clicked_row = False 
                        
                        imgui.same_line()
                        imgui.text(row["name"])
                        
                        if clicked_row and not io.key_shift and not io.key_ctrl:
                            toggle_file_selection(frel, not is_selected_row)
                            if not is_selected_row: state.file_checked[frel] = True

                        imgui.pop_id()
                        imgui.table_next_column()

                    if indent_w > 0:
                        imgui.unindent(indent_w)

        imgui.end_table()

    imgui.dummy(imgui.ImVec2(0, 30))
    imgui.end_child()
    imgui.columns(1)
    imgui.end()