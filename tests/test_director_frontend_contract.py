from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "js" / "director_desk.js").read_text(encoding="utf-8")


def test_frontend_uses_properties_hidden_state_and_lifecycle_cleanup():
    for token in [
        'const PROP_KEY = "jr_h3_director_state"',
        'const STATE_WIDGET = "director_state_json"',
        "node.properties[PROP_KEY]",
        "loadedGraphNode(node)",
        "afterConfigureGraph()",
        "onRemoved",
        "instances = new WeakMap()",
        "instance.activeDragCancel?.()",
        "flushInspectorEditor(instance)",
        "instance.pendingInspectorFlush",
        "instance.inspectorDraft",
        "saveInspectorDraft(instance, editor)",
        "cancelInspectorDraft(instance, editor)",
        "assetFingerprint",
        "lane_order",
        "instance.mediaElements",
        '"last_frame"',
        'Set as Last Frame',
        'isPointAnchor',
        "serialize: false",
        "hideOnZoom: false",
    ]:
        assert token in SOURCE


def test_frontend_commits_drag_once_and_keeps_node_resize_user_controlled():
    assert 'globalThis.addEventListener("pointermove", move, true)' in SOURCE
    assert 'globalThis.removeEventListener("pointermove", move, true)' in SOURCE
    assert "setPointerCapture" not in SOURCE
    assert "commit(instance, draft)" in SOURCE
    assert "graph?.beforeChange?.(node)" in SOURCE
    assert "graph?.afterChange?.(node)" in SOURCE
    assert SOURCE.count("this.setSize(") == 1
    assert "snapDelta" in SOURCE
    assert "onExecuted" not in SOURCE


def test_frontend_has_required_sections_actions_media_and_security_boundaries():
    for token in [
        'makeRow("SHOT")', 'makeRow("VISUAL")', 'makeRow("AUDIO")',
        'textContent = "Global Direction"', 'button("Duplicate"', 'button("Split"',
        'button(shotActions ? "Earlier" : "Lane ↑"', 'button(shotActions ? "Later" : "Lane ↓"',
        'button("Delete"', 'button("+ Image"', 'button("+ Video"', 'button("+ Audio"',
        'api.fetchApi("/upload/image"', 'api.fetchApi("/jr-h3/director/probe"',
        'new URLSearchParams({ filename: asset.filename, subfolder: asset.subfolder, type: asset.type })',
    ]:
        assert token in SOURCE
    assert "base64" not in SOURCE.lower()
    assert "file://" not in SOURCE.lower()


def test_inspector_uses_atomic_save_cancel_and_blocks_silent_navigation_loss():
    select_start = SOURCE.index("function selectItem")
    clone_start = SOURCE.index("const next = deepClone(instance.state);", select_start)
    flush_start = SOURCE.index("flushInspectorEditor(instance)", select_start)
    assert flush_start < clone_start
    assert 'button("Save", () => saveInspectorDraft(instance, editor)' in SOURCE
    assert 'button("Cancel", () => cancelInspectorDraft(instance, editor)' in SOURCE
    assert '"Unsaved Inspector changes. Click Save or Cancel before leaving this item."' in SOURCE
    assert 'const applied = updateItem(instance, editor.itemId, changes);' in SOURCE
    assert 'for (const actionButton of editor.actionButtons || []) actionButton.disabled = editor.dirty;' in SOURCE
    assert 'if (!selectItem(instance, item.id, { render: false })) return;' in SOURCE
