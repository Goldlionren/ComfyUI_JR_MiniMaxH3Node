"""CPU-only CI smoke checks for package registration and bundled workflows."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import tomllib


def _load_package(project: Path, comfy_root: Path):
    sys.path.insert(0, str(project.parent))
    sys.path.insert(0, str(comfy_root))

    from aiohttp import web

    import server

    server.PromptServer.instance = types.SimpleNamespace(
        routes=web.RouteTableDef(),
        node_replace_manager=types.SimpleNamespace(register=lambda value: None),
    )
    return importlib.import_module(project.name)


def _validate_workflows(project: Path, registered: set[str]) -> tuple[int, list[str]]:
    example_paths = sorted((project / "examples").glob("*.json"))
    comfytv_paths = sorted((project / "comfytv" / "workflows").glob("*.json"))
    stale_legacy_links: list[str] = []

    for path in example_paths + comfytv_paths:
        workflow = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(workflow, dict):
            raise AssertionError(f"{path}: workflow root must be an object")
        nodes = workflow.get("nodes")
        links = workflow.get("links")
        if not isinstance(nodes, list) or not isinstance(links, list):
            raise AssertionError(f"{path}: nodes and links must be lists")

        node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
        if len(node_ids) != len(nodes) or len(node_ids) != len(set(node_ids)):
            raise AssertionError(f"{path}: node ids must exist and be unique")
        known_ids = set(node_ids)

        missing_jr = sorted(
            {
                node.get("type")
                for node in nodes
                if isinstance(node, dict)
                and isinstance(node.get("type"), str)
                and node["type"].startswith(("JR_H3_", "JR_MiniMaxH3"))
                and node["type"] not in registered
            }
        )
        if missing_jr:
            raise AssertionError(f"{path}: unregistered JR node types: {missing_jr}")

        invalid_links = [
            link
            for link in links
            if not (
                isinstance(link, list)
                and len(link) >= 6
                and link[1] in known_ids
                and link[3] in known_ids
            )
        ]
        if path in comfytv_paths and invalid_links:
            raise AssertionError(f"{path}: ComfyTV workflow has dangling or malformed links")
        if invalid_links:
            stale_legacy_links.append(f"{path.name}:{len(invalid_links)}")

    return len(example_paths) + len(comfytv_paths), stale_legacy_links


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, required=True)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    comfy_root = args.comfy_root.resolve()
    if not (comfy_root / "server.py").is_file():
        raise SystemExit(f"ComfyUI source not found at {comfy_root}")

    package = _load_package(project, comfy_root)
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    expected_version = metadata["project"]["version"]
    if package.__version__ != expected_version:
        raise AssertionError(
            f"runtime version {package.__version__!r} != pyproject version {expected_version!r}"
        )

    registered = set(package.NODE_CLASS_MAPPINGS)
    if registered != set(package.NODE_DISPLAY_NAME_MAPPINGS):
        raise AssertionError("class/display registration keys differ")
    sequential = package.NODE_CLASS_MAPPINGS["JR_H3_SequentialVideoOutput"]
    if sequential.RETURN_TYPES != ("STRING", "STRING"):
        raise AssertionError("Sequential Video Output RETURN_TYPES changed")
    server_default = sequential.INPUT_TYPES()["optional"]["server_auto_continue"][1]["default"]
    if server_default is not False:
        raise AssertionError("server_auto_continue must default to False")

    workflow_count, stale_links = _validate_workflows(project, registered)
    for item in stale_links:
        print(f"::warning::Legacy example retains a tolerated stale link record: {item}")
    print(
        json.dumps(
            {
                "version": expected_version,
                "registered_nodes": len(registered),
                "sequential_output": "PASS",
                "workflow_files": workflow_count,
                "comfytv_strict_links": "PASS",
                "legacy_stale_link_records": stale_links,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
