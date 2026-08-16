"""Load this project through ComfyUI's real V1 custom-node loader."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy_root))

    import nodes
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server.PromptServer(loop)
    try:
        loaded = loop.run_until_complete(
            nodes.load_custom_node(str(args.project_root), ignore=set(), module_parent="custom_nodes")
        )
        node = nodes.NODE_CLASS_MAPPINGS.get("JR_H3_HybridLoader")
        result = {
            "loaded": loaded,
            "registered": node is not None,
            "return_types": list(node.RETURN_TYPES) if node else None,
            "category": node.CATEGORY if node else None,
            "node_count": 12 if loaded else None,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        if not loaded or node is None or node.RETURN_TYPES != ("MODEL",):
            raise SystemExit(1)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
