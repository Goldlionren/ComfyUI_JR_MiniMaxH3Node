import sys
from pathlib import Path
from types import SimpleNamespace

import comfy
import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))
# Native ComfyUI now also has a top-level utils package (assets/mime types).
# JR tests must import JR utilities by their qualified package name.
sys.path.insert(0, str(Path(next(iter(comfy.__path__))).resolve().parent))

if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
    from comfy.cli_args import args

    args.cpu = True

def _init_import_only_server():
    # Resolve after native path and CPU/CUDA options; never start a listener.
    from aiohttp import web

    import server

    if not hasattr(server.PromptServer, "instance"):
        server.PromptServer.instance = SimpleNamespace(
            routes=web.RouteTableDef(), node_replace_manager=SimpleNamespace(register=lambda value: None),
            send_sync=lambda *args, **kwargs: None,
        )


_init_import_only_server()


@pytest.fixture
def package_name():
    return PROJECT.name
