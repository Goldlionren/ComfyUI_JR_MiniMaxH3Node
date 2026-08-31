import sys
from pathlib import Path

import pytest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))

if not (torch.cuda.is_available() and torch.cuda.device_count() > 0):
    from comfy.cli_args import args

    args.cpu = True


@pytest.fixture
def package_name():
    return PROJECT.name
