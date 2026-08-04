import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))


@pytest.fixture
def package_name():
    return PROJECT.name
