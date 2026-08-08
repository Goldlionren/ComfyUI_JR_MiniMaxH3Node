"""Offline tests for the clean-room MiniMax H3 metadata layer."""

from __future__ import annotations

import importlib
import json
import socket
from pathlib import Path

import pytest

from utils import h3_official_resources as resources


def test_default_metadata_loads_with_utf8_contents() -> None:
    metadata = resources.load_upstream_metadata()

    assert metadata["repository"] == resources.UPSTREAM_REPOSITORY
    assert metadata["branch"] == "main"
    assert metadata["commit"] == resources.UPSTREAM_COMMIT
    assert metadata["redistribution_strategy"] == "clean-room-metadata-only"
    assert [source["path"] for source in metadata["official_sources"]] == [
        resources.SKILL_SOURCE_PATH,
        resources.BASE_SOURCE_PATH,
        resources.REF_SOURCE_PATH,
    ]
    assert metadata["official_sources"][0]["sha256"].startswith("3C0D6E13")
    assert metadata["license"]["name"] == "MiniMax H3 Community License"


def test_mode_specs_are_immutable_and_contain_only_format_facts() -> None:
    base = resources.get_spec_for_mode("T2VA")
    ref = resources.get_spec_for_mode("ref2va")

    assert base.mode == "base"
    assert base.sections == (
        "integrated_multimodal_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    assert ref.sections == (
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
    )
    assert base.visible_retention_values == (
        "fully_preserved",
        "partially_preserved",
        "attribute_transfer",
        "weak_reference",
    )
    assert ref.audio_retention_values == (
        "fully_copy",
        "partially_copy",
        "reference",
        "weak_reference",
    )
    with pytest.raises((AttributeError, TypeError)):
        base.sections += ("extra",)  # type: ignore[misc]


def test_missing_metadata_file_is_clear(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    with pytest.raises(resources.H3OfficialMetadataNotFoundError, match="not found"):
        resources.load_upstream_metadata(missing)


def test_malformed_json_is_clear(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")

    with pytest.raises(resources.H3OfficialMetadataMalformedError, match="malformed JSON"):
        resources.load_upstream_metadata(malformed)


def test_missing_required_key_is_clear(tmp_path: Path) -> None:
    metadata = json.loads((Path(resources.__file__).parent.parent / "resources/minimax_h3_spec/UPSTREAM.json").read_text(encoding="utf-8"))
    del metadata["official_sources"]
    candidate = tmp_path / "missing-key.json"
    candidate.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(resources.H3OfficialMetadataValidationError, match="missing required key 'official_sources'"):
        resources.load_upstream_metadata(candidate)


def test_import_is_offline_and_does_not_open_metadata_or_network(monkeypatch: pytest.MonkeyPatch) -> None:
    original_read_text = Path.read_text

    def fail_read_text(*args: object, **kwargs: object) -> str:
        raise AssertionError("metadata read occurred during import")

    def fail_socket(*args: object, **kwargs: object) -> socket.socket:
        raise AssertionError("network access occurred during import")

    monkeypatch.setattr(Path, "read_text", fail_read_text)
    monkeypatch.setattr(socket, "socket", fail_socket)
    importlib.reload(resources)
    monkeypatch.setattr(Path, "read_text", original_read_text)

    assert resources.BASE_SECTION_ORDER[0] == "integrated_multimodal_description"
