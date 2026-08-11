"""Semantic JSON fixtures for the deterministic official H3 formatter."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence


def base_semantic(
    descriptions: Sequence[str] = ("A paper boat moves across a shallow pool.",),
    *,
    starts: Sequence[float | None] | None = None,
    soundscape: str = "Water moves softly around the paper hull.",
    music: str = "N/A",
    dialogues: Sequence[Sequence[dict]] | None = None,
) -> str:
    if starts is None:
        starts = tuple(0.0 if index == 0 else float(index) for index in range(len(descriptions)))
    if dialogues is None:
        dialogues = tuple(() for _ in descriptions)
    return json.dumps(
        {
            "style": "",
            "shots": [
                {
                    "description": description,
                    "start_seconds": starts[index],
                    "dialogues": list(dialogues[index]),
                }
                for index, description in enumerate(descriptions)
            ],
            "overall_soundscape": soundscape,
            "non_diegetic_music": music,
            "task_types": [],
            "summary": "",
            "references": [],
        },
        ensure_ascii=False,
    )


def ref_semantic(
    labels: Iterable[str],
    descriptions: Sequence[str] = ("The referenced subject crosses the water.",),
    *,
    starts: Sequence[float | None] | None = None,
    soundscape: str = "Water moves softly.",
    music: str = "N/A",
    dialogues: Sequence[Sequence[dict]] | None = None,
) -> str:
    labels = tuple(labels)
    if starts is None:
        starts = tuple(0.0 if index == 0 else float(index) for index in range(len(descriptions)))
    if dialogues is None:
        dialogues = tuple(() for _ in descriptions)
    references = []
    for label in labels:
        family = label[1:].split(" ", 1)[0]
        references.append(
            {
                "label": label,
                "definition": f"the registered {family.lower()} reference",
                "retention": "fully_copy" if family == "Audio" else "fully_preserved",
                "retention_detail": "retain the source identity and intended characteristics",
            }
        )
    return json.dumps(
        {
            "style": "Cinematic naturalism with physically continuous motion.",
            "shots": [
                {
                    "description": description,
                    "start_seconds": starts[index],
                    "dialogues": list(dialogues[index]),
                }
                for index, description in enumerate(descriptions)
            ],
            "overall_soundscape": soundscape,
            "non_diegetic_music": music,
            "task_types": ["reference generation"],
            "summary": "The registered references guide a coherent target video.",
            "references": references,
        },
        ensure_ascii=False,
    )
