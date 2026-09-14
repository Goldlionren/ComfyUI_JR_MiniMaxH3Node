from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.h3_stream_plan import PRESETS, canonical_plan, round_half_even_ratio, validate_plan


def test_canonical_geometry():
    p = canonical_plan()
    assert tuple(s.group_count for s in p.phases) == (2, 2, 2, 1)
    assert [0] + [s.frame_stop for s in p.phases] == [0, 39, 73, 107, 124]
    assert [0] + [s.video_latent_stop for s in p.phases] == [0, 12, 22, 32, 37]
    assert [0] + [s.audio_latent_stop for s in p.phases] == [0, 65, 122, 178, 207]
    assert p.native_frame_count == 124 and p.video_latent_count == 37
    assert sum(s.duration_seconds for s in p.phases) == Fraction(31, 6)
    for left, right in zip(p.phases, p.phases[1:]):
        assert left.frame_stop == right.frame_start
        assert left.video_latent_stop == right.video_latent_start
        assert left.audio_latent_stop == right.audio_latent_start
    for s in p.phases:
        assert s.audio_latent_start == round(Fraction(s.frame_start * 40, 24))
        assert s.audio_latent_stop == round(Fraction(s.frame_stop * 40, 24))
    assert validate_plan(p) is p


@pytest.mark.parametrize("n", [-11, -5, -3, -1, 1, 3, 5, 11, 2**63 + 1])
def test_exact_half_even(n):
    assert round_half_even_ratio(n, 2) == round(Fraction(n, 2))


def test_immutable_and_invalid_plans():
    plan = canonical_plan()
    with pytest.raises(FrozenInstanceError):
        plan.native_frame_count = 1
    for bad in [None, {}, replace(plan, request_index=1), replace(plan, phases=plan.phases[:-1]),
                replace(plan, phases=(replace(plan.phases[0], audio_latent_stop=66),) + plan.phases[1:])]:
        with pytest.raises(ValueError, match="invalid stream plan"):
            validate_plan(bad)


@pytest.mark.parametrize("preset,windows,frames,video,audio", [
    (PRESETS[0], 1, 124, 37, 207), (PRESETS[1], 2, 243, 72, 405), (PRESETS[2], 3, 362, 107, 603),
])
def test_long_timeline_geometry(preset, windows, frames, video, audio):
    p = canonical_plan(preset)
    assert validate_plan(p) is p
    assert (p.native_frame_count, p.video_latent_count, p.audio_latent_count) == (frames, video, audio)
    assert len(p.phases) == 4 * windows and len(p.windows) == windows
    assert tuple(s.group_count for s in p.phases) == (2, 2, 2, 1) * windows
    assert sum(s.duration_seconds for s in p.phases) == Fraction(frames, 24)
    for attr, total in (("frame", frames), ("video_latent", video), ("audio_latent", audio)):
        assert getattr(p.phases[0], attr + "_start") == 0
        assert getattr(p.phases[-1], attr + "_stop") == total
        for left, right in zip(p.phases, p.phases[1:]):
            assert getattr(left, attr + "_stop") == getattr(right, attr + "_start")
    for phase in p.phases:
        assert phase.audio_latent_start == round(Fraction(phase.frame_start * 40, 24))
        assert phase.audio_latent_stop == round(Fraction(phase.frame_stop * 40, 24))
    for index, phases in enumerate(p.windows):
        assert phases[-1].frame_stop - phases[0].frame_start == (124 if index == 0 else 119)
        assert phases[-1].video_latent_stop - phases[0].video_latent_start == (37 if index == 0 else 35)
    # More windows do not increase the largest micro-phase/cache allocation.
    assert max(s.video_latent_count * 1462 + 2 * s.audio_latent_count for s in p.phases) == 17674
    for changed in (replace(p, native_frame_offset=124), replace(p, video_latent_offset=37),
                    replace(p, audio_latent_offset=207), replace(p, media_time_origin=Fraction(31, 6)),
                    replace(p, preset="unknown"), replace(p, preset=[]), replace(p, schema_version=2)):
        with pytest.raises(ValueError, match="invalid stream plan"):
            validate_plan(changed)


def test_planner_schema_keeps_old_sockets_and_adds_exact_lengths():
    from ComfyUI_JR_MiniMaxH3Node.nodes.h3_taomate_chunk_planner import JR_H3_TaoMateChunkPlanner

    node = JR_H3_TaoMateChunkPlanner()
    assert node.RETURN_NAMES[:2] == ("stream_plan", "status")
    options, settings = node.INPUT_TYPES()["required"]["preset"]
    assert tuple(options) == PRESETS and settings["default"] == PRESETS[0]
    for preset in options:
        p, status, frames, ticks = node.plan(preset)
        assert frames == p.native_frame_count and ticks == p.audio_latent_count
        assert "Geometry only" in status
