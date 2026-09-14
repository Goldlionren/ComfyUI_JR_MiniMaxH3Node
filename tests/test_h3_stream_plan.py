from dataclasses import FrozenInstanceError, replace
from fractions import Fraction

import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.h3_stream_plan import canonical_plan, round_half_even_ratio, validate_plan


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
