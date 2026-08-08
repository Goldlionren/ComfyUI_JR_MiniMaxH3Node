import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.h3_prompt_modes import (
    H3InputMode,
    contains_reference_label,
    route_h3_mode,
    validate_mode_inputs,
)


@pytest.mark.parametrize(
    ("first", "last", "count", "instructions", "expected"),
    [
        (False, False, 0, "", H3InputMode.T2VA),
        (True, False, 0, "", H3InputMode.I2VA),
        (True, True, 0, "", H3InputMode.FL2VA),
        (False, True, 0, "", H3InputMode.L2VA),
        (False, False, 1, "", H3InputMode.REF2VA),
        (True, True, 1, "", H3InputMode.REF2VA),
        (False, False, 0, "Use <Picture 2> here", H3InputMode.REF2VA),
    ],
)
def test_auto_routes_deterministically(first, last, count, instructions, expected):
    assert route_h3_mode(
        " auto ",
        has_first_frame=first,
        has_last_frame=last,
        reference_image_count=count,
        reference_instructions=instructions,
    ) is expected


def test_reference_label_detection_requires_supported_label():
    assert contains_reference_label("<Picture 1> <Video 3> <Audio 2> <Subject 1>")
    assert not contains_reference_label("<image 1>")
    assert not contains_reference_label("Picture 1")


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("T2VA", {"has_first_frame": True}),
        ("I2VA", {}),
        ("I2VA", {"has_first_frame": True, "has_last_frame": True}),
        ("FL2VA", {"has_first_frame": True}),
        ("L2VA", {"has_first_frame": True, "has_last_frame": True}),
        ("Ref2VA", {}),
    ],
)
def test_explicit_mode_validation_errors_are_descriptive(mode, kwargs):
    with pytest.raises(ValueError, match=mode):
        validate_mode_inputs(mode, **kwargs)


def test_explicit_modes_override_auto_when_compatible():
    assert route_h3_mode("I2VA", has_first_frame=True) is H3InputMode.I2VA
    assert route_h3_mode("Ref2VA", reference_instructions="<Picture 9>") is H3InputMode.REF2VA


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        ("T2VA", {}),
        ("I2VA", {"has_first_frame": True}),
        ("FL2VA", {"has_first_frame": True, "has_last_frame": True}),
        ("L2VA", {"has_last_frame": True}),
    ],
)
def test_labelled_reference_instructions_are_ref2va_only(mode, kwargs):
    with pytest.raises(ValueError, match="reference_instructions"):
        validate_mode_inputs(mode, reference_instructions="Use <Picture 1>", **kwargs)


def test_opaque_presence_values_never_use_truthiness():
    class Opaque:
        def __bool__(self):
            raise AssertionError("presence must not use truthiness")

    assert route_h3_mode("I2VA", has_first_frame=Opaque()) is H3InputMode.I2VA
