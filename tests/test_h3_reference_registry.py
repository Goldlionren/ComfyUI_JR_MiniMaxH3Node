import pytest
from ComfyUI_JR_MiniMaxH3Node.utils.h3_reference_registry import (
    H3ReferenceType,
    ReferenceRegistry,
    build_reference_registry,
)


def test_standard_registration_order_and_picture_labels():
    registry = build_reference_registry(
        first_frame=object(),
        last_frame=object(),
        ref_image_1=object(),
        ref_image_2=None,
        ref_image_3=object(),
    )
    entries = registry.entries()
    assert [entry.identifier for entry in entries] == ["<Picture 1>", "<Picture 2>", "<Picture 3>", "<Picture 4>"]
    assert [entry.source_input for entry in entries] == ["first_frame", "last_frame", "ref_image_1", "ref_image_3"]
    assert [entry.role for entry in entries] == ["first_frame", "last_frame", "reference", "reference"]


def test_type_counters_are_independent_and_resolution_is_stable():
    registry = ReferenceRegistry()
    picture = registry.register_picture("first_frame", "first_frame")
    video = registry.register_video("clip", "source")
    audio = registry.register_audio("audio", "source")
    subject = registry.register_subject("person", "subject", subject_binding=picture.identifier)
    assert [picture.identifier, video.identifier, audio.identifier, subject.identifier] == [
        "<Picture 1>",
        "<Video 1>",
        "<Audio 1>",
        "<Subject 1>",
    ]
    assert registry.resolve("<Video 1>") is video
    assert registry.resolve("clip") is video
    assert registry.unresolved_labels("<Picture 1> <Video 2>") == ["<Video 2>"]


def test_duplicate_source_key_and_label_are_rejected_but_suffixes_are_supported():
    registry = ReferenceRegistry()
    first = registry.register_picture("batch", "reference")
    second = registry.register_picture("batch", "reference", source_key="batch#2")
    assert first.source_key == "batch"
    assert second.source_key == "batch#2"
    with pytest.raises(ValueError, match="duplicate reference source key"):
        registry.register_picture("other", "reference", source_key="batch")
    with pytest.raises(ValueError, match="unique source_key suffix"):
        registry.register_picture("batch", "reference")
    with pytest.raises(ValueError, match="duplicate reference label"):
        registry.register_picture("other", "reference", identifier="<Picture 1>")


def test_unresolved_status_and_label_validation():
    registry = ReferenceRegistry()
    pending = registry.register_picture("pending", resolved_status=False)
    assert registry.unresolved_labels() == [pending.identifier]
    assert not registry.validate_references(raise_on_unresolved=False)
    with pytest.raises(ValueError, match=pending.identifier):
        registry.validate_references()


def test_reference_type_filter_and_subject_binding_are_retained():
    registry = ReferenceRegistry()
    registry.register_picture("picture")
    subject = registry.register_subject("subject", subject_binding="<Picture 1>")
    assert subject.subject_binding == "<Picture 1>"
    assert registry.labels(H3ReferenceType.PICTURE) == ["<Picture 1>"]
    assert registry.list_references("Subject")[0] is subject
