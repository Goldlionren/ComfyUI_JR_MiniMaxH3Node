import concurrent.futures
import json
import sys
import time
import types
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.nodes import prompt_review_pause as review_module
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_directed_video_conditioning import (
    JR_H3_DirectedVideoConditioning,
)
from ComfyUI_JR_MiniMaxH3Node.nodes.h3_openai_prompt_optimizer import (
    JR_H3_OpenAICompatiblePromptOptimizer,
)
from ComfyUI_JR_MiniMaxH3Node.nodes.prompt_review_pause import JR_H3_PromptReviewPause
from ComfyUI_JR_MiniMaxH3Node.utils.director_pipe import (
    RuntimeMedia,
    RuntimeMediaFile,
    build_director_pipe,
)
from ComfyUI_JR_MiniMaxH3Node.utils.director_state import (
    DEFAULT_DIRECTOR_STATE_JSON,
    director_state_from_dict,
)
from ComfyUI_JR_MiniMaxH3Node.utils.h3_directed_conditioning import (
    _load_audio,
    _load_video,
    _runtime,
    normalize_native_output,
    prepare_directed_inputs,
)
from ComfyUI_JR_MiniMaxH3Node.utils.prompt_review_state import PROMPT_REVIEW_STORE


def _asset(identifier, kind, filename, *, width=None, height=None, duration=None):
    return {
        "id": identifier,
        "kind": kind,
        "filename": filename,
        "subfolder": "director",
        "type": "input",
        "display_name": filename,
        "status": "ready",
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


def _pipe(*, first=False, last=False, refs=0, video=False, audio_role=None):
    raw = json.loads(DEFAULT_DIRECTOR_STATE_JSON)
    raw["timeline"] = {"duration_seconds": 5.0, "fps": 30.0}
    raw["shots"] = [
        {"id": "shot-1", "start": 0.0, "end": 5.0, "direction": "Move.", "notes": ""}
    ]
    visual = []
    runtime = []
    if first:
        visual.append({
            "id": "first", "kind": "image", "role": "first_frame", "start": 0.0,
            "end": 0.0, "source_in": None, "source_out": None, "direction": "", "notes": "",
            "registry_order": 1, "asset": _asset("asset-first", "image", "first.png", width=640, height=320),
        })
        runtime.append(RuntimeMedia("asset-first", "first", "image", torch.zeros(1, 8, 8, 3), (("width", 640), ("height", 320))))
    if last:
        visual.append({
            "id": "last", "kind": "image", "role": "last_frame", "start": 5.0,
            "end": 5.0, "source_in": None, "source_out": None, "direction": "", "notes": "",
            "registry_order": 2, "asset": _asset("asset-last", "image", "last.png", width=640, height=320),
        })
        runtime.append(RuntimeMedia("asset-last", "last", "image", torch.ones(1, 8, 8, 3), (("width", 640), ("height", 320))))
    for index in range(refs):
        item_id = f"picture-{index}"
        asset_id = f"asset-picture-{index}"
        visual.append({
            "id": item_id, "kind": "image", "role": "reference_image", "start": 0.0,
            "end": 5.0, "source_in": None, "source_out": None, "direction": "", "notes": "",
            "registry_order": index + 10, "asset": _asset(asset_id, "image", f"ref-{index}.png", width=512, height=512),
        })
        runtime.append(RuntimeMedia(asset_id, item_id, "image", torch.full((1, 8, 8, 3), index + 2.0)))
    if video:
        visual.append({
            "id": "video", "kind": "video", "role": "reference_video", "start": 0.0,
            "end": 5.0, "source_in": 0.0, "source_out": 5.0, "direction": "", "notes": "",
            "registry_order": 1, "asset": _asset("asset-video", "video", "ref.mp4", width=1280, height=720, duration=5.0),
        })
        runtime.append(RuntimeMedia("asset-video", "video", "video", torch.zeros(5, 8, 8, 3)))
    audio_items = []
    if audio_role:
        audio_items.append({
            "id": "audio", "role": audio_role, "start": 0.0, "end": 5.0,
            "source_in": 0.0, "source_out": 5.0, "direction": "", "notes": "",
            "registry_order": 1, "asset": _asset("asset-audio", "audio", "ref.wav", duration=5.0),
        })
        runtime.append(RuntimeMedia(
            "asset-audio", "audio", "audio",
            {"waveform": torch.zeros(1, 1, 1600), "sample_rate": 32000},
        ))
    raw["visual_items"] = visual
    raw["audio_items"] = audio_items
    return build_director_pipe(
        director_state_from_dict(raw), runtime_resolver=lambda _state: tuple(runtime)
    )


def _install_native(monkeypatch, native):
    package = types.ModuleType("comfy_extras")
    package.nodes_minimax_h3 = native
    monkeypatch.setitem(sys.modules, "comfy_extras", package)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_minimax_h3", native)


class _NodeOutput:
    def __init__(self, *values):
        self.result = values


class _I2V:
    calls = []

    @classmethod
    def execute(cls, **kwargs):
        cls.calls.append(kwargs)
        return _NodeOutput("i2v-positive", "i2v-latent")


class _Ref2V:
    calls = []

    @classmethod
    def execute(cls, **kwargs):
        cls.calls.append(kwargs)
        return _NodeOutput("ref-positive", "ref-latent")


@pytest.fixture
def native():
    _I2V.calls.clear()
    _Ref2V.calls.clear()
    return SimpleNamespace(
        nodes=SimpleNamespace(MAX_RESOLUTION=16384),
        adapt_canvas=lambda width, height: (768, 384) if width >= height else (384, 768),
        MiniMaxH3ImageToVideo=_I2V,
        MiniMaxH3ReferenceToVideo=_Ref2V,
    )


@pytest.mark.parametrize(
    ("kwargs", "mode"),
    [
        ({}, "Image to Video"),
        ({"first": True}, "Image to Video"),
        ({"first": True, "last": True}, "Image to Video"),
        ({"refs": 1}, "Reference to Video"),
        ({"first": True, "refs": 1}, "Reference to Video"),
        ({"video": True}, "Reference to Video"),
        ({"audio_role": "reference_audio"}, "Reference to Video"),
        ({"audio_role": "driving_audio"}, "Reference to Video"),
        ({"refs": 2, "video": True, "audio_role": "reference_audio"}, "Reference to Video"),
    ],
)
def test_auto_mode_matrix(kwargs, mode, native):
    prepared = prepare_directed_inputs(
        _pipe(**kwargs), mode_override="Auto", dimension_source="Prefer Node",
        width=1344, height=768, length=124, native_module=native,
    )
    assert prepared.mode == mode


def test_i2v_first_last_and_prompt_priority(native):
    pipe = _pipe(first=True, last=True).derive(
        optimized_prompt="optimized", reviewed_prompt="reviewed"
    )
    prepared = prepare_directed_inputs(
        pipe, mode_override="Auto", dimension_source="Prefer Pipe",
        width=1344, height=768, length=124, native_module=native,
    )
    assert prepared.prompt == "reviewed"
    assert prepared.first_frame is pipe.media_for_item("first").payload
    assert prepared.last_frame is pipe.media_for_item("last").payload
    assert (prepared.width, prepared.height, prepared.length) == (768, 384, 120)
    assert pipe.timeline.fps == 30.0  # H3 length still uses fixed 24 fps.


def test_prompt_fallback_and_prefer_node_dimensions(native):
    base = _pipe(first=True)
    optimized = base.derive(optimized_prompt="optimized")
    prepared = prepare_directed_inputs(
        optimized, mode_override="Auto", dimension_source="Prefer Node",
        width=1024, height=576, length=141, native_module=native,
    )
    assert prepared.prompt == "optimized"
    assert (prepared.width, prepared.height, prepared.length) == (1024, 576, 141)
    assert prepare_directed_inputs(
        base, mode_override="Auto", dimension_source="Prefer Node",
        width=1024, height=576, length=141, native_module=native,
    ).prompt == base.compiled_director_prompt


def test_reference_order_and_real_media_mapping(native):
    pipe = _pipe(first=True, last=True, refs=2, video=True, audio_role="driving_audio")
    prepared = prepare_directed_inputs(
        pipe, mode_override="Reference to Video", dimension_source="Prefer Node",
        width=1344, height=768, length=124, native_module=native,
    )
    picture_records = [record for record in pipe.reference_registry if record.family == "Picture"]
    assert [record.role for record in picture_records] == [
        "first_frame", "last_frame", "reference_image", "reference_image"
    ]
    assert [name for name, _ in prepared.ref_images] == [
        "ref_image_0", "ref_image_1", "ref_image_2", "ref_image_3"
    ]
    assert [name for name, _ in prepared.ref_videos] == ["ref_video_0"]
    assert [name for name, _ in prepared.ref_audios] == ["ref_audio_0"]
    assert prepared.ref_audios[0][1]["waveform"].shape[1] == 2


def test_explicit_mode_conflicts_missing_media_and_audio_vae(native, monkeypatch):
    with pytest.raises(ValueError, match="conflicts"):
        prepare_directed_inputs(
            _pipe(refs=1), mode_override="Image to Video", dimension_source="Prefer Node",
            width=1344, height=768, length=124, native_module=native,
        )
    with pytest.raises(ValueError, match="requires at least one"):
        prepare_directed_inputs(
            _pipe(), mode_override="Reference to Video", dimension_source="Prefer Node",
            width=1344, height=768, length=124, native_module=native,
        )
    _install_native(monkeypatch, native)
    with pytest.raises(ValueError, match="audio_vae"):
        JR_H3_DirectedVideoConditioning().condition(
            object(), object(), _pipe(audio_role="reference_audio"),
            "Auto", "Prefer Node", 1344, 768, 124, "match", None,
        )


def test_missing_runtime_media_and_native_reference_limits(native):
    missing = replace(_pipe(refs=1), runtime_media=())
    with pytest.raises(ValueError, match="no runtime media"):
        prepare_directed_inputs(
            missing, mode_override="Auto", dimension_source="Prefer Node",
            width=1344, height=768, length=124, native_module=native,
        )
    with pytest.raises(ValueError, match="at most 9"):
        prepare_directed_inputs(
            _pipe(refs=10), mode_override="Auto", dimension_source="Prefer Node",
            width=1344, height=768, length=124, native_module=native,
        )


def test_incompatible_native_api_has_clear_error(native, monkeypatch):
    del native.MiniMaxH3ReferenceToVideo
    _install_native(monkeypatch, native)
    with pytest.raises(RuntimeError, match="conditioning API is incompatible"):
        JR_H3_DirectedVideoConditioning().condition(
            object(), object(), _pipe(first=True),
            "Auto", "Prefer Node", 1344, 768, 124, "match", None,
        )

    native.MiniMaxH3ReferenceToVideo = _Ref2V
    del native.adapt_canvas
    with pytest.raises(RuntimeError, match="conditioning API is incompatible"):
        JR_H3_DirectedVideoConditioning().condition(
            object(), object(), _pipe(first=True),
            "Auto", "Prefer Node", 1344, 768, 124, "match", None,
        )


def test_invalid_ref_image_size_is_rejected_before_native_call(native, monkeypatch):
    _install_native(monkeypatch, native)
    with pytest.raises(ValueError, match="ref_image_size"):
        JR_H3_DirectedVideoConditioning().condition(
            object(), object(), _pipe(refs=1),
            "Auto", "Prefer Node", 1344, 768, 124, "invalid", None,
        )


def test_runtime_video_file_uses_public_api_source_trim_and_rejects_non_24fps(monkeypatch):
    calls = []

    class FakeVideo:
        def __init__(self, path, *, start_time, duration):
            calls.append((path, start_time, duration))

        def get_components(self):
            return SimpleNamespace(frame_rate=24, images=torch.zeros(6, 8, 8, 3))

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.VideoFromFile = FakeVideo
    comfy_api.latest = latest
    monkeypatch.setitem(sys.modules, "comfy_api", comfy_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", latest)
    item = SimpleNamespace(
        source_in=1.0, source_out=1.25,
        asset=SimpleNamespace(display_name="clip.mp4"),
    )
    media = RuntimeMedia(
        "asset", "video", "video", RuntimeMediaFile("safe.mp4", "video"),
        (("duration_seconds", 2.0), ("height", 8), ("width", 8)),
    )
    assert _load_video(media, item).shape == (6, 8, 8, 3)
    assert calls == [("safe.mp4", 1.0, 0.25)]

    class WrongRateVideo(FakeVideo):
        def get_components(self):
            return SimpleNamespace(frame_rate=30, images=torch.zeros(6, 8, 8, 3))

    latest.VideoFromFile = WrongRateVideo
    with pytest.raises(ValueError, match="must be 24 fps"):
        _load_video(media, item)


def test_runtime_audio_file_source_trim_and_mono_to_stereo(monkeypatch):
    comfy_extras = types.ModuleType("comfy_extras")
    nodes_audio = types.ModuleType("comfy_extras.nodes_audio")
    nodes_audio.load = lambda _path: (torch.zeros(1, 96000), 32000)
    comfy_extras.nodes_audio = nodes_audio
    monkeypatch.setitem(sys.modules, "comfy_extras", comfy_extras)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_audio", nodes_audio)
    item = SimpleNamespace(
        source_in=1.0, source_out=2.0,
        asset=SimpleNamespace(display_name="sound.wav"),
    )
    media = RuntimeMedia(
        "asset", "audio", "audio", RuntimeMediaFile("safe.wav", "audio"),
        (("channels", 1), ("duration_seconds", 3.0), ("sample_rate", 32000), ("size_bytes", 1024)),
    )
    audio = _load_audio(media, item)
    assert audio["waveform"].shape == (1, 2, 32000)
    assert audio["sample_rate"] == 32000


def test_runtime_file_decode_budgets_fail_closed(monkeypatch):
    video_item = SimpleNamespace(
        source_in=0.0, source_out=5.0,
        asset=SimpleNamespace(display_name="large.mp4"),
    )
    video_file = RuntimeMediaFile("safe.mp4", "video")
    with pytest.raises(ValueError, match="missing bounded width/height"):
        _load_video(RuntimeMedia("asset", "video", "video", video_file), video_item)
    oversized_video = RuntimeMedia(
        "asset", "video", "video", video_file,
        (("duration_seconds", 5.0), ("height", 4096), ("width", 4096)),
    )
    with pytest.raises(ValueError, match="bounded decode budget"):
        _load_video(oversized_video, video_item)

    audio_item = SimpleNamespace(
        source_in=0.0, source_out=1.0,
        asset=SimpleNamespace(display_name="large.wav"),
    )
    audio_file = RuntimeMediaFile("safe.wav", "audio")
    with pytest.raises(ValueError, match="missing bounded media metadata"):
        _load_audio(RuntimeMedia("asset", "audio", "audio", audio_file), audio_item)
    oversized_audio = RuntimeMedia(
        "asset", "audio", "audio", audio_file,
        (("channels", 2), ("duration_seconds", 1.0), ("sample_rate", 32000),
         ("size_bytes", 129 * 1024 * 1024)),
    )
    with pytest.raises(ValueError, match="bounded decode size"):
        _load_audio(oversized_audio, audio_item)
    long_audio = RuntimeMedia(
        "asset", "audio", "audio", audio_file,
        (("channels", 2), ("duration_seconds", 181.0), ("sample_rate", 32000),
         ("size_bytes", 1024)),
    )
    with pytest.raises(ValueError, match="exceeds 180 seconds"):
        _load_audio(long_audio, audio_item)
    high_rate_audio = RuntimeMedia(
        "asset", "audio", "audio", audio_file,
        (("channels", 2), ("duration_seconds", 180.0), ("sample_rate", 192000),
         ("size_bytes", 1024)),
    )
    with pytest.raises(ValueError, match="decoded-sample budget"):
        _load_audio(high_rate_audio, audio_item)


def test_runtime_file_is_revalidated_and_bound_to_descriptor(tmp_path, monkeypatch):
    expected = (tmp_path / "ref.mp4").resolve()
    expected.write_bytes(b"media")
    pipe = _pipe(video=True)
    original = pipe.media_for_item("video")
    media = RuntimeMedia(
        original.asset_id,
        original.item_id,
        original.kind,
        RuntimeMediaFile(str(expected), "video"),
        (("mtime_ns", 123), ("size_bytes", 5)),
    )
    pipe = replace(
        pipe,
        runtime_media=tuple(media if item.item_id == "video" else item for item in pipe.runtime_media),
    )
    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.utils.director_media.resolve_asset_path",
        lambda _asset: expected,
    )
    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.utils.director_media.probe_asset",
        lambda asset: (asset, {"size_bytes": 5, "mtime_ns": 123}),
    )
    assert _runtime(pipe, "video", "Video") is media

    changed = replace(
        pipe,
        runtime_media=tuple(
            replace(item, metadata=(("mtime_ns", 122), ("size_bytes", 5)))
            if item.item_id == "video" else item
            for item in pipe.runtime_media
        ),
    )
    with pytest.raises(ValueError, match="asset changed"):
        _runtime(changed, "video", "Video")

    wrong_path = replace(
        media,
        payload=RuntimeMediaFile(str(tmp_path / "other.mp4"), "video"),
    )
    wrong = replace(
        pipe,
        runtime_media=tuple(
            wrong_path if item.item_id == "video" else item for item in pipe.runtime_media
        ),
    )
    with pytest.raises(ValueError, match="does not match"):
        _runtime(wrong, "video", "Video")


def test_node_delegates_to_native_and_preserves_output_identity(native, monkeypatch):
    _install_native(monkeypatch, native)
    positive, latent = JR_H3_DirectedVideoConditioning().condition(
        object(), object(), _pipe(first=True),
        "Auto", "Prefer Node", 1344, 768, 124, "match", None,
    )
    assert (positive, latent) == ("i2v-positive", "i2v-latent")
    assert _I2V.calls[-1]["first_frame"] is not None
    positive, latent = JR_H3_DirectedVideoConditioning().condition(
        object(), object(), _pipe(refs=2, video=True),
        "Auto", "Prefer Node", 1344, 768, 124, "max", None,
    )
    assert (positive, latent) == ("ref-positive", "ref-latent")
    assert list(_Ref2V.calls[-1]["ref_images"]) == ["ref_image_0", "ref_image_1"]
    assert list(_Ref2V.calls[-1]["ref_videos"]) == ["ref_video_0"]
    assert _Ref2V.calls[-1]["ref_video_audios"] == {}


@pytest.mark.parametrize("bad", [None, (), ("only-one",), {"bad": "shape"}])
def test_native_output_shape_errors(bad):
    with pytest.raises(RuntimeError, match="incompatible output shape"):
        normalize_native_output(bad)


def test_node_metadata_and_types():
    inputs = JR_H3_DirectedVideoConditioning.INPUT_TYPES()
    assert list(inputs["required"]) == [
        "clip", "vae", "pipe", "mode_override", "dimension_source",
        "width", "height", "length", "ref_image_size",
    ]
    assert inputs["optional"] == {"audio_vae": ("VAE",)}
    assert JR_H3_DirectedVideoConditioning.RETURN_TYPES == ("CONDITIONING", "LATENT")
    assert JR_H3_DirectedVideoConditioning.RETURN_NAMES == ("positive", "latent")
    assert JR_H3_DirectedVideoConditioning.CATEGORY == "JR MiniMax H3/Generation"


def test_full_director_optimizer_review_conditioning_pipe_e2e(native, monkeypatch):
    pipe0 = _pipe(first=True, last=True, refs=1, video=True, audio_role="reference_audio")
    optimized = (
        "subject_definitions:\n<Subject 1> is defined by <Picture 1>, <Picture 2>, and <Picture 3>.\n"
        "summary: [reference generation] <Subject 1> moves through the scene.\n"
        "retention_analysis:\n"
        "<Picture 1>: fully_preserved - retain the opening appearance.\n"
        "<Picture 2>: fully_preserved - retain the closing appearance.\n"
        "<Picture 3>: fully_preserved - retain the reference identity.\n"
        "<Video 1>: partially_preserved - retain its motion rhythm.\n"
        "<Audio 1>: fully_copy - retain the reference sound.\n"
        "detailed_description: [Shot 1] <Subject 1> moves continuously.\n"
        "overall_soundscape: Reference ambience continues.\n"
        "non_diegetic_music: N/A"
    )
    monkeypatch.setattr(
        "ComfyUI_JR_MiniMaxH3Node.nodes.h3_prompt_optimizer_official.request_chat",
        lambda *_args, **_kwargs: {"choices": [{"message": {"content": optimized}}]},
    )
    result = JR_H3_OpenAICompatiblePromptOptimizer().optimize(
        prompt="", enable=True, api_base_url="http://127.0.0.1:10000", model="test",
        prompt_profile="Standard", duration_seconds=5, target_width=768, target_height=1152,
        temperature=0.0, top_p=1.0, max_tokens=1800, timeout_seconds=2,
        image_send_size=768, fail_mode="Stop Workflow", disable_reasoning=True,
        h3_input_mode="Auto", reference_instructions="", api_key="", pip=pipe0,
    )
    pipe1 = result[3]
    assert pipe1 is not pipe0 and pipe1.optimized_prompt == optimized
    for field in (
        "timeline", "shots", "visual_items", "audio_items", "compiled_director_prompt",
        "runtime_media", "reference_registry",
    ):
        assert getattr(pipe1, field) is getattr(pipe0, field)

    class Socket:
        closed = False

    class Server:
        client_id = "browser"
        sockets = {"browser": Socket()}

        def __init__(self):
            self.events = []

        def send_sync(self, event, data, sid=None):
            self.events.append((event, data, sid))

    server = Server()
    monkeypatch.setattr(review_module, "_prompt_server", lambda: server)
    PROMPT_REVIEW_STORE.clear()
    approved = "final reviewed director prompt"
    with concurrent.futures.ThreadPoolExecutor() as pool:
        future = pool.submit(JR_H3_PromptReviewPause().review, "", 3600, "review", pipe1)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not server.events:
            time.sleep(0.01)
        request = next(event for event in server.events if event[0] == "jr_h3_prompt_review_requested")
        assert request[1]["text"] == optimized
        PROMPT_REVIEW_STORE.submit(request[1]["review_id"], approved)
        reviewed, pipe2 = future.result(timeout=3)
    assert reviewed == approved
    assert pipe2 is not pipe1 and pipe2.reviewed_prompt == approved
    for field in (
        "timeline", "shots", "visual_items", "audio_items", "compiled_director_prompt",
        "runtime_media", "reference_registry",
    ):
        assert getattr(pipe2, field) is getattr(pipe0, field)

    _install_native(monkeypatch, native)
    positive, latent = JR_H3_DirectedVideoConditioning().condition(
        object(), object(), pipe2, "Auto", "Prefer Node", 1344, 768, 124, "match", object()
    )
    assert (positive, latent) == ("ref-positive", "ref-latent")
    assert _Ref2V.calls[-1]["prompt"] == approved
    assert list(_Ref2V.calls[-1]["ref_images"]) == [
        "ref_image_0", "ref_image_1", "ref_image_2"
    ]
    assert list(_Ref2V.calls[-1]["ref_videos"]) == ["ref_video_0"]
    assert list(_Ref2V.calls[-1]["ref_audios"]) == ["ref_audio_0"]
