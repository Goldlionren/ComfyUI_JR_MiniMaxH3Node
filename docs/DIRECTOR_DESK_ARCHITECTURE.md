# Director Desk architecture decision

Status: accepted for the initial `JR_H3_DirectorDesk` implementation.

## Decision

Director Desk is split into three strict layers:

1. `DirectorState` is the only workflow-persisted representation. It is schema-versioned JSON containing timeline semantics, stable item IDs, lightweight ComfyUI asset descriptors and small UI preferences. It never contains tensors, decoded media, base64 data or absolute paths.
2. The compiler validates and normalizes that state, builds one deterministic reference registry, and emits the raw Director Prompt. Shot ordering follows time; media ordering follows stable creation order and never visual lane placement.
3. `DirectorPipe` is an immutable runtime object. It contains the validated structured timeline, the exact compiled prompt, the same registry and execution-time media handles. Runtime media is never serialized back into workflow state.

The V1 node receives a hidden serialized state widget because `node.properties` is not an execution input. The frontend mirrors the same normalized state into `node.properties` for workflow persistence and into that hidden widget for queue execution. All pointer drags use a local draft and commit once at pointer release so ComfyUI undo remains usable.

## Timeline rules

- Times are normalized to 0.1-second precision in both frontend and backend.
- Shots may touch but never overlap.
- Reference images, reference videos and reference audio may overlap and are displayed in deterministic stacked lanes.
- First Frame is one IMAGE point marker at `0.0s`; Last Frame is one IMAGE point marker at timeline end. Neither is a duration clip.
- Driving Audio segments may be sequential but may not overlap.
- Video source in/out and timeline start/end are distinct fields.

## Media and security

Uploads use ComfyUI's existing upload endpoint. Persisted descriptors contain only `filename`, `subfolder`, `type` and bounded metadata. Backend resolution is restricted to ComfyUI input/temp/output roots. Images are decoded during Director execution; video and audio become validated runtime-only file handles. The optimizer sees only their labels, timing and direction text, while Directed Video Conditioning lazily decodes the real frames/waveforms immediately before calling the native H3 node. Media inspection uses bounded, timeout-controlled subprocess argument lists with no shell.

## Optimizer compatibility

`pip` is appended as an optional `JR_H3_DIRECTOR_PIPE` input to the existing optimizer. With no PIP, the legacy path remains unchanged and its new trailing PIP output is `None`. With PIP, the compiled Director Prompt, duration and reference registry are authoritative; legacy media/reference inputs are rejected as conflicts. Existing H3 mode routing, JPEG conversion, validation, single repair attempt and fail modes remain the only optimization pipeline. Successful optimization derives PIPE P1; Prompt Review derives PIPE P2. Both preserve the original immutable timeline, registry and runtime media.

## Native conditioning boundary

`JR_H3_DirectedVideoConditioning` is a thin compatibility layer over the installed ComfyUI `MiniMaxH3ImageToVideo` and `MiniMaxH3ReferenceToVideo` implementations. It does not monkey patch ComfyUI and does not copy upstream source. The current native API fixes output timing at 24 fps, accepts at most 9 images (including First/Last Frame when using Ref2V), 3 videos and 3 standalone audios, and has no arbitrary timeline gating. Reference videos are bounded to 15 decoded seconds, must be 24 fps with at least 5 trimmed frames, and are subsequently aligned by native reference logic to the `17k+5` grid. Reference mode has no first/last anchor inputs, so mixed anchor/reference projects consume those anchor images as ordinary references. Director Driving Audio maps to standalone reference audio because no distinct native driving-audio port exists. Prefer Pipe timelines above 150 seconds exceed the conditioning node's 3600-frame input bound and are rejected.

The runtime PIPE protocol is version 2 because it adds immutable `optimized_prompt` and `reviewed_prompt` stage values and validated video/audio file handles. Persisted Director state remains schema version 1 and contains no runtime payload. Final prompt selection is deterministic: reviewed, then optimized, then compiled Director Prompt.

## Reference boundary

ComfyTV, ComfyUI-qwenmultiangle, ComfyUI-mesh2motion and DaSiWa were used only for product and lifecycle research. No source, styles, runner, project database, queue engine, timeline library or bundled asset is copied. The implementation is clean-room and remains Apache-2.0.
