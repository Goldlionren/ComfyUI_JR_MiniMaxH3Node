# Director PIPE Builder / Unpack

## Purpose

`JR_H3_DirectorPipeBuilder` and `JR_H3_DirectorPipeUnpack` are the standard ComfyUI ingress and inspection adapters for `JR_H3_DIRECTOR_PIPE`.

```text
STRING + IMAGE + VIDEO + AUDIO
  -> Director PIPE Builder
       pip: JR_H3_DIRECTOR_PIPE
         -> Optimizer / Review / Directed Conditioning

any JR_H3_DIRECTOR_PIPE
  -> Director PIPE Unpack
       pip (unchanged)
       prompt stages + metadata + selected standard media
```

They do not replace Director Desk. Director Desk remains the full timeline editor with per-item timing, Direction, Notes, source ranges and saved UI state. Builder is for workflows that already have standard ComfyUI media values and need to enter the same authoritative PIPE pipeline without authoring a Director timeline.

## Builder contract

Required inputs:

- `prompt`: non-empty final text.
- `duration_seconds`: 0.1-second canonical timeline duration.
- `fps`: editing metadata. Native H3 generation remains 24 fps in Directed Conditioning.

Optional standard inputs:

- `first_frame`, `last_frame`: exactly one RGB IMAGE each.
- `reference_images`: one IMAGE batch; each batch member becomes a separate canonical Picture record.
- `reference_video`: one standard ComfyUI VIDEO.
- `reference_audio`: one standard ComfyUI AUDIO.
- `driving_audio`: one standard ComfyUI AUDIO.

The generated PIPE contains one Shot spanning the requested duration. The exact input prompt is stored as the current `optimized_prompt`, so direct Builder → Conditioning use is byte-preserving. It is also present in the compiled single-Shot Director context, allowing a later Prompt Optimizer stage to use the normal authoritative PIPE path and replace the optimized stage. `reviewed_prompt` starts empty.

First Frame and Last Frame keep their normal point-anchor roles. Reference media span the generated single Shot. Builder supports up to nine total Picture records across anchors and the reference IMAGE batch, matching the current native H3 Ref2V limit.

## Unpack contract

Unpack never mutates its input. Its first output is the identical PIPE object. It also returns:

- final `prompt` using `reviewed > optimized > director` priority;
- `director_prompt`, `optimized_prompt`, `reviewed_prompt` separately;
- duration, editing fps and the first available Picture/Video dimensions;
- First Frame and Last Frame;
- one Reference Image, Reference Video, Reference Audio and Driving Audio selected with independent 1-based indexes;
- registry JSON containing labels, family, role, timing, Direction and Notes;
- a count/selection status string.

An index beyond the available count returns `None` for that media output. This is intentional: the node stays compact while the passthrough PIPE remains the lossless multi-item data bus. Add another Unpack node with a different index when several individual standard outputs are required.

For Director Desk file-backed video, Unpack returns a lazy standard `VideoFromFile` value after the existing root-containment and fingerprint checks. File-backed audio uses the existing bounded decode path. In-memory Builder VIDEO/AUDIO values remain runtime objects; mono audio is normalized to stereo using the same H3 adapter behavior.

## Persistence and security

Builder creates runtime-only synthetic descriptors so registry ordering and PIPE validation remain deterministic. Actual IMAGE tensors, VIDEO objects, AUDIO waveform tensors, absolute paths and binary data are held only in `runtime_media`. They are not written into workflow JSON or registry JSON.

Neither node accesses the network, loads models, initializes CUDA or invokes FFmpeg at import time. Unpack decodes a selected file-backed audio item during execution; its standard VIDEO output remains lazy until a downstream consumer requests components.
