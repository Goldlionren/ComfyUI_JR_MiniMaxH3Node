# JR MiniMax H3 Director Desk

`JR_H3_DirectorDesk` is a timeline-aware multimodal director composer. It does not call an LLM and it is not a second Prompt Optimizer. It deterministically compiles the editor state into:

- `director_prompt: STRING` for inspection and debugging;
- `pip: JR_H3_DIRECTOR_PIPE` as the authoritative Director data bus.

The normal connection is:

```text
Director Desk.pip
  -> Prompt Optimizer.pip
       pip
         -> Prompt Review & Continue.pip
              pip -> JR MiniMax H3 Directed Video Conditioning.pipe
```

Do not also connect `director_prompt` to the optimizer. The PIP already contains the exact same compiled text together with the structured timeline and runtime media. STRING outputs are monitoring/debugging surfaces; the PIPE is the main workflow bus.

## Editor layout

The node starts at approximately `1000×650` and remains freely resizable. Its size is the native ComfyUI node size saved in the workflow; execution never resets it.

- **Global Direction** describes whole-video style, performance, continuity, camera language and hard constraints.
- **SHOT** contains non-overlapping time intervals and multiline Direction/Notes.
- **VISUAL** accepts Reference Image, First Frame, Last Frame and Reference Video items. Overlap is legal and is displayed in deterministic stacked lanes.
- **AUDIO** accepts Reference Audio and Driving Audio. Reference items may overlap; Driving items must be sequential and non-overlapping.
- **Inspector** is the main editor for role, timeline time, source in/out, Direction, Notes and media relinking.

Items support selection, drag, left/right resize, 0.1-second snapping, duplicate, midpoint split, delete and ordering actions. Right-click opens the same contextual operations; double-click focuses the Inspector. Inspector edits are committed before timeline selection or dragging, so a newly typed time is not lost when the user immediately clicks a Shot. Dragging uses a local draft and commits once on pointer release so one drag produces one ComfyUI undo transaction.

For Shots, `Earlier` / `Later` swaps the selected Shot with the previous or next chronological time slot. For Visual and Audio items, `Lane ↑` / `Lane ↓` changes only the saved stacked display order: it does not change timing or silently renumber `<Picture N>`, `<Video N>` or `<Audio N>` labels. `Duplicate` creates a new item (or the next non-overlapping Shot), while `Split` divides the selected non-point item at its midpoint.

### Time semantics

- All displayed, saved and compiled times use the same 0.1-second canonical value.
- Shots may touch at a boundary but may not overlap.
- First Frame is one IMAGE point marker fixed at `0.0s`; it is never a duration clip.
- Last Frame is one IMAGE point marker fixed at the timeline duration; the final Shot must end at that same time.
- Reference Image and Reference Video items may overlap at the same time. Three images over the same interval remain three independent references; they are not auto-segmented into keyframes.
- Video `timeline start/end` controls when the reference is active. `source in/out` identifies the source-media range and is validated separately.
- Reference Audio may overlap. Overlapping Driving Audio is rejected because the authoritative active source would be ambiguous.

## Media import and preview

Use **+ Image**, **+ Video**, **+ Audio**, drag a media file onto the editor, or use **Relink** in the Inspector. Upload uses ComfyUI's existing input upload mechanism and stores files under the `jr_h3_director` input subfolder. Preview uses ComfyUI `/view` descriptors:

```json
{
  "filename": "clip.mp4",
  "subfolder": "jr_h3_director",
  "type": "input"
}
```

The workflow saves only that relative descriptor plus bounded metadata such as duration and dimensions. It never saves absolute paths, tensors, decoded frames, base64, waveform samples or video/audio bytes. Video preview seeks a small distance into the file and creates an ephemeral, scaled browser poster; the poster is never serialized.

The JR media probe verifies image dimensions or uses bounded `ffprobe` metadata inspection when available. The execution resolver repeats root containment and file checks. Missing, corrupt, unsafe and unsupported assets produce clear errors without exposing arbitrary filesystem paths. Images are decoded during Director Desk execution. Video and audio use validated runtime-only file handles inside the PIP; the optimizer receives only their labels, timing and Direction/Notes text, while Directed Video Conditioning revalidates and decodes them for the native H3 call.

If ffprobe is unavailable, browser metadata may still provide a preview duration and the descriptor is marked `probe_unavailable`. Install an FFmpeg distribution that includes `ffprobe` for authoritative video/audio metadata validation.

## Reference labels

The compiler builds one canonical registry:

```text
<Picture 1>, <Picture 2>, ...
<Video 1>, <Video 2>, ...
<Audio 1>, <Audio 2>, ...
```

First Frame sorts first, Last Frame second, then Reference Images follow stable creation order and item ID. Display lane placement and array rendering order never change labels. Adding/removing a media item can legitimately recompile the registry; the STRING output, PIP registry, optimizer view and native conditioning input order always use the same result.

The raw compiled text contains the fixed sections `GLOBAL DIRECTION`, `REFERENCE MEDIA`, `TIMELINE` and `END STATE`. It is a Director Prompt, not the final strict H3 response. Prompt Optimizer remains the only component that selects H3 mode, calls the OpenAI-compatible endpoint, validates output and performs at most one format-only repair.

## PIP schema and persistence boundary

Runtime `DirectorPipe` is a frozen Python dataclass graph with schema:

```text
schema = jr_h3_director_pipe
schema_version = 2
timeline
global_direction
shots
visual_items
audio_items
compiled_director_prompt
optimized_prompt
reviewed_prompt
reference_registry
runtime_media
```

`runtime_media` contains IMAGE tensors or validated runtime-only video/audio file handles plus small probed metadata. PIP containers and metadata containers are immutable; an opaque tensor payload is not claimed to be deeply immutable. `DirectorPipe.to_persisted()` explicitly omits runtime media and all compiled/stage prompt output. The frontend continues to save compatible `jr_h3_director_state` schema version 1 JSON in `node.properties`; PIPE v2 is a runtime protocol and is never workflow JSON.

## Prompt Optimizer precedence

When `pip` is not connected, the optimizer follows its historical path byte-for-byte, including legacy inputs, H3 routing, success status format, validator, repair and fail modes.

When `pip` is connected:

- `pip.compiled_director_prompt`, PIP timeline duration, registry and runtime images are authoritative;
- the normal `prompt` widget must be empty (or exactly equal to the compiled prompt);
- legacy `first_frame`, `last_frame`, `ref_image_1..9` and `reference_instructions` must be disconnected;
- the existing `h3_input_mode` still selects or automatically routes T2VA/I2VA/FL2VA/L2VA/Ref2VA;
- PIP IMAGE tensors use the existing JPEG data-URL helper;
- VIDEO/AUDIO are represented as registry/timeline text only;
- successful optimization derives a new PIP with `optimized_prompt` and clears any stale reviewed stage;
- the input PIP is never mutated.

Conflicts produce a descriptive fallback status under Return Original or stop the workflow under Stop Workflow. PIP success appends `source=pip` to the normal status. Prompt Review derives another PIP with the approved `reviewed_prompt`; final prompt selection is `reviewed > optimized > compiled director`.

## Known limits

- The first release uses a lightweight DOM timeline rather than a full NLE or third-party timeline library.
- Audio uses the browser's native player. A decoded waveform is intentionally not persisted or generated on workflow reload.
- One execution accepts at most 32 unique runtime assets, 12 million pixels per IMAGE and 24 million IMAGE pixels in aggregate. These bounds prevent unbounded decode/probe work; larger projects should downscale reference images or split the timeline.
- If upload succeeds but subsequent media inspection fails, ComfyUI keeps the uploaded input file. Director Desk does not delete user input assets automatically; remove abandoned files manually from `input/jr_h3_director`.
- Media files in `temp` or `output` are valid descriptors but may be less portable than `input` assets.
- Browser-only editor actions require the normal ComfyUI frontend; API execution requires a valid serialized `director_state_json`.
- Director Desk does not replace an H3 model loader, sampler, VAE, acceleration, RTX or video output nodes. `JR_H3_DirectedVideoConditioning` replaces only the native I2V/Ref2V conditioning entry choice.

The internal clean-room and lifecycle decisions are recorded in [DIRECTOR_DESK_ARCHITECTURE.md](DIRECTOR_DESK_ARCHITECTURE.md). The end-to-end runtime contract is documented in [DIRECTOR_PIPELINE.md](DIRECTOR_PIPELINE.md).
