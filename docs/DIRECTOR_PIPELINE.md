# JR MiniMax H3 Director Pipeline

## Authoritative data flow

```text
JR MiniMax H3 Director Desk
  director_prompt: STRING (monitor/debug)
  pipe: JR_H3_DIRECTOR_PIPE
    -> JR MiniMax H3 Prompt Optimizer.pip
         optimized_prompt: STRING (monitor/debug)
         pip: JR_H3_DIRECTOR_PIPE
           -> JR MiniMax H3 Prompt Review & Continue.pip
                reviewed_prompt: STRING (monitor/debug)
                pip: JR_H3_DIRECTOR_PIPE
                  -> JR MiniMax H3 Directed Video Conditioning.pipe
                       positive: CONDITIONING
                       latent: LATENT
```

STRING outputs expose each prompt stage for inspection. They are not the Director data bus. The PIPE is authoritative and keeps the timeline, registry, media descriptors, runtime media and prompt stages together.

## Immutable stages

- P0 from Director Desk contains `compiled_director_prompt`.
- P1 from Prompt Optimizer adds `optimized_prompt` and clears a stale reviewed stage.
- P2 from Prompt Review adds `reviewed_prompt`.
- Directed Video Conditioning reads P2 without modifying it.

Every stage is a distinct frozen `DirectorPipe`. Its tuple containers, registry and runtime metadata containers are immutable. Opaque IMAGE/AUDIO tensors remain runtime objects and are never serialized. The prompt selected for generation is always:

```text
reviewed_prompt > optimized_prompt > compiled_director_prompt
```

An empty result across all three stages is an error.

## Optimizer and Review compatibility

Without a PIPE, Prompt Optimizer retains every legacy input and behavior. Its trailing PIP output is `None`. With a PIPE, the legacy `duration_seconds` widget must exactly equal the PIPE timeline duration; legacy first/last/reference media and reference instructions are conflicts; the normal prompt must be blank or byte-identical to the compiled Director Prompt. Images reuse the existing JPEG data-URL path. Video and audio binary are never sent to the OpenAI-compatible endpoint; only canonical labels, roles, directions and timeline/source text are sent. The endpoint returns semantic JSON only; Python uses authoritative PIPE timing/registry data to produce and validate the final official H3 text.

Prompt Review also retains STRING-only operation. With a PIPE, the review source is `optimized_prompt`, falling back to `compiled_director_prompt`. A nonblank STRING must match that source exactly. Clicking Next always derives P2, even when the user does not edit the text. Timeout, Stop, interruption, active-browser checks, refresh recovery, one-time approval and pending-state cleanup remain unchanged.

## Current native H3 mapping

The adapter calls the installed ComfyUI implementation in `comfy_extras.nodes_minimax_h3`:

| JR mode | Native implementation | Real media |
| --- | --- | --- |
| Image to Video | `MiniMaxH3ImageToVideo.execute` | `first_frame`, `last_frame` IMAGE anchors |
| Reference to Video | `MiniMaxH3ReferenceToVideo.execute` | dense `ref_image_N`, `ref_video_N`, `ref_audio_N` inputs |

Auto selects Reference to Video if any Reference Image, Reference Video, Reference Audio or Driving Audio exists. Otherwise it selects Image to Video. Explicit Image to Video rejects reference-only media instead of ignoring it. Explicit Reference to Video requires at least one reference or anchor image.

The current native limits are 9 reference images, 3 reference videos, 3 index-paired video soundtracks and 3 standalone reference audios. In Ref2V, First/Last Frame are ordinary Picture references and count toward the same 9-image total. Director does not synthesize video soundtrack records, so it deliberately passes no `ref_video_audio_N`. Reference Audio and Driving Audio are both sent through standalone `ref_audio_N`; the native API cannot distinguish their Director roles.

Reference order is the canonical registry order. First Frame sorts before Last Frame, followed by Reference Images; videos and audios retain their independent canonical order. Consequently `<Picture N>`, `<Video N>` and `<Audio N>` match the actual native input order.

## Dimensions, duration and length

MiniMax H3 generation is fixed at 24 fps in the current native implementation. Director timeline fps is editing metadata and does not change native H3 fps.

- **Prefer Pipe**: use the first Picture/Video runtime dimensions when available and adapt them with the native `adapt_canvas`; otherwise use node width/height. Convert timeline duration using `ceil(seconds × 24)` and let the native helper align frame count until `frames % 17 == 5`.
- **Prefer Node**: use node width, height and length. PIPE values remain available as metadata but do not override widgets.

Width and height must be multiples of 32 in the current native range. Length is validated as 5..3600 before native alignment.
With **Prefer Pipe**, a timeline longer than 150 seconds exceeds that 3600-frame input limit and is rejected. This is an adapter bound, not a general model-duration claim.

## Honest capability boundaries

- Native Ref2V has no first/last-frame anchor inputs. In a mixed anchor-plus-reference project, First/Last Frame are consumed as ordinary reference images. They are not hard endpoints.
- Timeline `start/end` and item Direction/Notes guide prompt semantics. The current native conditioning API does not time-gate reference tensors to arbitrary clip intervals.
- Video `source_in/source_out` trims the decoded frame batch. Reference videos must decode at 24 fps, retain at least 5 frames after trimming, are limited to 15 decoded seconds per item, and require trusted width/height metadata for a bounded pixel budget. The native implementation then trims reference frames to its `17k+5` frame grid. Audio source ranges trim the waveform; file audio requires trusted size/duration/sample-rate/channel metadata and is bounded by file size, duration, channels and decoded sample count before whole-file decoding.
- Driving Audio is not a target soundtrack replacement. It is a Director role that becomes prompt/timeline semantics plus a native standalone audio reference.
- Embedded video soundtracks are not auto-extracted. Native soundtrack labels are interleaved before their paired Video labels, which would break the existing Director registry unless soundtrack records are explicitly added in a future schema.

## Persistence and security

Workflow JSON stores only Director state schema v1 and relative ComfyUI asset descriptors. PIPE schema v2 exists only at runtime. It may contain IMAGE tensors and validated runtime-only video/audio file handles, but never writes tensors, waveform samples, decoded frames, base64, absolute paths or binary media into workflow JSON.

Conditioning re-resolves each file descriptor inside the allowed ComfyUI input/temp/output roots and verifies that the runtime handle still matches before decoding. Imports do not access the network, load models, initialize CUDA or execute FFmpeg.
