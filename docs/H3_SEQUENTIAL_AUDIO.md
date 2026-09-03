# H3 Sequential Audio Generation

This workflow generates a long audio-driven MiniMax H3 video as a series of independent ComfyUI prompts. It is different from `JR_H3_TemporalChunkSampler`: the temporal sampler divides one already-created long AV latent inside one execution, while the sequential audio workflow creates, samples, decodes and commits one complete H3-sized clip per execution.

## Recommended wiring

```text
Director Desk -> Prompt Optimizer -> Prompt Review -> Directed Video Conditioning
                                                |-> positive ----------------------┐
                                                |-> latent -> Sequential Audio     |
Full Load Audio ------------------------------------------------> Chunk Driver     |
Audio VAE ------------------------------------------------------> Chunk Driver     |
                                                                  | latent          |
                                                                  | context         |
                                                                  | seed -> Random Noise
                                                                  v                |
                                              Sequential Continuation Guide <-------┘
                                                                  | positive -> Guider
                                                                  | latent ---------┐
Sampler + Sigmas + Noise + Guider -----------------------------------------------> Sampler
                                                                                   |
                                                          Sequential Latent Checkpoint
                                                                                   |
                                                                                VAE Decode
                                                                                   |
                                                          Sequential Video Output (OUTPUT)
```

`Sequential Video Output` replaces Enhanced Video Combine on this branch. It must be an execution root so each decoded chunk is committed before the next prompt is queued.

## Prompt methodology

The default contract is **Same Audio Reactive Prompt**:

- Ref2VA official structure.
- One open-ended `[Shot 1]` only.
- No chunk boundary timestamps in prompt text.
- Every execution reuses the same reviewed/optimized prompt and positive conditioning.
- The current real audio slice changes; prompt text does not need to change.
- Avoid `begins`, `then`, `finally`, `at the end`, or another miniature beginning/middle/end arc that would restart in every chunk.

An appropriate `detailed_description` describes one continuous performance whose facial expression, mouth movement, body movement, rhythm, pauses and intensity follow `<Audio 1>`. The long source audio may also be registered as Director Driving Audio for Writer/Ref2VA semantics. The optimizer still sends only reference labels and instructions to an OpenAI-compatible endpoint; it never uploads audio binary.

## Exact H3 presets

| Preset | Frames at 24 fps | Video latent T | Audio latent ticks at 40 Hz | Hard-prefix stride frames / ticks |
| --- | ---: | ---: | ---: | ---: |
| 14.375 seconds | 345 | 102 | 575 | 306 / 510 |
| 10.125 seconds | 243 | 72 | 405 | 204 / 340 |
| 8.000 seconds | 192 | 57 | 320 | 153 / 255 |
| 5.875 seconds | 141 | 42 | 235 | 102 / 170 |

`Hard Latent Prefix` is the default and supports all four presets. Every preset uses a fixed 39-frame / 12-video-latent-step / 65-audio-tick continuation context. The stride is the selected raw window minus that context.

The Directed Video Conditioning `length` must match the selected preset's frame count. Raw audio boundaries are calculated from the absolute global source frame timeline with `round(frame_boundary × sample_rate / 24)`; they are never calculated by repeatedly rounding a duration. In hard-prefix mode adjacent raw audio slices intentionally overlap by 39 frames / 65 ticks, while the final mux still uses the original continuous PCM exactly once.

The final chunk is padded only for H3 audio-latent generation. Its decoded video is trimmed to the number of frames needed to cover the real remaining source samples. The final output mux uses the original full decoded PCM once, so audio is never separately encoded per segment.

## Node roles

### Sequential Audio Chunk Driver

Inputs: Directed H3 AV latent, full AUDIO, H3 audio VAE, preset, continuity/seed settings and job path. Outputs: current audio-driven AV latent, `JR_H3_AUDIO_CHUNK_CONTEXT`, deterministic seed, the real unpadded source audio slice and status.

On the first execution it writes two float32 PCM spools:

- source sample-rate PCM for the one-time final mux;
- one globally resampled audio-VAE-rate PCM stream for exact latent slicing.

It then encodes only the selected padded block and delegates H3 AV replacement/mask validation to the existing Audio Driven Latent Builder implementation. It never increments the chunk index itself.

### Sequential Continuation Guide

`Hard Latent Prefix` leaves chunk 1 unchanged. For every later chunk it loads the preceding full sampled AV checkpoint, copies the final 12 video latent steps into the current video's first 12 steps, and forces that video-mask prefix to zero. The remaining video mask retains its upstream semantics. The current chunk's absolute-time audio latent is never replaced with previous audio and its mask remains zero. This mode does not call `MiniMaxH3AddGuide` or append a duplicate `minimax_keyframes` tail guide.

`Previous Last Frame` remains a legacy fallback: it uses the connected initial frame for chunk 1, then loads the previous committed terminal PNG and delegates a local frame-zero anchor to ComfyUI's native `MiniMaxH3AddGuide`. `Independent MV` leaves positive conditioning unchanged.

The guide must be placed before the Guider is constructed. Prompt reuse alone does not guarantee pose or camera-state continuity.

### Sequential Latent Checkpoint

After the sampler, the two sampled tensors are atomically saved as `latents/chunk_NNNNN.safetensors`. The node returns an official CPU-backed H3 `NestedTensor` for decode. No Python pickle is loaded, and no Tensor is stored in workflow JSON.

### Sequential Video Output

The output node:

1. keeps all real frames for chunk 1 and trims exactly 39 decoded prefix frames from every later hard-prefix chunk;
2. encodes a silent H.264/MP4 segment;
3. validates it before commit;
4. saves its last frame;
5. atomically advances `manifest.json`;
6. queues exactly one next ComfyUI prompt after whole-prompt `execution_success` when the selected continuation mode is enabled;
7. after the last chunk, concatenates compatible video streams with `-c:v copy` and encodes the original continuous PCM to AAC once.

The first usable encoder is recorded in the manifest and every later segment must use exactly the same encoder. Encoder changes fail closed instead of risking a corrupt concat.

## Cache and recovery

Default relative cache path:

```text
ComfyUI/output/temp/JR_H3_audio_jobs/<job_name>/run_0001/
```

Absolute paths are supported. Relative paths cannot escape the ComfyUI output directory. `job_name` is sanitized and `run_id` selects a new immutable run directory. Existing runs are never recursively deleted or overwritten; increment `run_id` to start over.

The manifest is the authoritative state. It records and validates `continuation_mode`, `hard_context_frames`, `hard_context_latent_steps`, and the preset-specific `stride_frames`. Hard-prefix manifests use schema 2; an older run fails closed and requires a larger `run_id`. A chunk advances only after its video segment and continuation frame have been written and validated. A failed sampler, decode, encode or final mux leaves the current index unchanged. Queue the same workflow manually to resume.

There are two continuation controls:

- `auto_queue_next=true` is the browser mode. The frontend queues the next prompt after the chunk commit event and requires an active page. Closing the browser safely pauses after the committed chunk; reopen the workflow and queue once to resume from disk.
- `server_auto_continue=true` is the direct API/headless mode. It supersedes browser auto-queue, waits for the entire source prompt to finish successfully, and re-posts the saved API prompt through the normal loopback `/prompt` endpoint. The original `extra_data` context is preserved. Error or interrupt stops the chain, and the loopback request has a 30-second timeout.

Server continuation deliberately supports exactly one Sequential job per source API prompt. A duplicate callback for the same job is deduplicated. A second independent job (including a conflicting total-chunk lifecycle) blocks replay for the source prompt, retains already committed chunks, and pauses fail-closed; split independent chains into separate prompt/workflow submissions. Prompt state is removed when its watcher exits. For wrapped execution such as a ComfyTV stage, the outer saved API prompt does not contain `JR_H3_SequentialVideoOutput`; the replay guard therefore skips blind server replay and leaves continuation to the orchestrator.

## Memory boundary

At most one generated chunk is active in the sampling/decode path. Sampled AV tensors are checkpointed to CPU/disk before decode; decoded frames are streamed to one segment and then become unreachable after the prompt finishes. ComfyUI may retain the immediately preceding node output in its execution cache until the next prompt invalidates it, so this feature bounds growth by chunk count but cannot remove model weights or every framework cache allocation.

The final pipeline never loads every decoded segment back into one IMAGE batch. Final video concat is a stream copy and the full PCM spool is read by FFmpeg, avoiding an all-frames RAM/VRAM spike.

## Boundaries

- There is no cross-chunk DiT hidden-state carry.
- Hard-prefix mode locks the overlap latent exactly, but downstream model/decode behavior can still expose a visible or motion discontinuity; real generation must be evaluated separately.
- Hard-prefix mode supports the exact 345/243/192/141-frame H3 presets; arbitrary durations outside these temporal grids are not supported.
- Previous-frame guidance is retained only as a legacy fallback and cannot guarantee numerically seamless motion.
- `Independent MV` deliberately permits visual cuts and different deterministic per-chunk seeds.
- The current release supports H.264/MP4 segment caching and final MP4 output for concat safety.
- The final audio is AAC encoded once; “exact” refers to continuous sample slicing and absence of per-block trim/pad/encoder joins, not lossless delivery codec.
