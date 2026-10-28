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
- `first_latent`, `last_latent`: optional clean single-frame H3 video LATENTs, appended after the original inputs.

Additional `reference_videos` and `reference_audios` use native Autogrow sockets for references 2 and 3. Connect a slot to expose the next one. The original input names and order remain intact for saved workflows. Connected references are registered in numeric slot order; empty slots are skipped. Unpack indexes refer to that compact registry order. Directed Conditioning receives up to three videos and three standalone audios; Driving Audio retains its existing routing rules and is not a fourth standalone audio slot.

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
- original `first_latent` and `last_latent` at output indices 17 and 18; all existing output indices are unchanged.

An index beyond the available count returns `None` for that media output. This is intentional: the node stays compact while the passthrough PIPE remains the lossless multi-item data bus. Add another Unpack node with a different index when several individual standard outputs are required.

For Director Desk file-backed video, Unpack returns a lazy standard `VideoFromFile` value after the existing root-containment and fingerprint checks. File-backed audio uses the existing bounded decode path. In-memory Builder VIDEO/AUDIO values remain runtime objects; mono audio is normalized to stereo using the same H3 adapter behavior.

## Persistence and security

Builder creates runtime-only synthetic descriptors so registry ordering and PIPE validation remain deterministic. Actual IMAGE tensors, VIDEO objects, AUDIO waveform tensors, absolute paths and binary data are held only in `runtime_media`. They are not written into workflow JSON or registry JSON.

Neither node accesses the network, loads models, initializes CUDA or invokes FFmpeg at import time. Unpack decodes a selected file-backed audio item during execution; its standard VIDEO output remains lazy until a downstream consumer requests components.

## Experimental pre-encoded frame masters (2026-09-15)

For Hermes wallpaper continuity, pair `first_frame` with its corresponding `first_latent`, and optionally pair `last_frame` with `last_latent`. IMAGE still supplies Qwen/optimizer vision; the clean latent supplies the DiT keyframe without VAE encoding. Images and latents must depict the same frame and framing; this semantic correspondence cannot be inferred from tensor shapes.

Accepted latent shape: `[1,24,1,H,W]`, finite floating point, standard public H3 video-VAE latent domain. No AV NestedTensor, multi-frame clip, mask, or intermediate noisy sampler state. Shape validation cannot determine whether arbitrary values are genuinely clean H3 latents. Builder snapshots the sample tensor, retains extra LATENT dictionary metadata, and keeps it out of persisted JSON. Unpack returns the original-resolution master, never the generation-sized copy. Runtime PIPE values are not durable session storage; Hermes must separately save/reload matching IMAGE/LATENT pairs for later jobs and reset both on restoration.

Use Directed Conditioning's **Prefer Node** to select the actual video canvas independently of the wallpaper. For square video, 640×640 is 0.4096MP, 768×768 is 0.5898MP and 896×896 is 0.8028MP. A 1440×1440 master is 2.0736MP. `Prefer Pipe` retains the existing native canvas-selection behavior; it does not mean exact master-sized generation.

The pre-encoded adapter builds native `minimax_keyframes` at frame 0 / the last aligned frame. It stretches first-frame copies and center-crops last-frame copies, matching native IMAGE anchor geometry. Spatial resampling uses float32 area interpolation in latent space, then restores source dtype/device. It is not a learned upscaler, not an exact substitute for resize-then-VAE-encode, and not a promise of lossless multi-turn generation. It never changes the master or the AV target timeline.

With all frame anchors pre-encoded, Progressive Guided uses independent copies of each master for its low/high grids and needs no keyframe VAE round trip. Mixed IMAGE/latent anchors still need the same video VAE for the IMAGE-only side. Ordinary IMAGE-only workflows retain their previous official-node delegation and progressive decode/encode behavior.

Without a paired IMAGE, Directed Conditioning decodes the master once for vision only and never re-encodes that frame. Unpack's corresponding IMAGE output remains `None`. The upstream Prompt Optimizer has no VAE input, so a latent-only anchor there produces an actionable request to connect the corresponding IMAGE rather than inventing a visual representation.

When pre-encoded anchors are combined with independent references, the adapter keeps actual first/last `minimax_keyframes` alongside native `minimax_refs`; IMAGE-only legacy Ref2V routing is unchanged. Reference images/videos/audio still require their normal encoders. Driving Audio in PIPE retains its reference-audio routing; locking target audio still uses the separate Audio Driven Latent Builder. Combined-task checkpoint quality is not guaranteed by payload compatibility.

No VAE, CLIP or model is monkey-patched. This opt-in path uses an explicit adapter because the installed native conditioning nodes do not accept pre-encoded keyframes. Its current contract is tested against local ComfyUI 0.35.0; this does not resolve the existing GitHub CI pin to 0.34.0.

Validation: compare IMAGE-only with IMAGE+LATENT at fixed seed/size, check VAE encode call counts, verify original masters before/after, then repeat multiple Hermes continuation rounds. Test normal and Progressive Guided sampling separately. The separate experimental [Tail Frame Latent](H3_TAIL_FRAME_LATENT.md) node now supplies a true decoded endpoint through one explicit encode, keeping source resolution. Zero-encode extraction and high-resolution enhancement remain separate work.

Local automated checks (2026-09-15): 940 passed / 1 environment skip; 32 new anchor tests also passed with CUDA hidden. Miniature native H3 tests cover first/last anchors, independent refs, TST 0.2, locked target audio and fixed-seed progressive repeats. VAE/CLIP compute and learned upscaling are replaced in these contract tests; no full-checkpoint visual-quality claim is made. Import/workflow smoke and Ruff pass; four previously documented legacy stale-link warnings remain.
