# H3 Tail Frame Latent — experimental continuation baseline

`JR_H3_TailFrameLatent` / **JR MiniMax H3 Tail Frame Latent (Experimental)**

This first version obtains an actual video endpoint and converts it to the clean
single-image H3 representation consumed by Director `first_latent` / `last_latent`.
It does **one VAE encode per video endpoint**. It is not zero-encode, lossless
extraction, super-resolution, or a promise of drift-free multi-round continuation.

## Wiring

```text
Original high-resolution IMAGE -> Director first_frame -> existing video workflow
Final completed sampler -> Split AV Latent.video_latent -> Tail Frame Latent
Matching native H3 video VAE ---------------------------> vae

Tail Frame Latent.tail_image  -> next Director first_frame / wallpaper / Save Image
Tail Frame Latent.tail_latent -> next Director first_latent / Save Latent
```

Use **the final pass**, after any existing video upscale/refinement, not its input
noise or an intermediate sigma. The node keeps that video's resolution: a 768px
video produces a 768px tail, even if its original reference was 1440px. Additional
high-resolution tail refinement is a separate next step. There is no MP widget
in this baseline; ordinary interpolation would not establish high-detail quality.

For external latent upscale/refinement, use the appended **tail_context_latent**
output instead. It retains the original decoder window, not the re-encoded
single-frame keyframe. Existing outputs remain at indices 0/1/2; context is index 3.
The current native `tail_context` path outputs `[1,24,7,H,W]` for videos of at
least 7 tokens, yielding 22 decoded frames. Decode that context and select the
last IMAGE before testing any upscale or sampling. It must match `tail_image`
with the same VAE and no externally supplied frames.

Then connect context -> neural upscale -> AV Builder with
[Empty Audio Latent for Tail](H3_EMPTY_AUDIO_LATENT_FOR_TAIL.md). The audio node
now derives duration automatically (7 video tokens -> 22 frames -> 37 audio
ticks). Use native SamplerCustomAdvanced for ordinary second-pass refinement,
split the result, decode the video, and select its last IMAGE for the wallpaper.
The 7-token output is **not** accepted by Director `first_latent` / `last_latent`.
Converting the refined endpoint into a reusable single-image keyframe is still
a separate quality-validation problem; this change does not solve it.

User tests found darkening/stripes in the legacy single-frame encode/decode
round trip even without upscale or sampling. Do not treat the legacy
`tail_latent` as a visually validated reconstruction, or compensate with gamma.
The new context branch bypasses that re-encode for its samples, but the node
still performs the original one encode to produce the legacy output.

Across separate executions, save the paired IMAGE and LATENT; load the LATENT
with ComfyUI's native Load Latent and the image with Load Image. Do not wire a
cycle into one ComfyUI graph. The initial workflow needs only the original IMAGE;
it does not require a pre-encoded initial `first_latent`.

## Endpoint and memory behavior

- `tail_context` (default): for the reviewed native H3 temporal configuration,
  decode only the final **7 latent time slices** (or the complete shorter input).
  A native clip has `T=5k+2` and `17k+5` decoded frames. Its final five output frames
  are produced by the final 7-token decoder window; they are not blended with an
  earlier window. Keeping this context gives the same terminal frame as a full
  decode under that implementation. Unknown temporal configurations fall back to
  full decode; they are not assigned guessed chunk sizes.
- `full_video`: normal full VAE decode, useful for an endpoint A/B check.
  `tail_context_latent` retains the complete input in this mode, and also when
  the temporal contract is unknown. It is not guaranteed to be a short clip.
- Optional `decoded_frames`: matching pre-compression IMAGE frames at the same
  resolution; use their last frame and skip decode. This takes precedence over
  `decode_mode`. It also supports the actual played endpoint after a frame trim.
  Connect the frame batch before MP4/JPEG compression, not a reloaded movie.
  The caller must supply matching content; shape checks cannot establish identity.
  The appended context still represents the original native video endpoint,
  **not** a trimmed/edited supplied IMAGE endpoint. Status warns about this.
  Leave this input disconnected for the context equivalence acceptance test.
- Without `decoded_frames`, the endpoint is the last **native generated** frame.
  It may differ from the last displayed frame if a downstream node trims the video.
- Output IMAGE is the exact selected pre-encode frame. LATENT is its VAE encoding,
  not a guarantee that decoding it reconstructs the IMAGE pixel-for-pixel.
- A `T=1` input is already a keyframe: it is cloned and decoded, with no re-encode
  unless an authoritative `decoded_frames` IMAGE is supplied.

Legacy keyframe output is `[1,24,1,H/16,W/16]`, accepted by Director latent inputs. The output
owns its storage; input masks, AV audio, batch indices and video-specific metadata
are intentionally not copied onto this new single-image latent. Inputs stay
unchanged. Native VAE output dtype/device are respected. Completed sampling is a
caller precondition: tensor shape alone cannot detect residual diffusion noise.
Context also owns compact, detached storage, preserving source dtype/device and
all selected sample values. It starts a fresh local timeline and intentionally
omits source masks, audio, batch indices and conditioning metadata; migrate
guidance separately. A VAE cannot mutate the returned context through its input.

## First acceptance test

1. Generate the initial video through the working original IMAGE path.
2. Extract the final-pass tail and preview `tail_image` beside the decoded video's
   last frame. `full_video` and `tail_context` should agree using the same VAE.
3. For the next round, connect **both** outputs to Director. Start at the same
   video dimensions, keep seed/model/LoRA/prompt/audio/acceleration fixed between
   A/B runs. Compare against IMAGE-only continuation from that exact same tail.
4. Only after this baseline passes, test resolution changes and several rounds.

Known confounders are unchanged: the pre-encoded Director path uses temporal
anchors in mixed reference-media requests, unlike the legacy IMAGE-only Ref2V
path; Progressive can also resize latent-domain guides. These are not equivalent
conditions. Merely extracting a good tail does not prove that either difference
is harmless. For a strict equivalence test, use Image-to-Video without reference
audio and a non-progressive sampler at matching dimensions first.

Use the native H3 video VAE for the first test. Existing TRT wrappers remain
external; an encoder is required even when video decoding uses TRT. A decode-only
TRT VAE can instead supply its already decoded IMAGE frames, with the native VAE
connected here for the one image encode. No new engine is compiled or downloaded.

## Validation scope

Tests cover native temporal padding/blending across short and 5/10/15-second
grids, a small real native ViT decoder (random weights, not production weights),
full/tail endpoint equivalence, supplied/trimmed frames, single-frame reuse,
dtype/storage isolation, invalid AV/audio/encoder outputs and Director first/last
handoff. They establish contracts, not production visual quality or TRT-engine
acceptance. Full-checkpoint visual and repeated-round evaluation remains manual.
