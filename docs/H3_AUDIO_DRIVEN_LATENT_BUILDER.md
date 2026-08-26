# JR MiniMax H3 Audio Driven Latent Builder

## Purpose

`JR_H3_AudioDrivenLatentBuilder` is a latent-level adapter. It replaces the audio stream inside an existing official MiniMax H3 AV `NestedTensor`, preserves video denoising semantics, and locks the replacement audio stream during sampling.

```text
Load Audio
  -> VAE Encode Audio (using the appropriate MiniMax H3 Audio VAE)
  -> Audio Drive Latent ----------------------------------------------┐
                                                                    ├-> Audio Driven Latent Builder -> KSampler
JR MiniMax H3 Directed Video Conditioning -> AV Latent -------------┘
```

The source audio latent is used to drive H3 video generation while the audio denoise branch is locked. For final output audio quality, the original waveform can still be muxed into the decoded video separately.

## Interface

Inputs:

- `av_latent: LATENT`: an official H3 joint AV latent containing exactly `NestedTensor(video, template_audio)`.
- `audio_drive_latent: LATENT`: a plain H3 audio latent `{ "samples": Tensor[B,32,2,T] }` encoded by the appropriate MiniMax H3 Audio VAE.

Outputs:

- `audio_driven_av_latent: LATENT`: the reconstructed official H3 AV latent for a normal compatible H3 sampler path.
- `status: STRING`: shapes, mask handling, temporal fit, batch fit, dtype/device handling and drive mode.

## Exact audio-lock behavior

The node creates:

```text
samples = NestedTensor(original_video, fitted_audio_drive)
noise_mask = NestedTensor(video_mask, zeros_like(fitted_audio_drive))
```

If the incoming AV latent contains a valid official two-stream `noise_mask`, its exact video-mask Tensor is preserved. The incoming audio mask is discarded. If the AV latent has no mask, the video fallback is `ones_like(original_video)`. A malformed mask fails clearly instead of silently destroying first/last-frame, temporal, continuation or partial-regeneration semantics.

The input LATENT mapping is shallow-copied. Only `samples` and `noise_mask` are replaced; unrelated metadata is retained, and neither input dictionary is mutated.

## Temporal and batch fitting

The template audio stream inside `av_latent` is authoritative:

- equal `T`: preserve the Audio Drive tensor;
- longer `T`: trim on the final dimension;
- shorter `T`: append zeros on the final dimension;
- no interpolation, stretching, looping or generated tail.

For batch size, `N -> N` is unchanged and `1 -> N` uses safe expansion. Any other mismatch fails. The fitted audio is converted only as needed to the template audio stream's device and dtype; the video stream is never converted or copied.

## Validation and boundaries

The node requires the exact current ComfyUI `comfy.nested_tensor.NestedTensor` type with exactly two streams. It validates H3 video `[B,24,T,H,W]`, template/drive audio `[B,32,2,T]`, official H3 temporal compatibility, floating materialized storage, batch/device/dtype consistency of the incoming AV pair, and finite values. It does not accept duck-typed containers.

This node does not load audio, resample waveforms, load or run an Audio VAE, decode audio/video, detect faces or phonemes, generate audio, or mux final media. Audio encoding and final waveform muxing remain separate workflow responsibilities.
