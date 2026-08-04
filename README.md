# ComfyUI JR MiniMax H3 Node

Five focused ComfyUI V1 nodes for preparing prompts, calculating dimensions, optional RTX enhancement, encoding IMAGE batches, and carrying the last video frame into the next MiniMax H3 segment.

## Nodes

- **JR MiniMax H3 Prompt Optimizer (OpenAI Compatible)** calls `/v1/models` and `/v1/chat/completions`, accepts text plus up to nine IMAGE sockets, and always returns optimized prompt, original prompt, and status.
- **JR MiniMax H3 Resolution Scale Calculator** preserves a selected aspect ratio within a target pixel area and aligns both dimensions to 8, 16, or 32.
- **JR MiniMax H3 RTX Upscaler & Refiner** exposes denoise, deblur, VSR/high-bitrate and sizing controls while loading optional RTX dependencies only on execution.
- **JR MiniMax H3 Enhanced Video Combine** sends an IMAGE batch to FFmpeg, supports H.264/H.265/VP9, MP4/WebM/MKV, CRF quality, optional AUDIO, metadata, ping-pong, first/last PNG export, saved filename, and optional frame pass-through.
- **JR MiniMax H3 Last Frame** returns `frames[-1:].contiguous()` and therefore preserves `[1,H,W,C]`, dtype, device, channels, and values.

## Installation

Copy or link this project into `ComfyUI\custom_nodes`, then install ordinary dependencies in the same Python environment as ComfyUI:

```powershell
cd F:\ComfyUI-aki-v3\ComfyUI
python -m pip install -r F:\AI\custom_nodes\ComfyUI_JR_MiniMaxH3Node\requirements.txt
Copy-Item -Recurse -Force `
  'F:\AI\custom_nodes\ComfyUI_JR_MiniMaxH3Node' `
  'F:\ComfyUI-aki-v3\ComfyUI\custom_nodes\ComfyUI_JR_MiniMaxH3Node'
```

Do the copy only while ComfyUI is stopped. The development process does not modify the production directory.

FFmpeg must be installed and `ffmpeg.exe` must be on `PATH` (or available through `imageio-ffmpeg`). Video encoding is performed only when the combine node executes.

RTX processing is optional. It requires a compatible NVIDIA RTX GPU, current driver, NVIDIA Video Effects SDK, and a Python binding that provides `nvvfx`:

```powershell
python -m pip install -r F:\AI\custom_nodes\ComfyUI_JR_MiniMaxH3Node\requirements-rtx.txt
```

NVIDIA SDK packaging changes between releases; verify the SDK's Windows installation instructions. A missing SDK never prevents the other four nodes from loading.

## OpenAI-compatible prompt optimization

Example llama.cpp server (replace the placeholder model path):

```powershell
llama-server.exe `
  -m "F:\Models\YourModel\model.gguf" `
  --host 127.0.0.1 `
  --port 10000
```

Set `api_base_url` to any of these equivalent forms: `http://127.0.0.1:10000`, the same URL with `/`, `/v1`, `/v1/`, or `/v1/chat/completions`. The node normalizes them without duplicated path segments. `model` is a plain STRING: enter a model ID to avoid discovery, or leave it empty to select the first ID returned by `/v1/models` during execution.

Choose Standard, Cinematic Drama, Action, or Character Consistency. For example, a rough prompt such as “a pilot reaches a storm-lit landing pad and recognizes her brother” becomes a chronological shot description sized to the requested duration and resolution.

Connect up to nine reference IMAGE inputs. Every batch frame is JPEG-encoded after RGB/RGBA validation, white alpha compositing, clamping, aspect-preserving resize, and is sent after its own `[Picture N]` text separator. Picture aliases in text are normalized to `<Picture N>`.

Leave `api_key` blank for local llama.cpp. When non-empty, it is sent only as a Bearer header. Keys, Authorization headers, base64 images, and complete prompts are never logged. Use ComfyUI secrets/environment controls where possible and never publish workflows containing credentials.

With `disable_reasoning=true`, the first request includes common reasoning-disable extensions. An HTTP 400 triggers exactly one retry without those extensions. `Return Original` produces a safe fallback status; `Stop Workflow` raises an error.

## Scaling and RTX

Enter source dimensions, target megapixels, aspect and divisor in Resolution Scale Calculator. Outputs are aligned width, height, geometric scale factor and actual megapixels.

RTX Upscaler & Refiner returns RGB. If all effects are off it safely passes RGB through. VSR and High Bitrate use the installed `nvvfx.VideoSuperRes` DLPack API and process frames sequentially. Any actual RTX operation checks CUDA and the optional binding at execution. If the installed binding provides only VideoSuperRes, selecting Denoise or Deblur raises an explicit error rather than silently skipping the requested pass.

## Video combine and Last Frame

Connect an IMAGE batch, select frame rate/codec/container/quality, and queue the output node. `pingpong` appends reversed interior frames. Optional AUDIO is written to a temporary WAV and muxed, with cleanup in `finally`. Metadata is bounded. FFmpeg runs with an argument array, never `shell=True`, has a timeout, checks the return code, and limits surfaced stderr.

Filename input is reduced to a safe basename to prevent traversal; spaces and Unicode are supported. First/last exports are PNG files beside the video.

**To connect the `frames` output to JR MiniMax H3 Last Frame, enable `pass_frames`.** When false, the node deliberately returns an empty IMAGE batch. A saved last-frame PNG is a disk artifact and is not the graph's IMAGE output. See `examples/WORKFLOW_WIRING.md`.

## Troubleshooting

- **Connection refused / timeout:** start the OpenAI-compatible service and check its port. Local model discovery happens only when the node runs.
- **HTTP 401:** check the API key; it is not included in the displayed error.
- **No models:** enter the exact model ID or ensure `/v1/models` returns a non-empty `data` list.
- **FFmpeg not found:** install FFmpeg and restart ComfyUI so its PATH is refreshed.
- **WebM error:** select VP9 with WebM, or use Auto/H.264/MP4.
- **Last Frame says empty batch:** enable `pass_frames` on Enhanced Video Combine.
- **RTX error:** verify CUDA, the GPU/driver, NVIDIA SDK, and the SDK-specific `nvvfx` binding.

## Known limitations and licensing

- RTX SDK bindings are not standardized. VSR is implemented against `nvvfx.VideoSuperRes`; Denoise/Deblur require a binding that exposes those effects and are unavailable in the locally detected VideoSuperRes-only binding.
- Video Auto currently selects the broadly compatible H.264/MP4 path, rather than probing every advertised hardware encoder.
- FFmpeg encoding timeout is 300 seconds per node execution.
- The task's DaSiWa license description did not match the cloned repository. No GPL source was incorporated. See `DEVELOPMENT_NOTES.md`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` for exact commits and provenance.

The JR source is licensed under Apache-2.0. FFmpeg, NVIDIA SDKs, ComfyUI, and reference repositories retain their own licenses.
