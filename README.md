# ComfyUI JR MiniMax H3 Node

Five focused ComfyUI V1 nodes for preparing prompts, calculating dimensions, optional RTX enhancement, encoding IMAGE batches, and carrying the last video frame into the next MiniMax H3 segment.

## Nodes

- **JR MiniMax H3 Prompt Optimizer (OpenAI Compatible)** calls `/v1/models` and `/v1/chat/completions`, accepts text plus up to nine IMAGE sockets, and always returns optimized prompt, original prompt, and status. Its built-in Chinese H3 director prompt enforces reference-image mapping, timed shot headings, continuity, hard constraints, and a visible final state, with Standard, Cinematic Drama, Action, and Character Consistency profiles.
- **JR MiniMax H3 Resolution Scale Calculator** preserves a selected aspect ratio within a target pixel area and aligns both dimensions to 8, 16, or 32.
- **JR MiniMax H3 RTX Upscaler & Refiner** exposes denoise, deblur, VSR/high-bitrate and sizing controls while loading optional RTX dependencies only on execution.
- **JR MiniMax H3 Enhanced Video Combine** provides an in-node video player, Autoplay and Download controls, first/last-frame save controls, ComfyUI Assets publication, AV1/VP9/H.265/H.264 with GPU-to-software fallback, 8/10-bit output, MP4/WebM/MKV and animated WebP/AVIF, configurable audio, metadata, ping-pong, and optional frame pass-through.
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

RTX Upscaler & Refiner returns RGB. If all effects are off it safely passes RGB through. The NVIDIA binding represents VSR, High Bitrate, Denoise, and Deblur through one `nvvfx.VideoSuperRes` class; the node selects the operation with `QualityLevel` values such as `DENOISE_HIGH`, `DEBLUR_HIGH`, and `HIGHBITRATE_HIGH`. Enabled passes run in Denoise → Deblur → Upscale order, reuse their effects across the batch, clone every DLPack result, and release each SDK context after execution.

## Video combine and Last Frame

Connect an IMAGE batch, select frame rate/codec/container/bit depth/quality, and queue the output node. The completed video appears inside the node with native playback controls, resolution, duration, FPS, Autoplay and Download. Hovering enables preview audio; leaving mutes it. AV1, HEVC, 10-bit, MKV, and other browser-incompatible results use a streamed H.264 compatibility preview without changing or duplicating the saved source file.

`codec=Auto` runtime-tests AV1/WebM, VP9/WebM, then H.264/MP4. Explicit codecs prefer NVIDIA NVENC, Intel QSV, AMD AMF, VAAPI, then software encoding. A final H.264/MP4 fallback is always attempted. Explicit codecs can auto-detect 8- versus 10-bit frame quantization; Auto uses browser-safe 8-bit. Animated WebP/AVIF are explicit container choices and omit connected audio.

Optional AUDIO supports Auto/AAC/Opus/MP3 and 64k–320k bitrate selection; `crop_to_audio` limits output to the audio duration. Temporary raw audio, metadata and failed partial files are cleaned up. Frames are streamed to FFmpeg in bounded chunks, encoding progress is reported to ComfyUI, subprocess execution has a timeout, and surfaced stderr is bounded.

Filename prefixes support safe subfolders, Unicode, `%date%`, `%date:yyyy-MM-dd%`, and `%date:hhmmss%`. `save_output=true` writes to ComfyUI output; false writes to temp. The encoded file and selected native-resolution first/last PNGs are published to ComfyUI Assets. Existing workflows from the earlier JR node schema are migrated by the frontend when loaded.

See [`docs/ENHANCED_VIDEO_COMBINE.md`](docs/ENHANCED_VIDEO_COMBINE.md) for the complete codec, container, preview, audio, timeout, and fallback behavior.

**To connect the `frames` output to JR MiniMax H3 Last Frame, enable `pass_frames`.** When false, the node deliberately returns an empty IMAGE batch. A saved last-frame PNG is a disk artifact and is not the graph's IMAGE output. See `examples/WORKFLOW_WIRING.md`.

## Troubleshooting

- **Connection refused / timeout:** start the OpenAI-compatible service and check its port. Local model discovery happens only when the node runs.
- **HTTP 401:** check the API key; it is not included in the displayed error.
- **No models:** enter the exact model ID or ensure `/v1/models` returns a non-empty `data` list.
- **FFmpeg not found:** install FFmpeg and restart ComfyUI so its PATH is refreshed.
- **Preview unavailable:** confirm FFmpeg provides `libx264`; unsupported source formats use the `/jr-h3/enhanced-video-preview` compatibility stream.
- **Auto chose a lower-priority codec:** a listed hardware/software encoder failed a real runtime attempt, so the node continued to the next compatible choice.
- **Last Frame says empty batch:** enable `pass_frames` on Enhanced Video Combine.
- **RTX error:** verify CUDA, the GPU/driver, NVIDIA SDK, and the SDK-specific `nvvfx` binding.

## Known limitations and licensing

- RTX SDK bindings are not standardized. The installed binding must expose `VideoSuperRes` plus the requested VSR/Denoise/Deblur/High-Bitrate `QualityLevel` enum values.
- Animated WebP and Animated AVIF cannot mux AUDIO; the node logs that the connected audio was omitted.
- FFmpeg encoding has a 3600-second overall limit, a short no-progress guard for hardware candidates, and a 120-second progress-stall guard; preview stream reads have a 60-second inactivity timeout.
- The task's DaSiWa license description did not match the cloned repository. No GPL source was incorporated. See `DEVELOPMENT_NOTES.md`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` for exact commits and provenance.

The JR source is licensed under Apache-2.0. FFmpeg, NVIDIA SDKs, ComfyUI, and reference repositories retain their own licenses.
