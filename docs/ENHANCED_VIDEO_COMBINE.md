# JR MiniMax H3 Enhanced Video Combine

This output node encodes a ComfyUI `IMAGE` batch and displays the completed video inside the node. The preview toolbar contains **Save first frame**, **Save last frame**, **Autoplay**, and **Download**. Saved videos and selected PNG exports are also included in ComfyUI Assets.

## Automatic selection

- `codec=Auto` runtime-tests AV1/WebM, VP9/WebM, then H.264/MP4. H.265 is explicit-only.
- Explicit codecs try compatible containers and prefer NVENC, QSV, AMF, VAAPI, then software encoding.
- If requested combinations fail, H.264/MP4 is the mandatory final fallback.
- `bit_depth=Auto` detects 8/10-bit quantization for explicit codecs. Codec Auto stays at browser-compatible 8-bit.
- Animated WebP and Animated AVIF are explicit container choices. They ignore the codec selection and omit AUDIO.

## Preview and saved assets

H.264/MP4, VP9/WebM, and 8-bit AV1/WebM are offered directly to the browser. Other formats use a temporary streamed H.264 response from `/jr-h3/enhanced-video-preview`; the requested output file is not modified and no sidecar preview is saved. Preview audio unmutes only while the pointer is over the player.

The first/last checkboxes mirror the hidden backend widgets. Enabling them writes full-resolution PNG files beside the encoded video. `save_output=false` uses ComfyUI temp storage; otherwise files are permanent output assets. Download always targets the original encoded file.

## Audio, filenames and frames

Audio codec choices are Auto, AAC, Opus and MP3, with 64k through 320k bitrates. Auto uses Opus for WebM and AAC elsewhere, with compatible fallbacks. `crop_to_audio` stops at the connected audio duration.

Filename prefixes can contain safe subfolders and `%date%` tokens. An audio-bearing filename receives `-audio` before its extension. `pingpong` appends reversed interior frames. `pass_frames=true` returns the encoded frame sequence; false returns an empty IMAGE batch while still saving and previewing the file.
