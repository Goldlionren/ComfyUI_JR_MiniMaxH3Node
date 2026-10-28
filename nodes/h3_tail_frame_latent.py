"""Endpoint conversion for Hermes IMAGE / LATENT continuation tests."""

from ..utils.h3_tail_frame_latent import extract_h3_tail_frame_latent


class JR_H3_TailFrameLatent:
    CATEGORY = "JR MiniMax H3/Latent"
    FUNCTION = "extract"
    RETURN_TYPES = ("LATENT", "IMAGE", "STRING", "LATENT")
    RETURN_NAMES = ("tail_latent", "tail_image", "status", "tail_context_latent")
    DESCRIPTION = (
        "Experimental endpoint conversion: decodes the true final video frame and encodes it once "
        "as a reusable H3 image-keyframe latent. Not lossless or zero-encode. Keeps source resolution. "
        "Appended tail_context_latent preserves the original decoder window for video upscale/refinement, "
        "without re-encoding that output. Decode it then select the last image; not a Director keyframe. "
        "Use the finished final-pass video latent; never an intermediate noisy sampling state."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_latent": ("LATENT", {"tooltip": "Final sampling output -> Split AV Latent -> video_latent."}),
                "vae": ("VAE", {"tooltip": "Matching H3 video VAE with encoder. Native VAE recommended for the first test."}),
                "decode_mode": (["tail_context", "full_video"], {
                    "default": "tail_context",
                    "tooltip": "Tail context uses the reviewed H3 final 7-token window; unknown contracts fall back to full decode.",
                }),
            },
            "optional": {
                "decoded_frames": ("IMAGE", {
                    "tooltip": "Optional matching decoded/trimmed video frames at source resolution. Uses their last frame, skips decode. No MP4/JPEG round trip.",
                }),
            },
        }

    def extract(self, video_latent, vae, decode_mode="tail_context", decoded_frames=None):
        return extract_h3_tail_frame_latent(video_latent, vae, decode_mode, decoded_frames)
