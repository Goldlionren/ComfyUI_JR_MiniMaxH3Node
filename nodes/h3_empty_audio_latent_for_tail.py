"""Empty H3 audio matched to a local tail-context or legacy single-frame timeline."""

from ..utils.h3_av_latent_builder import AUDIO_LATENT_FPS, H3_FPS
from ..utils.h3_tail_frame_latent import tail_frame_count, validate_tail_video_latent


class JR_H3_EmptyAudioLatentForTail:
    CATEGORY = "JR MiniMax H3/Latent"
    FUNCTION = "create"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("audio_latent", "status")
    DESCRIPTION = (
        "Experimental empty audio matched to an H3 tail video or single frame. Uses the native "
        "24 fps / 40 audio ticks/s timeline (7 video tokens -> 22 frames -> 37 audio ticks; "
        "legacy single frame -> 2 ticks), matching the input dtype/device. Connect with the same video "
        "latent to AV Latent Builder, then a normal second-pass sampler. Not encoded silence, "
        "not an original-video audio crop, and not an audio lock."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_latent": ("LATENT", {
            "tooltip": "Connect the same video as AV Builder: tail_context_latent [1,24,5k+2,H,W] or legacy T=1. Not an AV pair or masked latent.",
        })}}

    def create(self, video_latent):
        video = validate_tail_video_latent(video_latent)
        if video_latent.get("noise_mask") is not None:
            raise ValueError("JR H3 Empty Audio Latent for Tail: connect a clean, unmasked video latent.")
        frames = tail_frame_count(video.shape[2])
        audio_t = round(frames * AUDIO_LATENT_FPS / H3_FPS)
        audio = video.new_zeros((1, 32, 2, audio_t))
        kind = "Single-frame refinement" if video.shape[2] == 1 else "Tail-context refinement"
        status = (f"{kind}: {frames} frames @ {H3_FPS} fps -> {audio_t} audio ticks @ {AUDIO_LATENT_FPS} Hz\n"
                  f"audio [1,32,2,{audio_t}], {audio.dtype}, {audio.device}; zero initializer, not encoded silence.\n"
                  "Timeline starts locally at zero; no original audio is retained. Discard the sampled audio output.")
        return {"samples": audio}, status
