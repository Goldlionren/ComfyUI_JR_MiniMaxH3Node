"""Return the final ComfyUI IMAGE frame without dropping its batch axis."""

import torch


class JR_H3_LastFrame:
    CATEGORY = "JR MiniMax H3/Utility"
    FUNCTION = "extract"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"frames": ("IMAGE",)}}

    def extract(self, frames):
        if not isinstance(frames, torch.Tensor):
            raise TypeError("JR MiniMax H3 Last Frame expects a torch.Tensor IMAGE batch.")
        if frames.ndim != 4:
            raise ValueError("JR MiniMax H3 Last Frame expects IMAGE shape [B,H,W,C].")
        if frames.shape[0] == 0:
            raise ValueError(
                "JR MiniMax H3 Last Frame received an empty IMAGE batch. "
                "Enable pass_frames on JR MiniMax H3 Enhanced Video Combine."
            )
        if frames.shape[1] < 1 or frames.shape[2] < 1 or frames.shape[3] not in (3, 4):
            raise ValueError("JR MiniMax H3 Last Frame requires non-empty RGB or RGBA frames.")
        return (frames[-1:].contiguous(),)
