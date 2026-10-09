"""Interactive local audio loader/cutter; the editor commits a portable selection."""

from ..utils.cut_audio import CutAudioError, parse_selection, resolve_audio, selection_audio, source_digest


class JR_CutAudio:
    CATEGORY = "JR MiniMax H3/Audio"
    FUNCTION = "cut"
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING")
    RETURN_NAMES = ("audio", "duration_seconds", "status")
    DESCRIPTION = (
        "Load/upload audio, inspect the waveform, audition and set start/end seconds. "
        "Press Cut to lock the AUDIO output. Playback volume does not change the output. "
        "Cut does not queue the downstream workflow. Files stay on your ComfyUI server."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("STRING", {"default": "", "tooltip": "Input-relative source audio filename (managed by the editor)."}),
            "selection": ("STRING", {"default": "", "multiline": True, "tooltip": "Locked sample selection created by Cut."}),
        }}

    @classmethod
    def VALIDATE_INPUTS(cls, audio, selection):
        try:
            resolve_audio(audio)
            parse_selection(selection, audio)
        except CutAudioError as error:
            return str(error)
        return True

    @classmethod
    def IS_CHANGED(cls, audio, selection):
        try:
            return source_digest(resolve_audio(audio))
        except (CutAudioError, OSError):
            return float("nan")

    def cut(self, audio, selection):
        return selection_audio(audio, selection)
