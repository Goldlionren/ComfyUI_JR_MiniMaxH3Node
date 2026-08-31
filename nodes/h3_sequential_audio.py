"""ComfyUI nodes for disk-backed sequential MiniMax H3 audio generation."""

from __future__ import annotations

import gc

from ..utils.h3_sequential_audio import (
    CHUNK_PRESET_LABELS,
    CONTINUITY_MODES,
    DEFAULT_CACHE_PATH,
    SEED_MODES,
    apply_continuation_guide,
    checkpoint_sampled_latent,
    commit_decoded_chunk,
    manifest_fingerprint,
    prepare_audio_chunk,
)


class JR_H3_SequentialAudioChunkDriver:
    CATEGORY = "JR MiniMax H3/Sequential Audio"
    FUNCTION = "prepare"
    RETURN_TYPES = ("LATENT", "JR_H3_AUDIO_CHUNK_CONTEXT", "INT", "AUDIO", "STRING")
    RETURN_NAMES = ("audio_driven_av_latent", "chunk_context", "chunk_seed", "audio_slice", "status")
    DESCRIPTION = (
        "Selects one exact frame-aligned slice from a full AUDIO input, encodes it once for the current H3 "
        "chunk, locks it into the Directed Video Conditioning latent, and advances only after the disk-backed "
        "video output node commits the chunk."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": (
                    "LATENT",
                    {"tooltip": "LATENT from JR MiniMax H3 Directed Video Conditioning."},
                ),
                "audio": (
                    "AUDIO",
                    {"tooltip": "Full continuous source audio. It is decoded/resampled once and sliced globally."},
                ),
                "audio_vae": (
                    "VAE",
                    {"tooltip": "MiniMax H3 audio VAE used to encode the current exact audio slice."},
                ),
                "chunk_preset": (
                    list(CHUNK_PRESET_LABELS),
                    {"default": CHUNK_PRESET_LABELS[0]},
                ),
                "continuity_mode": (
                    list(CONTINUITY_MODES),
                    {"default": "Previous Last Frame"},
                ),
                "seed_mode": (
                    list(SEED_MODES),
                    {"default": "Derived per chunk"},
                ),
                "base_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": False},
                ),
                "cache_path": (
                    "STRING",
                    {
                        "default": DEFAULT_CACHE_PATH,
                        "tooltip": "Relative paths are placed below ComfyUI/output; absolute paths are also supported.",
                    },
                ),
                "job_name": (
                    "STRING",
                    {"default": "audio_sequence", "tooltip": "Safe job folder name; never used as a raw path."},
                ),
                "run_id": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 2_147_483_647,
                        "tooltip": "Increment to start a new job. Existing runs are never deleted or overwritten.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(
        cls,
        av_latent,
        audio,
        audio_vae,
        chunk_preset,
        continuity_mode,
        seed_mode,
        base_seed,
        cache_path,
        job_name,
        run_id,
        unique_id=None,
    ):
        return manifest_fingerprint(cache_path, job_name, int(run_id))

    def prepare(
        self,
        av_latent,
        audio,
        audio_vae,
        chunk_preset=CHUNK_PRESET_LABELS[0],
        continuity_mode="Previous Last Frame",
        seed_mode="Derived per chunk",
        base_seed=0,
        cache_path=DEFAULT_CACHE_PATH,
        job_name="audio_sequence",
        run_id=1,
        unique_id=None,
    ):
        return prepare_audio_chunk(
            av_latent=av_latent,
            audio=audio,
            audio_vae=audio_vae,
            chunk_preset=chunk_preset,
            cache_path=cache_path,
            job_name=job_name,
            run_id=int(run_id),
            continuity_mode=continuity_mode,
            seed_mode=seed_mode,
            base_seed=int(base_seed),
        )


class JR_H3_SequentialContinuationGuide:
    CATEGORY = "JR MiniMax H3/Sequential Audio"
    FUNCTION = "apply"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "JR_H3_AUDIO_CHUNK_CONTEXT", "STRING")
    RETURN_NAMES = ("positive", "latent", "chunk_context", "status")
    DESCRIPTION = (
        "In Previous Last Frame mode, applies the initial image to chunk 1 and the previous committed terminal "
        "frame to later chunks through ComfyUI's native MiniMaxH3AddGuide at local frame 0. Independent MV mode "
        "passes conditioning through unchanged."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "chunk_context": ("JR_H3_AUDIO_CHUNK_CONTEXT",),
                "vae": ("VAE",),
            },
            "optional": {"initial_frame": ("IMAGE",)},
        }

    def apply(self, positive, latent, chunk_context, vae, initial_frame=None):
        guided, output_latent, status = apply_continuation_guide(
            positive=positive,
            latent=latent,
            context=chunk_context,
            vae=vae,
            initial_frame=initial_frame,
        )
        return guided, output_latent, chunk_context, status


class JR_H3_SequentialLatentCheckpoint:
    CATEGORY = "JR MiniMax H3/Sequential Audio"
    FUNCTION = "checkpoint"
    RETURN_TYPES = ("LATENT", "JR_H3_AUDIO_CHUNK_CONTEXT", "STRING")
    RETURN_NAMES = ("latent", "chunk_context", "status")
    DESCRIPTION = (
        "Atomically stores the sampled video/audio tensors as one safetensors checkpoint and returns a CPU-backed "
        "H3 AV latent to VAE Decode. The input context is preserved for the commit node."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampled_latent": ("LATENT",),
                "chunk_context": ("JR_H3_AUDIO_CHUNK_CONTEXT",),
            }
        }

    def checkpoint(self, sampled_latent, chunk_context):
        output, status = checkpoint_sampled_latent(sampled_latent, chunk_context)
        return output, chunk_context, status


class JR_H3_SequentialVideoOutput:
    CATEGORY = "JR MiniMax H3/Sequential Audio"
    FUNCTION = "commit"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("filename", "status")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Encodes and validates one silent H.264 segment, saves its terminal frame, commits the manifest, queues "
        "the next ComfyUI prompt when enabled, then stream-concats all segments and muxes the original continuous "
        "PCM exactly once. This replaces per-chunk Video Combine in the sequential branch."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Decoded frames for the current chunk."}),
                "chunk_context": ("JR_H3_AUDIO_CHUNK_CONTEXT",),
                "quality": ("INT", {"default": 20, "min": 0, "max": 51}),
                "bit_depth": (["8-bit", "10-bit"], {"default": "8-bit"}),
                "audio_bitrate": (
                    ["96k", "128k", "160k", "192k", "256k", "320k"],
                    {"default": "192k"},
                ),
                "filename_prefix": (
                    "STRING",
                    {"default": "video/%date:yyyy-MM-dd%/%date:HHmmss%"},
                ),
                "auto_queue_next": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Requires an active browser. Closing the browser pauses safely after commit.",
                    },
                ),
                "aggressive_memory_cleanup": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Collect Python references and request ComfyUI soft cache cleanup after commit.",
                    },
                ),
            },
            "optional": {
                "server_auto_continue": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Queue the next chunk server-side after this prompt succeeds. Works without a "
                        "browser (API/headless clients); supersedes auto_queue_next when enabled.",
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    def commit(
        self,
        images,
        chunk_context,
        quality=20,
        bit_depth="8-bit",
        audio_bitrate="192k",
        filename_prefix="video/%date:yyyy-MM-dd%/%date:HHmmss%",
        auto_queue_next=True,
        aggressive_memory_cleanup=True,
        server_auto_continue=False,
        unique_id=None,
    ):
        filename, status, has_next = commit_decoded_chunk(
            images=images,
            context=chunk_context,
            quality=int(quality),
            bit_depth=bit_depth,
            audio_bitrate=audio_bitrate,
            filename_prefix=filename_prefix,
        )
        if server_auto_continue and has_next:
            from ..utils.h3_sequential_server_continue import schedule_server_continue

            status = status + "\n" + schedule_server_continue(
                job_id=chunk_context.job_id,
                chunk_index=int(chunk_context.chunk_index),
                total_chunks=int(chunk_context.total_chunks),
            )
        if aggressive_memory_cleanup:
            gc.collect()
            try:
                from comfy.model_management import soft_empty_cache

                soft_empty_cache()
            except (ImportError, RuntimeError):
                pass
        try:
            from server import PromptServer

            server = getattr(PromptServer, "instance", None)
            if server is not None:
                server.send_sync(
                    "jr_h3.sequential_audio.chunk_committed",
                    {
                        "node_id": str(unique_id or ""),
                        "job_id": chunk_context.job_id,
                        "chunk_index": int(chunk_context.chunk_index),
                        "total_chunks": int(chunk_context.total_chunks),
                        "has_next": bool(has_next),
                        "auto_queue_next": bool(auto_queue_next and not server_auto_continue),
                        "filename": filename,
                    },
                    getattr(server, "client_id", None),
                )
        except (ImportError, RuntimeError):
            pass
        return filename, status


__all__ = [
    "JR_H3_SequentialAudioChunkDriver",
    "JR_H3_SequentialContinuationGuide",
    "JR_H3_SequentialLatentCheckpoint",
    "JR_H3_SequentialVideoOutput",
]
