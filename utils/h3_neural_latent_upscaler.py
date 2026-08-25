"""MiniMax H3-specific neural video-latent upscaling.

The implementation is intentionally self-contained. It does not vendor the
reference custom-node source or model weights. Compatible checkpoints are
loaded from ComfyUI's standard latent_upscale_models folder.
"""

from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ERROR_PREFIX = "JR MiniMax H3 Neural Latent Upscaler:"
MODEL_FOLDER = "latent_upscale_models"
VIDEO_CHANNELS = 24
SUPPORTED_DTYPES = {torch.float32, torch.float16, torch.bfloat16}
MEGAPIXEL = 1_000_000
MIN_SCALE = 1.0
MAX_SCALE = 4.0
TEMPORAL_CHUNK = 24


class H3NeuralLatentUpscalerError(ValueError):
    """Raised when a video latent or neural-upscaler checkpoint is invalid."""


@dataclass(frozen=True, slots=True)
class H3SpatialContract:
    vae_compression: int
    latent_alignment_h: int
    latent_alignment_w: int
    pixel_alignment_h: int
    pixel_alignment_w: int


@dataclass(frozen=True, slots=True)
class H3UpscalePlan:
    mode: str
    input_h: int
    input_w: int
    output_h: int
    output_w: int
    input_pixel_h: int
    input_pixel_w: int
    output_pixel_h: int
    output_pixel_w: int
    requested_scale: float | None
    requested_megapixels: float | None
    actual_megapixels: float
    effective_scale: float


def _error(message: str) -> H3NeuralLatentUpscalerError:
    return H3NeuralLatentUpscalerError(f"{ERROR_PREFIX}\n{message}")


@lru_cache(maxsize=1)
def get_h3_spatial_contract() -> H3SpatialContract:
    """Discover the canonical H3 VAE and DiT spatial grids from ComfyUI."""

    try:
        from comfy.ldm.minimax.model import MiniMaxH3Model
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    except ImportError:
        raise RuntimeError(
            f"{ERROR_PREFIX}\nThe installed ComfyUI does not provide the native MiniMax H3 implementation."
        ) from None

    vae_default = inspect.signature(MiniMaxH3VideoVAE.__init__).parameters["space_down"].default
    patch_default = inspect.signature(MiniMaxH3Model.__init__).parameters["patch_size"].default
    if not isinstance(vae_default, Sequence) or not isinstance(patch_default, Sequence):
        raise RuntimeError(f"{ERROR_PREFIX}\nUnable to discover the native H3 spatial contract.")
    compression = math.prod(int(value) for value in vae_default)
    latent_h = int(patch_default[-2])
    latent_w = int(patch_default[-1])
    if compression <= 0 or latent_h <= 0 or latent_w <= 0:
        raise RuntimeError(f"{ERROR_PREFIX}\nNative H3 reported an invalid spatial contract.")
    return H3SpatialContract(
        vae_compression=compression,
        latent_alignment_h=latent_h,
        latent_alignment_w=latent_w,
        pixel_alignment_h=compression * latent_h,
        pixel_alignment_w=compression * latent_w,
    )


def _nearest_multiples(value: float, alignment: int, minimum: int) -> set[int]:
    center = int(round(value / alignment)) * alignment
    return {max(minimum, center + offset * alignment) for offset in range(-5, 6)}


def _choose_aligned_size(
    *,
    ideal_h: float,
    ideal_w: float,
    input_h: int,
    input_w: int,
    alignment_h: int,
    alignment_w: int,
) -> tuple[int, int]:
    target_area = ideal_h * ideal_w
    target_aspect = ideal_w / ideal_h
    candidates_h = _nearest_multiples(ideal_h, alignment_h, input_h)
    candidates_w = _nearest_multiples(ideal_w, alignment_w, input_w)
    candidates: set[tuple[int, int]] = set()
    for height in candidates_h:
        width_from_aspect = height * target_aspect
        for width in _nearest_multiples(width_from_aspect, alignment_w, input_w):
            candidates.add((height, width))
    for width in candidates_w:
        height_from_aspect = width / target_aspect
        for height in _nearest_multiples(height_from_aspect, alignment_h, input_h):
            candidates.add((height, width))
    candidates.update((height, width) for height in candidates_h for width in candidates_w)

    def score(size: tuple[int, int]) -> tuple[float, float, float, int, int]:
        height, width = size
        area_error = abs(math.log((height * width) / target_area))
        aspect_error = abs(math.log((width / height) / target_aspect))
        dimension_error = abs(math.log(height / ideal_h)) + abs(math.log(width / ideal_w))
        return area_error + aspect_error, area_error, dimension_error, height, width

    return min(candidates, key=score)


def plan_h3_latent_upscale(
    *,
    input_h: int,
    input_w: int,
    resize_mode: str,
    scale: float,
    target_megapixels: float,
    contract: H3SpatialContract | None = None,
) -> H3UpscalePlan:
    contract = contract or get_h3_spatial_contract()
    if input_h <= 0 or input_w <= 0:
        raise _error("Input latent H and W must both be greater than zero.")
    if input_h % contract.latent_alignment_h or input_w % contract.latent_alignment_w:
        raise _error(
            "Input H3 latent is not aligned to the native spatial patch grid.\n"
            f"required latent alignment: H×{contract.latent_alignment_h}, W×{contract.latent_alignment_w}\n"
            f"received latent size: {input_w}x{input_h}"
        )

    input_pixel_h = input_h * contract.vae_compression
    input_pixel_w = input_w * contract.vae_compression
    requested_scale: float | None = None
    requested_mp: float | None = None
    if resize_mode == "scale":
        requested_scale = float(scale)
        if not math.isfinite(requested_scale) or not MIN_SCALE <= requested_scale <= MAX_SCALE:
            raise _error(f"scale must be finite and between {MIN_SCALE:.1f} and {MAX_SCALE:.1f}.")
        ideal_h = input_h * requested_scale
        ideal_w = input_w * requested_scale
    elif resize_mode == "megapixels":
        requested_mp = float(target_megapixels)
        if not math.isfinite(requested_mp) or not 0.01 <= requested_mp <= 64.0:
            raise _error("target_megapixels must be finite and between 0.01 and 64.0.")
        target_pixels = requested_mp * MEGAPIXEL
        input_pixels = input_pixel_h * input_pixel_w
        requested_scale = math.sqrt(target_pixels / input_pixels)
        if requested_scale < MIN_SCALE:
            raise _error(
                "target_megapixels would downscale the H3 latent, but this neural model only supports upscaling."
            )
        if requested_scale > MAX_SCALE:
            raise _error(
                f"target_megapixels requires approximately {requested_scale:.3f}x linear scale; "
                f"the compatible neural checkpoint supports at most {MAX_SCALE:.1f}x."
            )
        ideal_h = input_h * requested_scale
        ideal_w = input_w * requested_scale
    else:
        raise _error("resize_mode must be either 'scale' or 'megapixels'.")

    output_h, output_w = _choose_aligned_size(
        ideal_h=ideal_h,
        ideal_w=ideal_w,
        input_h=input_h,
        input_w=input_w,
        alignment_h=contract.latent_alignment_h,
        alignment_w=contract.latent_alignment_w,
    )
    output_pixel_h = output_h * contract.vae_compression
    output_pixel_w = output_w * contract.vae_compression
    effective_scale = math.sqrt((output_h * output_w) / (input_h * input_w))
    if effective_scale > MAX_SCALE + 1e-6:
        raise _error(f"Aligned output requires {effective_scale:.3f}x, above the supported {MAX_SCALE:.1f}x maximum.")
    return H3UpscalePlan(
        mode=resize_mode,
        input_h=input_h,
        input_w=input_w,
        output_h=output_h,
        output_w=output_w,
        input_pixel_h=input_pixel_h,
        input_pixel_w=input_pixel_w,
        output_pixel_h=output_pixel_h,
        output_pixel_w=output_pixel_w,
        requested_scale=requested_scale if resize_mode == "scale" else None,
        requested_megapixels=requested_mp,
        actual_megapixels=(output_pixel_h * output_pixel_w) / MEGAPIXEL,
        effective_scale=effective_scale,
    )


class _ScaleResidual3D(nn.Module):
    def __init__(self, channels: int, embedding_channels: int):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embedding_channels, channels * 2))
        self.out_norm = nn.GroupNorm(32, channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Identity()

    def forward(self, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        residual = self.in_layers(features)
        scale, shift = self.emb_layers(embedding).chunk(2, dim=1)
        residual = self.out_norm(residual) * (1 + scale[:, :, None, None, None])
        residual = residual + shift[:, :, None, None, None]
        return self.skip(features) + self.out_layers(residual)


class _TemporalResidual3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.pwconv(self.dwconv(F.silu(self.norm(features))))
        return features + residual


BlockSpec = tuple[str, int]


class _H3CheckpointNetwork(nn.Module):
    """3D checkpoint-compatible network assembled from state-dict structure."""

    def __init__(
        self,
        *,
        in_channels: int,
        channels: int,
        embedding_channels: int,
        input_layout: Sequence[BlockSpec],
        output_layout: Sequence[BlockSpec],
    ):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, kernel_size=3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, embedding_channels),
            nn.SiLU(),
            nn.Linear(embedding_channels, embedding_channels),
        )
        self.in_blocks = nn.ModuleList(self._make_blocks(channels, embedding_channels, input_layout))
        self.out_blocks = nn.ModuleList(self._make_blocks(channels, embedding_channels, output_layout))
        self.norm_out = nn.GroupNorm(32, channels)
        self.conv_out = nn.Conv3d(channels, in_channels, kernel_size=3, padding=1)

    @staticmethod
    def _make_blocks(channels: int, embedding_channels: int, layout: Sequence[BlockSpec]) -> list[nn.Module]:
        blocks: list[nn.Module] = []
        for kind, kernel in layout:
            if kind == "residual":
                blocks.append(_ScaleResidual3D(channels, embedding_channels))
            elif kind == "temporal":
                blocks.append(_TemporalResidual3D(channels, kernel))
            else:
                raise _error(f"Unsupported neural block type: {kind}.")
        return blocks

    @staticmethod
    def _run_blocks(blocks: nn.ModuleList, features: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            if isinstance(block, _ScaleResidual3D):
                features = block(features, embedding)
            else:
                features = block(features)
        return features

    def forward(self, latent: torch.Tensor, scale: float, target_h: int, target_w: int) -> torch.Tensor:
        embedding = self.embed(
            torch.full((latent.shape[0], 1), float(scale) - 1.0, dtype=latent.dtype, device=latent.device)
        )
        features = self._run_blocks(self.in_blocks, self.conv_in(latent), embedding)
        features = F.interpolate(
            features,
            size=(latent.shape[2], target_h, target_w),
            mode="trilinear",
            align_corners=False,
        )
        features = self._run_blocks(self.out_blocks, features, embedding)
        return self.conv_out(F.silu(self.norm_out(features)))


def _extract_state_dict(raw: Any) -> dict[str, torch.Tensor]:
    if isinstance(raw, Mapping) and isinstance(raw.get("model"), Mapping):
        raw = raw["model"]
    if not isinstance(raw, Mapping) or not raw:
        raise _error("Checkpoint does not contain a non-empty tensor state dictionary.")
    state = {str(key): value for key, value in raw.items() if isinstance(value, torch.Tensor)}
    if len(state) != len(raw):
        raise _error("Checkpoint state dictionary contains non-tensor values.")
    if any(key.startswith("upscaler.") for key in state):
        state = {key.removeprefix("upscaler."): value for key, value in state.items() if key.startswith("upscaler.")}
    return state


def _detect_block_layout(state: Mapping[str, torch.Tensor], prefix: str) -> tuple[BlockSpec, ...]:
    indexes = sorted(
        {
            int(match.group(1))
            for key in state
            if (match := re.match(rf"{re.escape(prefix)}\.(\d+)\.", key)) is not None
        }
    )
    if not indexes or indexes != list(range(indexes[-1] + 1)):
        raise _error(f"Checkpoint has an invalid {prefix} block sequence.")
    layout: list[BlockSpec] = []
    for index in indexes:
        stem = f"{prefix}.{index}."
        keys = [key for key in state if key.startswith(stem)]
        if any(".q." in key or ".k." in key or ".v." in key for key in keys):
            raise _error("Attention-enabled H3 upscaler checkpoints are not supported by this backend.")
        if any(key.startswith(f"{stem}in_layers.") for key in keys):
            layout.append(("residual", 0))
            continue
        temporal_key = f"{stem}dwconv.weight"
        if temporal_key in state:
            weight = state[temporal_key]
            if weight.ndim != 5 or weight.shape[2] % 2 != 1:
                raise _error(f"Checkpoint temporal block {prefix}.{index} has an invalid kernel.")
            layout.append(("temporal", int(weight.shape[2])))
            continue
        raise _error(f"Checkpoint contains an unrecognized block at {prefix}.{index}.")
    return tuple(layout)


def build_network_from_state_dict(state: Mapping[str, torch.Tensor]) -> _H3CheckpointNetwork:
    state = _extract_state_dict(state)
    required = ("conv_in.weight", "conv_out.weight", "embed.0.weight", "embed.2.weight")
    missing = [key for key in required if key not in state]
    if missing:
        raise _error(f"Checkpoint is not a compatible H3 neural latent upscaler; missing: {', '.join(missing)}.")
    conv_in = state["conv_in.weight"]
    conv_out = state["conv_out.weight"]
    if conv_in.ndim != 5 or conv_out.ndim != 5:
        raise _error("Checkpoint input/output convolutions must be 3D.")
    in_channels = int(conv_in.shape[1])
    channels = int(conv_in.shape[0])
    if in_channels != VIDEO_CHANNELS or int(conv_out.shape[0]) != VIDEO_CHANNELS:
        raise _error(
            f"Checkpoint must target {VIDEO_CHANNELS}-channel MiniMax H3 latents; "
            f"received input/output channels {in_channels}/{int(conv_out.shape[0])}."
        )
    if channels % 32:
        raise _error("Checkpoint feature channels must be divisible by 32 for its GroupNorm layers.")
    embedding_channels = int(state["embed.0.weight"].shape[0])
    input_layout = _detect_block_layout(state, "in_blocks")
    output_layout = _detect_block_layout(state, "out_blocks")
    with torch.device("meta"):
        model = _H3CheckpointNetwork(
            in_channels=in_channels,
            channels=channels,
            embedding_channels=embedding_channels,
            input_layout=input_layout,
            output_layout=output_layout,
        )
    try:
        model.load_state_dict(dict(state), strict=True, assign=True)
    except RuntimeError as error:
        message = str(error).splitlines()[0]
        raise _error(f"Checkpoint structure is incompatible with the JR 3D backend: {message}") from None
    return model.eval().requires_grad_(False)


@dataclass(slots=True)
class _CachedModel:
    path: str
    patcher: Any
    model: _H3CheckpointNetwork
    dtype: torch.dtype
    temporal_context: int


_MODEL_CACHE: dict[str, _CachedModel] = {}
_MODEL_CACHE_LOCK = RLock()


def _candidate_checkpoint_names() -> list[str]:
    try:
        import folder_paths
    except ImportError:
        raise RuntimeError(f"{ERROR_PREFIX}\nComfyUI folder_paths is unavailable.") from None
    try:
        names = folder_paths.get_filename_list(MODEL_FOLDER)
    except KeyError:
        path = Path(folder_paths.models_dir) / MODEL_FOLDER
        raise _error(f"ComfyUI has no '{MODEL_FOLDER}' model folder. Create it at: {path}") from None
    return [
        name
        for name in names
        if Path(name).suffix.lower() in {".safetensors", ".pth", ".pt"}
        and "h3" in name.lower()
        and "upscal" in name.lower()
    ]


def _select_checkpoint(input_dtype: torch.dtype) -> tuple[str, str]:
    try:
        import folder_paths
    except ImportError:
        raise RuntimeError(f"{ERROR_PREFIX}\nComfyUI folder_paths is unavailable.") from None
    candidates = _candidate_checkpoint_names()
    model_roots = folder_paths.get_folder_paths(MODEL_FOLDER)
    expected_path = str(Path(model_roots[0]) if model_roots else Path(folder_paths.models_dir) / MODEL_FOLDER)
    if not candidates:
        raise _error(
            "No compatible MiniMax H3 neural latent-upscaler checkpoint was found.\n"
            f"Place a user-supplied H3 checkpoint in: {expected_path}\n"
            "Automatic download and interpolation fallback are intentionally disabled."
        )
    dtype_tokens = {
        torch.bfloat16: ("bf16", "bfloat16"),
        torch.float16: ("fp16", "float16"),
        torch.float32: ("fp32", "float32"),
    }[input_dtype]

    def rank(name: str) -> tuple[int, int, str]:
        lowered = name.lower()
        dtype_rank = 0 if any(token in lowered for token in dtype_tokens) else 1
        safe_rank = 0 if lowered.endswith(".safetensors") else 1
        return dtype_rank, safe_rank, lowered

    selected = min(candidates, key=rank)
    path = folder_paths.get_full_path_or_raise(MODEL_FOLDER, selected)
    return selected, path


def _model_dtype(model: nn.Module) -> torch.dtype:
    dtype = next(model.parameters()).dtype
    if dtype not in SUPPORTED_DTYPES:
        raise _error(f"Checkpoint dtype {dtype} is unsupported; use fp32, fp16 or bf16 weights.")
    return dtype


def _load_cached_model(path: str) -> _CachedModel:
    resolved = str(Path(path).resolve(strict=True))
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(resolved)
        if cached is not None:
            return cached
        try:
            import comfy.model_management as model_management
            import comfy.model_patcher
            import comfy.utils
        except ImportError:
            raise RuntimeError(f"{ERROR_PREFIX}\nRequired ComfyUI model-management modules are unavailable.") from None
        raw = comfy.utils.load_torch_file(resolved, safe_load=True)
        model = build_network_from_state_dict(raw)
        dtype = _model_dtype(model)
        model_management.archive_model_dtypes(model)
        patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=model_management.get_torch_device(),
            offload_device=model_management.unet_offload_device(),
        )
        kernels = [block.dwconv.kernel_size[0] for block in model.modules() if isinstance(block, _TemporalResidual3D)]
        cached = _CachedModel(
            path=resolved,
            patcher=patcher,
            model=model,
            dtype=dtype,
            temporal_context=max(kernels, default=1),
        )
        _MODEL_CACHE[resolved] = cached
        return cached


def _normalization_tensors(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD
    except ImportError:
        raise RuntimeError(f"{ERROR_PREFIX}\nNative H3 latent normalization constants are unavailable.") from None
    mean = torch.as_tensor(LATENTS_MEAN, device=device, dtype=dtype).view(1, VIDEO_CHANNELS, 1, 1, 1)
    std = torch.as_tensor(LATENTS_STD, device=device, dtype=dtype).view(1, VIDEO_CHANNELS, 1, 1, 1)
    return mean, std


def _run_temporally_chunked(
    model: _H3CheckpointNetwork,
    latent: torch.Tensor,
    plan: H3UpscalePlan,
    temporal_context: int,
) -> torch.Tensor:
    total_t = int(latent.shape[2])
    if total_t <= TEMPORAL_CHUNK:
        return model(latent, plan.effective_scale, plan.output_h, plan.output_w)
    context = max(1, temporal_context)
    output = torch.empty(
        (latent.shape[0], latent.shape[1], total_t, plan.output_h, plan.output_w),
        dtype=latent.dtype,
        device=latent.device,
    )
    for start in range(0, total_t, TEMPORAL_CHUNK):
        end = min(total_t, start + TEMPORAL_CHUNK)
        read_start = max(0, start - context)
        read_end = min(total_t, end + context)
        segment = model(latent[:, :, read_start:read_end], plan.effective_scale, plan.output_h, plan.output_w)
        valid = segment[:, :, start - read_start : start - read_start + (end - start)]
        output[:, :, start:end].copy_(valid)
    return output


NeuralRunner = Callable[[torch.Tensor, H3UpscalePlan], torch.Tensor]


def _run_checkpoint_backend(samples: torch.Tensor, plan: H3UpscalePlan) -> tuple[torch.Tensor, str]:
    try:
        import comfy.model_management as model_management
    except ImportError:
        raise RuntimeError(f"{ERROR_PREFIX}\nComfyUI model management is unavailable.") from None
    model_name, path = _select_checkpoint(samples.dtype)
    cached = _load_cached_model(path)
    patcher = cached.patcher
    model_management.load_models_gpu([patcher], force_full_load=True)
    device = patcher.load_device
    try:
        source = samples.to(device=device, dtype=cached.dtype)
        mean, std = _normalization_tensors(device, cached.dtype)
        with torch.inference_mode():
            normalized = (source - mean) / std
            output = _run_temporally_chunked(cached.model, normalized, plan, cached.temporal_context)
            output = output * std + mean
            output = output.to(device=samples.device, dtype=samples.dtype)
        return output, model_name
    finally:
        model_management.unload_model_and_clones(
            patcher,
            unload_additional_models=False,
            all_devices=False,
        )


def _validate_video_latent(video_latent: Any) -> torch.Tensor:
    if not isinstance(video_latent, dict) or "samples" not in video_latent:
        raise _error("video_latent must be a LATENT dictionary containing 'samples'.")
    samples = video_latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise _error(
            "Expected MiniMax H3 video latent [B,24,T,H,W]. "
            "Use JR MiniMax H3 Split AV Latent before this node."
        )
    if samples.ndim != 5 or samples.shape[1] != VIDEO_CHANNELS:
        shape = "[" + ",".join(str(int(value)) for value in samples.shape) + "]"
        raise _error(f"Expected MiniMax H3 video latent [B,24,T,H,W], received {shape}.")
    if any(int(samples.shape[index]) <= 0 for index in (0, 2, 3, 4)):
        raise _error("Video latent B, T, H and W dimensions must all be greater than zero.")
    if samples.layout != torch.strided or samples.device.type == "meta":
        raise _error("Video latent must be a materialized strided tensor.")
    if samples.dtype not in SUPPORTED_DTYPES:
        raise _error(f"Video latent dtype must be fp32, fp16 or bf16, received {samples.dtype}.")
    if not bool(torch.isfinite(samples).all().item()):
        raise _error("Input video latent contains NaN or Inf values.")
    return samples


def _validate_backend_output(output: Any, source: torch.Tensor, plan: H3UpscalePlan) -> torch.Tensor:
    if not isinstance(output, torch.Tensor):
        raise _error("Neural backend did not return a torch.Tensor.")
    expected = (source.shape[0], VIDEO_CHANNELS, source.shape[2], plan.output_h, plan.output_w)
    if tuple(output.shape) != expected:
        raise _error(f"Neural backend returned shape {tuple(output.shape)}; expected {expected}.")
    output = output.to(device=source.device, dtype=source.dtype)
    if not bool(torch.isfinite(output).all().item()):
        raise _error("Neural backend produced NaN or Inf values.")
    return output


def _format_status(samples: torch.Tensor, plan: H3UpscalePlan, model_name: str) -> str:
    requested = (
        f"scale={plan.requested_scale:.3f}x"
        if plan.mode == "scale"
        else f"target={plan.requested_megapixels:.3f} MP"
    )
    return "\n".join(
        (
            "Success",
            f"model: {model_name}",
            f"mode: {plan.mode} ({requested})",
            f"input latent: [{samples.shape[0]},24,{samples.shape[2]},{plan.input_h},{plan.input_w}]",
            f"input pixels: {plan.input_pixel_w}x{plan.input_pixel_h}",
            f"output latent: [{samples.shape[0]},24,{samples.shape[2]},{plan.output_h},{plan.output_w}]",
            f"output pixels: {plan.output_pixel_w}x{plan.output_pixel_h} ({plan.actual_megapixels:.3f} MP)",
            f"effective scale: {plan.effective_scale:.3f}x",
            f"temporal: {samples.shape[2]} -> {samples.shape[2]}",
            f"dtype/device: {samples.dtype} / {samples.device}",
        )
    )


def upscale_h3_video_latent(
    video_latent: Any,
    resize_mode: str,
    scale: float,
    target_megapixels: float,
    *,
    neural_runner: NeuralRunner | None = None,
    contract: H3SpatialContract | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate, plan and neurally upscale one plain H3 video LATENT."""

    samples = _validate_video_latent(video_latent)
    plan = plan_h3_latent_upscale(
        input_h=int(samples.shape[3]),
        input_w=int(samples.shape[4]),
        resize_mode=resize_mode,
        scale=scale,
        target_megapixels=target_megapixels,
        contract=contract,
    )
    result = dict(video_latent)
    if plan.output_h == plan.input_h and plan.output_w == plan.input_w:
        result["samples"] = samples
        return result, _format_status(samples, plan, "identity (checkpoint not loaded)")
    if neural_runner is None:
        output, model_name = _run_checkpoint_backend(samples, plan)
    else:
        output = neural_runner(samples, plan)
        model_name = "test neural runner"
    result["samples"] = _validate_backend_output(output, samples, plan)
    return result, _format_status(samples, plan, model_name)


__all__ = [
    "ERROR_PREFIX",
    "H3NeuralLatentUpscalerError",
    "H3SpatialContract",
    "H3UpscalePlan",
    "build_network_from_state_dict",
    "get_h3_spatial_contract",
    "plan_h3_latent_upscale",
    "upscale_h3_video_latent",
]
