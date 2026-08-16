"""Selected-only REF tensor reads for the JR H3 Hybrid Loader."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

from .h3_hybrid_plan import HybridPlan, validate_plan_file_identity
from .h3_hybrid_tensor_family import H3HybridCompatibilityError

_TORCH_DTYPE_NAMES = {
    "BOOL": "torch.bool",
    "I8": "torch.int8",
    "U8": "torch.uint8",
    "I16": "torch.int16",
    "U16": "torch.uint16",
    "I32": "torch.int32",
    "U32": "torch.uint32",
    "I64": "torch.int64",
    "U64": "torch.uint64",
    "F16": "torch.float16",
    "BF16": "torch.bfloat16",
    "F32": "torch.float32",
    "F64": "torch.float64",
    "F8_E4M3": "torch.float8_e4m3fn",
    "F8_E5M2": "torch.float8_e5m2",
    "C64": "torch.complex64",
}


def _default_opener(path: str) -> AbstractContextManager:
    from safetensors import safe_open

    return safe_open(path, framework="pt", device="cpu")


def read_selected_ref_tensors(
    plan: HybridPlan,
    *,
    opener: Callable[[str], AbstractContextManager] | None = None,
) -> dict[str, Any]:
    """Read exactly the planned REF keys and return independent CPU tensors."""

    open_ref = opener or _default_opener
    owned: dict[str, Any] = {}
    expected = {tensor.key: tensor for tensor in plan.selected_tensors}
    validate_plan_file_identity(plan, check_fl=False, check_ref=True)
    try:
        with open_ref(str(plan.ref_path)) as handle:
            available = set(handle.keys())
            missing = [key for key in plan.selected_keys if key not in available]
            if missing:
                raise H3HybridCompatibilityError(f"Missing selected REF tensor during read: {missing[0]}.")
            for key in plan.selected_keys:
                tensor = handle.get_tensor(key)
                spec = expected[key]
                if (
                    tuple(tensor.shape) != spec.shape
                    or tensor.nelement() * tensor.element_size() != spec.nbytes
                    or str(tensor.dtype) != _TORCH_DTYPE_NAMES.get(spec.dtype, str(tensor.dtype))
                ):
                    raise H3HybridCompatibilityError(
                        f"Selected REF tensor changed after header validation: {key}."
                    )
                owned[key] = tensor.detach().to(device="cpu").clone()
                del tensor
    except H3HybridCompatibilityError:
        raise
    except Exception:
        raise H3HybridCompatibilityError(
            f"Failed to read selected REF tensor data from {plan.ref_path.name}."
        ) from None
    return owned
