# Frozen H3-to-LTX inference adapter

These five Python files are unchanged from NVlabs/Sana revision
`801f04055badca8cc1f3644e973188068f14ad98`, path
`models/minimax_h3/Sol-H3-RTX5090/runtime/stage2_ops/h3_ltx_adapter/`.
Source: https://github.com/NVlabs/Sana/tree/801f04055badca8cc1f3644e973188068f14ad98/models/minimax_h3/Sol-H3-RTX5090/runtime/stage2_ops/h3_ltx_adapter

The JR wrapper supplies native ComfyUI normalized H3 latents, CUDA memory
coordination and padded timeline metadata. Architecture, normalization and
geometry in these frozen files are not altered. Do not silently substitute
different temporal packing or temporally chunk the full-volume GroupNorm.

Weights are separate: Efficient-Large-Model/H3-to-LTX-Latent-Adapter at
`cfcd8a7cc36c135142287584728f7fe362df0867`; SHA-256
`170199a390c40ac97f5895bc9c8cc29817e74fb9193c858a85d8c0f1f30724ac`.
No model weights are included in this plugin. The model card assigns no new
weight license; H3 and LTX retain their own terms. Local integration is not
a change to the upstream components' terms.

The pinned repository README explicitly releases its code under Apache-2.0:
https://github.com/NVlabs/Sana/blob/801f04055badca8cc1f3644e973188068f14ad98/README.md#-license
The Apache-2.0 license text is reproduced in LICENSE.txt. The five Python
modules remain byte-for-byte unchanged; the JR wrapper and provenance are
separate additions. This code license does not grant rights to model weights.
