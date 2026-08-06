# Third-party notices

## ComfyUI-DaSiWa-Nodes

- Repository: https://github.com/darksidewalker/ComfyUI-DaSiWa-Nodes
- Shallow-clone commit: `a297af20318dfb7d8bdd2295a920172437551036`
- License observed at that commit: GNU General Public License v3.0 (`LICENSE` in the reference checkout)
- Referenced behavior: RTX Upscaler & Refiner, Resolution Scale Calculator, and Enhanced Video Combine.

The development task stated that this repository was Apache-2.0. The actual checkout did not support that statement. To avoid relicensing GPL code incorrectly, this project contains no copied or adapted DaSiWa source, comments, prompts, JavaScript, or assets. The JR nodes that cover similar workflows are independent implementations based on the written requirements. The `.reference` checkout is excluded from Git.

## Comfyui-minimaxh3-FBcache-shendumao

- Repository: https://github.com/signerzwb/Comfyui-minimaxh3-FBcache-shendumao
- Shallow-clone commit: `8d2aa84f7f5fe66398b2c6db192a2cbd3a1926af`
- Referenced behavior: OpenAI-compatible prompt optimization, multiple reference images, profile selection, original/optimized/status outputs, and safe fallback.

The OpenAI-compatible request layer and node implementation in this project are independent and do not bundle the upstream Python file, comments, function organization, assets, or Ollama request code. At the user's request, the H3 system-prompt constraint strategy was aligned with `_build_system_prompt` from the referenced commit, then reorganized and rewritten for this project's OpenAI-compatible workflow.

No top-level license file was present in the shallow checkout at the recorded commit. This notice provides attribution and does not grant rights to the upstream repository; review its current terms before reusing upstream material.

## Diffusion-cache research concepts

The Adaptive Cache design considered publicly documented concepts associated with EasyCache, TeaCache, First Block Cache, and CacheDiT: low-cost change estimation, residual reuse, periodic forced refresh, and partial-Block probing. These names identify research context only. No implementation source, comments, thresholds, function organization, or distinctive code structure from those projects is included.

The JR implementation uses a separately defined dual-stream relative-delta metric, local presets in its own metric scale, and interfaces verified directly in the Apache-2.0 ComfyUI MiniMax H3 implementation. Users should consult the original papers and repositories for their authors, licenses, and model-specific claims before reusing those projects.

## Runtime tools

FFmpeg is invoked as a separate executable and is not distributed with this project. NVIDIA Video Effects SDK and its Python binding are optional, separately installed runtime components and are not distributed here.
