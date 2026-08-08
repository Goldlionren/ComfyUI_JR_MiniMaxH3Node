# Third-party notices

## MiniMax-AI/MiniMax-H3 prompt-writing specification

- Repository: https://github.com/MiniMax-AI/MiniMax-H3
- Audited branch and commit: `main` at `8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea`
- Audited paths: `skills/h3-prompt-writing/SKILL.md`, `skills/h3-prompt-writing/references/base-en.txt`, and `skills/h3-prompt-writing/references/ref-en.txt`
- License linked by the upstream README: [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)

The audited GitHub commit did not contain a `LICENSE` or `NOTICE` file. The linked Community License grants rights only within its defined Applicable Territory and expressly excludes the European Union, United Kingdom, Republic of Korea, and United States. A public GitHub repository cannot enforce that downstream territorial restriction. Consequently, this project does **not** redistribute the official Skill, guide prose, or examples.

The implementation is independently written from public interoperability requirements. It retains only necessary format facts such as field names, label tokens, mode names, ordering, and fixed relationship markers, plus factual upstream URLs, paths, commit and file hashes. [`resources/minimax_h3_spec/UPSTREAM.json`](resources/minimax_h3_spec/UPSTREAM.json) records the audited sources. This notice is attribution and provenance documentation, not a sublicense for MiniMax H3 model weights or documentation.

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
