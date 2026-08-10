# Third-party notices

## Director Desk clean-room product research

- ComfyTV: https://github.com/jtydhr88/ComfyTV — audited product concepts at commit `6cb67572c82d7f3e4e51ee0005f8308b8c15de63`; GitHub identifies MIT.
- ComfyUI-qwenmultiangle: https://github.com/jtydhr88/ComfyUI-qwenmultiangle — audited at commit `93efd354a002f9c6add7e948663cf459528242da`; README/package metadata identifies MIT, but no LICENSE file was present in the audited tree.
- ComfyUI-mesh2motion: https://github.com/jtydhr88/ComfyUI-mesh2motion — audited product documentation at commit `11fe6b7aaa5eac60afa3d726389cd9dd870ed1f6`; GitHub identifies MIT and the project credits additional upstream work/dependencies.

These projects were used only to study lifecycle, state persistence, inspector and asset-editor product patterns. This repository does not copy their source, styles, Vue/Three code, iframe bridge, renderer, timeline libraries, assets, project database, runner, queue or stage system. Director Desk is an independent DOM/JavaScript and Python implementation. DaSiWa remains covered by its separate GPL-3.0 notice below and no DaSiWa code was copied.

## MiniMax-H3 Turbo LoRA and ComfyUI nodes

- Weights/model repository: https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora
- ComfyUI node repository: https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo
- Author/project owner: Larryvrh
- License displayed by both audited project pages on 2026-08-09: Apache-2.0
- Upstream source or weights vendored by this repository: **NO**

The `MiniMax H3 Turbo LoRA` shown in recommended workflows is an external component from Larryvrh, not an original JR node or model. This JR repository contributes its own workflow integration, compatibility/orchestration layer, Unified Acceleration node, testing, RTX post-processing, and other H3 workflow utilities.

`Goldlionren/ComfyUI_JR_MiniMaxH3Node` is this project. The separate `Goldlionren/ComfyUI-MiniMax-H3-Turbo` repository is a fork of `Larryvrh/ComfyUI-MiniMax-H3-Turbo`; these repositories and their authorship must not be conflated.

## ComfyUI-KJNodes (Unified Acceleration runtime dependency)

- Repository: https://github.com/kijai/ComfyUI-KJNodes
- Reference workflow commit: `60cd6bc1870db94c6eeb05fbe455147a8e91c4e9`
- Installed commit audited 2026-08-09: `60cd6bc1870db94c6eeb05fbe455147a8e91c4e9`
- Current upstream `main` audited 2026-08-09: `60cd6bc1870db94c6eeb05fbe455147a8e91c4e9`
- License observed: GNU General Public License v3.0 (`LICENSE`; `pyproject.toml` points to that file)
- Runtime behavior used: `PathchSageAttentionKJ`, `MiniMaxLowVRAMAttention`, and `MiniMaxChunkFeedForward`.
- Upstream source vendored: **NO**

The JR compatibility layer resolves the installed node classes from ComfyUI's runtime registry and invokes their public node interfaces. No KJNodes source, assets, CUDA code, comments, or object-patch implementation is redistributed in this repository.

## ComfyUI-SolAttn_triton (Unified Acceleration runtime dependency)

- Repository: https://github.com/kijai/ComfyUI-SolAttn_triton
- Reference workflow commit: `0e334dc981cfe3b0ed926ee13ad43f64914b7f5b`
- Installed commit audited 2026-08-09: `842c4eaa7d91dbaef3fee3ccdbf36a39521e82fc`
- Current upstream `main` audited 2026-08-09: `842c4eaa7d91dbaef3fee3ccdbf36a39521e82fc`
- License audit result: **No explicit license confirmed**. Neither audited commit contained a LICENSE/COPYING/NOTICE, packaging license metadata, or source license header.
- Runtime behavior used: `SolAttnPatch`.
- Upstream source vendored: **NO**

The installed/current commit changes Triton/kernel internals relative to the workflow reference while retaining the same node API. Because no explicit license grant was confirmed, this project does not copy or redistribute any Sol-Attn Python/Triton source. Users install and review that external dependency separately.

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
