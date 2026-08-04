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

The optimizer in this project is a clean-room implementation. It does not copy `h3_ollama_prompt_optimizer.py`, its comments, system prompts, large strings, function organization, variable names, or Ollama calls.

## Runtime tools

FFmpeg is invoked as a separate executable and is not distributed with this project. NVIDIA Video Effects SDK and its Python binding are optional, separately installed runtime components and are not distributed here.
