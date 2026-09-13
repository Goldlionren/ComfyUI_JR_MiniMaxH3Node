"""Opt-in existing-VAE final decode probe. Never downloads or compiles an engine.

Use --native-vae for a baseline, or --trt-node-file plus --decoder-engine for an
already installed trusted H3VAE_TRT source and locally compiled engine. Synthetic
input validates the protocol, NOT picture quality or full-workflow performance.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--native-vae", type=Path)
    group.add_argument("--trt-node-file", type=Path)
    parser.add_argument("--decoder-engine", type=Path)
    parser.add_argument("--latent-t", type=int, default=2)
    parser.add_argument("--latent-h", type=int, default=16)
    parser.add_argument("--latent-w", type=int, default=16)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    for path in (args.native_vae, args.trt_node_file, args.decoder_engine):
        if path is not None and not path.is_file():
            parser.error(f"Existing file required: {path}")
    if args.trt_node_file and (args.decoder_engine is None or args.decoder_engine.suffix != ".engine"):
        parser.error("TRT mode requires --decoder-engine pointing to an existing local .engine")
    if args.native_vae and args.decoder_engine:
        parser.error("--decoder-engine applies only to TRT mode")
    if min(args.latent_t, args.latent_h, args.latent_w) < 1:
        parser.error("Latent dimensions must be positive")
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project.parent))
    sys.path.insert(0, str(args.comfy_root))
    import comfy.sd
    import comfy.utils
    import torch
    # Utility has no package-relative imports; avoid starting JR's HTTP routes
    # just to inspect/decode a VAE in this standalone process.
    spec = importlib.util.spec_from_file_location("jr_h3_vae_compat_probe", project / "utils/h3_vae_compat.py")
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    if args.native_vae:
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.native_vae), safe_load=True))
    else:
        spec = importlib.util.spec_from_file_location("jr_external_h3_vae_probe", args.trt_node_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        runner = module.AutoEngineRunner(str(args.decoder_engine.resolve()))
        vae = module.ComfyTRTVAE(module.MiniMaxH3TRTVAE(decoder_runner=runner, encoder_runner=None))
    report = compat.inspect_h3_video_vae(vae)
    report["synthetic_input_only"] = True
    try:
        if not args.inspect_only:
            z = torch.zeros(1, 24, args.latent_t, args.latent_h, args.latent_w)
            times = []
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            for _ in range(2):
                started = time.perf_counter()
                pixels = compat.decode_h3_video_checked(vae, z)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
            report.update(decode_result="PASS", shape=list(pixels.shape),
                          cold_seconds=times[0], warm_seconds=times[1],
                          engine_tested=report["known_trt_wrapper"],
                          peak_torch_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
        print(json.dumps(report, indent=2))
    finally:
        # Explicit probe owns the external runner; do not leave its context resident.
        stage = getattr(vae, "first_stage_model", None)
        offload = getattr(stage, "offload_runners_to_ram", None)
        if callable(offload):
            offload()


if __name__ == "__main__":
    main()
