"""Isolated browser QA harness: synthetic audio only; never starts ComfyUI/GPU.

Run with ComfyUI's Python, --comfy-root and --scratch pointing to a dev directory.
The localhost-only server and temporary media are separate from production.
"""

import argparse
import importlib.util
import shutil
import sys
import tempfile
import types
from pathlib import Path

from aiohttp import web

HTML = """<!doctype html><meta charset="utf-8"><title>JR Cut Audio · isolated QA</title>
<style>body{background:#0b1119;color:#dae6ef;font:14px system-ui;margin:25px auto;width:640px}#mount{height:610px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}</style>
<h2>JR Cut Audio · isolated development preview</h2><p>Synthetic audio only · no ComfyUI workflow is queued</p>
<div id="mount"></div><button id="restore">Restore saved node state</button><button id="execute">Check AUDIO output</button><pre id="state"></pre><pre id="result"></pre>
<script type="module">
import {createCutAudioEditor} from '/editor.mjs';
let audio=localStorage.getItem('jr-test-audio') || 'demo.wav', selection=localStorage.getItem('jr-test-selection') || '';
const show=()=>document.querySelector('#state').textContent=JSON.stringify({audio,selection},null,2);
const editor=createCutAudioEditor({request:fetch,onFile:v=>{audio=v;localStorage.setItem('jr-test-audio',v);show();},onSelection:v=>{selection=v;localStorage.setItem('jr-test-selection',v);show();}});
document.querySelector('#mount').append(editor.panel);editor.restore(audio,selection);show();
document.querySelector('#restore').onclick=()=>editor.restore(audio,selection);
document.querySelector('#execute').onclick=async()=>{const r=await fetch('/test-output',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({audio,selection})});document.querySelector('#result').textContent=await r.text();};
</script>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--scratch", required=True)
    parser.add_argument("--port", type=int, default=8191)
    args = parser.parse_args()
    sys.path.insert(0, args.comfy_root)
    import folder_paths
    import torch

    repo = Path(__file__).resolve().parents[1]
    scratch = Path(tempfile.mkdtemp(prefix="jr-cut-audio-", dir=args.scratch))
    source, temp = scratch / "input", scratch / "temp"
    source.mkdir()
    folder_paths.get_input_directory = lambda: str(source)
    folder_paths.get_temp_directory = lambda: str(temp)
    for name in ["jr_audio_qa", "jr_audio_qa.utils", "jr_audio_qa.server"]:
        package = types.ModuleType(name)
        package.__path__ = []
        sys.modules[name] = package

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    util = load("jr_audio_qa.utils.cut_audio", repo / "utils/cut_audio.py")
    routes = load("jr_audio_qa.server.cut_audio_routes", repo / "server/cut_audio_routes.py")
    rate = 24000
    t = torch.arange(rate * 24) / rate
    envelope = (.2 + .7 * torch.sin(t * .9).square()) * torch.minimum(torch.ones_like(t), t)
    wave = torch.stack([envelope * torch.sin(t * 440 * 2 * torch.pi), envelope * torch.sin(t * 660 * 2 * torch.pi)])
    token = util.write_wav(wave, rate, "c" * 64)
    shutil.copyfile(util.media_path(token), source / "demo.wav")

    async def home(request):
        return web.Response(text=HTML, content_type="text/html")

    async def editor(request):
        return web.Response(body=(repo / "js/cut_audio_editor.mjs").read_bytes(), content_type="text/javascript")

    async def output(request):
        data = await routes._body(request)
        audio, duration, status = await routes._work(util.selection_audio, data.get("audio"), data.get("selection"))
        return web.json_response({"shape": list(audio["waveform"].shape), "sample_rate": audio["sample_rate"], "duration": duration, "status": status})

    app = web.Application()
    app.router.add_get("/", home)
    app.router.add_get("/editor.mjs", editor)
    app.router.add_post("/test-output", output)
    app.router.add_get("/jr-cut-audio/files", routes.audio_files)
    app.router.add_post("/jr-cut-audio/info", routes.audio_info)
    app.router.add_post("/jr-cut-audio/cut", routes.audio_cut)
    app.router.add_get("/jr-cut-audio/media/{token}.wav", routes.audio_media)
    app.router.add_get("/jr-cut-audio/original", routes.audio_original)
    print(f"Isolated scratch: {scratch}", flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
