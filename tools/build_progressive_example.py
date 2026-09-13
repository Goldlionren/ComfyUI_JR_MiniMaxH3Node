"""Mechanically derive the experimental T2VA workflow from the shipped reference."""

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def build(write=True):
    workflow = json.loads((PROJECT / "comfytv/workflows/jr_h3_t2va_turbo_comfytv.json").read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    sampler = nodes[125]
    sampler["type"] = "JR_H3_ProgressiveSampler"
    sampler["title"] = "JR Progressive / scale=1 for native baseline"
    sampler["properties"] = {"Node name for S&R": "JR_H3_ProgressiveSampler"}
    sampler["size"] = [440, 320]
    sampler["inputs"] = [{"name": name, "type": kind, "link": link} for name, kind, link in (
        ("model", "MODEL", 259), ("positive", "CONDITIONING", 237), ("noise", "NOISE", 231),
        ("sampler", "SAMPLER", 233), ("sigmas", "SIGMAS", 234), ("latent_image", "LATENT", 235))]
    sampler["outputs"] = [sampler["outputs"][0], {"name": "status", "type": "STRING", "links": None}]
    sampler["outputs"][0]["name"] = "output"
    sampler["widgets_values"] = [3, .5, 1, False]
    destinations = {259: 0, 237: 1, 231: 2, 233: 3, 234: 4, 235: 5}
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] != 126]
    workflow["links"] = [link for link in workflow["links"] if link[0] != 232]
    for link in workflow["links"]:
        if link[0] in destinations:
            link[3], link[4] = 125, destinations[link[0]]
    nodes[123]["widgets_values"] = ["euler"]
    nodes[129]["widgets_values"] = [1, "fixed"]
    nodes[140]["widgets_values"][0] = False
    nodes[140]["title"] = "Acceleration OFF for first A/B test"
    nodes[92]["widgets_values"][0] = "video/JR_H3_Progressive"
    nodes[131]["title"] = "T2VA only: leave first_frame / last_frame disconnected"
    nodes[133]["widgets_values"] = [5.0]
    nodes[133]["title"] = "Duration = 5 seconds (H3 snaps to 124 frames)"
    nodes[131]["widgets_values"][-1] = 124
    # Scheduler and sampler must observe the same effective MODEL/shift.
    for link in workflow["links"]:
        if link[0] == 230:
            link[1] = 140
    nodes[116]["widgets_values"] = [
        "# JR H3 Progressive Sampler — experimental\n\n"
        "Start with T2VA, Euler, 8 steps, denoise=1, transition_step=3, lowres_scale=0.5, fixed seed. "
        "The incoming empty AV latent defines FINAL resolution. No separate upscale or second sampler is needed.\n\n"
        "For A/B change ONLY lowres_scale to 1.0: this runs native full-resolution Euler with the same schedule. "
        "Keep the seed fixed. Acceleration is disabled for the initial test.\n\n"
        "Requires the existing minimax_h3_latent_upscaler_3d checkpoint in latent_upscale_models. "
        "Masks, image/keyframe guides, audio-driven input and hard-prefix/temporal continuation are unsupported. "
        "This is a user quality trial, not a validated speed/quality claim."
    ]
    # A new workflow identity prevents overwriting the original tab/document.
    workflow.pop("id", None)
    workflow["revision"] = 0
    # The source's named-widget shadow values must not override changed widgets.
    for node in workflow["nodes"]:
        node.pop("widgets_values_named", None)
        for output in node.get("outputs", []):
            output["links"] = []
    for link in workflow["links"]:
        nodes[link[1]]["outputs"][link[2]]["links"].append(link[0])
    target = PROJECT / "examples/JR_H3_Progressive_T2VA_Experimental.json"
    if write:
        target.write_text(json.dumps(workflow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(target)
    return workflow


if __name__ == "__main__":
    build()
