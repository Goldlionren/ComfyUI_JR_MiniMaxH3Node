import json
from pathlib import Path


def test_director_example_is_importable_and_has_authoritative_pip_wiring(package_name):
    package = __import__(package_name)
    path = Path(__file__).resolve().parents[1] / "examples" / "jr_minimax_h3_director_desk_workflow.json"
    workflow = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["id"]: node for node in workflow["nodes"]}
    jr_types = {node["type"] for node in nodes.values() if node["type"].startswith("JR_")}
    assert jr_types <= set(package.NODE_CLASS_MAPPINGS)
    assert nodes[1]["type"] == "JR_H3_DirectorDesk"
    assert nodes[2]["type"] == "JR_H3_OpenAICompatiblePromptOptimizer"
    assert nodes[2]["inputs"] == [{"name": "pip", "type": "JR_H3_DIRECTOR_PIPE", "link": 2}]
    assert [2, 1, 1, 2, 0, "JR_H3_DIRECTOR_PIPE"] in workflow["links"]
    persisted = nodes[1]["properties"]["jr_h3_director_state"]
    assert json.loads(nodes[1]["widgets_values"][0]) == persisted
    assert nodes[1]["size"][0] >= 1000 and nodes[1]["size"][1] >= 650
