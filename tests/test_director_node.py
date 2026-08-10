import math

from ComfyUI_JR_MiniMaxH3Node.nodes.director_desk import JR_H3_DirectorDesk
from ComfyUI_JR_MiniMaxH3Node.utils.director_pipe import DirectorPipe
from ComfyUI_JR_MiniMaxH3Node.utils.director_state import DEFAULT_DIRECTOR_STATE_JSON


def test_director_node_contract_and_compose_without_media():
    node = JR_H3_DirectorDesk()
    assert node.CATEGORY == "JR MiniMax H3/Director"
    assert node.RETURN_TYPES == ("STRING", "JR_H3_DIRECTOR_PIPE")
    assert node.RETURN_NAMES == ("director_prompt", "pip")
    prompt, pipe = node.compose(DEFAULT_DIRECTOR_STATE_JSON)
    assert isinstance(pipe, DirectorPipe)
    assert prompt == pipe.compiled_director_prompt
    assert node.VALIDATE_INPUTS(DEFAULT_DIRECTOR_STATE_JSON) is True
    assert math.isnan(node.IS_CHANGED(DEFAULT_DIRECTOR_STATE_JSON))


def test_director_node_invalid_state_returns_descriptive_validation():
    result = JR_H3_DirectorDesk.VALIDATE_INPUTS("{}")
    assert result is True  # empty JSON migrates to the safe default state
    result = JR_H3_DirectorDesk.VALIDATE_INPUTS('{"schema_version":99}')
    assert "schema_version" in result
