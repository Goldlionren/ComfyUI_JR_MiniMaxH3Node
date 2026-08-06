import json

from ComfyUI_JR_MiniMaxH3Node.tools.h3_cache_benchmark import FIELDS, load_records, write_records


def test_benchmark_tool_writes_json_csv_and_markdown(tmp_path):
    row = {field: 0 for field in FIELDS}
    row.update(model="H3", quantization="int8", attention_backend="sage", resolution="768x1344",
               cache_mode="JR Balanced", cache_device="GPU")
    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    md_path = tmp_path / "results.md"
    write_records([row], json_path)
    write_records(load_records(json_path), csv_path)
    write_records(load_records(json_path), md_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))[0]["cache_mode"] == "JR Balanced"
    assert "cache_mode" in csv_path.read_text(encoding="utf-8-sig")
    assert "| model |" in md_path.read_text(encoding="utf-8")
