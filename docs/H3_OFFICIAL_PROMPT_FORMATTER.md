# Official MiniMax H3 Prompt Formatter

## Fixed source

The formatter targets the MiniMax-H3 prompt-writing specification at commit `8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea`:

- `skills/h3-prompt-writing/SKILL.md`
- `skills/h3-prompt-writing/references/base-en.txt`
- `skills/h3-prompt-writing/references/ref-en.txt`

Source hashes and license handling are recorded in [`specs/minimax_h3_prompt/.../SOURCE.md`](../specs/minimax_h3_prompt/8d8824efaf94586c0cc9ac7ad8d0723d4d6420ea/SOURCE.md). The upstream prose is not redistributed.

## Runtime architecture

```text
Director PIPE or legacy inputs
  -> OpenAI-compatible model: semantic JSON only
  -> strict semantic schema
  -> deterministic Python formatter
  -> strict final-output validator
  -> optimized_prompt and derived PIPE
```

The model decides scene semantics: action, camera, atmosphere, sound intent, delivery and reference relationships. Python owns every structural token: section title and order, mode alignment preamble, Shot number and timestamp, reference label/order, retention enum, speaker ID, language tag, protected dialogue text and blank-line layout. Runtime behavior never reads GitHub or parses the upstream natural-language files.

## Mode contracts

- `T2VA`: Base three-section format without an alignment preamble.
- `I2VA`: exact 0.00-second `<Picture 1>` first-frame reference preamble, then the Base sections.
- `FL2VA`: exact first/last alignment preamble using the real final Shot number and target duration.
- `L2VA`: exact last-frame alignment preamble using the real final Shot number and target duration.
- `Ref2VA`: exact six sections in order: `subject_definitions`, `summary`, `retention_analysis`, `detailed_description`, `overall_soundscape`, `non_diegetic_music`.

Director PIPE Shot starts are authoritative. Shot 1 has no timestamp; Shot 2+ use the official `At MM:SS.mmm,` form and must be strictly increasing and inside the target duration. Legacy mode may accept model-proposed starts, but the same deterministic range/order checks apply.

## Dialogue and references

Explicit spoken text is extracted before the request and exposed to the model only by `literal_index`. The formatter inserts each literal exactly once, without translation or punctuation changes, detects its official language label, and assigns stable `S1`, `S2`, ... IDs by speaker key. Base dialogue is emitted only inside `integrated_multimodal_description`; Ref2VA dialogue is emitted only inside `detailed_description`. `overall_soundscape` and `non_diegetic_music` cannot repeat protected dialogue.

Ref2VA semantic JSON must return every registered `<Picture N>`, `<Video N>` and `<Audio N>` in the exact program-owned order. Visible and audio retention relationships use separate pinned controlled vocabularies. Unknown, reordered or invented labels are rejected before formatting.

## Repair and compatibility

Malformed semantic JSON receives at most one `temperature=0.1` structured repair request. Python never asks the repair call to rewrite the final H3 text. A formatter/validator failure is treated as a deterministic data or code error and follows the existing `Return Original` / `Stop Workflow` behavior.

The node ID, legacy inputs, mode routing, image-data URL behavior, status shape and four outputs are unchanged. With a PIPE, only `optimized_prompt` is derived; timeline, shots, registry and runtime media remain unchanged.
