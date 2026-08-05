# MiniMax H3 end-frame wiring

Create this connection manually in ComfyUI:

```text
MiniMax H3 IMAGE frames
  -> JR MiniMax H3 Enhanced Video Combine (images)
       pass_frames = true
       frames -> JR MiniMax H3 Last Frame (frames)
                    image -> Preview Image or the next H3 segment's first-frame input
       filename -> optional downstream STRING consumer
```

`save_last_frame=true` writes a PNG beside the video. That disk file is separate from the `frames` IMAGE output. To connect Last Frame in the graph, `pass_frames` must be enabled.

This wiring guide is supplied instead of inventing an unverified ComfyUI workflow JSON schema.

## Prompt review wiring

```text
JR MiniMax H3 Prompt Optimizer (OpenAI Compatible)
  optimized_prompt -> JR MiniMax H3 Prompt Review & Continue (prompt)
                        reviewed_prompt -> MiniMax H3 text prompt input
```

Queue from an open ComfyUI browser. The review node pauses execution, fills its temporary multiline editor, and waits for **Next / Continue**. The editor is deliberately excluded from workflow prompt serialization, so a previous review cannot become the next run's socket input.
