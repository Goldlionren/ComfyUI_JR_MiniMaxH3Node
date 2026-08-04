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
