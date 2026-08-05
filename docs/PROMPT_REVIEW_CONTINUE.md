# Prompt Review & Continue

`JR MiniMax H3 Prompt Review & Continue` is an interactive human-approval checkpoint for a STRING prompt.

## Wiring

```text
Prompt Optimizer.optimized_prompt
  -> Prompt Review & Continue.prompt
       reviewed_prompt -> downstream MiniMax H3 prompt input
```

The `prompt` input is a multiline STRING socket intended for an upstream connection. `timeout_seconds` defaults to 3600 and accepts 60–86400 seconds. The single output is `reviewed_prompt`.

## Execution

1. The node verifies that the workflow was queued by an active ComfyUI browser client.
2. It creates a random, single-use review ID and sends the prompt only to that client.
3. The node editor enters **Waiting for review** while the worker thread performs short interruptible waits.
4. Edit the prompt and click **Next / Continue**.
5. The server validates the review ID and text, wakes that execution, and returns the edited STRING unchanged.

The same input is reviewed again on every queue. The node uses ComfyUI V1 `IS_CHANGED` with a NaN result so cached output cannot bypass approval.

## Statuses

- `Idle`
- `Waiting for review`
- `Submitting`
- `Approved`
- `Timed out`
- `Cancelled`
- `Error`

Empty/whitespace-only submissions and text longer than 100,000 characters are rejected. A review ID can be approved only once.

## Stop, timeout, and refresh

ComfyUI **Stop** interrupts the short wait loop, removes pending state, and prevents downstream execution. Timeout behaves the same way and raises a clear workflow error; it never silently returns the unreviewed input.

ComfyUI normally preserves its browser client ID across refreshes. After reconnect, the frontend queries pending reviews belonging to that client and restores the editor. If the browser is closed without reconnecting, the job remains paused until the client reconnects, the workflow is stopped, or the timeout expires.

## Security and API behavior

Review events are targeted to the client that queued the running workflow and are not broadcast. Prompt text is held only in bounded in-memory pending state, sent in the required WebSocket/POST messages, and never written to ordinary logs or the browser console.

Headless and unattended API execution are unsupported by design. Without a live matching WebSocket client, the node fails immediately instead of waiting indefinitely.
