# Prompt Review & Continue

`JR_H3_PromptReviewPause` 是交互式人工审核节点。它同时支持 legacy multiline `STRING` 和 optional `JR_H3_DIRECTOR_PIPE`，暂停当前 ComfyUI 执行，等待发起任务的浏览器确认或编辑，然后输出 `reviewed_prompt` 与派生 PIPE。

## 接线

```text
Prompt Optimizer.pip
  -> Prompt Review & Continue.pip
       pip -> JR MiniMax H3 Directed Video Conditioning.pipe
```

参数：

- `prompt`：legacy multiline STRING；当前 ComfyUI 前端允许 widget 和 STRING 连接共存，PIPE 模式下应为空，非空时必须与 PIPE 的权威审核文本完全相同。
- `timeout_seconds`：默认 `3600`，范围 `60..86400` 秒。
- `pip`：optional `JR_H3_DIRECTOR_PIPE`。审核文本优先 `optimized_prompt`，再回退 `compiled_director_prompt`。
- 隐藏输入 `unique_id`：把审核与具体节点实例绑定。

## 执行过程

1. 节点确认当前任务有活动的 ComfyUI browser client 和 WebSocket。
2. 为本次执行创建随机、单次使用的 review ID。
3. 只向该 client 发送 incoming prompt。
4. 节点显示 **Waiting for review**，后端以最多 0.25 秒的短等待循环检查 Stop 和超时。
5. 用户编辑文本并点击 **Next / Continue**。
6. 服务端校验 review ID 与文本，唤醒当前执行并原样输出已批准文本。PIPE 模式会派生新 PIPE 并写入 `reviewed_prompt`，即使用户未修改直接 Next 也会写入。

Review 不会 mutate 输入 PIPE；timeline、registry 和 runtime media 均原样保留。STRING-only 模式继续返回 `(reviewed_prompt, None)`，因此旧 STRING 工作流保持兼容。

`IS_CHANGED` 返回 NaN，所以相同输入再次排队仍会重新审核，不会命中 ComfyUI 缓存而跳过。

## 前端尺寸行为

- 新节点最小宽度约 460、高度约 360。
- 文本框自身可以纵向拉伸。
- 收到新提示词、恢复 pending review 或重新执行时，只会把节点扩大到最低可用尺寸。
- 如果用户已把节点拉得更宽或更高，前端保留该尺寸，不会自动缩回默认大小。
- 旧工作流里 `timeout_seconds` 为 0、非数值或小于 60 时，前端会规范为 3600。

## 状态

- Idle
- Waiting for review
- Submitting
- Approved
- Timed out
- Cancelled
- Error

空白提交会被拒绝。文本最大长度为 100,000 字符。一个 review ID 只能成功提交一次。

## Stop、超时与刷新

ComfyUI **Stop** 会中断短等待循环、清理 pending state，并阻止下游执行。超时会抛出明确 workflow error，不会返回未经审核的原提示词。

浏览器通常在刷新后保持同一个 client ID。重连后，前端查询属于该 client 的 pending review 并恢复文本和按钮。如果浏览器关闭后一直不重连，任务会等待到 Stop 或 timeout。

## 安全和 API 边界

- review request/status 只发送给发起执行的 client，不广播。
- 提示词仅存放在有界内存状态、WebSocket 消息和提交 POST 中。
- 普通日志和浏览器 console 不记录完整提示词。
- 提交路由：`POST /jr_h3/prompt-review/continue`。
- 恢复路由：`GET /jr_h3/prompt-review/pending?client_id=...`。
- 无匹配活动浏览器时立即报错。

该节点按设计不支持无人值守、纯 API 或长期自动批处理。
