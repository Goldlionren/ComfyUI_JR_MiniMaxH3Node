import pytest
import torch
from ComfyUI_JR_MiniMaxH3Node.utils.h3_stream_cache import JR_H3_CleanAVKVCache, select_layers


def stage(cache, index, fail=False):
    cache.begin_clean_commit(index)
    for layer in cache.layers:
        for start in range(cache.heads):
            v = torch.full((1, 1, 3, 8), float(index))
            a = torch.full((1, 1, 4, 8), float(index))
            cache.stage(layer, start, v, v, a, a)
    if fail:
        cache.rollback()
    else:
        cache.commit()
        cache.trim()


def test_owned_transaction_and_rollback():
    cache = JR_H3_CleanAVKVCache((0, 1), 2)
    stage(cache, 0)
    metrics = cache.metrics()
    stage(cache, 1, fail=True)
    assert cache.metrics() == metrics
    cache.begin_clean_commit(1)
    with pytest.raises(RuntimeError, match="incomplete clean layers"):
        cache.commit()
    cache.rollback()
    stage(cache, 1)
    assert cache.metrics()["retained_commits"] == 2
    cache.clear()
    assert cache.nbytes == 0 and cache.staged_bytes == 0 and not cache.active


@pytest.mark.parametrize("policy,commits,video,audio", [
    ("previous_only", 1, 3, 4), ("sink_plus_recent_1", 2, 6, 4), ("sink_plus_recent_2", 3, 9, 8)])
def test_retention_bounded_and_audio_reset(policy, commits, video, audio):
    cache = JR_H3_CleanAVKVCache((0, 3), 2, retention=policy)
    plateau = None
    for i in range(100):
        stage(cache, i)
        if i >= 4:
            if plateau is None:
                plateau = cache.nbytes
            assert cache.nbytes == plateau
    m = cache.metrics()
    assert (m["retained_commits"], m["video_history_tokens"], m["audio_history_tokens"]) == (commits, video, audio)
    keys, values = cache.history(0, 0, 2, device="cpu", dtype=torch.float32)
    assert sum(k.shape[2] for k in keys) == video + audio
    assert all(k.shape == v.shape for k, v in zip(keys, values))
    cache.drop_audio_history()
    assert cache.metrics()["audio_history_tokens"] == 0
    assert cache.nbytes < plateau


def test_nonfinite_shape_head_and_budget_guards():
    c = JR_H3_CleanAVKVCache((0,), 2)
    x = torch.ones(1, 1, 3, 8)
    with pytest.raises(RuntimeError, match="active"):
        c.stage(0, 0, x, x, x, x)
    c.begin_clean_commit(0)
    with pytest.raises(ValueError, match="non-finite"):
        c.stage(0, 0, x * float("nan"), x, x, x)
    c.stage(0, 0, x, x, x, x)
    x.zero_()
    assert bool(c._staged[0][0].video_k.eq(1).all())
    with pytest.raises(RuntimeError, match="head group"):
        c.stage(0, 0, x, x, x, x)
    with pytest.raises(RuntimeError, match="incomplete clean heads"):
        c.commit()
    with pytest.raises(RuntimeError, match="trim staged"):
        c.trim()
    c.rollback()
    small = JR_H3_CleanAVKVCache((0,), 1, max_bytes=1)
    small.begin_clean_commit(0)
    with pytest.raises(ValueError, match="budget"):
        small.stage(0, 0, x, x, x, x)


def test_layer_selection():
    assert len(select_layers("all", 50)) == 50
    assert len(select_layers("every_2", 50)) == 25
    assert len(select_layers("every_4", 50)) == 13
    assert select_layers("custom", 50, "0,4,8,49") == (0, 4, 8, 49)
    for bad in ["", "1,1", "-1", "50", "1,a"]:
        with pytest.raises(ValueError):
            select_layers("custom", 50, bad)
