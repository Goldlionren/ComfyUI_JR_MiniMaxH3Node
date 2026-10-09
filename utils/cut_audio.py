"""Local, sample-exact audio selection. No diffusion/VAE, gain or resampling."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".aif", ".aiff", ".wma"}
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_PCM_BYTES = 256 * 1024 * 1024
MAX_SECONDS = 7200
MAX_SELECTION_BYTES = 4096
DECODE_TIMEOUT = 90
_WORK_LOCK = threading.Lock()


class CutAudioError(ValueError):
    pass


def _error(message):
    return CutAudioError("JR Cut Audio: " + message)


def resolve_audio(name):
    import folder_paths

    if (not isinstance(name, str) or not name or len(name) > 1024 or
            any(ord(c) < 32 for c in name) or "\\" in name or ":" in name or
            Path(name).is_absolute() or ".." in Path(name).parts):
        raise _error("Choose a local audio file from the ComfyUI input directory.")
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() not in AUDIO_EXTENSIONS:
        raise _error("Unsupported audio filename or path outside the input directory.")
    if not path.is_file():
        raise _error("Audio file is missing; upload it again or choose an existing file.")
    if not 0 < path.stat().st_size <= MAX_FILE_BYTES:
        raise _error("Audio file must be nonempty and at most 512 MiB.")
    return path


def list_audio_inputs():
    import folder_paths

    root = Path(folder_paths.get_input_directory())
    if not root.is_dir():
        return []
    names = []
    for index, path in enumerate(root.rglob("*")):
        if index >= 20000 or len(names) >= 2000:
            break
        if path.suffix.lower() in AUDIO_EXTENSIONS and path.is_file():
            name = path.relative_to(root).as_posix()
            try:
                resolve_audio(name)
                names.append(name)
            except CutAudioError:
                continue
    return sorted(names, key=str.casefold)


def source_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        total = 0
        while block := handle.read(1024 * 1024):
            total += len(block)
            if total > MAX_FILE_BYTES:
                raise _error("Audio file exceeds the 512 MiB limit.")
            digest.update(block)
    return digest.hexdigest()


def decode_audio(path):
    """PyAV (already shipped by ComfyUI); retain rate/layout, convert PCM to f32."""
    import av

    started = time.monotonic()
    pieces, total = [], 0
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise _error("No decodable audio stream in this file.")
            stream = container.streams.audio[0]
            rate, channels = stream.codec_context.sample_rate, stream.channels
            if not rate or not 0 < rate <= 768000 or not 1 <= channels <= 8:
                raise _error("Expected audio with a valid sample rate and 1–8 channels.")
            converter = av.AudioResampler(format="fltp", layout=stream.layout.name, rate=rate)

            def append(frame):
                nonlocal total
                data = frame.to_ndarray()
                total += data.shape[-1]
                if total * channels * 4 > MAX_PCM_BYTES or total / rate > MAX_SECONDS:
                    raise _error("Decoded audio exceeds 256 MiB PCM or 2 hours; use a shorter source file.")
                if time.monotonic() - started > DECODE_TIMEOUT:
                    raise _error("Audio decoding timed out; use a shorter source file.")
                if data.shape[0] != channels or not np.isfinite(data).all():
                    raise _error("Decoder returned invalid PCM audio.")
                pieces.append(data)

            for frame in container.decode(stream):
                for converted in converter.resample(frame):
                    append(converted)
            for converted in converter.resample(None):
                append(converted)
        if not pieces:
            raise _error("Audio file contains no samples.")
        return torch.from_numpy(np.concatenate(pieces, axis=1)), int(rate)
    except CutAudioError:
        raise
    except Exception as error:
        raise _error("Cannot decode this audio. Check the file and installed FFmpeg/PyAV codec support.") from error


def read_source(name, expected_digest=None):
    path = resolve_audio(name)
    before = source_digest(path)
    if expected_digest is not None and before != expected_digest:
        raise _error("Source audio changed since selection. Reload it and press Cut again.")
    waveform, rate = decode_audio(path)
    if source_digest(path) != before:
        raise _error("Source changed while decoding. Reload and retry.")
    return waveform, rate, before


def sample_range(start, end, rate, count):
    if (type(start) not in (int, float) or type(end) not in (int, float) or
            not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end <= count / rate):
        raise _error("Use finite seconds with 0 ≤ start < end ≤ audio duration.")
    # Consistent half-up rounding, [first, last) sample interval.
    first = min(count, math.floor(start * rate + .5))
    last = min(count, math.floor(end * rate + .5))
    if first >= last:
        raise _error("Selection must contain at least one audio sample.")
    return first, last


def parse_selection(value, name):
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_SELECTION_BYTES:
        raise _error("Invalid locked selection.")
    try:
        selected = json.loads(value)
    except (ValueError, RecursionError) as error:
        raise _error("Set start/end and press Cut to lock the output first.") from error
    if (not isinstance(selected, dict) or type(selected.get("version")) is not int or selected.get("version") != 1 or selected.get("audio") != name or
            not isinstance(selected.get("source_id"), str) or len(selected["source_id"]) != 64 or
            any(c not in "0123456789abcdef" for c in selected["source_id"])):
        raise _error("Locked selection does not match this audio. Press Cut again.")
    for key in ("sample_rate", "source_samples", "start_sample", "end_sample"):
        if type(selected.get(key)) is not int:
            raise _error("Invalid sample indices in locked selection.")
    if not (0 < selected["sample_rate"] <= 768000 and
            0 <= selected["start_sample"] < selected["end_sample"] <= selected["source_samples"]):
        raise _error("Locked selection has invalid bounds.")
    return selected


def selection_audio(name, selection):
    selected = parse_selection(selection, name)
    with _WORK_LOCK:
        waveform, rate, _ = read_source(name, selected["source_id"])
        if rate != selected["sample_rate"] or waveform.shape[-1] != selected["source_samples"]:
            raise _error("Decoded source no longer matches the locked selection. Cut again.")
        result = waveform[:, selected["start_sample"]:selected["end_sample"]].clone().unsqueeze(0)
    duration = result.shape[-1] / rate
    return {"waveform": result, "sample_rate": rate}, duration, (
        f"Locked: {selected['start_sample'] / rate:.6f}–{selected['end_sample'] / rate:.6f} s | "
        f"{duration:.6f} s | {rate} Hz | {result.shape[1]} channels | unchanged PCM gain"
    )


def media_path(token):
    import folder_paths

    if not isinstance(token, str) or len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise _error("Invalid preview identifier.")
    root = Path(folder_paths.get_temp_directory()).resolve()
    target = root / "jr_cut_audio" / (token + ".wav")
    if not target.resolve().is_relative_to(root):
        raise _error("Preview directory must stay inside ComfyUI temp.")
    return target


def write_wav(waveform, rate, token):
    """IEEE float32 WAV: same decoded PCM as AUDIO, no lossy re-encoding."""
    path = media_path(token)
    path.parent.mkdir(parents=True, exist_ok=True)
    channels, count = waveform.shape
    data_bytes = count * channels * 4
    header = (b"RIFF" + struct.pack("<I", 48 + data_bytes) + b"WAVEfmt " +
              struct.pack("<IHHIIHH", 16, 3, channels, rate, rate * channels * 4, channels * 4, 32) +
              b"fact" + struct.pack("<II", 4, count) + b"data" + struct.pack("<I", data_bytes))
    temporary = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as output:
            output.write(header)
            for start in range(0, count, 65536):
                output.write(waveform[:, start:start + 65536].t().contiguous().numpy().astype("<f4", copy=False).tobytes())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return token


def analyze_audio(name):
    with _WORK_LOCK:
        waveform, rate, digest = read_source(name)
        count = waveform.shape[-1]
        bins = min(1600, count)
        peaks = []
        for index in range(bins):
            part = waveform[:, index * count // bins:(index + 1) * count // bins]
            # Keep extrema across channels; opposite-phase stereo never cancels.
            peaks.append([float(part.min()), float(part.max())])
        token = hashlib.sha256(("jr-audio-preview-v1:" + digest).encode()).hexdigest()
        write_wav(waveform, rate, token)
    return {"audio": name, "source_id": digest, "sample_rate": rate, "samples": count,
            "channels": waveform.shape[0], "duration": count / rate, "peaks": peaks, "media_token": token}


def commit_cut(name, start, end, source_id):
    if not isinstance(source_id, str) or len(source_id) != 64:
        raise _error("Reload the source waveform before cutting.")
    with _WORK_LOCK:
        waveform, rate, digest = read_source(name, source_id)
        first, last = sample_range(start, end, rate, waveform.shape[-1])
        selected = {"version": 1, "audio": name, "source_id": digest, "sample_rate": rate,
                    "source_samples": waveform.shape[-1], "start_sample": first, "end_sample": last}
        serialized = json.dumps(selected, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        token = hashlib.sha256(("jr-audio-cut-v1:" + serialized).encode()).hexdigest()
        write_wav(waveform[:, first:last], rate, token)
    return {"selection": serialized, "media_token": token, "start_seconds": first / rate,
            "end_seconds": last / rate, "duration": (last - first) / rate}
