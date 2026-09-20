"""Find out when each line is actually sung, so the words can follow the song.

Nothing in YuE2 records that. The model emits codec tokens; the words go in at
the top and never come back out with times attached. So the take's own audio is
transcribed and the transcript is aligned against the lyrics we already have.

That last part is what makes a small model enough: this is not transcription,
where a wrong word is a wrong answer, but alignment, where a wrong word only
costs one anchor among hundreds. Whisper mishears sung words often; it still
puts recognisable ones in the right place, and the lines in between are carried
by interpolation.

The result is written next to the take as lyrics_timing.json and is advisory:
the console falls back to spacing the lines evenly when it is missing.
"""
from __future__ import annotations

import difflib
import json
import re
import time
from pathlib import Path

import numpy as np

TIMING_FILE = "lyrics_timing.json"
SAMPLE_RATE = 16000
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def is_tag(line):
    return bool(re.fullmatch(r"\s*\[.*\]\s*", line or ""))


def words_of(text):
    return [w.lower() for w in WORD.findall(text or "")]


def load_audio(path):
    """Mono 16 kHz for the recogniser, without pulling in a resampler.

    The take is 48 kHz, an exact three to one, so averaging each group of three
    samples both decimates and low-passes it in one step.
    """
    import soundfile as sf

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if rate % SAMPLE_RATE == 0:
        factor = rate // SAMPLE_RATE
        usable = (len(mono) // factor) * factor
        mono = mono[:usable].reshape(-1, factor).mean(axis=1)
    elif rate != SAMPLE_RATE:
        # An odd rate is rare here; a linear read is enough for a recogniser.
        target = int(round(len(mono) * SAMPLE_RATE / rate))
        mono = np.interp(np.linspace(0, len(mono) - 1, target),
                         np.arange(len(mono)), mono).astype("float32")
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if peak > 0:
        mono = mono / peak
    return mono.astype("float32"), len(audio) / rate


def transcribe(samples, model_id, device=None, progress=None):
    """Word-level times from the take's own audio."""
    import torch
    from transformers import pipeline

    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    if progress:
        progress("loading " + model_id)
    recogniser = pipeline("automatic-speech-recognition", model=model_id,
                          device=device, torch_dtype=torch.float32,
                          chunk_length_s=30, stride_length_s=5)
    if progress:
        progress("listening to the take")
    try:
        result = recogniser(samples, return_timestamps="word",
                            generate_kwargs={"task": "transcribe"})
    finally:
        del recogniser
        with_cuda = device != -1
        if with_cuda:
            torch.cuda.empty_cache()
    found = []
    for chunk in result.get("chunks", []):
        stamp = chunk.get("timestamp") or (None, None)
        text = (chunk.get("text") or "").strip()
        tokens = words_of(text)
        if not tokens or stamp[0] is None:
            continue
        start = float(stamp[0])
        end = float(stamp[1]) if stamp[1] is not None else start
        # A chunk is one word here, but guard against a model that groups them.
        step = (end - start) / len(tokens) if len(tokens) > 1 else 0.0
        for index, token in enumerate(tokens):
            found.append({"word": token,
                          "start": start + index * step,
                          "end": start + (index + 1) * step if step else end})
    return found


def align(lines, heard, duration):
    """Give every sung line a start and an end, matching what was recognised."""
    tokens, owners = [], []
    for index, line in enumerate(lines):
        if is_tag(line) or not line.strip():
            continue
        for token in words_of(line):
            tokens.append(token)
            owners.append(index)

    times = {}
    if tokens and heard:
        matcher = difflib.SequenceMatcher(None, tokens, [w["word"] for w in heard], autojunk=False)
        for block in matcher.get_matching_blocks():
            for step in range(block.size):
                line_index = owners[block.a + step]
                word = heard[block.b + step]
                bounds = times.setdefault(line_index, [word["start"], word["end"]])
                bounds[0] = min(bounds[0], word["start"])
                bounds[1] = max(bounds[1], word["end"])

    sung = [index for index, line in enumerate(lines) if not is_tag(line) and line.strip()]
    anchored = sorted(times)
    out = []
    for index in sung:
        if index in times:
            start, end = times[index]
            out.append({"index": index, "start": max(0.0, start), "end": min(duration, max(end, start + .3)),
                        "matched": True})
        else:
            out.append({"index": index, "start": None, "end": None, "matched": False})

    # Lines the recogniser never caught sit evenly between the ones it did.
    known = [(entry["index"], entry["start"], entry["end"]) for entry in out if entry["matched"]]
    if not known:
        return [], anchored
    for position, entry in enumerate(out):
        if entry["matched"]:
            continue
        before = [k for k in known if k[0] < entry["index"]]
        after = [k for k in known if k[0] > entry["index"]]
        low = before[-1][2] if before else 0.0
        high = after[0][1] if after else duration
        span = [e for e in out if not e["matched"] and
                (not before or e["index"] > before[-1][0]) and
                (not after or e["index"] < after[0][0])]
        share = (high - low) / max(1, len(span))
        slot = span.index(entry)
        entry["start"] = low + slot * share
        entry["end"] = low + (slot + 1) * share

    # Time only ever moves forward.
    previous = 0.0
    for entry in out:
        entry["start"] = max(previous, min(duration, entry["start"]))
        entry["end"] = max(entry["start"] + .2, min(duration, entry["end"]))
        previous = entry["start"]
    return out, anchored


def align_take(directory, lyrics, model_id="openai/whisper-small", device=None, progress=None):
    directory = Path(directory)
    audio_path = directory / "audio.flac"
    if not audio_path.is_file():
        raise FileNotFoundError("This take has no audio to listen to")
    started = time.perf_counter()
    samples, duration = load_audio(audio_path)
    heard = transcribe(samples, model_id, device=device, progress=progress)
    lines = (lyrics or "").split("\n")
    timed, anchored = align(lines, heard, duration)
    if not timed:
        raise RuntimeError("Nothing in the take matched the words; the timing is unchanged")
    payload = {"version": 1, "model": model_id, "duration": duration,
               "words_heard": len(heard), "lines_matched": len(anchored),
               "lines_total": len(timed), "seconds": round(time.perf_counter() - started, 1),
               "lines": [{"index": e["index"],
                          "text": lines[e["index"]].strip(),
                          "start": round(float(e["start"]), 2),
                          "end": round(float(e["end"]), 2),
                          "matched": bool(e["matched"])} for e in timed]}
    (directory / TIMING_FILE).write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    return payload


def read_timing(directory):
    path = Path(directory) / TIMING_FILE
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
