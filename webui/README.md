# YuE2 Console

A local web console for YuE2: write a style prompt and lyrics, watch the score get
written token by token, then play the finished song and keep the take.

## Run it

```powershell
.\webui\start.ps1
```

Then open <http://127.0.0.1:7865>.

`YUE2_HOST`, `YUE2_PORT` and `YUE2_OUTPUTS` override the bind address and the take
folder (takes default to `outputs/` in the repository root).

## Setup

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install .
# PyPI ships a CPU-only torch on Windows; replace it with the CUDA build
.\.venv\Scripts\python.exe -m pip install --force-reinstall "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r webui\requirements.txt
```

On Blackwell (RTX 50-series) use the nightly index instead:
`--pre --index-url https://download.pytorch.org/whl/nightly/cu128`.

Model weights download from Hugging Face on the first run and stay resident in VRAM
between runs. Needs an NVIDIA GPU with BF16 and about 24 GB of VRAM.

## What the screens do

**Start from an idea** — type one line ("a slow song about missing the last train
home") and a local Ollama model writes the title, style prompt and lyrics into the
form. Pick the writer from the dropdown; it lists whatever Ollama has installed.

The two models share one GPU, so the console hands VRAM between them: YuE2 is
unloaded before the writer runs (the checkbox, on by default), and the writer is
asked to unload the moment it answers — `keep_alive: 0` on the request, then an
explicit unload, then `/api/ps` is polled to confirm it is really gone. The status
line says `writer unloaded`, or turns red if it did not. Turn the checkbox off for a
small writer that fits alongside YuE2.

To use a GGUF you already have, register it with Ollama once:

```powershell
# Modelfile
# FROM C:\path\to\model.gguf
ollama create my-writer -f Modelfile
```

**Length** — the *Length* control sets a target running time, and it works on both
halves of the job: the writer is told how many sung lines that length needs (about
one line per 4.3 seconds), and the run gets a token ceiling from it, since the
semantic stream is a steady 25 tokens per second of audio. It is a target, not a
guarantee — lyrics decide the real length, the ceiling carries 15% headroom, and a
song that ends on its own before the cap is the good outcome. *Follow the lyrics*
leaves the model's own limit in place.

**Compose** — style prompt, lyrics with `[Verse]` / `[Chorus]` tags, planning mode,
seed, an optional supplied ABC score, and per-stage sampling. Planning mode maps
straight to the pipeline's `cot` setting:

| Mode | `cot` | Use it for |
|---|---|---|
| Full plan | `full` | New songs — melody and chords are written first |
| Melody only | `melody` | Covers — melody is fixed, accompaniment is free |
| Direct | `off` | Straight from lyrics and style, no editable score |

**Take** — the live run as a four-stage chain (plan → music tokens → synthesis →
decode) with token counts and rates, the score appearing as it is written, then the
player, the engraved score, and the run metadata. *Edit and re-render* copies a
finished score back into the compose form so you can change the harmony or form and
render it again — the white-box loop from the project README.

**Engine** — model and decoder choice, device, backend, quantization, VRAM budget,
ODE steps and offline mode. Saving unloads the current pipeline; the next run loads
the new one. Settings persist in `webui/settings.json`.

Keyboard: `1` `2` `3` switch screens, `space` plays and pauses.

## How it hooks into YuE2

The server drives the public pipeline (`YuE2Pipeline.__call__`) with `cancelled` and
`on_token` callbacks, and swaps the pipeline's private `_status` reporter for one
that publishes to the browser over Server-Sent Events instead of writing to stderr.
If a future YuE2 release renames that reporter, progress bars go quiet but
generation itself is unaffected.

One GPU means one run at a time; further requests queue behind the active run.

Each take is a full `save_artifacts` directory — `audio.flac`, `score.abc`,
`semantic.npy`, `latent.npy`, `request.json`, `config.json`, `result.json` — plus a
`webui.json` holding the title and prompt for the library list. Deleting a take from
the library removes that directory.

## Covers and remixes

A cover keeps the melody and rebuilds everything around it. The *Cover or remix*
drawer takes a melody from either source, drops the chord symbols so the new style
can rebuild the harmony, and switches the run to **Melody only** mode:

- **From a take** — reuses a finished take's score. Works out of the box; this is
  the remix path: same melody, new style prompt, new arrangement.
- **From a recording** — transcribes an audio file with SheetSage2 (see below).

Two things the model genuinely cannot do, so the console does not pretend otherwise:

- **There is no audio-to-audio path.** `SongRequest` accepts style, lyrics, an ABC
  score, cot, seed and cfg — no audio input of any kind. Every cover goes through a
  score.
- **A recording cannot be used as a style reference.** Style is text. To follow a
  reference, transcribe it for the melody and describe its sound in the style prompt.

### Installing SheetSage2 (optional, for audio input)

SheetSage2 pins different dependency versions to YuE2, so it needs its own
environment. The console looks for `.venv-sheetsage2/` and `models/SheetSage2/` in
the repository root and enables the *Transcribe* button when both exist.

```powershell
py -3.11 -m venv .venv-sheetsage2
.\.venv-sheetsage2\Scripts\python.exe -m pip install huggingface-hub==0.36.0
.\.venv-sheetsage2\Scripts\huggingface-cli download m-a-p/SheetSage2 --local-dir models\SheetSage2
.\.venv-sheetsage2\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv-sheetsage2\Scripts\python.exe -m pip install -r models\SheetSage2\requirements.txt
```

It also needs **FFmpeg 6.1 and its shared libraries** on PATH. The upstream setup in
[docs/covers.md](../docs/covers.md) is written for Linux; on Windows the FFmpeg
requirement is the part most likely to need work.

Transcription and generation share the GPU, so the console parks YuE2 before
running SheetSage2.

## GPU memory

One GPU runs the song model, and optionally the writer and SheetSage2, so the
console hands memory between them rather than hoping it fits:

- After every run — finished, failed or cancelled — the weights are moved back to
  the CPU and the CUDA allocator's cache is dropped. Without that last step a run's
  freed blocks stay *reserved*, and after a few songs the card reads as full while
  sitting idle, which starves the next run.
- The writer is unloaded the moment it answers, and YuE2 is parked before it runs.
- The top bar shows live VRAM. **Engine → Free VRAM** releases it on demand.

## Not wired up

- **SheetSage2 transcription** for covers runs in a separate environment; supply its
  ABC through the *Supply your own score* drawer.
- **GGUF weights** ([audio-cpp/Yue2-3B-GGUF](https://huggingface.co/audio-cpp/Yue2-3B-GGUF))
  need the audio.cpp runtime, which is not released yet. When it lands, it would be a
  new backend alongside `torch` and `vllm`.
