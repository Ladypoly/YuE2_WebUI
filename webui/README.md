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

That CUDA 12.8 wheel covers `sm_70` through `sm_120`, so Ampere, Ada and Blackwell all
install the same way — a 5090 needs no nightly build.

Model weights download from Hugging Face on the first run and stay resident in VRAM
between runs. Needs an NVIDIA GPU with BF16 and about 24 GB of VRAM.

## What the screens do

**Start from an idea** — type one line ("a slow song about missing the last train
home") and a local model writes the title, style prompt, lyrics and a description for
the cover art. Two backends:

- **llama.cpp**, installed into the project folder from *Engine → Writer*, with a
  catalogue of GGUF writer models. The server runs only for the length of one request
  and is killed afterwards, so nothing of it survives.
- **Ollama**, if you already run it. The model is asked to unload with
  `keep_alive: 0`, then unloaded explicitly, then `/api/ps` is polled to confirm.

Either way, GGUF files already in your Hugging Face cache are found and offered; add
your own folders under *Engine → Writer*. Reasoning models are run with reasoning off:
with it on, Qwen3.8 spends the answer on a plan and returns a one-word lyric.

The writer is also told which cover model it is writing for, and shapes that line
accordingly — full sentences for the LLM-encoder families like Z-Image, a short tag
list for CLIP-based SD and SDXL checkpoints.

Lyrics are checked against a list of the phrases that make a song sound
machine-written — neon, echoes, whispers, dancing shadows and the stock metaphors
around them. Any of the worst offenders buys one rewrite, and the rewrite is kept only
if it is actually cleaner.

**Write and generate** — tick it and the whole pipeline runs from the one line: brief,
cover, then the song, without waiting for you to press anything.

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

**Engine** — model and decoder choice, device, backend, quantization, VRAM budget and
offline mode, plus the writer and cover-art backends with their model catalogues and
your own model folders. Saving weights or compute settings unloads the current
pipeline; writer and art settings do not. Everything persists in
`webui/settings.json`.

Keyboard: `space` plays and pauses, `Esc` closes the Engine panel.

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

## Remixing a take

A remix keeps the melody and rebuilds everything around it. The *Remix a take* drawer
takes the score from a song you already made, drops the chord symbols so the new style
can rebuild the harmony, and switches the run to **Melody only**. A ballad becomes a
techno track on the same tune; a German song becomes an English one.

This is the white-box property the project is built on, and it needs no extra setup.

Two things the model genuinely cannot do, so the console does not pretend otherwise:

- **There is no audio-to-audio path.** `SongRequest` accepts style, lyrics, an ABC
  score, cot, seed and cfg — no audio input of any kind. Every cover goes through a
  score.
- **A recording cannot be used as a style reference.** Style is text. To follow a
  reference, describe its sound in the style prompt.

### Covering someone else's recording (not wired into the UI)

`POST /api/cover/from-audio` still exists and transcribes an uploaded file with
SheetSage2, but the console does not show it. SheetSage2 pins different dependency
versions — Python 3.10/3.11, torch 2.8, and an FFmpeg 6.x whose shared libraries
torchaudio can bind — so it needs its own environment:

```powershell
py -3.11 -m venv .venv-sheetsage2
.\.venv-sheetsage2\Scripts\python.exe -m pip install huggingface-hub==0.36.0
.\.venv-sheetsage2\Scripts\huggingface-cli download m-a-p/SheetSage2 --local-dir models\SheetSage2
.\.venv-sheetsage2\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv-sheetsage2\Scripts\python.exe -m pip install -r models\SheetSage2
equirements.txt
```

That is roughly 6 GB — torch, the 2.5 GB MERT-v2-FullSong encoder it loads, and
SheetSage2 itself. The FFmpeg version is the part most likely to need work on Windows:
a system FFmpeg 7 will not satisfy a torchaudio built against FFmpeg 6. The console
finds the environment at `.venv-sheetsage2/` and `models/SheetSage2/` if you build it.

## GPU memory

One GPU runs the song model, the writer and the cover-art model, so the console hands
memory between them rather than hoping it fits:

- After every run — finished, failed or cancelled — the weights are moved back to
  the CPU and the CUDA allocator's cache is dropped. Without that last step a run's
  freed blocks stay *reserved*, and after a few songs the card reads as full while
  sitting idle, which starves the next run.
- The writer is unloaded the moment it answers, and YuE2 is parked before it runs.
- The cover is drawn before the song model loads, in the window where the GPU is idle,
  so it costs no extra wall-clock time.
- The top bar shows live VRAM. **Engine → Free VRAM** releases it on demand.

## Not wired up

- **Transcribing someone else's recording** needs SheetSage2 in its own environment;
  see above. Remixing your own takes needs none of it.
- **GGUF weights** ([audio-cpp/Yue2-3B-GGUF](https://huggingface.co/audio-cpp/Yue2-3B-GGUF))
  need the audio.cpp runtime, which is not released yet. When it lands, it would be a
  new backend alongside `torch` and `vllm`.
