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

## The acoustic solver

**Engine -> Compute** has *ODE steps* and *Acoustic solver*. The released protocol
is `midpoint` at 32 steps: a midpoint step evaluates the model twice, so a take
costs 64 evaluations. `dpmpp_2m` is a second-order multistep solver that keeps the
previous step's velocity instead of taking a probe evaluation, so a step costs one.

Measured here on one 47-second take, re-running only the acoustic stage from the
same semantic tokens on a 24 GiB card:

| Solver and steps | Acoustic stage | Difference from the reference latents |
|---|---:|---:|
| `midpoint` 32 (default) | 13.9 s | — |
| `dpmpp_2m` 32 | 7.5 s | 1.2% |
| `dpmpp_2m` 8 | 4.2 s | 13.6% |
| `dpmpp_2m` 6 | 3.9 s | 19.5% |
| `midpoint` 6 | 4.5 s | 7.0% |

So the same 32 steps under `dpmpp_2m` halve the acoustic stage while barely moving
the result, and that is the setting worth trying first. Cutting to six steps saves
little beyond that — each chunk's prefill is a fixed cost the step count cannot
touch — while changing the output materially. Judge six steps by ear, not by the
table. Note also what the acoustic stage is a share of: in that take the whole run
was 31 s, of which the acoustic stage was 9 s, so the headline is a fraction of the
song's wall clock, not its whole.

`midpoint` remains the default and the only validated setting; the solver used is
recorded in each take's `config.json`.

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

## From your phone

```powershell
.\webui\start.ps1 -Lan        # or from Explorer:  start-console.bat lan
```

The console then answers on your Wi-Fi as well as on this machine, and prints the
address and a six-digit PIN. Open the printed link on the phone — it carries the PIN
— or type the address and enter the PIN once; either way a cookie keeps you in for a
month, so artwork and audio load without asking again. **Engine -> Phone access**
shows the address and PIN again, and copies the link for you. A machine with Hyper-V,
WSL or a VPN has several addresses; the card lists the one the routing table would
actually use first, and that is the one to try.

A phone is redirected to `/m`, a console of its own: two screens, a player fixed at
the bottom, and none of the desktop's furniture. See **The phone console** below.
`/?desktop=1` opens the full console on a phone anyway, and the phone one is always
at `/m` from any browser.

Windows Firewall blocks the port until you allow it, and it does not always ask.
If the phone cannot reach the console at all, **Engine -> Phone access** prints the
one command to run in an *administrator* PowerShell; it allows the port for your own
subnet only:

```powershell
New-NetFirewallRule -DisplayName 'YuE2 Console' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 7865 -RemoteAddress 192.168.0.0/24
```

That rule covers every firewall profile, so a network Windows has classed as Public
works without reclassifying it. The layout folds to one column on a phone and the controls are sized
for a thumb, so writing a brief, starting a run, watching it and playing the take all
work from the sofa.

What the PIN is and is not: it stops the other devices on your network from driving
your GPU by accident. This is plain HTTP and the PIN is six digits, so keep it to a
network you trust and never forward the port on your router. Eight wrong guesses lock
that device out for five minutes. The PIN lives in `webui/settings.json`, which is
not tracked by git; delete `access_pin` there to have a new one made.

## The phone console

`/m` is a separate page, not the console squeezed smaller. Two tabs at the bottom —
**Create** and **Listen** — and a player that stays put, like any music app: tap a
song to play it, tap the next when it ends, scrub by tapping the line above it.

Create has one switch, *Simple* or *Custom*:

- **Simple** — one line of idea. The local writer turns it into a title, style and
  words, and the song follows straight on.
- **Custom** — your own title, style and lyrics, no writer.

The song sheet sits under both, in three drawers (Sound, Voice, Words and length)
rather than nine dropdowns in a row. In Simple it steers the writer; in Custom the
musical half can be pushed into the style box.

On Listen, tapping a song plays it and tapping its sleeve opens the song itself: the
full artwork, the style prompt, the lyrics and the cover prompt, a live meter off the
audio while it plays, the words over the sleeve as it goes, and **Save on this device**. Saving
keeps the audio in the browser, so the song plays from the phone rather than the wire
and survives walking out of range; once kept, a download button hands you the file. A
kept song shows a dot in the list. Cache Storage and service workers need a secure
origin, which a console on plain HTTP does not have, so the copy lives in IndexedDB,
which has no such rule.

Everything else is decided for you, because a phone is not where you tune things:
the full chord-annotated plan every time, the writer model the console is set to,
and no sampling, no remix, no score of your own, no engine settings. Those all stay
in the full console, which is one tap away at the bottom of either screen.

### Timing the words

YuE2 never says when a line is sung: the words go in at the top and codec tokens come
out at the bottom, with nothing joining the two. So the song view spaces the lines
evenly by length at first, which follows the song without knowing it and drifts as
soon as there is an instrumental bar.

**Time the words to the song** fixes that by listening to the take. It transcribes the
finished audio with Whisper and lines that transcript up against the lyrics already on
file, which is a much easier job than transcription: a misheard word costs one anchor
out of hundreds rather than being a wrong answer, and lines nothing was heard in are
placed between the ones that were. The result is written beside the take as
`lyrics_timing.json` and the phone uses it automatically from then on.

Measured on a 93-second take here: 58 seconds on the GPU, 99 words heard, 20 of the 23
lines anchored in the audio and 3 placed between them. The model is
`openai/whisper-small` by default, about 900 MB fetched once on first use; the
`align_model` setting takes any Whisper checkpoint if you want a bigger one. No extra
install: it runs on the `transformers` and `torch` the console already has.

## Adapters

LoRA adapters are merged into the weights under **Engine -> Adapters**. Point the
console at the folder you keep them in (a ComfyUI `models/loras` folder is the usual
one), tick the adapter, and set a strength; the model reloads on the next run with
the adapter folded in, which costs nothing per song.

The adapters published for YuE2 are trained on the NAR branch — the acoustic half
that turns the written score into sound — so they change the voice and the mix,
not the composition. They arrive in the ComfyUI repackaging's key names, where the
NAR expert is called the diffusion model and its q/k/v and gate/up projections are
fused; `src/yue2/lora.py` splits those back apart by the checkpoint's own layer
widths, so a file that does not fit this model is refused instead of half-applied.
Only files whose header really carries adapter keys are offered, so an image LoRA
sitting in the same folder is simply not listed.

Adapters need the PyTorch engine. audio.cpp reads GGUF weights and cannot merge
them, and a run is refused rather than quietly ignoring the selection. From the
command line: `yue2 song --lora path	odapter.safetensors:0.8 ...`.

## The song sheet

Under *Start from an idea* is a **Song sheet**: genre, tempo, meter, key, lyrics,
language, voice, theme and length, as pickers. Anything set there the writer has to
honour, and the musical half — genre, tempo, meter, key, voice — can be pushed
straight into the style prompt with one button, for when you are writing the words
yourself. A field left on *let the writer decide* is not mentioned to the writer at
all, so an empty sheet behaves exactly as before.

The vocabulary (472 entries across the nine fields) is the curated list from the
structured prompt node of ComfyUI-MiniMax-Music-Production-Toolkit, MIT licensed and
credited in `THIRD_PARTY_NOTICES.md`. It lives in `webui/vocabulary.json`, so you can
add your own entries to a field by editing that file; the console reads it at start.
Your last sheet is remembered in the browser, not on the server.

## Song engines

Two engines run the same model, chosen under **Engine -> Song engine**:

| | PyTorch (default) | audio.cpp (GGUF) |
|---|---|---|
| Weights | `model.safetensors`, bf16 | `yue2-3b-q4_0/q8_0/bf16.gguf` |
| VRAM for one take | about 12.5 GiB | 7.8 GiB at q4_0, 8.9 at q8_0 |
| Where it runs | in this process, weights stay in VRAM | its own process, VRAM handed back at the end |
| Score written by the model | kept as `score.abc`, streamed while it writes | not returned by the CLI |

The VRAM figures are audio.cpp's own RTX 5090 measurements for a three-minute take;
the quantized runs are also slightly faster there (0.199 RTF against 0.269). Neither
engine changes how long a song can be: the protocol stops at 9000 semantic tokens on a
24576-token context either way. What the small weights change is whether a 8-12 GiB
card reaches that ceiling instead of running out of memory on the way.

The console fetches the weights it needs from
[audio-cpp/Yue2-3B-GGUF](https://huggingface.co/audio-cpp/Yue2-3B-GGUF) on the first
run, or from the list under Engine.

**The binary.** YuE2 landed on audio.cpp's `dev` branch after its most recent release,
so *Install audio.cpp* only takes a release that actually carries the model: it checks
each build's source tree before downloading one, and says so plainly when none does.
Until that release exists, build audio.cpp from `dev` yourself and give the console the
path to your `audiocpp_cli` under Engine. A build without YuE2 is refused with the
reason rather than failing mid-run.

## Not wired up

- **Transcribing someone else's recording** needs SheetSage2 in its own environment;
  see above. Remixing your own takes needs none of it.
- **The score audio.cpp writes** is not saved with a GGUF take: its CLI returns audio
  only, so `score.abc` is written only when the score came from here. ABC input works
  in both engines; ABC output works in the PyTorch one.
- **The token-by-token score view** is PyTorch only, for the same reason.
