"""Local web console for YuE2: compose, watch the run, keep the takes."""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Set before torch is ever imported: the allocator otherwise fragments across a
# run and a late 1.4 GiB attention block fails while gigabytes sit reserved.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
OUTPUTS = Path(os.environ.get("YUE2_OUTPUTS", ROOT / "outputs")).resolve()
OUTPUTS.mkdir(parents=True, exist_ok=True)

VAE_CHOICES = {"standard": "m-a-p/YuE2-Vae", "legacy": "m-a-p/YuE2-Vae-legacy"}

# A release can ship the weights next to the code. resolve_model() takes a
# directory as-is, so a bundled copy is used instead of downloading 7 GB.
BUNDLED = ROOT / "models"


def weights_on_disk():
    """True when the song model is already local, bundled or in the HF cache."""
    if (BUNDLED / "YuE2-3B" / "config.json").is_file():
        return True
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    folder = "models--" + str(SETTINGS.model).replace("/", "--")
    return (cache / folder).is_dir()


def bundled_or_hub(repo):
    local = BUNDLED / str(repo).split("/")[-1]
    return str(local) if (local / "config.json").is_file() else repo

STAGES = [
    ("plan", "Planning score", "Writing the melody and chord plan"),
    ("semantic", "Generating song", "Turning the score into music tokens"),
    ("synthesize", "Synthesizing audio", "Refining acoustic latents"),
    ("decode", "Decoding audio", "Rendering 48 kHz stereo"),
]
# Pipeline stage labels that are setup work rather than musical work.
SETUP_LABELS = {"Resolving model files", "Loading model", "Loading audio decoder", "Using provided score"}
LABEL_TO_STAGE = {label: key for key, label, _ in STAGES}


# --------------------------------------------------------------------------- bus

class EventBus:
    """Append-only event log; SSE clients read forward from a cursor."""

    def __init__(self):
        self._events = []
        self._lock = threading.Lock()

    def publish(self, kind, payload):
        with self._lock:
            self._events.append({"seq": len(self._events) + 1, "kind": kind, "data": payload})
            if len(self._events) > 4000:
                del self._events[:2000]

    def since(self, cursor):
        with self._lock:
            return [e for e in self._events if e["seq"] > cursor], (self._events[-1]["seq"] if self._events else 0)


BUS = EventBus()


# --------------------------------------------------------------------------- settings

@dataclasses.dataclass
class Settings:
    model: str = "m-a-p/YuE2-3B"
    vae: str = "standard"
    device: str = "auto"
    backend: str = "torch"
    quantization: str = "none"
    memory_budget_gib: float = 0.0      # 0 = follow the card
    offload_ar: bool = True
    ode_steps: int = 32
    offline: bool = False
    ollama_url: str = "http://127.0.0.1:11434"
    muse_model: str = ""
    muse_free_engine: bool = True
    writer_backend: str = "auto"        # auto | ollama | llamacpp
    llamacpp_model: str = ""
    llamacpp_gpu_layers: int = 999
    art_model: str = ""
    art_auto: bool = False
    writer_dirs: list = dataclasses.field(default_factory=list)
    art_dirs: list = dataclasses.field(default_factory=list)

    def to_dict(self):
        return dataclasses.asdict(self)


SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"
SETTINGS = Settings()
if SETTINGS_FILE.is_file():
    with contextlib.suppress(Exception):
        stored = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        known = {f.name for f in dataclasses.fields(Settings)}
        SETTINGS = Settings(**{k: v for k, v in stored.items() if k in known})


def save_settings():
    SETTINGS_FILE.write_text(json.dumps(SETTINGS.to_dict(), indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- engine

def release_cuda_cache():
    """Return the caching allocator's reserved-but-free blocks to the driver."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def resolved_budget():
    """VRAM the pipeline may use. 0 means follow whatever card is present, so a
    24 GiB default does not throttle a 32 GiB one or overcommit a 16 GiB one."""
    if SETTINGS.memory_budget_gib and SETTINGS.memory_budget_gib > 0:
        return float(SETTINGS.memory_budget_gib)
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            return float(max(6, int(total)))
    except Exception:
        pass
    return 24.0


def vram():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return {"used_gib": round((total - free) / 2 ** 30, 2),
                "total_gib": round(total / 2 ** 30, 2),
                "reserved_gib": round(torch.cuda.memory_reserved() / 2 ** 30, 2),
                "allocated_gib": round(torch.cuda.memory_allocated() / 2 ** 30, 2)}
    except Exception:
        return None


class Engine:
    """Owns the single GPU pipeline and the one run that may use it."""

    def __init__(self):
        self.pipe = None
        self.loaded_with = None
        self.status = "idle"          # idle | loading | busy | error
        self.detail = ""
        self.ever_loaded = False

    def describe(self):
        return {"status": self.status, "detail": self.detail, "loaded_with": self.loaded_with}

    def unload(self):
        if self.pipe is not None:
            with contextlib.suppress(Exception):
                self.pipe.close()
        self.pipe = None
        self.loaded_with = None
        release_cuda_cache()

    def park(self):
        """Between runs, keep the weights but hand the VRAM back.

        The pipeline parks the language model and the decoder on the CPU as it
        goes, but the caching allocator keeps their freed blocks reserved, so a
        run leaves several GiB apparently in use until the cache is dropped.
        """
        pipe = self.pipe
        if pipe is None:
            return
        with contextlib.suppress(Exception):
            if getattr(pipe, "_model", None) is not None:
                pipe._model.to("cpu")
        with contextlib.suppress(Exception):
            if getattr(pipe, "_vae", None) is not None:
                pipe._vae.to("cpu")
        release_cuda_cache()

    def ensure(self):
        budget = resolved_budget()
        want = {"model": SETTINGS.model, "vae": SETTINGS.vae, "device": SETTINGS.device,
                "backend": SETTINGS.backend, "quantization": SETTINGS.quantization,
                "memory_budget_gib": budget, "offload_ar": SETTINGS.offload_ar,
                "ode_steps": SETTINGS.ode_steps, "offline": SETTINGS.offline}
        if self.pipe is not None and self.loaded_with == want:
            return self.pipe
        self.unload()
        from yue2 import YuE2Pipeline
        from yue2.protocol import GenerationConfig

        self.status = "loading"
        self.detail = "Loading YuE2"
        BUS.publish("engine", self.describe())
        config = dataclasses.replace(GenerationConfig(), ode_steps=int(SETTINGS.ode_steps))
        pipe = YuE2Pipeline.from_pretrained(
            bundled_or_hub(SETTINGS.model),
            vae=bundled_or_hub(VAE_CHOICES.get(SETTINGS.vae, SETTINGS.vae)),
            local_files_only=SETTINGS.offline,
            device=SETTINGS.device,
            backend=SETTINGS.backend,
            quantization=SETTINGS.quantization,
            memory_budget_gib=budget,
            offload_ar=SETTINGS.offload_ar,
            generation_config=config,
            progress=True,
        )
        self.pipe = pipe
        self.loaded_with = want
        self.ever_loaded = True
        self.status = "idle"
        self.detail = ""
        BUS.publish("engine", self.describe())
        return pipe


ENGINE = Engine()


# --------------------------------------------------------------------------- runs

class Job:
    def __init__(self, spec):
        self.id = "%x%04x" % (int(time.time() * 1000), random.randrange(16 ** 4))
        self.spec = spec
        self.state = "queued"          # queued | running | done | cancelled | failed
        self.error = ""
        self.created = time.time()
        self.started = None
        self.finished = None
        self.cancel_flag = threading.Event()
        self.stage = None
        self.stages = {key: {"key": key, "label": label, "note": note, "state": "waiting",
                             "completed": 0, "total": None, "unit": None, "seconds": 0.0}
                       for key, label, note in STAGES}
        self.setup = ""
        self.setup_at = None
        self.cover_ready = False
        self.abc_partial = ""
        self.abc_tokens = []
        self._last_push = 0.0
        self.take = None               # output folder name once saved

    def snapshot(self):
        return {"id": self.id, "state": self.state, "error": self.error, "stage": self.stage,
                "setup": self.setup, "setup_at": self.setup_at,
                "created": self.created, "started": self.started,
                "finished": self.finished, "take": self.take,
                "title": self.spec.get("title") or self.spec.get("id") or "Untitled take",
                "seed": self.spec.get("seed"), "cot": self.spec.get("cot"),
                "stages": list(self.stages.values()), "abc_partial": self.abc_partial,
                "cover_ready": self.cover_ready}

    def push(self, force=False):
        now = time.monotonic()
        if force or now - self._last_push > 0.18:
            self._last_push = now
            BUS.publish("job", self.snapshot())


JOBS = {}
JOB_ORDER = []
WORK = queue.Queue()


class _StageProxy:
    """Stands in for yue2.progress._Stage so run progress reaches the browser."""

    def __init__(self, job, label, total, unit):
        self.job = job
        self.key = LABEL_TO_STAGE.get(label)
        self.start = time.monotonic()
        if self.key is None:
            job.setup = label
            job.setup_at = time.time()
            job.push(force=True)
            return
        entry = job.stages[self.key]
        entry.update(state="running", completed=0, total=total, unit=unit, seconds=0.0)
        job.stage = self.key
        job.setup = ""
        job.push(force=True)

    def _tick(self, force=False):
        if self.key is None:
            return
        self.job.stages[self.key]["seconds"] = time.monotonic() - self.start
        self.job.push(force=force)

    def advance(self, count=1):
        if self.key is None:
            return
        self.job.stages[self.key]["completed"] += count
        self._tick()

    def update(self, completed, total=None):
        if self.key is None:
            return
        entry = self.job.stages[self.key]
        entry["completed"] = completed
        if total is not None:
            entry["total"] = total
        self._tick()

    def token(self, phase, token):
        self.advance()

    def finish(self, status="completed"):
        if self.key is None:
            return
        self.job.stages[self.key]["state"] = status
        self._tick(force=True)

    def close(self):
        if self.key is None:
            self.job.setup = ""
            self.job.setup_at = None
            self.job.push(force=True)
            return
        entry = self.job.stages[self.key]
        if entry["state"] == "running":
            entry["state"] = "completed"
        self._tick(force=True)


def _status_factory(job):
    @contextlib.contextmanager
    def _status(label, *, total=None, unit=None):
        stage = _StageProxy(job, label, total, unit)
        try:
            yield stage
        finally:
            stage.close()

    return _status


def _sampling(overrides, base):
    from yue2.protocol import Sampling
    names = {f.name for f in dataclasses.fields(Sampling)}
    values = {k: v for k, v in (overrides or {}).items() if k in names and v is not None}
    return dataclasses.replace(base, **values) if values else None


def is_oom(exc):
    """True for a CUDA out-of-memory error, whatever wrapper it arrives in."""
    if exc.__class__.__name__ in {"OutOfMemoryError", "CudaOutOfMemoryError"}:
        return True
    return "out of memory" in str(exc).lower()


def run_job(job):
    from yue2.protocol import GenerationConfig

    job.state = "running"
    job.started = time.time()
    job.push(force=True)
    ENGINE.status = "busy"
    BUS.publish("engine", ENGINE.describe())
    spec = job.spec

    if SETTINGS.art_auto and spec.get("cover") and ART.exe() is not None and art_model_path():
        # The GPU is free right now; once YuE2 loads it is busy for minutes.
        job.setup = "Drawing the cover"
        job.setup_at = time.time()
        job.push(force=True)
        with contextlib.suppress(Exception):
            job.cover_seconds = ART.draw(art_model_path(), spec["cover"],
                                         PENDING_COVER, seed=int(spec["seed"]))["seconds"]
            job.cover_ready = PENDING_COVER.is_file()
        job.setup = ""
        job.setup_at = None
        job.push(force=True)

    if ENGINE.pipe is None:
        # Model resolution happens before the progress reporter is swapped in.
        # Only promise a download when the weights are genuinely absent; after a
        # server restart they are on disk and nothing is fetched.
        job.setup = ("Loading model and decoder into VRAM" if weights_on_disk()
                     else "Loading model and decoder (downloading weights, about 7 GB)")
        job.setup_at = time.time()
        job.push(force=True)
    pipe = ENGINE.ensure()
    job.setup = ""
    job.setup_at = None
    pipe._status = _status_factory(job)
    defaults = GenerationConfig()

    def on_token(phase, token):
        if phase != "abc" or spec.get("abc"):
            return
        job.abc_tokens.append(int(token))
        if len(job.abc_tokens) % 16 == 0:
            with contextlib.suppress(Exception):
                job.abc_partial = pipe.tokenizer.decode(job.abc_tokens)

    semantic_overrides = dict(spec.get("semantic_sampling") or {})
    def render():
        return pipe(
            style=spec["style"],
            lyrics=spec["lyrics"],
            cot=spec.get("cot", "full"),
            seed=int(spec["seed"]),
            id=spec.get("id", "song"),
            abc=spec.get("abc") or None,
            cfg_scale=spec.get("cfg_scale"),
            abc_sampling=_sampling(spec.get("abc_sampling"), defaults.abc),
            semantic_sampling=_sampling(semantic_overrides, defaults.semantic),
            cancelled=job.cancel_flag.is_set,
            on_token=on_token,
        )

    try:
        result = render()
    except Exception as exc:
        if is_oom(exc) and not job.cancel_flag.is_set():
            # The protocol pins the acoustic context to 24576, so there is no
            # smaller setting to fall back to. Say what does help instead of
            # retrying something that cannot work.
            raise RuntimeError(
                "Ran out of VRAM while synthesising. Synthesis memory grows with the "
                "song, so the usual fixes are a shorter lyric, closing other GPU "
                "programs, or switching quantization to fp8 under Engine. "
                "Original error: " + str(exc)) from exc
        raise

    folder = time.strftime("%Y%m%d-%H%M%S") + "-" + spec.get("id", "song")
    directory = OUTPUTS / folder
    summary = result.save_artifacts(directory)
    meta = {"title": spec.get("title") or spec.get("id", "song"),
            "style": spec["style"], "lyrics": spec["lyrics"], "cot": spec.get("cot", "full"),
            "seed": int(spec["seed"]), "created": time.time(), "job": job.id,
            "seconds": summary["audio_seconds"], "truncated": summary["truncated"],
            "timing": summary["timing"], "identity": summary["identity"],
            "cover_prompt": spec.get("cover", ""),
            "vae": SETTINGS.vae, "backend": SETTINGS.backend,
            "provided_score": bool(spec.get("abc"))}
    (directory / "webui.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if PENDING_COVER.is_file():
        with contextlib.suppress(Exception):
            shutil.move(str(PENDING_COVER), str(directory / "cover.png"))
    job.take = folder
    job.abc_partial = result.abc or ""
    if SETTINGS.art_auto and not (directory / "cover.png").is_file()             and ART.exe() is not None and art_model_path():
        with contextlib.suppress(Exception):
            draw_cover(folder)


def worker():
    while True:
        job = WORK.get()
        if job.cancel_flag.is_set():
            job.state = "cancelled"
            job.finished = time.time()
            job.push(force=True)
            WORK.task_done()
            continue
        try:
            run_job(job)
            job.state = "done"
        except InterruptedError:
            job.state = "cancelled"
        except BaseException as exc:  # surface the real reason in the browser
            if job.cancel_flag.is_set():
                job.state = "cancelled"
            else:
                job.state = "failed"
                job.error = "%s: %s" % (type(exc).__name__, exc)
                traceback.print_exc()
        finally:
            job.finished = time.time()
            for entry in job.stages.values():
                if entry["state"] == "running":
                    entry["state"] = "cancelled" if job.state == "cancelled" else "failed"
            job.stage = None
            job.setup = ""
            job.push(force=True)
            if ENGINE.pipe is not None:
                with contextlib.suppress(Exception):
                    del ENGINE.pipe._status
            # Hand the VRAM back before the next run, however this one ended.
            ENGINE.park()
            ENGINE.status = "idle"
            BUS.publish("engine", ENGINE.describe())
            BUS.publish("library", {"reason": "run", "job": job.id})
            WORK.task_done()


threading.Thread(target=worker, daemon=True, name="yue2-worker").start()


# ----------------------------------------------------------------------------- muse
# Turn a one-line idea into a YuE2 brief using a local Ollama model. The model is
# asked to unload the moment it answers: the song model needs the same VRAM.

MUSE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Song title, two to four words, no quotes"},
        "style": {"type": "string", "description":
                  "One comma-separated line naming, in order: language, genre and mood, "
                  "voice type, two to four instruments, phrasing, and a tempo in BPM. "
                  "Never a sentence. Example: 'English, warm piano pop, expressive female "
                  "voice, acoustic piano, rounded bass and light drums, unhurried phrasing, 88 BPM'"},
        "cover": {"type": "string", "description":
                  "The album cover as a picture: subject, setting, light, colour palette "
                  "and art style, in the form the user's message asks for. Never mention "
                  "the music itself, and never ask for text, titles or lettering in the image"},
        "lyrics": {"type": "string", "description":
                   "A complete song of 24 to 32 sung lines using [Verse], [Chorus] and "
                   "[Bridge] tags, each tag alone on its line, the chorus repeated word "
                   "for word. No title line, no commentary"},
    },
    "required": ["title", "style", "cover", "lyrics"],
}

MUSE_SYSTEM = """You write briefs for YuE2, a song generation model.

You are given a one-line idea. Return exactly four fields: title, style, cover and lyrics.

style is a control string, not prose. It is a single comma-separated line listing,
in this order: language, genre and mood, voice type, two to four instruments,
phrasing, tempo in BPM. Write it the way a producer writes a track sheet.

Good style: English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, unhurried phrasing, 88 BPM
Good style: Mandarin, late-night jazz ballad, smoky male voice, upright bass, brushed drums, Rhodes, laid-back phrasing, 72 BPM
Bad style: instruments
Bad style: A beautiful song about love that features piano.

lyrics must be a COMPLETE song, never a sketch. The user's message gives the
section order to follow; follow it exactly, in that order, with no extra sections.

Rules for lyrics:
- Four to eight words per line, singable in one breath.
- Each section tag sits alone on its own line, spelled exactly as given.
- Repeat the chorus verbatim every time; never reword it.
- A pre-chorus is two lines that lift into the chorus, the same both times.
- Write in the language named in style.
- No title line, no commentary, no explanations.

cover is the album artwork described as a picture. Describe what is seen: the
subject, where it is, the light, the colour palette and the art style. Never name
an instrument, a genre, a tempo or the song title, and never ask for text,
letters or a title inside the image. The user's message says which shape to
write it in.

Write like a person who was there, not like a machine describing a mood.

- Name concrete things: an object, a place, a time, something someone did.
  "Your coat still on the hook" beats "memories of you".
- One image per line, and let it carry the feeling instead of naming it.
  Write the evidence, not the emotion.
- Never use these, in any language: neon, echoes, whispers, dancing shadows,
  city lights, fading light, endless night, burning desire, fire inside,
  broken heart, shattered dreams, breaking chains, spreading wings, endless
  road, chasing dreams, weathering the storm, hidden scars, lost in time,
  rising from the ashes, standing tall, feeling alive.
- Avoid rhyming on fire/desire, night/light, heart/apart, rain/pain.

title must never be empty, and cover must never be empty.

This is the exact shape of the lyrics field, tags included:

[Verse]
Neon fades along the lane
Footsteps keep the time of rain
[Chorus]
Let the day come into view
Every road begins with you

Answer with JSON only. No commentary, no markdown fences."""

RETRY_NOTE = ("Your previous answer had no section tags. Rewrite the lyrics with the section "
              "order given above, each tag alone on its own line, spelled exactly as shown, "
              "for example a line containing only [Chorus].")

CLICHE_NOTE = ("Your previous answer used these worn-out phrases: %s. Rewrite the lyrics "
               "without them and without any near-synonym of them. Replace each one with a "
               "concrete detail: a named object, a place, a time of day, or something "
               "somebody actually does. Keep the same structure, section tags and language.")


def has_sections(lyrics):
    return any(line.strip().startswith("[") for line in str(lyrics).splitlines())

# The score vocabulary documents verse, chorus, bridge and interlude. Pre-chorus,
# intro and outro are passed through as written and are not attested tags.
STRUCTURES = {
    "simple": ["[Verse]", "[Chorus]", "[Verse]", "[Chorus]"],
    "bridge": ["[Verse]", "[Chorus]", "[Verse]", "[Chorus]", "[Bridge]", "[Chorus]"],
    "prechorus": ["[Verse]", "[Pre-Chorus]", "[Chorus]", "[Verse]", "[Pre-Chorus]",
                  "[Chorus]", "[Bridge]", "[Chorus]"],
    "full": ["[Intro]", "[Verse]", "[Pre-Chorus]", "[Chorus]", "[Verse]", "[Pre-Chorus]",
             "[Chorus]", "[Bridge]", "[Chorus]", "[Outro]"],
}
DEFAULT_STRUCTURE = "bridge"

# Phrases that turn up in AI lyrics again and again. Collected from what
# songwriting communities and lyric checkers actually complain about, not from
# taste: "neon", "echoes", "shadows", "whispers" and the stock metaphors below
# are the ones people name when they say a song sounds machine-written.
CLICHES = [
    # the notorious four
    "neon", "echoes", "echo of", "whisper", "shadows dance", "dancing in the shadow",
    "dance in the shadow", "chasing shadows", "city lights",
    # light and dark
    "fading light", "endless night", "into the night", "dead of night",
    "burning bright", "blinding light", "silver moon", "moonlit",
    # fire
    "burning desire", "fire inside", "hearts on fire", "flames of",
    "rise from the ashes", "phoenix", "set the night on fire",
    # breakage
    "broken heart", "shattered dreams", "shattered glass", "picking up the pieces",
    "break these chains", "breaking free", "spread my wings", "silent scream",
    # journeys
    "endless road", "path unknown", "long road", "chasing dreams", "no turning back",
    # weather
    "weather the storm", "drowning in", "eye of the storm", "tears like rain",
    "storm inside",
    # pain and time
    "hidden scars", "unseen tears", "lost in time", "memories fade", "frozen in time",
    "against all odds", "rise again", "stand tall", "feel alive", "come alive",
    # synth-pop filler
    "electric dreams", "concrete jungle", "velvet sky", "crimson sky",
    # the same in German, since songs here are often written in it
    "neonlicht", "neonlichter", "im schatten tanz", "tanz im schatten",
    "zerbrochene träume", "gebrochenes herz", "ketten sprengen",
    "asche", "flügel", "sterne verglühen", "endlose nacht", "im regen stehen",
]


# The ones people name first when a song sounds machine-written. A single
# occurrence of any of these is worth a rewrite; the rest only matter in bulk.
WORST = {"neon", "echoes", "whisper", "shadows dance", "dancing in the shadow",
         "city lights", "neonlicht", "neonlichter"}


def find_cliches(text):
    lowered = str(text).lower()
    return sorted({phrase for phrase in CLICHES if phrase in lowered})


THINK_OPEN, THINK_CLOSE = "<think>", "</think>"

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so the sibling module
from llamacpp import LlamaCpp  # noqa: E402  imports however the app is launched
from sdcpp import StableDiffusionCpp  # noqa: E402

LLAMA = LlamaCpp(ROOT, on_event=BUS.publish)
ART = StableDiffusionCpp(ROOT, on_event=BUS.publish)


def sync_model_dirs():
    LLAMA.extra_dirs = list(SETTINGS.writer_dirs or [])
    ART.extra_dirs = list(SETTINGS.art_dirs or [])


def writer_backend():
    """Which writer to use. 'auto' prefers a local llama.cpp install, since it
    needs nothing installed system-wide, and falls back to Ollama."""
    choice = (SETTINGS.writer_backend or "auto").lower()
    if choice in {"ollama", "llamacpp"}:
        return choice
    if LLAMA.server_exe() is not None and (SETTINGS.llamacpp_model or LLAMA.status()["models"]):
        return "llamacpp"
    return "ollama"


def llamacpp_model_path():
    sync_model_dirs()
    return LLAMA.resolve_choice(SETTINGS.llamacpp_model)


ART_GUIDANCE = {
    "natural": ("The cover model reads prompts with a language model, so write cover as two or "
                "three full sentences: name the subject, the setting, the light, the colour "
                "palette and the art style, in that order. Be concrete and visual."),
    "tags": ("The cover model reads prompts with CLIP and only sees about 75 tokens, so write "
             "cover as a short comma-separated list of visual tags, strongest first, no "
             "sentences. Around twelve tags."),
}


def _art_request():
    """Tell the writer which image model its cover line has to feed."""
    try:
        sync_model_dirs()
        style = ART.prompt_style(SETTINGS.art_model)
    except Exception:
        style = "natural"
    return "\n\n" + ART_GUIDANCE.get(style, ART_GUIDANCE["natural"])


def _structure_request(structure):
    sections = STRUCTURES.get(structure or DEFAULT_STRUCTURE, STRUCTURES[DEFAULT_STRUCTURE])
    lines = ["\n\nSection order, exactly:"]
    for section in sections:
        if section in ("[Intro]", "[Outro]"):
            lines.append("%s — instrumental: write the tag alone, with no words under it" % section)
        elif section == "[Bridge]":
            lines.append("[Bridge] — two to four lines, new words")
        elif section == "[Pre-Chorus]":
            lines.append("[Pre-Chorus] — two lines, identical each time")
        elif section == "[Chorus]":
            lines.append("[Chorus] — four lines, identical each time")
        else:
            lines.append("[Verse] — four lines, new words each time")
    return "\n".join(lines)


def _ollama(path, payload=None, timeout=900):
    import urllib.request
    url = SETTINGS.ollama_url.rstrip("/") + path
    if payload is None:
        request = urllib.request.Request(url)
    else:
        request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _strip_thinking(text):
    """Reasoning models may narrate before the JSON; keep the JSON."""
    while THINK_OPEN in text and THINK_CLOSE in text:
        start = text.index(THINK_OPEN)
        text = text[:start] + text[text.index(THINK_CLOSE) + len(THINK_CLOSE):]
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else text


def ollama_models():
    try:
        listed = _ollama("/api/tags", timeout=15).get("models", [])
    except Exception as exc:
        return {"available": False, "error": str(exc), "models": [], "loaded": []}
    models = [{"name": m.get("name", ""), "size_gib": round(m.get("size", 0) / 2 ** 30, 2)} for m in listed]
    models.sort(key=lambda m: m["name"])
    try:
        loaded = [m.get("name", "") for m in _ollama("/api/ps", timeout=15).get("models", [])]
    except Exception:
        loaded = []
    return {"available": True, "models": models, "loaded": loaded}


def default_muse_model(models):
    names = [m["name"] for m in models]
    for wanted in ("qwen3.8", "qwen3", "qwen"):
        for name in names:
            if wanted in name.lower():
                return name
    return names[0] if names else ""


def muse_unload(model):
    """Ask Ollama to drop the model now, then confirm it is gone."""
    with contextlib.suppress(Exception):
        _ollama("/api/generate", {"model": model, "prompt": "", "keep_alive": 0, "stream": False}, timeout=120)
    for _ in range(10):
        try:
            loaded = [m.get("name", "") for m in _ollama("/api/ps", timeout=10).get("models", [])]
        except Exception:
            return False
        if model not in loaded:
            return True
        time.sleep(1)
    return False


# ---------------------------------------------------------------------- transcribe
# Audio in, score out. SheetSage2 pins different dependency versions to YuE2, so
# it lives in its own environment and the two exchange files, exactly as
# docs/covers.md describes. YuE2 itself takes no audio input of any kind.

SHEETSAGE_VENV = ROOT / ".venv-sheetsage2"
SHEETSAGE_MODEL = ROOT / "models" / "SheetSage2"
UPLOADS = OUTPUTS / "_uploads"
# A cover drawn before the run has nowhere to live yet; it waits here.
PENDING_COVER = OUTPUTS / "_pending-cover.png"


def sheetsage_python():
    for candidate in (SHEETSAGE_VENV / "Scripts" / "python.exe", SHEETSAGE_VENV / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


def sheetsage_status():
    python = sheetsage_python()
    infer = SHEETSAGE_MODEL / "infer.py"
    return {"available": bool(python and infer.is_file()),
            "python": str(python) if python else None,
            "model": str(SHEETSAGE_MODEL) if infer.is_file() else None,
            "env_dir": str(SHEETSAGE_VENV), "model_dir": str(SHEETSAGE_MODEL)}


def strip_chord_symbols(abc):
    """Drop "Dm"-style chord symbols so a cover's accompaniment can be rebuilt.

    docs/covers.md asks for a score without chord symbols in melody mode; ABC
    puts them in double quotes ahead of the note they colour. Information fields
    quote things too -- V: lines carry name="Vocal Melody" -- so only music lines
    are touched, or the header would lose its voice names.
    """
    out = []
    for line in abc.split("\n"):
        if re.match(r"\s*[A-Za-z]:", line) or line.lstrip().startswith("%"):
            out.append(line)
        else:
            out.append(re.sub(r'"[^"\n]*"', "", line))
    return "\n".join(out)


def transcribe_audio(source: Path, destination: Path, melody_only=True):
    """Run SheetSage2 in its own interpreter and return the ABC it wrote."""
    status = sheetsage_status()
    if not status["available"]:
        raise HTTPException(
            501, "SheetSage2 is not installed. Transcription needs its own environment "
                 "(Python 3.10/3.11, torch 2.8+cu126, FFmpeg 6.1) — see webui/README.md.")
    command = [status["python"], str(SHEETSAGE_MODEL / "infer.py"), str(source),
               "--output", str(destination)]
    if melody_only:
        command.append("--melody-only")
    finished = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    score = destination / "score.abc"
    if finished.returncode != 0 or not score.is_file():
        detail = (finished.stderr or finished.stdout or "").strip()[-1200:]
        raise HTTPException(502, "SheetSage2 could not transcribe this audio.\n" + detail)
    return score.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- library

def read_take(directory):
    result_file = directory / "result.json"
    if not result_file.is_file():
        return None
    try:
        result = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    meta, request = {}, {}
    if (directory / "webui.json").is_file():
        with contextlib.suppress(Exception):
            meta = json.loads((directory / "webui.json").read_text(encoding="utf-8"))
    if (directory / "request.json").is_file():
        with contextlib.suppress(Exception):
            request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    score = ""
    if (directory / "score.abc").is_file():
        with contextlib.suppress(Exception):
            score = (directory / "score.abc").read_text(encoding="utf-8")
    timing = result.get("timing", {})
    return {
        "name": directory.name,
        "title": meta.get("title") or request.get("id") or directory.name,
        "style": meta.get("style") or request.get("style", ""),
        "lyrics": meta.get("lyrics") or request.get("lyrics", ""),
        "cot": meta.get("cot") or request.get("cot", "full"),
        "seed": meta.get("seed", request.get("seed")),
        "created": meta.get("created") or directory.stat().st_mtime,
        "seconds": result.get("audio_seconds", 0),
        "sample_rate": result.get("sample_rate", 48000),
        "truncated": result.get("truncated", {}),
        "identity": result.get("identity", ""),
        "vae": meta.get("vae", "standard"),
        "provided_score": meta.get("provided_score", False),
        "cover_prompt": meta.get("cover_prompt", ""),
        "score": score,
        "e2e_seconds": timing.get("e2e_seconds"),
        "has_audio": (directory / "audio.flac").is_file(),
        "has_cover": (directory / "cover.png").is_file(),
    }


def library():
    takes = []
    if OUTPUTS.is_dir():
        for entry in OUTPUTS.iterdir():
            if entry.is_dir():
                take = read_take(entry)
                if take:
                    takes.append(take)
    takes.sort(key=lambda t: t["created"], reverse=True)
    return takes


def take_dir(name):
    directory = (OUTPUTS / name).resolve()
    if directory.parent != OUTPUTS or not directory.is_dir():
        raise HTTPException(404, "No such take")
    return directory


# --------------------------------------------------------------------------- api

app = FastAPI(title="YuE2 Console")


class GenerateBody(BaseModel):
    style: str
    lyrics: str
    cover: str = ""
    cot: str = "full"
    seed: int | None = None
    id: str = "song"
    title: str = ""
    abc: str | None = None
    cfg_scale: float | None = None
    abc_sampling: dict = Field(default_factory=dict)
    semantic_sampling: dict = Field(default_factory=dict)


class MuseBody(BaseModel):
    idea: str
    backend: str | None = None
    model: str | None = None
    free_engine: bool | None = None
    structure: str | None = None


class SettingsBody(BaseModel):
    model: str | None = None
    vae: str | None = None
    device: str | None = None
    backend: str | None = None
    quantization: str | None = None
    memory_budget_gib: float | None = None
    offload_ar: bool | None = None
    ode_steps: int | None = None
    offline: bool | None = None
    ollama_url: str | None = None
    muse_model: str | None = None
    muse_free_engine: bool | None = None
    writer_backend: str | None = None
    llamacpp_model: str | None = None
    llamacpp_gpu_layers: int | None = None
    art_model: str | None = None
    art_auto: bool | None = None
    writer_dirs: list[str] | None = None
    art_dirs: list[str] | None = None


def hardware():
    info = {"torch": None, "cuda": False, "gpu": None, "vram_gib": None, "bf16": False}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["vram_gib"] = round(props.total_memory / 2 ** 30, 1)
            info["bf16"] = bool(torch.cuda.is_bf16_supported())
    except Exception as exc:
        info["error"] = str(exc)
    return info


def defaults_payload():
    from yue2.protocol import GenerationConfig
    config = GenerationConfig()
    return {"abc": dataclasses.asdict(config.abc), "semantic": dataclasses.asdict(config.semantic),
            "ode_steps": config.ode_steps}


@app.get("/api/state")
def state():
    return {"engine": ENGINE.describe(), "settings": SETTINGS.to_dict(), "hardware": hardware(),
            "vram": vram(), "budget_gib": resolved_budget(),
            "bundled_weights": (BUNDLED / "YuE2-3B" / "config.json").is_file(),
            "defaults": defaults_payload(), "outputs": str(OUTPUTS),
            "jobs": [JOBS[j].snapshot() for j in JOB_ORDER[-12:]]}


@app.get("/api/library")
def api_library():
    return {"takes": library()}


@app.get("/api/library/{name}/audio")
def api_audio(name: str):
    path = take_dir(name) / "audio.flac"
    if not path.is_file():
        raise HTTPException(404, "This take has no audio file")
    return FileResponse(path, media_type="audio/flac", filename=name + ".flac")


@app.get("/api/library/{name}/file/{filename}")
def api_file(name: str, filename: str):
    if filename not in {"score.abc", "request.json", "result.json", "config.json", "webui.json", "plan.json"}:
        raise HTTPException(404, "Not an exportable artifact")
    path = take_dir(name) / filename
    if not path.is_file():
        raise HTTPException(404, "Artifact missing from this take")
    return FileResponse(path, filename=name + "-" + filename)


@app.delete("/api/library/{name}")
def api_delete(name: str):
    shutil.rmtree(take_dir(name))
    BUS.publish("library", {"reason": "delete", "name": name})
    return {"ok": True}


@app.post("/api/generate")
def api_generate(body: GenerateBody):
    if not body.style.strip() or not body.lyrics.strip():
        raise HTTPException(400, "A style prompt and lyrics are both required")
    spec = body.model_dump()
    spec["seed"] = random.randrange(2 ** 31) if body.seed is None else int(body.seed)
    spec["title"] = body.title.strip() or body.id
    job = Job(spec)
    JOBS[job.id] = job
    JOB_ORDER.append(job.id)
    WORK.put(job)
    job.push(force=True)
    return {"job": job.snapshot()}


@app.post("/api/jobs/{job_id}/cancel")
def api_cancel(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "No such run")
    job.cancel_flag.set()
    job.push(force=True)
    return {"ok": True}


@app.post("/api/settings")
def api_settings(body: SettingsBody):
    if ENGINE.status in {"busy", "loading"}:
        raise HTTPException(409, "Finish or cancel the current run before changing the engine")
    # Only weights and compute settings invalidate a loaded pipeline; the writer
    # settings do not, so changing them must not cost a model reload.
    for key in ("writer_dirs", "art_dirs"):
        value = getattr(body, key)
        if value is not None:
            setattr(SETTINGS, key, [str(v) for v in value if str(v).strip()])
            save_settings()
            sync_model_dirs()
    engine_keys = {"model", "vae", "device", "backend", "quantization",
                   "memory_budget_gib", "offload_ar", "ode_steps", "offline"}
    changed, reload_needed = False, False
    for key, value in body.model_dump(exclude_none=True).items():
        if getattr(SETTINGS, key) != value:
            setattr(SETTINGS, key, value)
            changed = True
            reload_needed = reload_needed or key in engine_keys
    if changed:
        save_settings()
    if reload_needed:
        ENGINE.unload()
        BUS.publish("engine", ENGINE.describe())
    return {"settings": SETTINGS.to_dict(), "saved": changed, "reloaded": reload_needed}


@app.post("/api/load")
def api_load():
    if ENGINE.status in {"busy", "loading"}:
        raise HTTPException(409, "The engine is already working")

    def go():
        try:
            ENGINE.ensure()
        except Exception as exc:
            ENGINE.status = "error"
            ENGINE.detail = "%s: %s" % (type(exc).__name__, exc)
            BUS.publish("engine", ENGINE.describe())

    threading.Thread(target=go, daemon=True).start()
    return {"ok": True}


@app.get("/api/muse/models")
def api_muse_models():
    listing = ollama_models()
    selected = SETTINGS.muse_model or default_muse_model(listing["models"])
    return dict(listing, selected=selected, free_engine=SETTINGS.muse_free_engine)


@app.post("/api/muse")
def api_muse(body: MuseBody):
    idea = body.idea.strip()
    if not idea:
        raise HTTPException(400, "Describe the song in a line or two first")
    if ENGINE.status in {"busy", "loading"}:
        raise HTTPException(409, "The song model is working; wait for the run to finish")

    backend = (body.backend or writer_backend()).lower()
    messages = [{"role": "system", "content": MUSE_SYSTEM},
                {"role": "user", "content": "Idea: " + idea + _structure_request(body.structure) +
                                            _art_request()}]

    free_engine = SETTINGS.muse_free_engine if body.free_engine is None else body.free_engine
    freed = False
    if free_engine and ENGINE.pipe is not None:
        # A writer and a resident YuE2 do not both fit on one consumer card.
        ENGINE.unload()
        freed = True
        BUS.publish("engine", ENGINE.describe())

    def ask(history):
        if backend == "llamacpp":
            return _muse_llamacpp(history)
        return _muse_ollama(history, body.model)

    start = time.perf_counter()
    content, model, extra = ask(messages)
    try:
        parsed = json.loads(_strip_thinking(content))
    except Exception:
        raise HTTPException(502, "The writer did not return usable JSON. Try again, or pick another model.")

    # Section tags are what YuE2 reads as structure. Smaller writers drop them,
    # so check and give the model one corrective pass before accepting the brief.
    retried = False
    note = None
    if not has_sections(parsed.get("lyrics", "")):
        note = RETRY_NOTE
    else:
        # Asking nicely is not enough on its own: check the words that come back
        # and give the model one chance to do better.
        found = find_cliches(parsed.get("lyrics", ""))
        if len(found) >= 2 or any(phrase in WORST for phrase in found):
            note = CLICHE_NOTE % ", ".join(found[:8])

    if note:
        retried = True
        second = messages + [{"role": "assistant", "content": content},
                             {"role": "user", "content": note}]
        try:
            content2, model, extra2 = ask(second)
            parsed2 = json.loads(_strip_thinking(content2))
            better = has_sections(parsed2.get("lyrics", "")) and (
                len(find_cliches(parsed2.get("lyrics", ""))) <=
                len(find_cliches(parsed.get("lyrics", ""))))
            if better:
                parsed, extra = parsed2, extra2
        except Exception:
            pass

    if freed:
        BUS.publish("engine", ENGINE.describe())

    lyrics = str(parsed.get("lyrics", "")).strip()
    sung = [line for line in lyrics.splitlines() if line.strip() and not line.strip().startswith("[")]
    title = str(parsed.get("title", "")).strip()
    if not title:
        # Writers occasionally return an empty title; a hook line beats "song".
        title = (sung[0] if sung else idea)[:60].strip(" ,.!?-")
    return dict({"title": title,
                 "style": str(parsed.get("style", "")).strip(),
                 "cover": str(parsed.get("cover", "")).strip(),
                 "lyrics": lyrics,
                 "sections": [l.strip() for l in lyrics.splitlines() if l.strip().startswith("[")],
                 "backend": backend, "model": model, "retried": retried,
                 "cliches": find_cliches(lyrics),
                 "seconds": round(time.perf_counter() - start, 1),
                 "lines": len(sung), "freed_engine": freed}, **extra)


def _muse_ollama(messages, requested):
    listing = ollama_models()
    if not listing["available"]:
        raise HTTPException(502, "Ollama is not answering at " + SETTINGS.ollama_url +
                            " - start it, switch the writer to llama.cpp, or change the address under Engine")
    model = (requested or SETTINGS.muse_model or default_muse_model(listing["models"])).strip()
    if not model:
        raise HTTPException(400, "No Ollama model is installed - pull one first")

    payload = {"model": model, "stream": False, "keep_alive": 0, "format": MUSE_SCHEMA,
               "options": {"temperature": 0.9, "top_p": 0.95, "num_ctx": 8192, "num_predict": 4096},
               "messages": messages}
    try:
        # Reasoning costs ~10x the tokens here and writes a worse brief.
        try:
            response = _ollama("/api/chat", dict(payload, think=False))
        except Exception:
            response = _ollama("/api/chat", payload)
    except Exception as exc:
        muse_unload(model)
        raise HTTPException(502, "Ollama call failed: %s: %s" % (type(exc).__name__, exc))

    content = (response.get("message") or {}).get("content", "")
    unloaded = muse_unload(model)
    if SETTINGS.muse_model != model:
        SETTINGS.muse_model = model
        save_settings()
    return content, model, {"unloaded": unloaded, "tokens": response.get("eval_count"),
                            "done_reason": response.get("done_reason")}


def _muse_llamacpp(messages):
    if LLAMA.server_exe() is None:
        raise HTTPException(501, "llama.cpp is not installed yet - install it under Engine")
    path = llamacpp_model_path()
    if not path:
        raise HTTPException(400, "No writer model downloaded yet - pick one under Engine")
    try:
        LLAMA.start(path, gpu_layers=SETTINGS.llamacpp_gpu_layers)
        content, usage = LLAMA.chat(messages, schema=MUSE_SCHEMA)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, "llama.cpp call failed: %s: %s" % (type(exc).__name__, exc))
    finally:
        # Killing the process is the unload; nothing of it survives.
        unloaded = LLAMA.stop()
        BUS.publish("writer", {"busy": ""})
    return content, Path(path).name, {"unloaded": unloaded,
                                      "tokens": (usage or {}).get("completion_tokens"),
                                      "done_reason": "stop"}


@app.get("/api/writer/status")
def api_writer_status():
    sync_model_dirs()
    listing = ollama_models()
    return {"backend": writer_backend(), "configured": SETTINGS.writer_backend,
            "ollama": dict(listing, selected=SETTINGS.muse_model or default_muse_model(listing["models"])),
            "llamacpp": dict(LLAMA.status(), selected=SETTINGS.llamacpp_model,
                             dirs=list(SETTINGS.writer_dirs or [])),
            "free_engine": SETTINGS.muse_free_engine}


@app.post("/api/writer/llamacpp/install")
def api_llamacpp_install():
    if LLAMA.busy:
        raise HTTPException(409, "Already busy: " + LLAMA.busy)

    def go():
        try:
            LLAMA.install()
            BUS.publish("writer", {"busy": "", "installed": True})
        except Exception as exc:
            BUS.publish("writer", {"busy": "", "error": "%s: %s" % (type(exc).__name__, exc)})

    threading.Thread(target=go, daemon=True).start()
    return {"started": True}


@app.post("/api/writer/llamacpp/model")
def api_llamacpp_model(model_id: str = Form(...)):
    if LLAMA.busy:
        raise HTTPException(409, "Already busy: " + LLAMA.busy)

    def go():
        try:
            result = LLAMA.download_model(model_id)
            if not SETTINGS.llamacpp_model:
                SETTINGS.llamacpp_model = result["file"]
                save_settings()
            BUS.publish("writer", {"busy": "", "downloaded": result["file"]})
        except Exception as exc:
            BUS.publish("writer", {"busy": "", "error": "%s: %s" % (type(exc).__name__, exc)})

    threading.Thread(target=go, daemon=True).start()
    return {"started": True}


@app.delete("/api/writer/llamacpp/model/{filename}")
def api_llamacpp_model_delete(filename: str):
    try:
        LLAMA.delete_model(filename)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    if SETTINGS.llamacpp_model == filename:
        SETTINGS.llamacpp_model = ""
        save_settings()
    BUS.publish("writer", {"busy": "", "deleted": filename})
    return {"ok": True}


@app.get("/api/cover/status")
def api_cover_status():
    return {"sheetsage": sheetsage_status(),
            "takes": [{"name": t["name"], "title": t["title"], "has_score": bool(t["score"])}
                      for t in library() if t["score"]]}


@app.post("/api/cover/from-take")
def api_cover_from_take(name: str = Form(...), melody_only: bool = Form(True)):
    """Reuse a finished take's score so it can be rendered in another style."""
    take = read_take(take_dir(name))
    if not take or not take["score"]:
        raise HTTPException(404, "That take has no score to cover; it was made in Direct mode")
    score = take["score"]
    if melody_only:
        score = strip_chord_symbols(score)
    return {"abc": score, "lyrics": take["lyrics"], "style": take["style"],
            "title": take["title"], "chords_removed": melody_only}


@app.post("/api/cover/from-audio")
async def api_cover_from_audio(file: UploadFile = File(...), melody_only: bool = Form(True)):
    """Transcribe an uploaded recording into a score with SheetSage2.

    Kept working but deliberately not surfaced in the console: SheetSage2 needs
    its own environment and an FFmpeg 6.x whose shared libraries torchaudio can
    bind, which is a lot of setup for the one case it serves -- covering someone
    else's recording. Remixing a take you already made needs none of it.
    """
    if ENGINE.status in {"busy", "loading"}:
        raise HTTPException(409, "The song model is working; transcription needs the same GPU")
    suffix = Path(file.filename or "source.wav").suffix.lower()
    if suffix not in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac"}:
        raise HTTPException(400, "Upload a wav, flac, mp3, m4a, ogg, opus or aac file")
    UPLOADS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    source = UPLOADS / (stamp + suffix)
    destination = UPLOADS / (stamp + "-score")
    with source.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    # SheetSage2 wants the GPU to itself, so park YuE2 first.
    ENGINE.park()
    start = time.perf_counter()
    abc = transcribe_audio(source, destination, melody_only=melody_only)
    return {"abc": abc, "source": source.name, "seconds": round(time.perf_counter() - start, 1),
            "melody_only": melody_only}


@app.post("/api/free")
def api_free(unload: bool = False):
    """Give the GPU back: park the weights, or drop them entirely."""
    if ENGINE.status in {"busy", "loading"}:
        raise HTTPException(409, "The engine is working; cancel the run first")
    before = vram()
    ENGINE.unload() if unload else ENGINE.park()
    BUS.publish("engine", ENGINE.describe())
    return {"before": before, "after": vram(), "unloaded": bool(unload)}


@app.get("/api/pending-cover")
def api_pending_cover():
    """The sleeve drawn before the song, so the run has something to show."""
    if not PENDING_COVER.is_file():
        raise HTTPException(404, "No cover is waiting")
    return FileResponse(PENDING_COVER, media_type="image/png",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/vram")
def api_vram():
    return {"vram": vram(), "engine": ENGINE.describe()}


@app.get("/api/art/status")
def api_art_status():
    sync_model_dirs()
    return dict(ART.status(), selected=SETTINGS.art_model, auto=SETTINGS.art_auto,
                dirs=list(SETTINGS.art_dirs or []))


@app.post("/api/art/install")
def api_art_install():
    if ART.busy:
        raise HTTPException(409, "Already busy: " + ART.busy)

    def go():
        try:
            ART.install()
            BUS.publish("art", {"busy": "", "installed": True})
        except Exception as exc:
            BUS.publish("art", {"busy": "", "error": "%s: %s" % (type(exc).__name__, exc)})

    threading.Thread(target=go, daemon=True).start()
    return {"started": True}


@app.post("/api/art/model")
def api_art_model(model_id: str = Form(...)):
    if ART.busy:
        raise HTTPException(409, "Already busy: " + ART.busy)

    def go():
        try:
            result = ART.download_model(model_id)
            if not SETTINGS.art_model:
                SETTINGS.art_model = result["file"]
                save_settings()
            BUS.publish("art", {"busy": "", "downloaded": result["file"]})
        except Exception as exc:
            BUS.publish("art", {"busy": "", "error": "%s: %s" % (type(exc).__name__, exc)})

    threading.Thread(target=go, daemon=True).start()
    return {"started": True}


@app.delete("/api/art/model/{filename}")
def api_art_model_delete(filename: str):
    try:
        ART.delete_model(filename)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    if SETTINGS.art_model == filename:
        SETTINGS.art_model = ""
        save_settings()
    BUS.publish("art", {"busy": "", "deleted": filename})
    return {"ok": True}


# Words in a style prompt that describe sound, not a picture. Matched as whole
# words: a substring test would throw away "synthwave" for containing "synth".
MUSIC_WORDS = {"voice", "vocal", "vocals", "guitar", "guitars", "piano", "bass", "drums",
               "drum", "synth", "synths", "strings", "rhodes", "percussion", "bpm",
               "phrasing", "tempo", "male", "female", "instrumental", "acoustic",
               "electric", "upright", "brushed", "gated", "sub", "analog", "backing"}
LANGUAGES = {"english", "german", "deutsch", "mandarin", "chinese", "spanish", "french",
             "japanese", "korean", "italian", "portuguese", "russian"}


def visual_from_style(style):
    """Turn a track sheet into something a diffusion model can draw.

    Only genre and mood survive; instruments, tempo and the language are dropped,
    because "urgent male voice, sub bass, 118 BPM" draws nothing at all.
    """
    keep = []
    for chunk in style.split(","):
        cleaned = chunk.strip()
        words = {w.strip(".;:-").lower() for w in cleaned.split()}
        if not cleaned or words & MUSIC_WORDS or words <= LANGUAGES:
            continue
        keep.append(cleaned)
    mood = ", ".join(keep[:3])
    return mood if mood else "abstract album artwork"


def art_model_path():
    sync_model_dirs()
    return ART.resolve_choice(SETTINGS.art_model)


def draw_cover(take_name, seed=-1):
    """Draw a sleeve for one take from its own style prompt and title."""
    directory = take_dir(take_name)
    take = read_take(directory)
    if not take:
        raise HTTPException(404, "No such take")
    if ART.exe() is None:
        raise HTTPException(501, "stable-diffusion.cpp is not installed yet - install it under Engine")
    path = art_model_path()
    if not path:
        raise HTTPException(400, "No art model downloaded yet - pick one under Engine")
    # A style prompt is a track sheet -- instruments and BPM -- which makes a poor
    # picture. Prefer the visual line the writer produced for this song.
    prompt = take.get("cover_prompt") or visual_from_style(take["style"])
    ENGINE.park()
    result = ART.draw(path, prompt, directory / "cover.png",
                      seed=take["seed"] if seed == -1 and take["seed"] else seed)
    BUS.publish("library", {"reason": "cover", "name": take_name})
    return dict(result, take=take_name, prompt=prompt)


@app.post("/api/art/cover/{name}")
def api_art_cover(name: str, seed: int = Form(-1)):
    if ENGINE.status in {"busy", "loading"}:
        raise HTTPException(409, "The song model is working; the picture needs the same GPU")
    try:
        return draw_cover(name, seed=seed)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, str(exc))


@app.get("/api/library/{name}/cover")
def api_cover_image(name: str):
    path = take_dir(name) / "cover.png"
    if not path.is_file():
        raise HTTPException(404, "This take has no cover yet")
    return FileResponse(path, media_type="image/png", filename=name + ".png")


@app.get("/api/examples")
def api_examples():
    out = {}
    song = ROOT / "examples" / "song.json"
    if song.is_file():
        out["song"] = json.loads(song.read_text(encoding="utf-8"))
    for key, filename in (("melody", "melody.abc"), ("score", "score.abc"), ("jazz", "score-jazz.abc")):
        path = ROOT / "examples" / filename
        if path.is_file():
            out[key] = path.read_text(encoding="utf-8")
    return out


@app.get("/api/events")
async def api_events(cursor: int = 0):
    async def stream():
        position = cursor
        yield "event: hello\ndata: " + json.dumps({"cursor": position}) + "\n\n"
        idle = 0
        while True:
            events, latest = BUS.since(position)
            if events:
                position = latest
                idle = 0
                for event in events:
                    yield "event: " + event["kind"] + "\ndata: " + json.dumps(event["data"]) + "\n\n"
            else:
                idle += 1
                if idle >= 60:      # keep browsers and proxies from closing the pipe
                    idle = 0
                    yield ": keepalive\n\n"
            await asyncio.sleep(0.2)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the page with its assets versioned, so an edited console never
    loads against a browser-cached script from an earlier build."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    stamp = str(max(int((STATIC / name).stat().st_mtime) for name in ("app.js", "app.css")))
    html = html.replace("/static/app.js", "/static/app.js?v=" + stamp)
    html = html.replace("/static/app.css", "/static/app.css?v=" + stamp)
    return HTMLResponse(html, headers={"Cache-Control": "no-store, must-revalidate"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main():
    import uvicorn
    host = os.environ.get("YUE2_HOST", "127.0.0.1")
    port = int(os.environ.get("YUE2_PORT", "7865"))
    print("YuE2 Console  ->  http://%s:%d" % (host, port), file=sys.stderr, flush=True)
    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("YUE2_LOG", "info"),
                access_log=True)


if __name__ == "__main__":
    main()
