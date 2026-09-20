"""A second song engine: audio.cpp, which runs YuE2 from quantized GGUF weights.

The PyTorch pipeline keeps the whole 3B model in BF16, so a run needs about
12.5 GiB and a smaller card runs out of memory part way through a song. The
audio.cpp build of the same model reads Q8_0 or Q4_0 weights and reports 8.9 and
7.8 GiB for the same take, which is the difference between finishing a song and
losing it. It is a separate process, so the VRAM is handed back completely once
the take is done.

The maximum song length itself does not change: both engines stop at 9000
semantic tokens on a 24576-token context. What changes is whether a small card
reaches that ceiling instead of an out-of-memory error on the way.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

GITHUB_API = "https://api.github.com/repos/0xShug0/audio.cpp"
GITHUB_RELEASES = GITHUB_API + "/releases?per_page=15"
WEIGHTS_REPO = "audio-cpp/Yue2-3B-GGUF"

# The main model, cheapest first. The VRAM figures are audio.cpp's own RTX 5090
# measurements for one three-minute take, each paired with the F16 decoder.
MAIN_WEIGHTS = [
    {"id": "q4_0", "file": "yue2-3b-q4_0.gguf", "size_gb": 2.5, "vram_gib": 7.8,
     "note": "Smallest. Held to be safe upstream; the quality is not formally validated."},
    {"id": "q8_0", "file": "yue2-3b-q8_0.gguf", "size_gb": 4.0, "vram_gib": 8.9,
     "note": "The balanced choice, and audio.cpp's own default."},
    {"id": "bf16", "file": "yue2-3b-bf16.gguf", "size_gb": 6.8, "vram_gib": 12.5,
     "note": "Unquantized: the same weights as the PyTorch engine, and no VRAM saved."},
]
VAE_WEIGHTS = [
    {"id": "f16", "file": "yue2-vae-f16.gguf", "size_gb": 0.25, "note": "Half precision decoder."},
    {"id": "f32", "file": "yue2-vae-f32.gguf", "size_gb": 0.5, "note": "Full precision decoder."},
]
# Config and tokenizer the runtime reads from the model folder. Not optional.
SIDECARS = ["sidecars/yue2-model-config.json", "sidecars/yue2-generation-config.json",
            "sidecars/yue2-qwen.tiktoken", "sidecars/yue2-vae-config.json"]

# Timing markers audio.cpp prints under --log. Most arrive as a phase ends, so
# they move the console on to the next stage rather than fill a bar.
TIMING = re.compile(r"^\[TIMING [^\]]*\]\s+(\S+)\s+(.*)$")
# audio.cpp says what it loaded at Info level. Keeping the line is the only
# evidence a finished take was really made with the adapter.
ADAPTER = re.compile(r"(yue2\.(?:nar|ar)_lora):\s*(.+?),\s*projections=(\d+),\s*scale=([\d.]+)")
PHASE_DONE = [
    ("yue2.semantic.abc_generated_tokens", "plan"),
    ("yue2.plan.abc_tokens", "plan"),
    ("yue2.semantic.tokens", "semantic"),
    ("yue2.nar.synthesize.chunks", "synthesize"),
    ("yue2.vae_decode.output_frames", "decode"),
]
# Printed once per chunk or tile, so these two do fill a bar.
PHASE_TICK = {"yue2.nar.chunk.total_ms": "synthesize",
              "yue2.vae_decode.tile_decode_ms": "decode"}


class AudioCppError(RuntimeError):
    pass


class AudioCpp:
    def __init__(self, root: Path, on_event=None):
        self.home = Path(root) / "audiocpp"
        self.bin_dir = self.home / "bin"
        self.models_dir = Path(root) / "models" / "Yue2-3B-GGUF"
        self.on_event = on_event or (lambda kind, payload: None)
        self.override = ""
        self.busy = ""
        self.lock = threading.Lock()
        self.process = None
        self._families = {}

    # ------------------------------------------------------------- install

    def exe_name(self):
        return "audiocpp_cli.exe" if os.name == "nt" else "audiocpp_cli"

    def cli_exe(self):
        """A build you point at wins; otherwise the one downloaded here."""
        if self.override:
            candidate = Path(self.override).expanduser()
            if candidate.is_dir():
                candidate = candidate / self.exe_name()
            return candidate if candidate.is_file() else None
        candidate = self.bin_dir / self.exe_name()
        return candidate if candidate.is_file() else None

    def supports_yue2(self, exe=None):
        """Ask the binary itself: a build from before YuE2 landed will say no."""
        exe = exe or self.cli_exe()
        if exe is None:
            return False
        key = (str(exe), exe.stat().st_mtime_ns)
        if key in self._families:
            return self._families[key]
        try:
            finished = subprocess.run([str(exe), "--task", "gen", "--help"],
                                      capture_output=True, text=True, timeout=120,
                                      cwd=str(exe.parent))
            answer = bool(re.search(r"^\s*yue2\s*$", finished.stdout or "", re.M))
        except (OSError, subprocess.SubprocessError):
            answer = False
        self._families = {key: answer}
        return answer

    def _release_has_yue2(self, tag):
        """One path-scoped tree call, rather than downloading a build to find out."""
        url = "%s/git/trees/%s:src/models" % (GITHUB_API, urllib.parse.quote(tag))
        request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                tree = json.load(response).get("tree", [])
        except Exception:
            return False
        return any(entry.get("path") == "yue2" for entry in tree)

    def _pick_asset(self):
        request = urllib.request.Request(GITHUB_RELEASES, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            releases = json.load(response)
        for release in releases:
            tag = release.get("tag_name") or ""
            assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
            for cuda in ("13.3", "12.4"):
                name = "audio-%s-bin-windows-x64-cuda%s.zip" % (tag, cuda)
                if name in assets and self._release_has_yue2(tag):
                    return tag, cuda, assets[name]
        raise AudioCppError(
            "No audio.cpp release carries YuE2 yet: the model landed on the dev branch "
            "after the most recent build. Either wait for the next release, or build "
            "audio.cpp from dev yourself and point the console at that audiocpp_cli.")

    def _say(self, message):
        self.busy = message
        self.on_event("song_engine", {"busy": message})

    def _download(self, url, destination, label):
        with urllib.request.urlopen(url, timeout=180) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done, last = 0, 0.0
            with destination.open("wb") as handle:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if now - last > 0.5:
                        last = now
                        self._say("%s  %.0f%%" % (label, (done / total * 100) if total else 0))

    def install(self):
        """Fetch a prebuilt CUDA build that actually contains YuE2."""
        if os.name != "nt":
            raise AudioCppError("The bundled audio.cpp install path is Windows-only here; "
                                "build it yourself and set the binary path instead")
        tag, cuda, url = self._pick_asset()
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        staging = self.home / "download"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            archive = staging / url.rsplit("/", 1)[-1]
            self._download(url, archive, "audio.cpp " + tag)
            self._say("unpacking audio.cpp " + tag)
            with zipfile.ZipFile(archive) as bundle:
                for member in bundle.infolist():
                    if member.is_dir():
                        continue
                    # The archive nests a build folder; flatten it into bin/.
                    target = self.bin_dir / Path(member.filename).name
                    with bundle.open(member) as source, target.open("wb") as handle:
                        shutil.copyfileobj(source, handle)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self._say("")
        exe = self.bin_dir / self.exe_name()
        if not exe.is_file():
            raise AudioCppError("audiocpp_cli was not in the downloaded build")
        return {"tag": tag, "cuda": cuda, "bin_dir": str(self.bin_dir),
                "yue2": self.supports_yue2(exe)}

    # ------------------------------------------------------------- weights

    def weight_path(self, name):
        return self.models_dir / name

    def have(self, name):
        path = self.weight_path(name)
        return path.is_file() and path.stat().st_size > 0

    def weights_ready(self, main_file, vae_file):
        missing = [name for name in ([main_file, vae_file] + SIDECARS) if not self.have(name)]
        return not missing, missing

    def download_weights(self, main_id, vae_id):
        from huggingface_hub import hf_hub_download
        main, vae = _by_id(MAIN_WEIGHTS, main_id), _by_id(VAE_WEIGHTS, vae_id)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        wanted = [(main["file"], main["id"] + " weights"),
                  (vae["file"], "the " + vae["id"] + " decoder")]
        wanted += [(name, "the model configuration") for name in SIDECARS]
        try:
            for name, label in wanted:
                if self.have(name):
                    continue
                self._say("downloading " + label)
                hf_hub_download(WEIGHTS_REPO, name, local_dir=str(self.models_dir))
        finally:
            self._say("")
        return {"model_dir": str(self.models_dir), "main": main["file"], "vae": vae["file"]}

    def delete_weight(self, name):
        """Only the GGUF files this console downloaded may be removed."""
        if name not in {entry["file"] for entry in MAIN_WEIGHTS + VAE_WEIGHTS}:
            raise ValueError("Only YuE2 GGUF weights downloaded here can be removed")
        path = self.weight_path(name)
        if path.is_file():
            path.unlink()

    def status(self, main_id="q4_0", vae_id="f16"):
        exe = self.cli_exe()
        main, vae = _by_id(MAIN_WEIGHTS, main_id), _by_id(VAE_WEIGHTS, vae_id)
        ready, missing = self.weights_ready(main["file"], vae["file"])
        return {
            "installed": exe is not None,
            "yue2": self.supports_yue2(exe) if exe is not None else False,
            "exe": str(exe) if exe else "",
            "override": self.override,
            "supported": os.name == "nt" and platform.machine().lower() in {"amd64", "x86_64"},
            "bin_dir": str(self.bin_dir), "models_dir": str(self.models_dir),
            "weights_ready": ready, "missing": missing,
            "main": [dict(entry, installed=self.have(entry["file"])) for entry in MAIN_WEIGHTS],
            "vae": [dict(entry, installed=self.have(entry["file"])) for entry in VAE_WEIGHTS],
            "busy": self.busy,
            "running": self.process is not None and self.process.poll() is None,
        }

    # ------------------------------------------------------------ generate

    def adapter_dir(self):
        return self.models_dir / "adapters"

    def prepare_adapters(self, adapters, model_dir):
        """audio.cpp takes the same adapter, in the checkpoint's own layout.

        It refuses a ComfyUI file by name rather than guessing, so the fused one
        is converted here, once, and kept beside the GGUF weights.
        """
        from yue2.lora import convert_for_audiocpp

        prepared = []
        for entry in adapters or []:
            source = Path(entry["path"])
            if not source.is_file():
                raise AudioCppError("No adapter file at " + str(source))
            target = self.adapter_dir() / (source.stem + ".audiocpp.safetensors")
            if not target.is_file() or target.stat().st_mtime < source.stat().st_mtime:
                self._say("converting " + source.name)
                convert_for_audiocpp(source, model_dir, target)
                self._say("")
            prepared.append({"file": target.relative_to(self.models_dir).as_posix(),
                             "strength": float(entry.get("strength", 1.0))})
        if len(prepared) > 1:
            raise AudioCppError("audio.cpp takes one NAR adapter at a time; "
                                "leave one selected under Engine")
        return prepared

    def _command(self, spec, settings, out_wav, abc_file):
        exe = self.cli_exe()
        if exe is None:
            raise AudioCppError("audio.cpp is not installed yet")
        if not self.supports_yue2(exe):
            raise AudioCppError("This audiocpp_cli build has no YuE2 family. Install a newer "
                                "build, or point the console at one built from the dev branch.")
        main, vae = _by_id(MAIN_WEIGHTS, settings["main"]), _by_id(VAE_WEIGHTS, settings["vae"])
        ready, missing = self.weights_ready(main["file"], vae["file"])
        if not ready:
            raise AudioCppError("These GGUF files are still missing: " + ", ".join(missing))
        command = [str(exe), "--task", "gen", "--family", "yue2",
                   "--model", str(self.models_dir), "--backend", settings["backend"],
                   "--threads", str(settings["threads"]),
                   "--session-option", "yue2.model_gguf=" + main["file"],
                   "--session-option", "yue2.vae_gguf=" + vae["file"],
                   "--lyrics", spec.get("lyrics", ""),
                   "--request-option", "style=" + spec.get("style", ""),
                   "--request-option", "cot=" + spec.get("cot", "full"),
                   "--request-option", "num_inference_steps=%d" % int(settings.get("ode_steps", 32)),
                   "--seed", str(int(spec.get("seed", 831001))),
                   "--out", str(out_wav), "--log"]
        for adapter in settings.get("adapters") or []:
            command += ["--session-option", "yue2.nar_lora=" + adapter["file"],
                        "--session-option", "yue2.nar_lora_scale=%s" % adapter["strength"]]
        if spec.get("cfg_scale") is not None:
            command += ["--request-option", "cfg_scale=%s" % float(spec["cfg_scale"])]
        if abc_file is not None:
            command += ["--request-option", "abc_file=" + str(abc_file)]
        # The console's sampling names match the CLI's once prefixed by phase.
        for phase, key in (("abc", "abc_sampling"), ("semantic", "semantic_sampling")):
            for name, value in (spec.get(key) or {}).items():
                if value is not None:
                    command += ["--request-option", "%s_%s=%s" % (phase, name, value)]
        return command

    def generate(self, spec, settings, out_wav, *, on_phase=None, cancelled=None, abc_file=None):
        """Run one take to completion, reporting phases as the log reports them."""
        on_phase = on_phase or (lambda phase, event, value, marker: None)
        cancelled = cancelled or (lambda: False)
        command = self._command(spec, settings, out_wav, abc_file)
        started = time.perf_counter()
        timings, tail, loaded = {}, [], []
        with self.lock:
            self.process = subprocess.Popen(
                command, cwd=str(self.cli_exe().parent), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        process = self.process
        # A cancelled run can go quiet before it ends, so the flag is watched on
        # its own thread rather than between log lines.
        stop = threading.Event()

        def watch():
            while not stop.wait(1.0):
                if cancelled():
                    process.terminate()
                    return

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            for line in process.stdout:
                line = line.rstrip()
                tail.append(line)
                del tail[:-60]
                self._observe(line, timings, on_phase)
                found = ADAPTER.search(line)
                if found:
                    loaded.append({"option": found.group(1), "file": found.group(2).strip(),
                                   "projections": int(found.group(3)),
                                   "scale": float(found.group(4))})
            code = process.wait()
        finally:
            stop.set()
            with self.lock:
                self.process = None
        if cancelled():
            raise InterruptedError("cancelled")
        if code != 0 or not Path(out_wav).is_file():
            raise AudioCppError("audio.cpp could not finish this take (exit %s).\n%s"
                                % (code, "\n".join(tail[-25:])))
        timings["e2e_seconds"] = time.perf_counter() - started
        return {"wav": str(out_wav), "timing": timings, "adapters_loaded": loaded}

    def _observe(self, line, timings, on_phase):
        match = TIMING.match(line)
        if match is None:
            return
        name, value = match.group(1), match.group(2).strip()
        try:
            timings[name] = float(value)
        except ValueError:
            timings[name] = value
        phase = PHASE_TICK.get(name)
        if phase is not None:
            on_phase(phase, "tick", timings[name], name)
            return
        for marker, phase in PHASE_DONE:
            if name == marker:
                on_phase(phase, "done", timings[name], name)
                return

    def stop(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()


def _by_id(catalog, value):
    for entry in catalog:
        if entry["id"] == value:
            return entry
    raise ValueError("Unknown YuE2 GGUF choice: %s" % value)
