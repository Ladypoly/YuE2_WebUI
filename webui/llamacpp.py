"""A second writer backend: llama.cpp, installed and fed from the console.

Ollama has to be installed separately and manages its own model store. This
backend keeps everything inside the project folder: it fetches the prebuilt
llama-server binaries, downloads GGUF writer models, and runs the server only
for the length of one request.

Unloading is therefore total rather than advisory -- the process that held the
weights is gone, which matters because the song model wants the same VRAM.
"""
from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

GITHUB_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=12"

# Writer models that fit a single consumer card and follow instructions well.
# Verified present and ungated on the Hub.
CATALOG = [
    {"id": "qwen2.5-3b", "name": "Qwen2.5 3B Instruct", "size_gb": 2.0,
     "repo": "bartowski/Qwen2.5-3B-Instruct-GGUF",
     "file": "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
     "note": "Fastest. Fine for briefs, a few seconds per song."},
    {"id": "qwen2.5-7b", "name": "Qwen2.5 7B Instruct", "size_gb": 4.7,
     "repo": "bartowski/Qwen2.5-7B-Instruct-GGUF",
     "file": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
     "note": "The balanced choice: better lyrics, still quick."},
    {"id": "llama3.1-8b", "name": "Llama 3.1 8B Instruct", "size_gb": 4.9,
     "repo": "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
     "file": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
     "note": "Alternative voice; strong English phrasing."},
    {"id": "qwen2.5-14b", "name": "Qwen2.5 14B Instruct", "size_gb": 9.0,
     "repo": "bartowski/Qwen2.5-14B-Instruct-GGUF",
     "file": "Qwen2.5-14B-Instruct-Q4_K_M.gguf",
     "note": "Best writing here. Needs the song model unloaded first."},
]


class LlamaCpp:
    def __init__(self, root: Path, on_event=None):
        self.home = root / "llamacpp"
        self.bin_dir = self.home / "bin"
        self.models_dir = root / "models" / "writers"
        self.on_event = on_event or (lambda kind, payload: None)
        self.process = None
        self.port = None
        self.lock = threading.Lock()
        self.busy = ""
        self.extra_dirs = []

    # ------------------------------------------------------------ install

    def server_exe(self):
        exe = self.bin_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
        return exe if exe.is_file() else None

    SUFFIXES = {".gguf"}
    # A GGUF is not automatically a chat model: vision projectors, speech and
    # embedding models sit in the same cache and cannot write a brief.
    NOT_A_WRITER = ("mmproj", "-asr", "asr-", "whisper", "embed", "rerank",
                    "bge-", "clip", "-vae", "vision-f16", "tts")

    def _scan(self, directory, source, limit=400):
        """Find usable model files without walking a whole drive."""
        found = []
        base = Path(directory).expanduser()
        if not base.is_dir():
            return found
        try:
            for path in base.rglob("*"):
                if len(found) >= limit:
                    break
                if path.suffix.lower() in self.SUFFIXES and path.is_file():
                    lowered = path.name.lower()
                    if any(mark in lowered for mark in self.NOT_A_WRITER):
                        continue
                    # Multi-part GGUFs are loaded from their first shard only.
                    if "-of-" in path.name and not path.name.split("-of-")[0].endswith("00001"):
                        continue
                    found.append({"file": path.name, "path": str(path), "source": source,
                                  "size_gb": round(path.stat().st_size / 2 ** 30, 2)})
        except (OSError, PermissionError):
            pass
        return found

    def discover(self):
        """Downloaded models first, then anything already on this machine."""
        models, seen = [], set()
        for entry in self._scan(self.models_dir, "downloaded"):
            models.append(entry)
            seen.add(Path(entry["path"]).resolve())
        sources = [(Path.home() / ".cache" / "huggingface" / "hub", "hugging face cache")]
        sources += [(Path(d), "added folder") for d in self.extra_dirs]
        for directory, label in sources:
            for entry in self._scan(directory, label):
                resolved = Path(entry["path"]).resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                models.append(entry)
        return models

    def status(self):
        models = self.discover()
        installed = {m["file"] for m in models}
        catalog = [dict(entry, installed=entry["file"] in installed) for entry in CATALOG]
        return {"installed": self.server_exe() is not None,
                "supported": os.name == "nt" and platform.machine().lower() in {"amd64", "x86_64"},
                "bin_dir": str(self.bin_dir), "models_dir": str(self.models_dir),
                "models": models, "catalog": catalog, "busy": self.busy,
                "running": self.process is not None and self.process.poll() is None}

    def _pick_assets(self):
        """Newest build-tagged release; the 'latest' tag is not a binary build."""
        request = urllib.request.Request(GITHUB_RELEASES, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            releases = json.load(response)
        for release in releases:
            assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
            for cuda in ("13.3", "12.4"):
                binary = "llama-%s-bin-win-cuda-%s-x64.zip" % (release.get("tag_name"), cuda)
                runtime = "cudart-llama-bin-win-cuda-%s-x64.zip" % cuda
                if binary in assets and runtime in assets:
                    return release.get("tag_name"), cuda, assets[binary], assets[runtime]
        raise RuntimeError("No prebuilt Windows CUDA build found in the recent llama.cpp releases")

    def _download(self, url, destination, label):
        with urllib.request.urlopen(url, timeout=120) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            last = 0.0
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
                        self.busy = "%s  %.0f%%" % (label, (done / total * 100) if total else 0)
                        self.on_event("writer", {"busy": self.busy})

    def install(self):
        """Fetch llama-server plus the CUDA runtime it links against."""
        if os.name != "nt":
            raise RuntimeError("The bundled llama.cpp install path is Windows-only here")
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        tag, cuda, binary_url, runtime_url = self._pick_assets()
        staging = self.home / "download"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            for url, label in ((binary_url, "llama.cpp " + tag), (runtime_url, "CUDA " + cuda + " runtime")):
                archive = staging / url.rsplit("/", 1)[-1]
                self._download(url, archive, label)
                self.busy = "unpacking " + label
                self.on_event("writer", {"busy": self.busy})
                with zipfile.ZipFile(archive) as bundle:
                    for member in bundle.infolist():
                        if member.is_dir():
                            continue
                        # The archives nest a build folder; flatten into bin/.
                        target = self.bin_dir / Path(member.filename).name
                        with bundle.open(member) as source, target.open("wb") as handle:
                            shutil.copyfileobj(source, handle)
                archive.unlink(missing_ok=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self.busy = ""
            self.on_event("writer", {"busy": ""})
        if self.server_exe() is None:
            raise RuntimeError("llama-server was not found in the downloaded build")
        return {"tag": tag, "cuda": cuda, "bin_dir": str(self.bin_dir)}

    def download_model(self, model_id):
        entry = next((e for e in CATALOG if e["id"] == model_id), None)
        if entry is None:
            raise ValueError("Unknown writer model")
        from huggingface_hub import hf_hub_download
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.busy = "downloading " + entry["name"]
        self.on_event("writer", {"busy": self.busy})
        try:
            path = hf_hub_download(entry["repo"], entry["file"], local_dir=str(self.models_dir))
        finally:
            self.busy = ""
            self.on_event("writer", {"busy": ""})
        return {"file": Path(path).name, "path": path}

    def resolve_choice(self, choice):
        """Accept a full path or a bare filename, whichever the setting holds."""
        models = self.discover()
        if choice:
            for model in models:
                if choice in (model["path"], model["file"]):
                    return model["path"]
        return models[0]["path"] if models else None

    def delete_model(self, filename):
        """Only files this console downloaded may be deleted."""
        path = (self.models_dir / filename).resolve()
        if path.parent != self.models_dir.resolve() or not path.is_file():
            raise ValueError("Only models downloaded here can be removed")
        path.unlink()

    # -------------------------------------------------------------- serve

    def _free_port(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def start(self, model_path, gpu_layers=999, context=8192):
        exe = self.server_exe()
        if exe is None:
            raise RuntimeError("llama.cpp is not installed yet")
        self.stop()
        self.port = self._free_port()
        # Reasoning models otherwise spend the answer on a plan and return a
        # near-empty brief: Qwen3.8 fills "lyrics" with a single word. Ollama has
        # think=false for this; llama-server has --reasoning off.
        command = [str(exe), "--model", str(model_path), "--port", str(self.port),
                   "--n-gpu-layers", str(gpu_layers), "--ctx-size", str(context),
                   "--host", "127.0.0.1", "--no-webui", "--reasoning", "off"]
        self.process = subprocess.Popen(command, cwd=str(self.bin_dir),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(0.4)
        if self.process.poll() is not None:
            output = (self.process.stdout.read() or b"").decode("utf-8", "replace")
            self.process = None
            if "reasoning" in output.lower():
                self.process = subprocess.Popen(command[:-2], cwd=str(self.bin_dir),
                                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            else:
                raise RuntimeError("llama-server exited during startup: " + output[-1500:])
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                output = (self.process.stdout.read() or b"").decode("utf-8", "replace")[-1500:]
                self.process = None
                raise RuntimeError("llama-server exited during startup:\n" + output)
            try:
                with urllib.request.urlopen("http://127.0.0.1:%d/health" % self.port, timeout=2):
                    return self.port
            except Exception:
                time.sleep(0.5)
        self.stop()
        raise RuntimeError("llama-server did not become ready within 180 seconds")

    def stop(self):
        """Kill the server. The weights go with the process, so this is final."""
        process, self.process = self.process, None
        if process is None:
            return False
        with contextlib.suppress(Exception):
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        return True

    def chat(self, messages, schema=None, temperature=0.9, max_tokens=4096):
        payload = {"messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": False}
        if schema is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "brief", "schema": schema, "strict": True}}
        request = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % self.port,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            body = json.load(response)
        choice = (body.get("choices") or [{}])[0]
        return choice.get("message", {}).get("content", ""), body.get("usage", {})
