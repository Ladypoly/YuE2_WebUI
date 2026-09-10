"""Album art for a take, drawn locally by stable-diffusion.cpp.

Same shape as the llama.cpp writer: fetch the prebuilt CUDA build into the
project folder, download a checkpoint on demand, and run the binary for exactly
one image. sd.cpp is a one-shot CLI rather than a server, so the process ends
with the picture and takes its VRAM with it.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

GITHUB_RELEASES = "https://api.github.com/repos/leejet/stable-diffusion.cpp/releases?per_page=12"

# Checkpoints that draw a decent sleeve on a consumer card. Verified on the Hub.
# Newer architectures ship as separate parts -- a diffusion transformer, a VAE
# and a text encoder -- so an entry may list several files instead of one.
CATALOG = [
    {"id": "z-image-turbo", "name": "Z-Image Turbo", "size_gb": 6.4,
     "steps": 8, "cfg": 1.0, "size": 1024, "flags": ["--diffusion-fa"],
     "note": "Best quality here, and fast: 8 steps, strong prompt following.",
     "prompt_style": "natural",
     "parts": [
         {"role": "diffusion", "repo": "leejet/Z-Image-Turbo-GGUF", "file": "z_image_turbo-Q4_K.gguf"},
         {"role": "llm", "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"},
         {"role": "vae", "repo": "Comfy-Org/z_image_turbo", "file": "split_files/vae/ae.safetensors"},
     ]},
    {"id": "krea2-turbo", "name": "Krea 2 Turbo", "size_gb": 13.0,
     "steps": 8, "cfg": 1.0, "size": 1024, "flags": ["--diffusion-fa", "--offload-to-cpu"],
     "note": "Newest and most detailed. Large download, slower than Z-Image.",
     "prompt_style": "natural",
     "parts": [
         {"role": "diffusion", "repo": "realrebelai/KREA-2_GGUFs", "file": "TURBO/Krea-2-Turbo-Q4_K_M.gguf"},
         {"role": "llm", "repo": "Qwen/Qwen3-VL-4B-Instruct-GGUF", "file": "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"},
         {"role": "vae", "repo": "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
          "file": "split_files/vae/wan_2.1_vae.safetensors"},
     ]},
]

STYLE_SUFFIX = "album cover artwork, square composition, bold graphic design"
NEGATIVE = "text, letters, words, watermark, signature, frame, border, blurry, low quality"


class StableDiffusionCpp:
    def __init__(self, root: Path, on_event=None):
        self.home = root / "sdcpp"
        self.bin_dir = self.home / "bin"
        self.models_dir = root / "models" / "art"
        self.on_event = on_event or (lambda kind, payload: None)
        self.busy = ""
        self.extra_dirs = []
        self._sniffed = {}

    # Recent builds ship sd-cli; older ones called it sd.
    EXE_NAMES = ("sd-cli.exe", "sd.exe") if os.name == "nt" else ("sd-cli", "sd")

    def exe(self):
        for name in self.EXE_NAMES:
            binary = self.bin_dir / name
            if binary.is_file():
                return binary
        return None

    SUFFIXES = {".safetensors", ".gguf", ".ckpt"}
    # sd.cpp reads a complete single-file SD/SDXL checkpoint: a UNet, its VAE and
    # a text encoder together. Requiring the VAE alongside the denoiser is what
    # separates those from video and audio diffusion models, which carry the
    # denoiser keys too but none of the rest.
    UNET_KEYS = ("model.diffusion_model", "down_blocks", "input_blocks")
    VAE_KEYS = ("first_stage_model", "decoder.up_blocks", "vae.")
    TEXT_KEYS = ("cond_stage_model", "conditioner.", "text_model", "text_encoders")

    def _is_diffusion(self, path):
        """Sniff the safetensors header rather than trusting the file name."""
        key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        cached = self._sniffed.get(key)
        if cached is not None:
            return cached
        verdict = True
        if path.suffix.lower() == ".safetensors":
            try:
                with path.open("rb") as handle:
                    length = int.from_bytes(handle.read(8), "little")
                    if 0 < length <= 64 * 1024 * 1024:
                        header = handle.read(length).decode("utf-8", "replace")
                        verdict = (any(k in header for k in self.UNET_KEYS)
                                   and any(k in header for k in self.VAE_KEYS)
                                   and any(k in header for k in self.TEXT_KEYS))
                    else:
                        verdict = False
            except (OSError, ValueError):
                verdict = False
        self._sniffed[key] = verdict
        return verdict

    def _scan(self, directory, source, limit=400):
        found = []
        base = Path(directory).expanduser()
        if not base.is_dir():
            return found
        try:
            for path in base.rglob("*"):
                if len(found) >= limit:
                    break
                if path.suffix.lower() in self.SUFFIXES and path.is_file():
                    # Skip the small companions that sit beside a checkpoint.
                    if path.stat().st_size < 300 * 1024 * 1024:
                        continue
                    # A bare model.safetensors is the generic Hub layout, not a
                    # diffusion checkpoint sd.cpp can read.
                    if path.name.lower() in {"model.safetensors", "diffusion_pytorch_model.safetensors"}:
                        continue
                    if not self._is_diffusion(path):
                        continue
                    found.append({"file": path.name, "path": str(path), "source": source,
                                  "size_gb": round(path.stat().st_size / 2 ** 30, 2)})
        except (OSError, PermissionError):
            pass
        return found

    def discover(self):
        """Single-file checkpoints found on disk, catalogued or your own."""
        models, seen = [], set()
        for entry in self._scan(self.models_dir, "downloaded"):
            models.append(entry)
            seen.add(Path(entry["path"]).resolve())
        for directory in self.extra_dirs:
            for entry in self._scan(Path(directory), "added folder"):
                resolved = Path(entry["path"]).resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                models.append(entry)
        return models

    def part_path(self, entry, part):
        return self.models_dir / entry["id"] / Path(part["file"]).name

    def parts_ready(self, entry):
        return all(self.part_path(entry, part).is_file() for part in entry.get("parts", []))

    def catalog_state(self):
        """Catalogue rows, marked installed whether they are one file or many."""
        on_disk = {Path(m["path"]).name for m in self.discover()}
        rows = []
        for entry in CATALOG:
            if entry.get("parts"):
                installed = self.parts_ready(entry)
            else:
                installed = entry["file"] in on_disk
            rows.append(dict(entry, installed=installed,
                             multipart=bool(entry.get("parts"))))
        return rows

    def resolve_choice(self, choice):
        """Return what draw() needs: a catalogue id for a multi-part model, or a
        path for a single checkpoint."""
        for entry in CATALOG:
            if entry.get("parts") and choice == entry["id"] and self.parts_ready(entry):
                return {"entry": entry,
                        "parts": {p["role"]: str(self.part_path(entry, p)) for p in entry["parts"]}}
        models = self.discover()
        if choice:
            for model in models:
                if choice in (model["path"], model["file"]):
                    return {"path": model["path"]}
        for entry in CATALOG:
            if entry.get("parts") and self.parts_ready(entry):
                return {"entry": entry,
                        "parts": {p["role"]: str(self.part_path(entry, p)) for p in entry["parts"]}}
        return {"path": models[0]["path"]} if models else None

    def status(self):
        """What the console can offer: ready multi-part models first, then any
        single checkpoint found on disk."""
        models = []
        for entry in CATALOG:
            if entry.get("parts") and self.parts_ready(entry):
                total = sum(self.part_path(entry, part).stat().st_size for part in entry["parts"])
                models.append({"file": entry["id"], "path": entry["id"], "name": entry["name"],
                               "source": "downloaded", "size_gb": round(total / 2 ** 30, 2)})
        models.extend(self.discover())
        return {"installed": self.exe() is not None,
                "supported": os.name == "nt" and platform.machine().lower() in {"amd64", "x86_64"},
                "bin_dir": str(self.bin_dir), "models_dir": str(self.models_dir),
                "models": models, "busy": self.busy,
                "catalog": self.catalog_state()}

    def _pick_assets(self):
        request = urllib.request.Request(GITHUB_RELEASES, headers={"Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(request, timeout=60) as response:
            releases = json.load(response)
        for release in releases:
            assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}
            binary = next((n for n in assets if "bin-win-cuda12-x64" in n), None)
            runtime = next((n for n in assets if n.startswith("cudart-sd-bin-win-cu12")), None)
            if binary and runtime:
                return release.get("tag_name"), assets[binary], assets[runtime]
        raise RuntimeError("No prebuilt Windows CUDA build found in the recent stable-diffusion.cpp releases")

    def _download(self, url, destination, label):
        with urllib.request.urlopen(url, timeout=120) as response:
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
                        self.busy = "%s  %.0f%%" % (label, (done / total * 100) if total else 0)
                        self.on_event("art", {"busy": self.busy})

    def install(self):
        if os.name != "nt":
            raise RuntimeError("The bundled stable-diffusion.cpp install path is Windows-only here")
        self.bin_dir.mkdir(parents=True, exist_ok=True)
        tag, binary_url, runtime_url = self._pick_assets()
        staging = self.home / "download"
        staging.mkdir(parents=True, exist_ok=True)
        try:
            for url, label in ((binary_url, "stable-diffusion.cpp"), (runtime_url, "CUDA runtime")):
                archive = staging / url.rsplit("/", 1)[-1]
                self._download(url, archive, label)
                self.busy = "unpacking " + label
                self.on_event("art", {"busy": self.busy})
                with zipfile.ZipFile(archive) as bundle:
                    for member in bundle.infolist():
                        if member.is_dir():
                            continue
                        target = self.bin_dir / Path(member.filename).name
                        with bundle.open(member) as source, target.open("wb") as handle:
                            shutil.copyfileobj(source, handle)
                archive.unlink(missing_ok=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            self.busy = ""
            self.on_event("art", {"busy": ""})
        if self.exe() is None:
            raise RuntimeError("No sd-cli executable was found in the downloaded build")
        return {"tag": tag, "bin_dir": str(self.bin_dir)}

    def download_model(self, model_id):
        entry = next((e for e in CATALOG if e["id"] == model_id), None)
        if entry is None:
            raise ValueError("Unknown art model")
        from huggingface_hub import hf_hub_download
        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            if entry.get("parts"):
                target = self.models_dir / entry["id"]
                target.mkdir(parents=True, exist_ok=True)
                for index, part in enumerate(entry["parts"], 1):
                    self.busy = "%s: part %d of %d (%s)" % (entry["name"], index,
                                                            len(entry["parts"]), part["role"])
                    self.on_event("art", {"busy": self.busy})
                    fetched = hf_hub_download(part["repo"], part["file"], local_dir=str(target))
                    final = target / Path(part["file"]).name
                    if Path(fetched).resolve() != final.resolve():
                        shutil.move(fetched, final)
                return {"file": entry["id"], "path": str(target)}
            self.busy = "downloading " + entry["name"]
            self.on_event("art", {"busy": self.busy})
            path = hf_hub_download(entry["repo"], entry["file"], local_dir=str(self.models_dir))
            return {"file": Path(path).name, "path": path}
        finally:
            self.busy = ""
            self.on_event("art", {"busy": ""})

    def delete_model(self, filename):
        """Only files this console downloaded may be deleted."""
        path = (self.models_dir / filename).resolve()
        if path.parent != self.models_dir.resolve() or not path.is_file():
            raise ValueError("Only checkpoints downloaded here can be removed")
        path.unlink()

    def prompt_style(self, choice):
        """How the active checkpoint likes to be prompted.

        Z-Image and Krea2 read the prompt with an LLM text encoder and reward a
        full descriptive sentence. SD and SDXL use CLIP, which truncates around
        75 tokens and responds better to a comma-separated tag list.
        """
        spec = self.resolve_choice(choice)
        if not spec:
            return "natural"
        if spec.get("entry"):
            return spec["entry"].get("prompt_style", "natural")
        name = Path(spec["path"]).name.lower()
        for entry in CATALOG:
            if entry.get("file", "").lower() == name:
                return entry.get("prompt_style", "tags")
        return "tags"

    def settings_for(self, filename):
        for entry in CATALOG:
            if entry["file"] == filename:
                return entry
        lowered = filename.lower()
        # Your own checkpoints get sensible defaults from their family: SDXL is
        # trained at 1024, SD 1.5 at 512, and "turbo"/"lightning" want few steps.
        extra_large = "xl" in lowered or "illustrious" in lowered or "pony" in lowered
        fast = any(mark in lowered for mark in ("turbo", "lightning", "lcm", "hyper"))
        return {"steps": 6 if fast else 22,
                "cfg": 1.5 if fast else 6.5,
                "size": 1024 if extra_large else 512}

    def draw(self, spec, prompt, destination, seed=-1, steps=None, cfg=None, size=None):
        """Render one image. The process exits with it, so nothing stays resident."""
        binary = self.exe()
        if binary is None:
            raise RuntimeError("stable-diffusion.cpp is not installed yet")
        if not spec:
            raise RuntimeError("No art model selected")

        command = [str(binary), "--mode", "img_gen"]
        if spec.get("parts"):
            entry = spec["entry"]
            preset = entry
            command += ["--diffusion-model", spec["parts"]["diffusion"],
                        "--vae", spec["parts"]["vae"],
                        "--llm", spec["parts"]["llm"]]
            command += entry.get("flags", [])
        else:
            preset = self.settings_for(Path(spec["path"]).name)
            command += ["--model", spec["path"]]

        command += ["--prompt", prompt + ", " + STYLE_SUFFIX,
                    "--negative-prompt", NEGATIVE,
                    "--output", str(destination),
                    "--steps", str(steps or preset["steps"]),
                    "--cfg-scale", str(cfg if cfg is not None else preset["cfg"]),
                    "--width", str(size or preset["size"]),
                    "--height", str(size or preset["size"]),
                    "--seed", str(seed)]

        self.busy = "drawing the cover"
        self.on_event("art", {"busy": self.busy})
        start = time.perf_counter()
        try:
            finished = subprocess.run(command, cwd=str(self.bin_dir), capture_output=True,
                                      text=True, timeout=3600)
        finally:
            self.busy = ""
            self.on_event("art", {"busy": ""})
        if finished.returncode != 0 or not Path(destination).is_file():
            detail = (finished.stderr or finished.stdout or "").strip()[-1200:]
            raise RuntimeError("stable-diffusion.cpp failed: " + detail)
        return {"path": str(destination), "seconds": round(time.perf_counter() - start, 1),
                "model": spec.get("entry", {}).get("name") or Path(spec.get("path", "")).name}
