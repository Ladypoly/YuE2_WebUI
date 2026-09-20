"""Finding YuE2 adapters on this machine, and keeping the chosen ones honest.

Adapters are ordinary safetensors files that usually live in a ComfyUI loras
folder rather than next to this project, so the console scans the folders you
name. Only files whose header actually carries YuE2 adapter keys are offered:
an image LoRA in the same folder is not a song adapter, and finding that out
costs one header read rather than a failed run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from yue2.lora import LoraError, read_metadata  # noqa: E402


class Loras:
    def __init__(self, root: Path):
        self.models_dir = Path(root) / "models" / "loras"
        self.extra_dirs = []

    def _scan(self, directory, source, limit=200):
        found = []
        base = Path(directory).expanduser()
        if not base.is_dir():
            return found
        try:
            for path in sorted(base.rglob("*.safetensors")):
                if len(found) >= limit:
                    break
                try:
                    info = read_metadata(path)
                except (LoraError, OSError, ValueError, UnicodeDecodeError):
                    continue
                if not info["branches"]:
                    continue        # a LoRA for some other model
                found.append({"file": info["name"], "path": str(path.resolve()), "source": source,
                              "size_mb": round(info["bytes"] / 2 ** 20, 1),
                              "branch": "NAR" if info["branches"] == ["diffusion_model"] else
                                        "AR" if info["branches"] == ["text_encoders"] else "AR+NAR",
                              "layers": info["layers"], "ranks": info["ranks"],
                              "note": info["metadata"].get("base_model", "")})
        except (OSError, PermissionError):
            pass
        return found

    def discover(self):
        found, seen = [], set()
        sources = [(self.models_dir, "project folder")]
        sources += [(Path(d), "added folder") for d in self.extra_dirs]
        for directory, label in sources:
            for entry in self._scan(directory, label):
                if entry["path"] in seen:
                    continue
                seen.add(entry["path"])
                found.append(entry)
        return found

    def resolve(self, selected):
        """Turn the saved selection into what the pipeline takes, dropping nothing silently."""
        adapters, missing = [], []
        for entry in selected or []:
            path = Path(str(entry.get("path", "")))
            if not path.is_file():
                missing.append(str(path))
                continue
            adapters.append({"path": str(path), "strength": float(entry.get("strength", 1.0))})
        return adapters, missing

    def status(self, selected):
        available = self.discover()
        adapters, missing = self.resolve(selected)
        return {"available": available, "selected": list(selected or []),
                "missing": missing, "dirs": list(self.extra_dirs),
                "models_dir": str(self.models_dir), "active": len(adapters)}
