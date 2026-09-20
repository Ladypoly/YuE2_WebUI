"""Merge community LoRA adapters into the loaded mixture-of-transformers weights.

Adapters for YuE2 are published in the ComfyUI repackaging's key names, which
differ from the checkpoint's in two ways. The NAR expert is called the diffusion
model and the AR expert the text encoder, so a NAR adapter's keys arrive under
``diffusion_model.`` and belong to this checkpoint's ``nar_*`` modules; and that
packaging fuses q/k/v into one projection and gate/up into another, as a
block-diagonal stack of the separate adapters. The fused delta is therefore
sliced back apart by the target modules' own output widths rather than by any
number written here, so a checkpoint with different head counts either matches
or is refused.

Merging is one-way. The deltas are folded into the weights in place, which costs
nothing at inference time, and a different adapter selection means loading the
model again rather than subtracting an adapter back out in bfloat16.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from torch import nn

# Which expert an adapter's keys belong to, and the module prefix they map onto.
BRANCHES = {"diffusion_model": "nar_", "text_encoders": ""}
LAYER = re.compile(r"^(diffusion_model|text_encoders)\.model\.layers\.(\d+)\.(.+)\.(lora_up|lora_down)\.weight$")
DIRECT = re.compile(r"^(diffusion_model|text_encoders)\.(llm2vae|vae2llm)\.(diff|diff_b)$")
# Fused adapter target -> the separate projections it stacks, in order.
FUSED = {"self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
         "mlp.gate_up_proj": ("mlp.gate_proj", "mlp.up_proj")}
PLAIN = {"self_attn.o_proj", "mlp.down_proj", "self_attn.q_proj", "self_attn.k_proj",
         "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"}


class LoraError(ValueError):
    pass


def read_metadata(path):
    """The adapter's own description, without loading a single weight."""
    path = Path(path)
    with path.open("rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        if not 0 < length <= 100 * 2 ** 20:
            raise LoraError("Not a safetensors adapter: " + path.name)
        header = json.loads(handle.read(length))
    metadata = header.pop("__metadata__", {}) or {}
    branches = sorted({key.split(".", 1)[0] for key in header} & set(BRANCHES))
    layers = {int(match.group(2)) for match in map(LAYER.fullmatch, header) if match}
    ranks = sorted({tuple(header[key]["shape"])[0] for key in header if key.endswith("lora_down.weight")})
    return {"name": path.name, "path": str(path), "bytes": path.stat().st_size,
            "branches": branches, "layers": len(layers), "ranks": ranks,
            "tensors": len(header), "metadata": {str(k): str(v) for k, v in metadata.items()}}


def _module(model, name):
    try:
        target = model.get_submodule(name)
    except AttributeError:
        raise LoraError("This checkpoint has no module " + name) from None
    if not isinstance(target, nn.Linear):
        raise LoraError("Expected a Linear layer at " + name + ", found " + type(target).__name__)
    if not isinstance(target.weight, nn.Parameter) or target.weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise LoraError("Cannot merge into the weights at " + name +
                        "; load without quantization to use adapters")
    return target


def _add(target, delta, name):
    if tuple(delta.shape) != tuple(target.weight.shape):
        raise LoraError("Adapter shape %s does not fit %s %s"
                        % (tuple(delta.shape), name, tuple(target.weight.shape)))
    with torch.no_grad():
        target.weight.add_(delta.to(target.weight.dtype))


def _add_bias(target, delta, name):
    if target.bias is None:
        raise LoraError(name + " has no bias for this adapter to change")
    if tuple(delta.shape) != tuple(target.bias.shape):
        raise LoraError("Adapter bias %s does not fit %s" % (tuple(delta.shape), name))
    with torch.no_grad():
        target.bias.add_(delta.to(target.bias.dtype))


def _merge_layer(model, prefix, index, target, pair, strength, device):
    """One adapter entry, split across as many projections as it was fused from."""
    up = pair["lora_up"].to(device=device, dtype=torch.float32)
    down = pair["lora_down"].to(device=device, dtype=torch.float32)
    if up.shape[1] != down.shape[0]:
        raise LoraError("Adapter rank mismatch at layer %d %s" % (index, target))
    delta = (up @ down) * strength
    names = FUSED.get(target)
    if names is None:
        if target not in PLAIN:
            raise LoraError("Unsupported adapter target: " + target)
        names = (target,)
    modules = [_module(model, "model.layers.%d.%s%s" % (index, prefix, name)) for name in names]
    widths = [module.out_features for module in modules]
    if sum(widths) != delta.shape[0]:
        raise LoraError("Fused adapter for %s covers %d rows, this checkpoint needs %d"
                        % (target, delta.shape[0], sum(widths)))
    offset = 0
    for module, width, name in zip(modules, widths, names):
        _add(module, delta[offset:offset + width], name)
        offset += width
    return len(names)


def apply_loras(model, adapters, device=None):
    """Fold every adapter into the model in place, in the order given."""
    if model is None:
        raise LoraError("Load the model before merging adapters")
    applied = getattr(model, "_yue2_loras", None)
    if applied:
        raise LoraError("Adapters are already merged into this model; reload it to change them")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    record = []
    for adapter in adapters:
        path = Path(adapter["path"] if isinstance(adapter, dict) else adapter)
        strength = float(adapter.get("strength", 1.0)) if isinstance(adapter, dict) else 1.0
        if not path.is_file():
            raise LoraError("No adapter file at " + str(path))
        if not -4 <= strength <= 4:
            raise LoraError("Adapter strength must be between -4 and 4")
        record.append(_merge_one(model, path, strength, device))
    object.__setattr__(model, "_yue2_loras", record)
    return record


def _merge_one(model, path, strength, device):
    from safetensors.torch import load_file

    state = load_file(str(path))
    pairs, merged, touched = {}, 0, 0
    for key, value in state.items():
        layer = LAYER.fullmatch(key)
        direct = DIRECT.fullmatch(key)
        if layer is not None:
            branch, index, target, half = layer.group(1), int(layer.group(2)), layer.group(3), layer.group(4)
            pairs.setdefault((branch, index, target), {})[half] = value
        elif direct is not None:
            branch, name, kind = direct.groups()
            # The latent projections live inside the packaged diffusion model
            # because only the NAR path reads them; the AR expert never does.
            if not BRANCHES[branch]:
                raise LoraError("An AR adapter cannot change the latent projection " + name)
            module = _module(model, name)
            delta = value.to(device=device, dtype=torch.float32) * strength
            if kind == "diff":
                _add(module, delta, name)
            else:
                _add_bias(module, delta, name)
            touched += 1
        else:
            raise LoraError("Unexpected key in " + path.name + ": " + key)

    for (branch, index, target), pair in sorted(pairs.items()):
        if set(pair) != {"lora_up", "lora_down"}:
            raise LoraError("Adapter entry %s at layer %d is missing half its pair" % (target, index))
        touched += _merge_layer(model, BRANCHES[branch], index, target, pair, strength, device)
        merged += 1

    if not touched:
        raise LoraError(path.name + " changed nothing in this checkpoint")
    return {"name": path.name, "path": str(path), "strength": strength,
            "entries": merged, "modules": touched,
            "branches": sorted({branch for branch, _, _ in pairs})}


def lora_status(model):
    return {"merged": list(getattr(model, "_yue2_loras", None) or [])}
