"""Adapter merging and the audio.cpp conversion, on a checkpoint small enough to build."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from yue2.lora import (AUDIOCPP_PROJECTIONS, LATENT, LoraError, apply_loras,
                       convert_for_audiocpp, read_metadata)

HIDDEN, KV, FF, LATENT_DIM, RANK, LAYERS = 32, 16, 64, 8, 4, 2


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)


class Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, FF, bias=False)
        self.up_proj = nn.Linear(HIDDEN, FF, bias=False)
        self.down_proj = nn.Linear(FF, HIDDEN, bias=False)


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.mlp = Attention(), Mlp()
        self.nar_self_attn, self.nar_mlp = Attention(), Mlp()


class Stub(nn.Module):
    """The same module names as the checkpoint, at a size a test can hold."""

    def __init__(self, layers=LAYERS):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(Layer() for _ in range(layers))
        self.llm2vae = nn.Linear(HIDDEN, LATENT_DIM)
        self.vae2llm = nn.Linear(LATENT_DIM, HIDDEN)


def block_diagonal(widths, rank, generator):
    """How these adapters are really packaged: separate LoRAs stacked, not fused.

    Each projection keeps its own rank block, so the off-block parts are zero and
    the stack can be taken apart again without loss.
    """
    up = torch.zeros(sum(widths), rank * len(widths))
    offset = 0
    for position, rows in enumerate(widths):
        up[offset:offset + rows, position * rank:(position + 1) * rank] =             torch.randn(rows, rank, generator=generator)
        offset += rows
    return up


def comfy_adapter(path, layers=LAYERS, seed=0, fused=False):
    """A ComfyUI NAR adapter: fused q/k/v and gate/up, latent projections as deltas."""
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for layer in range(layers):
        stem = "diffusion_model.model.layers.%d." % layer
        for name, widths, rank in (
            ("self_attn.qkv_proj", [HIDDEN, KV, KV], RANK),
            ("self_attn.o_proj", [HIDDEN], RANK),
            ("mlp.gate_up_proj", [FF, FF], RANK),
        ):
            total = rank * len(widths)
            state[stem + name + ".lora_up.weight"] = (
                torch.randn(sum(widths), total, generator=generator) if fused
                else block_diagonal(widths, rank, generator))
            state[stem + name + ".lora_down.weight"] = torch.randn(total, HIDDEN, generator=generator)
        state[stem + "mlp.down_proj.lora_up.weight"] = torch.randn(HIDDEN, RANK, generator=generator)
        state[stem + "mlp.down_proj.lora_down.weight"] = torch.randn(RANK, FF, generator=generator)
    state["diffusion_model.llm2vae.diff"] = torch.randn(LATENT_DIM, HIDDEN, generator=generator) * .01
    state["diffusion_model.llm2vae.diff_b"] = torch.randn(LATENT_DIM, generator=generator) * .01
    state["diffusion_model.vae2llm.diff"] = torch.randn(HIDDEN, LATENT_DIM, generator=generator) * .01
    state["diffusion_model.vae2llm.diff_b"] = torch.randn(HIDDEN, generator=generator) * .01
    save_file(state, str(path), metadata={"yue2_lora_branch": "nar"})
    return state


@pytest.fixture
def adapter(tmp_path):
    path = tmp_path / "adapter.safetensors"
    return path, comfy_adapter(path)


def test_merge_changes_only_the_nar_branch(adapter):
    path, _ = adapter
    model = Stub().to(torch.float32)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}

    record = apply_loras(model, [{"path": str(path), "strength": 1.0}], device="cpu")

    assert record[0]["branches"] == ["diffusion_model"]
    changed = {n for n, p in model.named_parameters() if not torch.equal(p, before[n])}
    assert changed == {n for n in before if ".nar_" in n or n.startswith(("llm2vae", "vae2llm"))}


def test_a_truly_fused_adapter_is_refused_rather_than_split(tmp_path):
    path = tmp_path / "fused.safetensors"
    comfy_adapter(path, fused=True)
    _write_base(tmp_path / "model")
    with pytest.raises(LoraError, match="shares ranks"):
        convert_for_audiocpp(path, tmp_path / "model", tmp_path / "out.safetensors")


def test_fused_entries_are_split_by_the_checkpoints_own_widths(adapter):
    path, state = adapter
    model = Stub().to(torch.float32)
    before = model.model.layers[0].nar_self_attn.k_proj.weight.detach().clone()
    apply_loras(model, [{"path": str(path)}], device="cpu")

    fused = (state["diffusion_model.model.layers.0.self_attn.qkv_proj.lora_up.weight"]
             @ state["diffusion_model.model.layers.0.self_attn.qkv_proj.lora_down.weight"])
    delta = model.model.layers[0].nar_self_attn.k_proj.weight - before
    torch.testing.assert_close(delta, fused[HIDDEN:HIDDEN + KV], atol=1e-6, rtol=1e-5)


def test_strength_scales_the_delta(adapter):
    path, _ = adapter
    full, half = Stub().to(torch.float32), Stub().to(torch.float32)
    half.load_state_dict(full.state_dict())
    base = full.model.layers[0].nar_mlp.up_proj.weight.detach().clone()

    apply_loras(full, [{"path": str(path), "strength": 1.0}], device="cpu")
    apply_loras(half, [{"path": str(path), "strength": 0.5}], device="cpu")
    whole = full.model.layers[0].nar_mlp.up_proj.weight - base
    part = half.model.layers[0].nar_mlp.up_proj.weight - base
    torch.testing.assert_close(part, whole * 0.5, atol=1e-6, rtol=1e-5)


def test_a_second_merge_and_a_wrong_size_are_refused(adapter):
    path, _ = adapter
    model = Stub().to(torch.float32)
    apply_loras(model, [{"path": str(path)}], device="cpu")
    with pytest.raises(LoraError, match="already merged"):
        apply_loras(model, [{"path": str(path)}], device="cpu")

    with pytest.raises(LoraError, match="no module"):
        apply_loras(Stub(layers=1).to(torch.float32), [{"path": str(path)}], device="cpu")


def test_metadata_reads_without_loading_weights(adapter):
    path, _ = adapter
    info = read_metadata(path)
    assert info["branches"] == ["diffusion_model"] and info["layers"] == LAYERS
    assert info["metadata"]["yue2_lora_branch"] == "nar"


def _write_base(directory):
    """The handful of tensors the conversion reads out of the checkpoint."""
    model = Stub().to(torch.float32)
    state = {name: tensor for name, tensor in model.state_dict().items()}
    directory.mkdir(parents=True, exist_ok=True)
    save_file(state, str(directory / "model.safetensors"))
    return state


def test_conversion_produces_exactly_what_audiocpp_walks_for(adapter, tmp_path):
    path, _ = adapter
    base = _write_base(tmp_path / "model")
    out = tmp_path / "converted.safetensors"
    report = convert_for_audiocpp(path, tmp_path / "model", out)

    from safetensors.torch import load_file
    converted = load_file(str(out))

    expected = set(LATENT)
    for layer in range(LAYERS):
        for projection in AUDIOCPP_PROJECTIONS:
            stem = "layers.%d.nar_%s" % (layer, projection)
            expected |= {stem + ".lora_A", stem + ".lora_B"}
    assert set(converted) == expected
    assert report["layers"] == LAYERS

    # A is [rank, in] and B is [out, rank], the orientation the loader requires.
    a = converted["layers.0.nar_self_attn.v_proj.lora_A"]
    b = converted["layers.0.nar_self_attn.v_proj.lora_B"]
    assert tuple(a.shape) == (RANK, HIDDEN) and tuple(b.shape) == (KV, RANK)

    # The latent projections leave as whole tensors, not deltas.
    torch.testing.assert_close(converted["llm2vae.weight"],
                               base["llm2vae.weight"] + _delta(path, "llm2vae.diff"),
                               atol=1e-6, rtol=1e-5)


def _delta(path, key):
    from safetensors.torch import load_file
    return load_file(str(path))["diffusion_model." + key]


def test_converted_and_merged_adapters_agree(adapter, tmp_path):
    """The two engines are handed the same arithmetic, in two layouts."""
    path, _ = adapter
    _write_base(tmp_path / "model")
    out = tmp_path / "converted.safetensors"
    convert_for_audiocpp(path, tmp_path / "model", out)

    from safetensors.torch import load_file
    converted = load_file(str(out))

    model = Stub().to(torch.float32)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    apply_loras(model, [{"path": str(path)}], device="cpu")

    for layer in range(LAYERS):
        for projection in AUDIOCPP_PROJECTIONS:
            stem = "layers.%d.nar_%s" % (layer, projection)
            merged = (model.get_submodule("model." + stem).weight
                      - before["model." + stem + ".weight"])
            rebuilt = converted[stem + ".lora_B"] @ converted[stem + ".lora_A"]
            torch.testing.assert_close(merged, rebuilt, atol=1e-6, rtol=1e-5)


def test_a_converted_file_is_refused_as_a_merge_source(adapter, tmp_path):
    """The two layouts are not interchangeable, and saying so beats half-applying."""
    path, _ = adapter
    _write_base(tmp_path / "model")
    out = tmp_path / "converted.safetensors"
    convert_for_audiocpp(path, tmp_path / "model", out)
    with pytest.raises(LoraError, match="Unexpected key"):
        apply_loras(Stub().to(torch.float32), [{"path": str(out)}], device="cpu")
