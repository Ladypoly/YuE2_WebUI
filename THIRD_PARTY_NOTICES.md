# Third-party code notices

The Oobleck VAE and SnakeBeta implementation in `modeling_vae.py` is derived
from stable-audio-tools commit `a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45`.
The module hierarchy, weight normalization and activation equations preserve
the checkpoint's original inference implementation.

- Oobleck / stable-audio-tools: Copyright (c) 2023 Stability AI, MIT.
  Full text: `licenses/stable-audio-tools-MIT.txt`.
- SnakeBeta / BigVGAN: Copyright (c) 2022 NVIDIA CORPORATION, MIT.
  Full text: `licenses/SnakeBeta-NVIDIA-MIT.txt`.

The field vocabulary in `webui/vocabulary.json` — the genre, tempo, meter, key,
lyrics, language, voice, theme and length lists offered by the console's song
sheet — is taken from the structured prompt node of
ComfyUI-MiniMax-Music-Production-Toolkit 3.1.2.

- ComfyUI-MiniMax-Music-Production-Toolkit: Copyright (c) Johannes Plenio, MIT.
  Source: https://github.com/jplenio/ComfyUI-MiniMax-Music-Production-Toolkit

These notices cover the identified source code and retain its original licenses.
The YuE2 model checkpoint weights are separately licensed under CC BY-NC 4.0;
see MODEL_LICENSE for the scope and full terms. This does not relicense third-party code.
