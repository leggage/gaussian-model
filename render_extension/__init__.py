from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parent
_BUILD_DIR = _ROOT / "build"
_BUILD_DIR.mkdir(exist_ok=True)

_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        _EXTENSION = load(
            name="render_extension_cuda",
            sources=[
                str(_ROOT / "render.cpp"),
                str(_ROOT / "render_kernel.cu"),
            ],
            build_directory=str(_BUILD_DIR),
            verbose=False,
        )
    return _EXTENSION


def render_forward(colors, cens, inv_covs, weights, inv_k, height, width):
    if not torch.cuda.is_available():
        raise RuntimeError("render_extension requires CUDA")
    return _load_extension().render_forward(
        colors,
        cens,
        inv_covs,
        weights,
        inv_k,
        int(height),
        int(width),
    )
