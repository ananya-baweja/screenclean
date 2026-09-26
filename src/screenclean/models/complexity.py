"""Model size and cost: parameters and FLOPs (counted on the "meta" device, so no memory is used).

FLOPs are counted by ``torch.utils.flop_counter.FlopCounterMode``: one multiply-add = 2 FLOPs.
Only convolutions and matrix products are counted; element-wise work (norms, gates) is not.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from screenclean.models.registry import build_model, model_names

SIZES = {"512x512": (512, 512), "1920x1080": (1080, 1920)}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(name: str, size: tuple[int, int]) -> int:
    """Forward-pass FLOPs of a registered model for one (h, w) image."""
    with torch.device("meta"):
        model = build_model(name).eval()
        x = torch.empty(1, 3, *size)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(x)
    return int(counter.get_total_flops())


def complexity_table(names: list[str] | None = None) -> list[dict]:
    """Parameters (millions) and GFLOPs at 512x512 and 1920x1080 for each model."""
    rows = []
    for name in names or model_names():
        with torch.device("meta"):
            params = count_params(build_model(name))
        row = {"model": name, "params_m": round(params / 1e6, 3)}
        for label, size in SIZES.items():
            row[f"gflops_{label}"] = round(count_flops(name, size) / 1e9, 1)
        rows.append(row)
    return rows


def markdown_table(rows: list[dict]) -> str:
    lines = ["| Model | Parameters | GFLOPs 512x512 | GFLOPs 1920x1080 |", "|---|---:|---:|---:|"]
    for r in rows:
        gflops = f"{r['gflops_512x512']:.1f} | {r['gflops_1920x1080']:.1f}"
        lines.append(f"| `{r['model']}` | {r['params_m']:.2f} M | {gflops} |")
    return "\n".join(lines)
