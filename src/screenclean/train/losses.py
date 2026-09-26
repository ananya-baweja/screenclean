"""Training losses: Charbonnier + FFT amplitude (+ optional VGG perceptual).

``L = charbonnier(y, gt) + fft_weight * fft_amplitude(y, gt) + perc_weight * perceptual(y, gt)``

- **Charbonnier** ``sqrt((y - gt)^2 + eps^2)``: like L1 (robust to outliers, keeps edges sharp) but
  smooth at zero, so gradients don't flip sign abruptly near the target.
- **FFT amplitude** ``mean | |FFT(y)| - |FFT(gt)| |`` per channel: moiré is a set of strong peaks in
  the spectrum, and this term penalises them directly wherever they are in the image. Amplitude
  ignores phase, so it can't pin down *where* things are; the Charbonnier term does that.
  Always computed in float32 with autocast off: cuFFT in half precision only accepts sizes that
  are powers of two (a 384 crop would crash), and float16 spectra lose precision.
- **Perceptual** (VGG16 features, off by default): compares textures instead of pixels.
"""

from __future__ import annotations

import torch
from torch import nn


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    diff = pred.float() - target.float()
    return torch.sqrt(diff * diff + eps * eps).mean()


def fft_amplitude(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute difference of the per-channel 2-D spectrum magnitudes (orthonormal FFT)."""
    with torch.autocast(device_type=pred.device.type, enabled=False):
        p = torch.fft.rfft2(pred.float(), norm="ortho")
        t = torch.fft.rfft2(target.float(), norm="ortho")
        return (p.abs() - t.abs()).abs().mean()


class VGGPerceptual(nn.Module):
    """L1 distance between VGG16 feature maps (relu1_2, relu2_2, relu3_3), with frozen weights.

    ``weights="DEFAULT"`` downloads the ImageNet weights from PyTorch's servers once (no login);
    ``weights=None`` gives random features (tests).
    """

    LAYERS = (3, 8, 15)

    def __init__(self, weights: str | None = "DEFAULT"):
        super().__init__()
        from torchvision.models import VGG16_Weights
        from torchvision.models.vgg import cfgs, make_layers

        features = make_layers(cfgs["D"])  # VGG16's feature layers only (not its 120 M-parameter classifier)
        if weights is not None:
            state = VGG16_Weights[weights].get_state_dict(progress=False)
            features.load_state_dict(
                {k[len("features.") :]: v for k, v in state.items() if k.startswith("features.")}
            )
        features = features[: max(self.LAYERS) + 1].eval()
        for p in features.parameters():
            p.requires_grad_(False)
        self.features = features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = (pred - self.mean) / self.std
        y = (target - self.mean) / self.std
        loss = pred.new_zeros(())
        for i, layer in enumerate(self.features):
            x, y = layer(x), layer(y)
            if i in self.LAYERS:
                loss = loss + (x - y).abs().mean()
        return loss


class TrainLoss(nn.Module):
    """The weighted sum, plus each term as a detached tensor (turn them into numbers with
    :meth:`as_floats` only when logging: reading a GPU value waits for the GPU)."""

    def __init__(
        self,
        fft_weight: float = 0.05,
        perc_weight: float = 0.0,
        eps: float = 1e-3,
        perceptual: nn.Module | None = None,
    ):
        super().__init__()
        self.fft_weight, self.perc_weight, self.eps = fft_weight, perc_weight, eps
        self.perceptual = (
            perceptual if perceptual is not None else (VGGPerceptual() if perc_weight > 0 else None)
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts = {"charbonnier": charbonnier(pred, target, self.eps)}
        if self.fft_weight > 0:
            parts["fft"] = fft_amplitude(pred, target)
        if self.perc_weight > 0 and self.perceptual is not None:
            parts["perceptual"] = self.perceptual(pred, target)
        total = parts["charbonnier"]
        total = (
            total + self.fft_weight * parts.get("fft", 0.0) + self.perc_weight * parts.get("perceptual", 0.0)
        )
        detached = {k: v.detach() for k, v in parts.items()}
        detached["total"] = total.detach()
        return total, detached

    def as_floats(self, parts: dict[str, torch.Tensor]) -> dict[str, float]:
        """Numbers for the log, plus ``fft_share``: the FFT term's part of the total (aim for 10-30%)."""
        out = {k: float(v) for k, v in parts.items()}
        if "fft" in out:
            out["fft_share"] = self.fft_weight * out["fft"] / max(out["total"], 1e-12)
        return out


def suggest_fft_weight(charbonnier_value: float, fft_value: float, share: float = 0.2) -> float:
    """The FFT weight that makes the FFT term ``share`` of the total at these loss values."""
    return share / (1 - share) * charbonnier_value / max(fft_value, 1e-12)
