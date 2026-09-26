"""Training losses (train/losses.py)."""

import pytest
import torch

from screenclean.train.losses import TrainLoss, VGGPerceptual, charbonnier, fft_amplitude, suggest_fft_weight


def test_charbonnier():
    x = torch.rand(2, 3, 16, 16)
    assert charbonnier(x, x, eps=1e-3) == pytest.approx(1e-3)
    assert charbonnier(x + 0.5, x) == pytest.approx(0.5, rel=1e-4)  # like L1 far from zero


def test_fft_amplitude_zero_for_identical_and_gradients_flow():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 24, 40)
    assert fft_amplitude(x, x) == 0
    pred = x.clone().requires_grad_(True)
    noisy = x + 0.1 * torch.randn_like(x)
    loss = fft_amplitude(pred, noisy)
    loss.backward()
    assert loss > 0 and torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_fft_amplitude_sees_stripes_but_ignores_circular_shifts():
    x = torch.rand(1, 3, 32, 32)
    shifted = torch.roll(x, shifts=(3, 5), dims=(2, 3))
    assert fft_amplitude(shifted, x) < 1e-5 < charbonnier(shifted, x)  # amplitude ignores position
    yy = torch.arange(32).view(1, 1, 32, 1).float()
    stripes = x + 0.1 * torch.sin(2 * torch.pi * yy / 4)  # a moiré-like periodic pattern
    assert fft_amplitude(stripes, x) > 1e-3  # one peak, averaged over all frequencies


def test_fft_term_is_float32_under_autocast_at_odd_sizes():
    pred = torch.rand(1, 3, 48, 36, requires_grad=True)
    target = torch.rand(1, 3, 48, 36)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = fft_amplitude(pred * 1.0, target)
    assert loss.dtype == torch.float32
    loss.backward()
    assert pred.grad is not None


def test_train_loss_parts_and_share():
    torch.manual_seed(0)
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    target = torch.rand(2, 3, 32, 32)
    loss_fn = TrainLoss(fft_weight=0.05)
    total, tensors = loss_fn(pred, target)
    assert all(not t.requires_grad for t in tensors.values())
    parts = loss_fn.as_floats(tensors)
    assert parts["total"] == pytest.approx(parts["charbonnier"] + 0.05 * parts["fft"], rel=1e-5)
    assert 0 < parts["fft_share"] < 1
    total.backward()
    assert pred.grad.abs().sum() > 0
    w = suggest_fft_weight(parts["charbonnier"], parts["fft"], share=0.2)
    assert w * parts["fft"] / (parts["charbonnier"] + w * parts["fft"]) == pytest.approx(0.2)


def test_perceptual_term_without_downloading_weights():
    torch.manual_seed(0)
    perc = VGGPerceptual(weights=None)
    pred = torch.rand(1, 3, 32, 32, requires_grad=True)
    loss_fn = TrainLoss(fft_weight=0.0, perc_weight=0.1, perceptual=perc)
    total, tensors = loss_fn(pred, torch.rand(1, 3, 32, 32))
    parts = loss_fn.as_floats(tensors)
    total.backward()
    assert parts["perceptual"] > 0 and "fft" not in parts and pred.grad.abs().sum() > 0
    assert not any(p.requires_grad for p in perc.parameters())
