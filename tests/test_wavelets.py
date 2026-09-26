"""Haar wavelet layers (models/wavelets.py): exactness, orthonormality, sub-band meaning, ONNX export."""

import numpy as np
import pytest
import torch

from screenclean.models.wavelets import HaarDWT, HaarIDWT, haar_dwt, haar_idwt


@pytest.mark.parametrize("shape", [(1, 3, 8, 8), (2, 5, 32, 18), (1, 1, 2, 2)])
def test_perfect_reconstruction_and_energy(shape):
    x = torch.randn(shape, dtype=torch.float64)
    y = haar_dwt(x)
    assert y.shape == (shape[0], 4 * shape[1], shape[2] // 2, shape[3] // 2)
    assert (haar_idwt(y) - x).abs().max() < 1e-12
    assert torch.allclose((y**2).sum(), (x**2).sum())  # orthonormal: energy preserved
    x32 = x.float()
    assert (haar_idwt(haar_dwt(x32)) - x32).abs().max() < 1e-6


def test_subbands_mean_what_they_say():
    flat = torch.full((1, 1, 4, 4), 0.3)
    ll, lh, hl, hh = haar_dwt(flat).chunk(4, dim=1)
    assert torch.allclose(ll, torch.full_like(ll, 0.6))
    assert lh.abs().max() == hl.abs().max() == hh.abs().max() == 0
    stripes = torch.tensor([[1.0, 0.0] * 2] * 4)[None, None]  # alternating columns: vertical edges
    ll, lh, hl, hh = haar_dwt(stripes).chunk(4, dim=1)
    assert lh.abs().min() > 0 and hl.abs().max() == 0 and hh.abs().max() == 0


def test_odd_sizes_are_rejected():
    with pytest.raises(ValueError):
        haar_dwt(torch.zeros(1, 3, 5, 6))
    with pytest.raises(ValueError):
        haar_idwt(torch.zeros(1, 6, 4, 4))


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # the legacy exporter announces its removal
@pytest.mark.parametrize("dynamo", [False, pytest.param(True, marks=pytest.mark.slow)])  # dynamo: ~20 s
def test_dwt_idwt_exports_to_onnx_and_runs(tmp_path, dynamo):
    ort = pytest.importorskip("onnxruntime")

    class RoundTrip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dwt, self.idwt = HaarDWT(), HaarIDWT()

        def forward(self, x):
            y = self.dwt(x)
            return self.idwt(y * 1.5), y

    model = RoundTrip().eval()
    x = torch.rand(1, 3, 16, 24)
    path = tmp_path / "haar.onnx"
    if dynamo:
        half_h, half_w = (
            torch.export.Dim("half_h", min=1, max=4096),
            torch.export.Dim("half_w", min=1, max=4096),
        )
        dynamic = {"dynamic_shapes": ({2: 2 * half_h, 3: 2 * half_w},), "verbose": False}  # any even size
    else:
        dynamic = {"dynamic_axes": {"x": {2: "h", 3: "w"}}, "opset_version": 18}
    torch.onnx.export(
        model, (x,), str(path), input_names=["x"], output_names=["xr", "bands"], dynamo=dynamo, **dynamic
    )
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    other = torch.rand(1, 3, 40, 32)  # a different size: the axes are dynamic
    for inp in (x, other):
        xr, bands = sess.run(None, {"x": inp.numpy()})
        ref_xr, ref_bands = model(inp)
        np.testing.assert_allclose(xr, ref_xr.detach().numpy(), atol=1e-5)
        np.testing.assert_allclose(bands, ref_bands.detach().numpy(), atol=1e-5)
