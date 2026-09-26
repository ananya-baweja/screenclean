"""ScreenCleanNet and variants (models/): shapes, sizes, identity start, gradients, Haar = 2x2 conv."""

import pytest
import torch
from torch import nn

from screenclean.models.blocks import LayerNorm2d, NAFBlock
from screenclean.models.complexity import complexity_table, count_flops, count_params, markdown_table
from screenclean.models.registry import build_model, model_names
from screenclean.models.wavelets import haar_dwt


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    return build_model("scnet_tiny")


def test_registry():
    assert model_names() == ["plain_unet", "scnet_base", "scnet_nodil", "scnet_tiny"]
    assert build_model("scnet_base", enc_blocks=1, widths=[8, 16, 32, 64], bottleneck=64).spec.enc_blocks == 1
    with pytest.raises(KeyError):
        build_model("resnet")


@pytest.mark.parametrize("shape", [(1, 3, 37, 53), (2, 3, 64, 48), (1, 3, 5, 7), (1, 3, 16, 16)])
def test_any_size_in_same_size_out_and_identity_at_start(tiny, shape):
    x = torch.rand(shape)
    with torch.no_grad():
        y = tiny(x)
    assert y.shape == x.shape
    assert torch.equal(y, x)  # the last conv starts at zero: an untrained model returns the photo


def test_parameter_counts():
    with torch.device("meta"):
        sizes = {name: count_params(build_model(name)) for name in model_names()}
    assert sizes["scnet_tiny"] < 1_000_000
    assert 2_000_000 <= sizes["scnet_base"] <= 6_000_000
    # the ablations change structure, not size
    assert sizes["plain_unet"] == sizes["scnet_nodil"] == sizes["scnet_base"]


@pytest.mark.parametrize("name", ["scnet_tiny", "scnet_base", "plain_unet"])
def test_forward_backward_on_cpu(name):
    torch.manual_seed(0)
    model = build_model(name)
    x, gt = torch.rand(2, 3, 128, 128), torch.rand(2, 3, 128, 128)
    loss = (model(x) - gt).abs().mean()
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
    assert grads["ending.weight"].abs().sum() > 0


def test_every_layer_learns_within_three_steps():
    torch.manual_seed(0)
    model = build_model("scnet_tiny")
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    x, gt = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    # step 1 trains only the zero-started last conv; step 2 reaches each block's zero-started beta/gamma;
    # from step 3 on, gradients reach the convolutions inside the blocks too
    for _ in range(3):
        opt.zero_grad()
        (model(x) - gt).pow(2).mean().backward()
        opt.step()
    assert model.intro.weight.grad.abs().sum() > 0 and model.middle[0].conv1.weight.grad.abs().sum() > 0


def test_naf_block_starts_as_identity_and_layernorm_normalises():
    block = NAFBlock(8)
    x = torch.randn(2, 8, 10, 10)
    assert torch.equal(block(x), x)  # beta = gamma = 0
    y = LayerNorm2d(8)(x * 5 + 3)
    assert torch.allclose(y.mean(1), torch.zeros(2, 10, 10), atol=1e-5)
    assert torch.allclose(y.var(1, unbiased=False), torch.ones(2, 10, 10), atol=1e-3)


def test_haar_plus_1x1_conv_equals_a_2x2_stride2_conv():
    """The wavelet down step spans exactly the same functions as NAFNet's 2x2 stride-2 conv."""
    torch.manual_seed(0)
    c, o = 3, 5
    conv = nn.Conv2d(c, o, 2, stride=2)
    # a..d = the 2x2 positions (0,0), (0,1), (1,0), (1,1); each is a mix of the LL, LH, HL, HH bands
    inv = 0.5 * torch.tensor(
        [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=torch.float32
    )
    w = conv.weight.detach().reshape(o, c, 4)  # positions in order a, b, c, d
    w1 = torch.einsum("ocp,pk->okc", w, inv).reshape(o, 4 * c, 1, 1)  # band-major, like haar_dwt's output
    x = torch.rand(2, c, 12, 10)
    wavelet_path = nn.functional.conv2d(haar_dwt(x), w1, conv.bias)
    assert torch.allclose(wavelet_path, conv(x), atol=1e-5)


def test_complexity():
    assert count_flops("scnet_tiny", (64, 64)) > 0
    rows = complexity_table(["scnet_tiny"])
    assert rows[0]["params_m"] < 1 and rows[0]["gflops_1920x1080"] > rows[0]["gflops_512x512"]
    assert "`scnet_tiny`" in markdown_table(rows)
