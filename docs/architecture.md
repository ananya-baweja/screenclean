# ScreenCleanNet architecture

A small U-Net that removes moiré from photos of screens. It is built to be **small** (a few million
parameters), to **see far** (moiré bands can span hundreds of pixels), and to **export to ONNX** and run
in a browser (no FFT inside the network; the FFT is only used in the training loss).

```
x (B,3,H,W) in [0,1] ── reflect-pad to a multiple of 16
 └ conv3x3 3→32
 L1 (H)    : NAFBlock ×2 (32) ──────────────────────────────────────────────┐ skip (add)
   down    : Haar DWT (32→128 ch, H/2) → conv1x1 → 64                       │
 L2 (H/2)  : NAFBlock ×2 (64) ─────────────────────────────────────┐        │
   down    : Haar DWT → conv1x1 → 128                               │        │
 L3 (H/4)  : NAFBlock ×2 (128) ───────────────────────────┐        │        │
   down    : Haar DWT → conv1x1 → 256                      │        │        │
 L4 (H/8)  : NAFBlock ×2 (256) ──────────────────┐        │        │        │
   down    : Haar DWT → conv1x1 → 256             │        │        │        │
 B  (H/16) : dilated NAFBlock ×4 (256; depthwise 5×5, dilations 1,2,3,4)   │
   up      : conv1x1 256→1024 → inverse Haar DWT → 256 (H/8) + skip L4 → NAFBlock ×1
   up      : conv1x1 → IDWT → 128 (H/4) + skip L3 → NAFBlock ×1
   up      : conv1x1 → IDWT → 64  (H/2) + skip L2 → NAFBlock ×1
   up      : conv1x1 → IDWT → 32  (H)   + skip L1 → NAFBlock ×1
 └ conv3x3 32→3 (starts at zero) → y = x + residual → remove padding (clamp to [0,1] at inference)
```

Code: `src/screenclean/models/` (`wavelets.py`, `blocks.py`, `scnet.py`, `registry.py`, `complexity.py`).

## Design choices

**Residual output, starting as the identity.** The network predicts a correction that is added to the
photo. The last convolution starts at zero, so an untrained network returns the photo unchanged: training
starts at the input's PSNR and only has to learn what moiré looks like. (Gradients still flow: the first
step trains the last layer, the second each block's scale, and from the third on everything.)

**NAFBlocks** (NAFNet, ECCV 2022): LayerNorm, 1×1 conv, depthwise conv, *SimpleGate* (split the
channels and multiply the halves, the only non-linearity), simplified channel attention, 1×1 conv; then
a small gated feed-forward half. Each half is added back scaled by a learned β or γ that starts at zero,
so the blocks also start as the identity. LayerNorm statistics are computed in float32, which keeps
fp16 training stable.

**Seeing far, cheaply.** Moiré bands are large and slow-varying. At 1/16 resolution, four depthwise 5×5
convolutions with dilations 1, 2, 3 and 4 reach 4·(1+2+3+4) = 40 pixels, about **640 pixels of the
input**, for little compute. Channel attention adds a global summary of each feature map. The bottleneck
holds 41% of the parameters (1.90 M of 4.65 M) but little of the compute, because it works on 1/256 of the
pixels.

**Haar wavelets for down- and up-sampling.** The orthonormal Haar transform splits every 2×2 block into
an average (LL) and horizontal, vertical and diagonal detail (LH, HL, HH), without losing anything. It is
written with slicing and `pixel_shuffle`, so it exports to ONNX (tested). An honest note: a Haar step
followed by a learned 1×1 conv computes **exactly the same family of functions** as NAFNet's 2×2 stride-2
convolution (both are linear maps of each 2×2 block; a unit test checks this), and 1×1 conv + inverse Haar
equals 1×1 conv + PixelShuffle. So the wavelet version can't represent anything new; what differs is the
starting point (an energy-preserving, frequency-split view) and so the training dynamics. The `plain_unet`
ablation measures whether that matters.

**No FFT inside.** The network uses only convolutions, element-wise operations and reshapes, all
supported by ONNX Runtime Web. Frequency awareness comes from the wavelets and from the FFT loss.

## Variants

| Name | What changes | Used for |
|---|---|---|
| `scnet_base` | the diagram above | the model trained and reported |
| `scnet_tiny` | widths 16/32/64/128, one block per level, 2 bottleneck blocks (dilations 1, 2) | CPU tests, the browser demo |
| `plain_unet` | 2×2 stride-2 conv down, 1×1 conv + PixelShuffle up, no dilation | ablation: wavelets and dilation |
| `scnet_nodil` | `scnet_base` without dilation | ablation: dilation alone |

## Size and cost

FLOPs are counted by `torch.utils.flop_counter` (one multiply-add = 2 FLOPs; convolutions and matrix
products only). Regenerate with `python -c "from screenclean.models.complexity import complexity_table,
markdown_table; print(markdown_table(complexity_table()))"`.

| Model | Parameters | GFLOPs 512x512 | GFLOPs 1920x1080 |
|---|---:|---:|---:|
| `plain_unet` | 4.65 M | 52.1 | 415.2 |
| `scnet_base` | 4.65 M | 52.1 | 415.2 |
| `scnet_nodil` | 4.65 M | 52.1 | 415.2 |
| `scnet_tiny` | 0.79 M | 9.8 | 77.9 |

For comparison, the ESDNet reference measured in job 0005 has 5.93 M parameters.

## Training (details in `src/screenclean/train/`)

- **Loss:** Charbonnier (a smooth L1) + λ·FFT amplitude (per channel, orthonormal, always in float32) +
  optional VGG perceptual (off). λ starts at 0.05 and is calibrated so the FFT term is about 20% of the loss
  at the start (measured by job 0008).
- **Optimiser:** AdamW (lr 3e-4, betas 0.9/0.99, weight decay 1e-4), 1k-iteration warmup then cosine decay to
  1e-6, gradient clipping at 1.0, fp16 autocast with a gradient scaler (the T4 has no bf16), EMA of the weights
  (decay 0.999) used for every evaluation.
- **Data:** random crops (384) with flips and 90° rotations. Every sample is fixed by its number in the run,
  so a run that resumes after a Colab disconnect sees exactly the data it would have seen; sources (UHDM,
  synthetic text) can be mixed by weight.
- **Checkpoints:** `last.pt` every 15 minutes and at the time budget (written to a temp file, then renamed),
  `best.pt` by validation PSNR; runs resume automatically.
