# Results: UHDM test set (all 500 full-resolution pairs; used once, in P9)

Generated from `results/jobs/*/summary.json` by `python -m screenclean tables`; don't edit by hand.

| Method | PSNR (dB) ↑ | Δ PSNR vs input | SSIM ↑ | LPIPS ↓ | s / image | Images | Job |
|---|---|---|---|---|---|---|---|
| *(no results yet)* | | | | | | | |

- Metrics on uint8 RGB at full resolution; SSIM with a Gaussian window (σ = 1.5).
- Δ PSNR is the mean per-image difference against the unprocessed input.
- Timing: classical baselines on the Colab CPU (2 vCPU); neural networks on a T4 GPU.
- *Reference* rows are published models run with their authors' weights, not our method.
