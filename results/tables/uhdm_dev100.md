# Results: UHDM dev100 (100 full-resolution test pairs)

Generated from `results/jobs/*/summary.json` by `python -m screenclean tables`; don't edit by hand.

| Method | PSNR (dB) ↑ | Δ PSNR vs input | SSIM ↑ | LPIPS ↓ | s / image | Images | Job |
|---|---|---|---|---|---|---|---|
| *(no results yet)* | | | | | | | |

- Metrics on uint8 RGB at full resolution; SSIM with a Gaussian window (σ = 1.5).
- Δ PSNR is the mean per-image difference against the unprocessed input.
- Timing: classical baselines on Colab CPU (2 vCPU); reference models on a T4 GPU.
- *Reference* rows are published models run with their authors' weights, not our method.
