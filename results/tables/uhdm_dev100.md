# Results: UHDM dev100 (100 full-resolution test pairs)

Generated from `results/jobs/*/summary.json` by `python -m screenclean tables`; don't edit by hand.

| Method | PSNR (dB) ↑ | Δ PSNR vs input | SSIM ↑ | LPIPS ↓ | s / image | Images | Job |
|---|---|---|---|---|---|---|---|
| Input (no cleaning) | 17.10 | – | 0.5033 | 0.5201 | 0.04 | 100 | `0005_esdnet_ref_dev100` |
| fft_notch_local | 17.41 | +0.31 | 0.5321 | – | 13.73 | 100 | `0004_eval_baselines_dev100` |
| fft_notch | 17.17 | +0.07 | 0.5045 | – | 12.67 | 100 | `0004_eval_baselines_dev100` |
| chroma_lowpass | 17.13 | +0.03 | 0.5057 | – | 0.78 | 100 | `0004_eval_baselines_dev100` |
| ESDNet (reference) | 21.76 | +4.66 | 0.7881 | 0.2696 | 1.16 | 100 | `0005_esdnet_ref_dev100` |

- Metrics on uint8 RGB at full resolution; SSIM with a Gaussian window (σ = 1.5).
- Δ PSNR is the mean per-image difference against the unprocessed input.
- Timing: classical baselines on Colab CPU (2 vCPU); reference models on a T4 GPU.
- *Reference* rows are published models run with their authors' weights, not our method.

Settings:

- **fft_notch_local**: channels=y, k=4, r0=0.1, sigma=2 (from params/baselines.yaml (job 0003_tune_notch))
- **fft_notch**: channels=ycc, k=3, r0=0.08, sigma=4 (from params/baselines.yaml (job 0003_tune_notch))
- **chroma_lowpass**: sigma=8 (from params/baselines.yaml (job 0003_tune_notch))
- **ESDNet (reference)**: defaults (from config); inference: {'full': 100}
