# Results: UHDM test set (all 500 full-resolution pairs; used once, in P9)

Generated from `results/jobs/*/summary.json` by `python -m screenclean tables`; don't edit by hand.

| Method | PSNR (dB) ↑ | Δ PSNR vs input | SSIM ↑ | LPIPS ↓ | s / image | Images | Job |
|---|---|---|---|---|---|---|---|
| Input (no cleaning) | 17.12 | – | 0.5089 | 0.5314 | 0.10 | 500 | `0022_eval_test500_classical` |
| ScreenCleanNet (before fine-tune) | 20.15 | +3.04 | 0.7581 | 0.3012 | 2.42 | 500 | `0021_eval_test500_models` |
| ScreenCleanNet (final) | 19.95 | +2.83 | 0.7483 | 0.3116 | 2.43 | 500 | `0021_eval_test500_models` |
| fft_notch_local | 17.46 | +0.35 | 0.5392 | – | 16.20 | 500 | `0022_eval_test500_classical` |
| fft_notch | 17.19 | +0.07 | 0.5105 | – | 15.18 | 500 | `0022_eval_test500_classical` |
| chroma_lowpass | 17.14 | +0.03 | 0.5112 | – | 0.96 | 500 | `0022_eval_test500_classical` |
| ESDNet (reference) | 22.12 | +5.00 | 0.7949 | 0.2524 | 1.33 | 500 | `0021_eval_test500_models` |

- Metrics on uint8 RGB at full resolution; SSIM with a Gaussian window (σ = 1.5).
- Δ PSNR is the mean per-image difference against the unprocessed input.
- Timing: classical baselines on the Colab CPU (2 vCPU); neural networks on a T4 GPU.
- *Reference* rows are published models run with their authors' weights, not our method.

Settings:

- **ScreenCleanNet (before fine-tune)**: fp16=True, tile=full image (from checkpoint runs/0009_train_sc_base/best.pt)
- **ScreenCleanNet (final)**: fp16=True, tile=full image (from checkpoint runs/0011_finetune_text/best.pt)
- **fft_notch_local**: channels=y, k=4, r0=0.1, sigma=2 (from params/baselines.yaml (job 0003_tune_notch))
- **fft_notch**: channels=ycc, k=3, r0=0.08, sigma=4 (from params/baselines.yaml (job 0003_tune_notch))
- **chroma_lowpass**: sigma=8 (from params/baselines.yaml (job 0003_tune_notch))
- **ESDNet (reference)**: authors' pretrained weights, 5.93 M parameters, fp16; 91 images at full resolution
