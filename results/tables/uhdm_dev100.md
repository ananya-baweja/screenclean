# Results: UHDM dev100 (100 full-resolution test pairs)

Generated from `results/jobs/*/summary.json` by `python -m screenclean tables`; don't edit by hand.

| Method | PSNR (dB) ↑ | Δ PSNR vs input | SSIM ↑ | LPIPS ↓ | s / image | Images | Job |
|---|---|---|---|---|---|---|---|
| Input (no cleaning) | 17.10 | – | 0.5033 | 0.5201 | 0.03 | 100 | `0017_eval_ablations_dev100` |
| ScreenCleanNet (ours, whole image) | 19.98 | +2.88 | 0.7506 | 0.3111 | 2.37 | 100 | `0010_eval_dev100_models` |
| ScreenCleanNet (ours, 512 px tiles) | 19.83 | +2.73 | 0.7508 | 0.3006 | 3.63 | 100 | `0010_eval_dev100_models` |
| ScreenCleanNet + text fine-tune | 19.81 | +2.72 | 0.7418 | 0.3166 | 2.24 | 100 | `0017_eval_ablations_dev100` |
| Ablation: no FFT loss | 19.33 | +2.24 | 0.7281 | 0.3594 | 2.24 | 100 | `0017_eval_ablations_dev100` |
| Ablation: full model, short schedule | 19.32 | +2.22 | 0.7293 | 0.3371 | 2.21 | 100 | `0017_eval_ablations_dev100` |
| Ablation: plain U-Net (no wavelets, no dilation) | 19.26 | +2.16 | 0.7240 | 0.3463 | 2.06 | 100 | `0017_eval_ablations_dev100` |
| Ablation: no dilation | 19.25 | +2.15 | 0.7314 | 0.3396 | 2.23 | 100 | `0017_eval_ablations_dev100` |
| fft_notch_local | 17.41 | +0.31 | 0.5321 | – | 13.73 | 100 | `0004_eval_baselines_dev100` |
| fft_notch | 17.17 | +0.07 | 0.5045 | – | 12.67 | 100 | `0004_eval_baselines_dev100` |
| chroma_lowpass | 17.13 | +0.03 | 0.5057 | – | 0.78 | 100 | `0004_eval_baselines_dev100` |
| Ablation: simulator data only | 14.78 | -2.32 | 0.5916 | 0.4417 | 2.22 | 100 | `0017_eval_ablations_dev100` |
| ESDNet (reference) | 21.76 | +4.66 | 0.7881 | 0.2696 | 1.16 | 100 | `0005_esdnet_ref_dev100` |

- Metrics on uint8 RGB at full resolution; SSIM with a Gaussian window (σ = 1.5).
- Δ PSNR is the mean per-image difference against the unprocessed input.
- Timing: classical baselines on the Colab CPU (2 vCPU); neural networks on a T4 GPU.
- *Reference* rows are published models run with their authors' weights, not our method.

Settings:

- **ScreenCleanNet (ours, whole image)**: fp16=True, tile=full image (from checkpoint runs/0009_train_sc_base/best.pt)
- **ScreenCleanNet (ours, 512 px tiles)**: fp16=True, tile=512 (from checkpoint runs/0009_train_sc_base/best.pt)
- **ScreenCleanNet + text fine-tune**: fp16=True, tile=full image (from checkpoint runs/0011_finetune_text/best.pt)
- **Ablation: no FFT loss**: fp16=True, tile=full image (from checkpoint runs/0013_abl_no_fftloss/last.pt)
- **Ablation: full model, short schedule**: fp16=True, tile=full image (from checkpoint runs/0012_abl_full/last.pt)
- **Ablation: plain U-Net (no wavelets, no dilation)**: fp16=True, tile=full image (from checkpoint runs/0014_abl_plain_unet/last.pt)
- **Ablation: no dilation**: fp16=True, tile=full image (from checkpoint runs/0015_abl_no_dilation/last.pt)
- **fft_notch_local**: channels=y, k=4, r0=0.1, sigma=2 (from params/baselines.yaml (job 0003_tune_notch))
- **fft_notch**: channels=ycc, k=3, r0=0.08, sigma=4 (from params/baselines.yaml (job 0003_tune_notch))
- **chroma_lowpass**: sigma=8 (from params/baselines.yaml (job 0003_tune_notch))
- **Ablation: simulator data only**: fp16=True, tile=full image (from checkpoint runs/0016_abl_sim_only/last.pt)
- **ESDNet (reference)**: authors' pretrained weights, 5.93 M parameters, fp16; 100 images at full resolution
