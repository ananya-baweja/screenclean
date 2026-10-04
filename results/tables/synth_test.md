# Results: Synthetic text test (300 simulated 512 px crops of text pages)

Generated from `results/jobs/*/summary.json` by `python -m screenclean tables`; don't edit by hand.

| Method | PSNR (dB) ↑ | Δ PSNR vs input | SSIM ↑ | LPIPS ↓ | s / image | Images | Job |
|---|---|---|---|---|---|---|---|
| Input (no cleaning) | 19.37 | – | 0.5728 | 0.4600 | 0.00 | 300 | `0018_eval_synth_test` |
| Ablation: simulator data only | 27.94 | +8.57 | 0.9301 | 0.0956 | 0.07 | 300 | `0018_eval_synth_test` |
| ScreenCleanNet + text fine-tune | 26.21 | +6.85 | 0.8927 | 0.1432 | 0.07 | 300 | `0018_eval_synth_test` |
| ScreenCleanNet (ours, whole image) | 22.53 | +3.17 | 0.8039 | 0.2215 | 0.07 | 300 | `0018_eval_synth_test` |
| Ablation: no dilation | 21.75 | +2.39 | 0.7677 | 0.2694 | 0.07 | 300 | `0018_eval_synth_test` |
| Ablation: full model, short schedule | 21.73 | +2.37 | 0.7688 | 0.2699 | 0.07 | 300 | `0018_eval_synth_test` |
| Ablation: plain U-Net (no wavelets, no dilation) | 21.53 | +2.16 | 0.7616 | 0.2837 | 0.06 | 300 | `0018_eval_synth_test` |
| Ablation: no FFT loss | 20.82 | +1.45 | 0.7494 | 0.3242 | 0.07 | 300 | `0018_eval_synth_test` |
| fft_notch_local | 19.10 | -0.27 | 0.5602 | 0.4658 | 0.27 | 300 | `0018_eval_synth_test` |

- Metrics on uint8 RGB at full resolution; SSIM with a Gaussian window (σ = 1.5).
- Δ PSNR is the mean per-image difference against the unprocessed input.
- Timing: classical baselines on the Colab CPU (2 vCPU); neural networks on a T4 GPU.
- *Reference* rows are published models run with their authors' weights, not our method.

Settings:

- **Ablation: simulator data only**: fp16=True, tile=full image (from checkpoint runs/0016_abl_sim_only/last.pt)
- **ScreenCleanNet + text fine-tune**: fp16=True, tile=full image (from checkpoint runs/0011_finetune_text/best.pt)
- **ScreenCleanNet (ours, whole image)**: fp16=True, tile=full image (from checkpoint runs/0009_train_sc_base/best.pt)
- **Ablation: no dilation**: fp16=True, tile=full image (from checkpoint runs/0015_abl_no_dilation/last.pt)
- **Ablation: full model, short schedule**: fp16=True, tile=full image (from checkpoint runs/0012_abl_full/last.pt)
- **Ablation: plain U-Net (no wavelets, no dilation)**: fp16=True, tile=full image (from checkpoint runs/0014_abl_plain_unet/last.pt)
- **Ablation: no FFT loss**: fp16=True, tile=full image (from checkpoint runs/0013_abl_no_fftloss/last.pt)
- **fft_notch_local**: channels=y, k=4, r0=0.1, sigma=2 (from params/baselines.yaml (job 0003_tune_notch))
