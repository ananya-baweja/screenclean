"""Plot training curves of ingested ``train`` jobs: results/figures/curves_<job id>.png.

Panels: training loss (raw and smoothed) with the Charbonnier and FFT parts, the FFT term's share,
the learning rate, validation PSNR / SSIM against the input's PSNR, and speed.

    python tools/plot_curves.py 0009_train_sc_base [more job ids] [--results results]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    return {k: np.array([float(r[k]) if r[k] not in ("", None) else np.nan for r in rows]) for k in rows[0]}


def smooth(y: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over ``n`` points (shorter at the ends)."""
    if len(y) < 3 or n < 2:
        return y
    k = np.ones(min(n, len(y))) / min(n, len(y))
    num = np.convolve(np.nan_to_num(y), k, mode="same")
    den = np.convolve(np.ones_like(y), k, mode="same")
    return num / den


def plot(job_dir: Path, out: Path) -> Path:
    tr = read_csv(job_dir / "train_log.csv")
    va = read_csv(job_dir / "val_log.csv") if (job_dir / "val_log.csv").exists() else {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    it = tr["iter"]
    n = max(3, len(it) // 25)

    ax = axes[0, 0]
    ax.plot(it, tr["loss"], color="0.8", lw=0.8, label="loss (each log step)")
    ax.plot(it, smooth(tr["loss"], n), color="C0", lw=2, label="loss (smoothed)")
    ax.plot(it, smooth(tr["charbonnier"], n), color="C1", lw=1.2, label="Charbonnier part")
    ax.set_title("Training loss (lower is better)")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if va:
        ax.plot(va["iter"], va["psnr"], "o-", color="C2", label="EMA model (val)")
        ax.axhline(va["input_psnr"][0], color="0.4", ls="--", label=f"input {va['input_psnr'][0]:.2f} dB")
        best = int(np.nanargmax(va["psnr"]))
        ax.annotate(f"best {va['psnr'][best]:.2f} dB", (va["iter"][best], va["psnr"][best]),
                    textcoords="offset points", xytext=(-30, -16), fontsize=8)  # fmt: skip
        ax2 = ax.twinx()
        ax2.plot(va["iter"], va["ssim"], "s:", color="C4", ms=4, label="SSIM (right axis)")
        ax2.set_ylabel("SSIM")
        ax2.legend(fontsize=8, loc="lower right")
    ax.set_title("Validation PSNR on 400 crops (higher is better)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("dB")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    ax.plot(it, tr["lr"], color="C3")
    ax.set_title("Learning rate (warmup, then cosine decay)")
    ax.set_xlabel("iteration")
    if "fft_share" in tr:
        ax2 = ax.twinx()
        ax2.plot(it, smooth(tr["fft_share"], n), color="C5", lw=1, label="FFT share of the loss (right)")
        ax2.set_ylim(0, max(0.05, float(np.nanmax(tr["fft_share"])) * 1.2))
        ax2.legend(fontsize=8, loc="upper right")

    ax = axes[1, 1]
    ax.plot(it, tr["it_per_s"], color="C0", lw=1, label="iterations / s")
    ax.plot(it, tr["data_wait"] * 10, color="C1", lw=1, label="waiting for data (share x 10)")
    ax.set_title("Speed")
    ax.set_xlabel("iteration")
    ax.legend(fontsize=8)

    fig.suptitle(job_dir.name)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    for job in args.jobs:
        path = plot(args.results / "jobs" / job, args.results / "figures" / f"curves_{job}.png")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
