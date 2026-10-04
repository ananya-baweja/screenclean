"""``docs/RESULTS.md`` and its figures, generated from the job results (P9.5).

Every number comes from ``results/jobs/*/summary.json`` (or a CSV next to it), so nothing is
typed by hand. Sections whose jobs haven't run yet say which job will fill them. Re-run after
ingesting results::

    python -m screenclean report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from screenclean.eval.tables import TABLES, collect

log = logging.getLogger(__name__)

# Reference chart palette (light surface): categorical slots in fixed order, ink and surface.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
ENGINE_COLORS = [BLUE, ORANGE, AQUA]


# --------------------------------------------------------------------------- loading


def load_summaries(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Job id -> summary, for every ingested job."""
    out = {}
    for p in sorted(results_dir.glob("*/summary.json")):
        out[p.parent.name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def find_job(summaries: dict[str, dict[str, Any]], task: str, contains: str = "") -> dict[str, Any] | None:
    """The newest finished job of ``task`` whose id contains ``contains``."""
    hits = [s for jid, s in sorted(summaries.items()) if s.get("task") == task and contains in jid]
    return hits[-1] if hits else None


# --------------------------------------------------------------------------- markdown helpers


def _ci(entry: dict[str, float], digits: int = 3, signed: bool = False) -> str:
    fmt = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{fmt.format(entry['mean'])} [{fmt.format(entry['lo'])}, {fmt.format(entry['hi'])}]"


def _pending(job: str, what: str) -> list[str]:
    return [f"*Pending: job `{job}` ({what}).*", ""]


def quality_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Method | PSNR (dB) ↑ | Δ PSNR vs input | SSIM ↑ | LPIPS ↓ | s / image | Images |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        gain = r.get("psnr_gain_vs_input")
        lpips = r.get("lpips")
        label = (
            f"{r['label']} *(reference)*"
            if r.get("reference") and "reference" not in r["label"]
            else r["label"]
        )
        lines.append(
            f"| {label} | {r['psnr']:.2f} | {'–' if gain is None else f'{gain:+.2f}'} | {r['ssim']:.4f} | "
            f"{'–' if lpips is None else f'{lpips:.4f}'} | {r['seconds']:.2f} | {r['n']} |"
        )
    return lines + [""]


def ablation_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Ablation variants with dev100 and synthetic-text PSNR, relative to the short-schedule reference."""
    dev = {r["label"]: r for r in collect(results_dir, TABLES["uhdm_dev100"][0])}
    syn = {r["label"]: r for r in collect(results_dir, TABLES["synth_test"][0])}
    ref = "Ablation: full model, short schedule"
    if ref not in dev:
        return []
    rows = []
    for label in [ref, *sorted(k for k in dev if k.startswith("Ablation:") and k != ref)]:
        row = {"label": label, "dev_psnr": dev[label]["psnr"], "dev_lpips": dev[label].get("lpips")}
        row["dev_delta"] = row["dev_psnr"] - dev[ref]["psnr"]
        if label in syn and ref in syn:
            row["syn_psnr"], row["syn_delta"] = syn[label]["psnr"], syn[label]["psnr"] - syn[ref]["psnr"]
        rows.append(row)
    return rows


def ocr_tables(ocr: dict[str, Any], methods: list[str] | None = None) -> list[str]:
    """One table per engine: CER and word F1 with 95% intervals, and the paired change vs the baseline."""
    lines = []
    for engine, data in ocr.items():
        per = data["methods"]
        order = [m for m in (methods or per) if m in per]
        lines += [
            f"**{engine}**",
            "",
            "| Method | CER ↓ [95% CI] | Δ CER vs baseline [95% CI] | Word F1 ↑ [95% CI] | Pages |",
            "|---|---|---|---|---|",
        ]
        for m in order:
            e = per[m]
            delta = _ci(e["cer_vs_input"], signed=True) if "cer_vs_input" in e else "–"
            lines.append(f"| {m} | {_ci(e['cer'])} | {delta} | {_ci(e['word_f1'])} | {e['n']} |")
        lines.append("")
        for c in data.get("comparisons", []):
            verdict = "reads worse" if c["a_reads_worse"] else "not worse"
            lines.append(
                f"- {c['a']} minus {c['b']}: CER {_ci(c['cer_a_minus_b'], signed=True)}, "
                f"word F1 {_ci(c['word_f1_a_minus_b'], signed=True)} ({verdict})"
            )
        lines.append("")
    return lines


# --------------------------------------------------------------------------- figures


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK_2})
    return plt


def fig_ablations(rows: list[dict[str, Any]], out: Path) -> Path | None:
    """Two panels (never two y-axes): change vs the reference on dev100, and on synthetic text."""
    rows = [r for r in rows if not r["label"].endswith("short schedule")]
    if not rows:
        return None
    plt = _plt()
    labels = [r["label"].removeprefix("Ablation: ") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.55 * len(rows) + 1.4), sharey=True, facecolor=SURFACE)
    for ax, key, title in zip(
        axes, ("dev_delta", "syn_delta"), ("UHDM dev100 (real photos)", "Synthetic text crops"), strict=True
    ):
        vals = [r.get(key, np.nan) for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, vals, height=0.5, color=[BLUE if v >= 0 else ORANGE for v in vals])
        ax.axvline(0, color=INK_2, lw=0.8)
        for yi, v in zip(y, vals, strict=True):
            if np.isfinite(v):
                ax.annotate(f"{v:+.2f}", (v, yi), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                            ha="left" if v >= 0 else "right", va="center", fontsize=8, color=INK)  # fmt: skip
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.set_xlabel("PSNR change vs the full model, same schedule (dB)")
        finite = [abs(v) for v in vals if np.isfinite(v)]
        lim = max([1.0, *(1.35 * v for v in finite)])
        ax.set_xlim(-lim, lim)
        _style(ax)
    fig.suptitle(
        "Ablations: what each ingredient is worth (8,000 iterations each, one run)", x=0.01, ha="left"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_psnr_vs_params(results_dir: Path, out: Path, split_key: str) -> Path | None:
    """Measured PSNR against model size (classical filters at 0 parameters)."""
    rows = [
        r for r in collect(results_dir, split_key) if r["name"] != "identity" and "Ablation" not in r["label"]
    ]
    rows = [r for r in rows if "tiles" not in r["label"]]
    if not rows:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    base = next((r["psnr"] - r["psnr_gain_vs_input"] for r in rows if "psnr_gain_vs_input" in r), None)
    classical = [r for r in rows if not r.get("parameters")]
    best_classical = max(classical, key=lambda r: r["psnr"]) if classical else None
    xmax = max(8.0, max(r.get("parameters", 0) / 1e6 for r in rows) * 1.6)
    for r in rows:
        p = r.get("parameters", 0) / 1e6
        ours = r["name"] == "scnet"
        color = BLUE if ours else (ORANGE if r.get("reference") else INK_2)
        ax.scatter(p, r["psnr"], s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        if r in classical and r is not best_classical:
            continue  # the classical filters sit on top of each other: label the best one only
        label = f"classical filters (best: {r['label']})" if r is best_classical else r["label"]
        ax.annotate(label, (p, r["psnr"]), xytext=(8, 2), textcoords="offset points", fontsize=8, color=INK)
    if base is not None:
        ax.axhline(base, color=GRID, lw=1, ls="--")
        ax.annotate("input (no cleaning)", (xmax, base), xytext=(-4, 4), textcoords="offset points",
                    fontsize=8, color=INK_2, ha="right")  # fmt: skip
    ax.set_xlabel("parameters (millions; classical filters have none)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlim(-0.5, xmax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    _style(ax)
    ax.set_title(f"Quality vs size, measured on {split_key} (blue: ours, orange: published reference)",
                 fontsize=10, loc="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_ocr(ocr: dict[str, Any], out: Path, title: str, methods: list[str] | None = None) -> Path | None:
    """Mean CER with 95% intervals per method, one bar per engine."""
    if not ocr:
        return None
    plt = _plt()
    engines = list(ocr)
    names = methods or list(next(iter(ocr.values()))["methods"])
    names = [m for m in names if any(m in ocr[e]["methods"] for e in engines)]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(names) * len(engines) + 1.5), facecolor=SURFACE)
    h = 0.8 / len(engines)
    y = np.arange(len(names))
    for i, (engine, color) in enumerate(zip(engines, ENGINE_COLORS, strict=False)):
        per = ocr[engine]["methods"]
        mean = np.array([per[m]["cer"]["mean"] if m in per else np.nan for m in names])
        lo = np.array([per[m]["cer"]["lo"] if m in per else np.nan for m in names])
        hi = np.array([per[m]["cer"]["hi"] if m in per else np.nan for m in names])
        ys = y + (i - (len(engines) - 1) / 2) * h
        ax.barh(ys, mean, height=h * 0.85, color=color, label=engine)
        ax.errorbar(mean, ys, xerr=[mean - lo, hi - mean], fmt="none", ecolor=INK_2, elinewidth=1, capsize=2)
    ax.set_yticks(y, [m.split(":", 1)[-1] for m in names])
    ax.invert_yaxis()
    ax.set_xlabel("character error rate (lower is better; bars: 95% bootstrap interval)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(title, fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


def _strip_panels(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """An evaluation sample (input | output | target side by side) as its three panels."""
    from screenclean.utils.io import read_image

    img = read_image(path)
    w = img.shape[1] // 3
    return img[:, :w], img[:, w : 2 * w], img[:, 2 * w : 3 * w]


def fig_before_after(samples_dir: Path, out: Path, columns: dict[str, str], n_rows: int = 3) -> Path | None:
    """Rows of sample crops: input, then each method's output, then the target."""
    from screenclean.utils.io import write_image

    keys = sorted({p.name.split("__", 1)[1] for p in samples_dir.glob("*.jpg")})[:n_rows]
    if not keys:
        return None
    rows = []
    for key in keys:
        panels = []
        for i, (prefix, _) in enumerate(columns.items()):
            f = samples_dir / f"{prefix}__{key}"
            if not f.exists():
                return None
            inp, outp, tgt = _strip_panels(f)
            panels += ([inp] if i == 0 else []) + [outp]
        panels.append(tgt)
        rows.append(
            np.concatenate([np.pad(p, ((4, 4), (4, 4), (0, 0)), constant_values=1.0) for p in panels], 1)
        )
    write_image(out, np.concatenate(rows, 0), quality=88)
    return out


def _log_spectrum(img: np.ndarray) -> np.ndarray:
    luma = img @ np.array([0.299, 0.587, 0.114])
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(luma - luma.mean()))))


def fig_spectra(samples_dir: Path, out: Path, method_prefix: str) -> Path | None:
    """Log power spectra (luma) of input, output and target for the sample whose input differs most
    from its target in the spectrum (moiré adds strong frequencies the clean page doesn't have)."""
    files = sorted(samples_dir.glob(f"{method_prefix}__*.jpg"))
    if not files:
        return None
    strips = [_strip_panels(f) for f in files]
    panels = max(strips, key=lambda s: float(np.mean(np.abs(_log_spectrum(s[0]) - _log_spectrum(s[2])))))
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), facecolor=SURFACE)
    for ax, img, title in zip(axes, panels, ("input (photo)", "final model", "target"), strict=True):
        spec = _log_spectrum(img)
        ax.imshow(spec, cmap="Greys", vmin=np.percentile(spec, 5), vmax=np.percentile(spec, 99.9))
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Power spectra of a synthetic text crop (centre = low frequencies; darker = stronger)",
                 x=0.01, ha="left")  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- report


def build_report(repo_root: Path) -> list[Path]:
    """Write ``docs/RESULTS.md`` and the figures it uses; returns the files written."""
    results = repo_root / "results" / "jobs"
    figs = repo_root / "results" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    s = load_summaries(results)
    written: list[Path] = []
    md = [
        "# Results",
        "",
        "Generated by `python -m screenclean report` from `results/jobs/*/summary.json`; don't edit by hand.",
        "Intervals are 95% paired-bootstrap confidence intervals (10,000 resamples, seed 0).",
        "",
    ]

    # 1. UHDM
    md += ["## 1. Moiré removal on real 4K photos (UHDM)", ""]
    test500 = collect(results, TABLES["uhdm_test500"][0])
    md += ["### Full test set (500 pairs; used once)", ""]
    md += (
        quality_table(test500)
        if test500
        else _pending("0021/0022_eval_test500", "the 500-pair UHDM test set")
    )
    md += ["### dev100 (100 test pairs used during development)", ""]
    md += quality_table(
        [r for r in collect(results, TABLES["uhdm_dev100"][0]) if "Ablation" not in r["label"]]
    )

    # 2. synthetic text, image quality
    md += ["## 2. Synthetic text pages (300 simulated crops)", ""]
    md += quality_table(
        [r for r in collect(results, TABLES["synth_test"][0]) if "Ablation" not in r["label"]]
    )

    # 3. ablations
    rows = ablation_rows(results)
    md += [
        "## 3. Ablations",
        "",
        "Each variant trained from scratch for 8,000 iterations and scored on its final weights",
        "(one run each; differences under ~0.1 dB are within noise).",
    ]
    md += [
        "",
        "| Variant | dev100 PSNR | Δ vs full | dev100 LPIPS | text PSNR | Δ vs full |",
        "|---|---|---|---|---|---|",
    ]
    nan = float("nan")
    for r in rows:
        md.append(
            f"| {r['label'].removeprefix('Ablation: ')} | {r['dev_psnr']:.2f} | {r['dev_delta']:+.2f} | "
            f"{r['dev_lpips']:.3f} | {r.get('syn_psnr', nan):.2f} | {r.get('syn_delta', nan):+.2f} |"
        )
    md.append("")
    if p := fig_ablations(rows, figs / "ablations.png"):
        written.append(p)
        md += ["![Ablations](../results/figures/ablations.png)", ""]

    # 4. OCR on synthetic text
    md += ["## 4. OCR on synthetic text crops", ""]
    synth = find_job(s, "eval_synth_ocr")
    if synth:
        md += [f"Engines: {_engines(synth)}", ""] + ocr_tables(synth["ocr"])
        if p := fig_ocr(synth["ocr"], figs / "ocr_synth.png", "OCR errors on 300 synthetic text crops"):
            written.append(p)
            md += ["![OCR on synthetic text](../results/figures/ocr_synth.png)", ""]
    else:
        md += _pending("0019_eval_synth_ocr", "PaddleOCR and Tesseract on the synthetic text crops")

    # 5 + 6. real photos
    real = find_job(s, "eval_real_ocr")
    md += ["## 5. OCR on real phone photos (same geometry for every method)", ""]
    if real:
        oracle = {
            e: {**d, "methods": {k: v for k, v in d["methods"].items() if k.startswith("oracle:")},
                "comparisons": [c for c in d.get("comparisons", []) if c["a"].startswith("oracle:")]}
            for e, d in real["ocr"].items()
        }  # fmt: skip
        md += [f"{real['n_photos']} photos. Engines: {_engines(real)}", ""] + ocr_tables(oracle)
        if p := fig_ocr(oracle, figs / "ocr_real.png", f"OCR errors on {real['n_photos']} real phone photos"):
            written.append(p)
            md += ["![OCR on real photos](../results/figures/ocr_real.png)", ""]
    else:
        md += _pending("0025_eval_real_ocr", "needs the phone photos on Drive and job 0024")
    md += ["## 6. The product end to end", ""]
    if real and real.get("detection"):
        d = real["detection"]
        md += [
            f"- Screen found: {d['detected_rate']:.0%} of photos; corners right ({d['success_rule']}): "
            f"**{d['success_rate']:.0%}**; median corner error {d['median_mean_err_px']:.1f} px.",
            "",
        ]
        product = {
            e: {**v, "methods": {k: x for k, x in v["methods"].items() if k.startswith("product:")},
                "comparisons": [c for c in v.get("comparisons", []) if c["a"].startswith("product:")]}
            for e, v in real["ocr"].items()
        }  # fmt: skip
        md += ocr_tables(product)
        md += ["| Cleaner | Seconds per photo (median) | of which cleaning |", "|---|---|---|"]
        for label, t in sorted(real.get("seconds_per_photo", {}).items()):
            md.append(
                f"| {label} | {t.get('total', float('nan')):.1f} | {t.get('clean', float('nan')):.1f} |"
            )
        md.append("")
    else:
        md += _pending("0025_eval_real_ocr", "screen detection, end-to-end CER by cleaner, seconds per photo")

    # 7. efficiency
    md += ["## 7. Size and speed", ""]
    eff = find_job(s, "efficiency")
    if eff:
        sizes = list(eff["sizes"])
        head = (
            "| Model | Parameters | "
            + " | ".join(f"GFLOPs {k} | T4 ms {k} | CPU s {k}" for k in sizes)
            + " |"
        )
        md += [head, "|" + "---|" * (2 + 3 * len(sizes))]
        for r in eff["rows"]:
            cells = [f"{r.get(f'gflops_{k}', float('nan')):.0f} | {r.get(f'gpu_ms_{k}', float('nan')):.0f} | "
                     f"{r.get(f'cpu_s_{k}', float('nan')):.2f}" for k in sizes]  # fmt: skip
            md.append(f"| {r['label']} | {r['params_m']:.2f} M | " + " | ".join(cells) + " |")
        md += ["", f"GPU: {eff['gpu']} (fp16); CPU: {eff['cpu_threads']} threads (fp32). {eff['notes']}.", ""]
    else:
        md += _pending("0023_efficiency", "parameters, FLOPs, T4 and CPU latency")

    # figures that need no new job
    md += ["## Figures", ""]
    split_key = TABLES["uhdm_test500"][0] if test500 else TABLES["uhdm_dev100"][0]
    if p := fig_psnr_vs_params(results, figs / "psnr_vs_params.png", split_key):
        written.append(p)
        md += ["![PSNR vs parameters](../results/figures/psnr_vs_params.png)", ""]
    samples = results / "0018_eval_synth_test" / "samples"
    cols = {
        "fft_notch_local": "classical",
        "ScreenCleanNet_ours_whole_image": "before",
        "ScreenCleanNet_text_fine-tune": "final",
    }
    if p := fig_before_after(samples, figs / "before_after_text.jpg", cols):
        written.append(p)
        md += [
            "Synthetic text crops, left to right: photo, classical filter, ScreenCleanNet before the text",
            "fine-tune, the final model, target.",
            "",
            "![Before and after](../results/figures/before_after_text.jpg)",
            "",
        ]
    if p := fig_spectra(samples, figs / "spectra.png", "ScreenCleanNet_text_fine-tune"):
        written.append(p)
        md += [
            "Power spectra of the crop whose photo differs most from its target. The photo's faint lines",
            "near the corners are aliased copies of the display's pixel grid; the model's output has none,",
            "but it is also lighter than the target far from the centre: some fine detail of the letters",
            "is smoothed away.",
            "",
            "![Spectra](../results/figures/spectra.png)",
            "",
        ]
    for job in ("0009_train_sc_base", "0011_finetune_text"):
        if (figs / f"curves_{job}.png").exists():
            md += [f"![Training curves {job}](../results/figures/curves_{job}.png)", ""]

    out = repo_root / "docs" / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md).rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    return [out, *written]


def _engines(summary: dict[str, Any]) -> str:
    parts = []
    for name, rec in summary.get("engines", {}).items():
        used = rec.get("engine")
        versions = {
            k: v
            for k, v in (rec.get("versions") or {}).items()
            if k in ("paddleocr", "tesseract", "rapidocr")
        }
        note = f"{used} {', '.join(str(v) for v in versions.values())}".strip() if used else f"{name} failed"
        if used and used != name:
            note += f" (fallback for {name})"
        parts.append(note)
    return "; ".join(parts)
