"""Build Markdown result tables from the job results in ``results/jobs/``.

Numbers are never typed by hand: every cell comes from a job's ``summary.json``, and each
row names the job it came from. Re-run whenever new results are ingested::

    python -m screenclean tables
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPLIT_TITLES = {"dev100": "UHDM dev100 (100 full-resolution test pairs)"}


def _fmt(v: Any, digits: int) -> str:
    return "–" if v is None else f"{v:.{digits}f}"


def collect(results_dir: Path, split_name: str) -> list[dict[str, Any]]:
    """One entry per method across all finished ``eval`` jobs on ``split_name`` (newest job wins)."""
    rows: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        if s.get("task") != "eval" or Path(s.get("split", "")).name != split_name:
            continue
        for label, m in s.get("methods", {}).items():
            prev = rows.get(label)
            if prev is None or m.get("n", 0) >= prev["n"]:
                rows[label] = {"label": label, "job": s["id"], "complete": s.get("complete", False), **m}
    return sorted(
        rows.values(), key=lambda r: (r["name"] != "identity", r.get("reference", False), -r["psnr"])
    )


def render(rows: list[dict[str, Any]], split_name: str) -> str:
    title = SPLIT_TITLES.get(split_name, split_name)
    lines = [
        f"# Results: {title}",
        "",
        "Generated from `results/jobs/*/summary.json` by `python -m screenclean tables`; don't edit by hand.",
        "",
        "| Method | PSNR (dB) ↑ | Δ PSNR vs input | SSIM ↑ | LPIPS ↓ | s / image | Images | Job |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        label = (
            f"{r['label']} *(reference)*"
            if r.get("reference") and "reference" not in r["label"]
            else r["label"]
        )
        gain = r.get("psnr_gain_vs_input")
        cells = [
            label,
            _fmt(r["psnr"], 2),
            "–" if gain is None else f"{gain:+.2f}",
            _fmt(r["ssim"], 4),
            _fmt(r.get("lpips"), 4),
            _fmt(r.get("seconds"), 2),
            f"{r['n']}" + ("" if r["complete"] else " (partial)"),
            f"`{r['job']}`",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not rows:
        lines.append("| *(no results yet)* | | | | | | | |")
    lines += [
        "",
        "- Metrics on uint8 RGB at full resolution; SSIM with a Gaussian window (σ = 1.5).",
        "- Δ PSNR is the mean per-image difference against the unprocessed input.",
        "- Timing: classical baselines on Colab CPU (2 vCPU); reference models on a T4 GPU.",
        "- *Reference* rows are published models run with their authors' weights, not our method.",
    ]
    notes = [r for r in rows if r.get("params_source") and r["name"] != "identity"]
    if notes:
        lines += ["", "Settings:", ""]
        for r in notes:
            params = ", ".join(
                f"{k}={v}"
                for k, v in sorted(r.get("params", {}).items() if isinstance(r.get("params"), dict) else [])
            )
            if r.get("reference"):
                modes = r.get("inference_modes", {})
                runs = [f"{modes['full']} images at full resolution"] if modes.get("full") else []
                runs += [f"{modes['tiled']} in tiles (GPU memory)"] if modes.get("tiled") else []
                lines.append(
                    f"- **{r['label']}**: authors' pretrained weights, {r.get('parameters', 0) / 1e6:.2f} M "
                    f"parameters, {'fp16' if r.get('fp16') else 'fp32'}; {', '.join(runs) or 'n/a'}"
                )
                continue
            lines.append(f"- **{r['label']}**: {params or 'defaults'} (from {r['params_source']})")
    return "\n".join(lines) + "\n"


def build_tables(repo_root: Path, splits: tuple[str, ...] = ("dev100",)) -> list[Path]:
    """Write ``results/tables/uhdm_<split>.md`` for each split; returns the written paths."""
    out_dir = repo_root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split in splits:
        path = out_dir / f"uhdm_{split}.md"
        path.write_text(
            render(collect(repo_root / "results" / "jobs", split), split), encoding="utf-8", newline="\n"
        )
        written.append(path)
    return written
