"""The ``train`` job task: stage data from Drive, optionally calibrate, train within the time budget.

Config (``configs/train/*.yaml``)::

    run: runs/<job id>            # run folder on Drive (checkpoints, logs); default runs/<job id>
    data:
      train: [data/uhdm_v1/train] # one or more split folders on Drive (mixed by train.source_weights)
      val: data/uhdm_v1/val
      local_dir: /content/train_data
    calibrate: {crops: [256, 384], batches: [4, 6, 8], iters: 30, warmup: 8, plan_minutes: 300}
    train: {...}                  # trainer settings (see train/trainer.py DEFAULTS)
    stop_margin_min: 8

The run resumes from ``last.pt`` whenever the job runs again. The results zip gets ``summary.json``,
the logs, the latest sample grid and (if run) ``calibration.json``.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from screenclean.data.pairs_dataset import PairsDataset
from screenclean.jobs import JobContext, TaskResult, register
from screenclean.train.calibrate import calibrate
from screenclean.train.trainer import Trainer
from screenclean.utils.drive import atomic_write_json, read_json, stage_shards

log = logging.getLogger(__name__)


def _stage(ctx: JobContext, split: str, local_root: Path) -> list[Path]:
    dest = local_root / split.replace("/", "_")
    ctx.progress(f"copying {split} from Drive to local disk")
    return stage_shards(ctx.layout.root / split, dest)


@register("train")
def train_task(ctx: JobContext) -> TaskResult:
    cfg = ctx.config
    data = cfg["data"]
    local_root = Path(data.get("local_dir", ctx.work_dir / "local_data"))
    sources = [PairsDataset(_stage(ctx, split, local_root)) for split in data["train"]]
    val = PairsDataset(_stage(ctx, data["val"], local_root)) if data.get("val") else None
    run_dir = ctx.layout.root / cfg.get("run", f"runs/{ctx.spec.id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("training sources: %s pairs; val: %s", [len(s) for s in sources], len(val) if val else 0)

    cal_path = run_dir / "calibration.json"
    if cfg.get("calibrate") and not cal_path.exists():
        c = cfg["calibrate"]
        ctx.progress(f"calibrating {len(c['crops']) * len(c['batches'])} crop/batch settings")
        result = calibrate(
            cfg.get("train", {}).get("model", "scnet_base"),
            sources,
            c["crops"],
            c["batches"],
            iters=c.get("iters", 30),
            warmup=c.get("warmup", 8),
            plan_minutes=c.get("plan_minutes", 300),
            workers=cfg.get("train", {}).get("workers", 2),
        )
        atomic_write_json(cal_path, result)
        ctx.progress(f"calibration: recommended {json.dumps(result.get('recommended'))}")

    margin = 60 * cfg.get("stop_margin_min", 8)
    trainer = Trainer(
        cfg.get("train", {}),
        run_dir,
        sources,
        val,
        should_stop=lambda: ctx.time_up(margin),
        progress=ctx.progress,
    )
    summary = trainer.fit()

    for name in ("train_log.csv", "val_log.csv", "calibration.json"):
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, ctx.results_dir / name)
    samples = sorted((run_dir / "samples").glob("val_*.jpg"))
    if samples:
        shutil.copy2(samples[-1], ctx.results_dir / "val_samples_latest.jpg")
    if cal_path.exists():
        summary["calibration"] = read_json(cal_path).get("recommended")
    summary["run_dir"] = str(run_dir.relative_to(ctx.layout.root)).replace("\\", "/")
    if summary["state"] == "partial":
        return TaskResult(
            "partial",
            summary,
            f"Trained to iteration {summary['iteration']} of {summary['iters']}; "
            "run the notebook again to continue from the last checkpoint.",
        )
    best = summary["best"]
    return TaskResult(
        "done",
        summary,
        f"Training finished at iteration {summary['iteration']}: best val PSNR {best['psnr']:.2f} dB "
        f"(input {summary['input_psnr']:.2f} dB) at iteration {best['iter']}."
        if summary.get("input_psnr") is not None
        else "",
    )
