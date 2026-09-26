"""The training loop: AdamW + warmup/cosine, fp16 autocast, EMA, time-budgeted checkpoint/resume.

Everything a run needs lives in its run folder (on Drive for Colab jobs):

- ``last.pt``: model, EMA, optimizer, scheduler, grad scaler, iteration, random states. Written every
  ``ckpt_minutes`` and when the time budget runs out; the next run resumes from it automatically.
- ``best.pt``: the EMA weights with the best validation PSNR so far.
- ``train_log.csv``, ``val_log.csv``, ``samples/`` (input | output | target), and TensorBoard files
  in ``tb/`` when TensorBoard is installed.

Validation always uses the EMA weights, on full validation crops, scored like the benchmark
(uint8 PSNR / SSIM).
"""

from __future__ import annotations

import copy
import csv
import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from screenclean.data.pairs_dataset import PairsDataset
from screenclean.eval.metrics import psnr, ssim
from screenclean.models.complexity import count_params
from screenclean.models.registry import build_model
from screenclean.train.data import MixedCrops, train_loader
from screenclean.train.losses import TrainLoss
from screenclean.train.state import EMA, load_checkpoint, rng_state, save_checkpoint, set_rng_state
from screenclean.utils.image import to_uint8
from screenclean.utils.io import write_image
from screenclean.utils.seed import seed_everything

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "model": "scnet_base",
    "model_overrides": {},
    "crop": 384,
    "batch": 6,
    "workers": 2,
    "source_weights": None,  # one weight per training source (default: equal)
    "iters": 60000,  # length of the whole schedule, across resumed runs; or "auto" (see budget_minutes)
    "budget_minutes": None,  # with iters "auto": training minutes to fill, all runs together
    "val_seconds": 100,  # time one validation takes (for iters "auto"; 0008 measured 93 s on a T4)
    "compile": False,  # torch.compile: True, False, or "auto" (time both, keep the faster)
    "probe_iters": 40,  # iterations timed per variant (the second half of them) for "auto" settings
    "iters_round": 1000,  # iters "auto" is rounded down to a multiple of this
    "optim": {"lr": 3e-4, "betas": [0.9, 0.99], "weight_decay": 1e-4, "warmup": 1000, "min_lr": 1e-6},
    "clip": 1.0,
    "loss": {"fft_weight": 0.05, "perc_weight": 0.0, "eps": 1e-3},
    "amp": True,  # fp16 autocast + GradScaler (GPU only; the T4 has no bf16)
    "ema": 0.999,
    "val_every": 2000,
    "val_max": 400,
    "val_batch": 8,
    "sample_grid": 6,
    "log_every": 50,
    "ckpt_minutes": 15,
    "progress_minutes": 2,
    "seed": 0,
}

TRAIN_FIELDS = ["iter", "loss", "charbonnier", "fft", "fft_share", "lr", "grad_norm", "it_per_s", "data_wait",
                "gpu_mem_gb", "elapsed_min"]  # fmt: skip
VAL_FIELDS = ["iter", "psnr", "ssim", "input_psnr", "seconds"]


def merge(defaults: dict[str, Any], cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Deep-merge ``cfg`` into a copy of ``defaults``."""
    out = copy.deepcopy(defaults)
    for k, v in (cfg or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def lr_factor(it: int, warmup: int, total: int, min_ratio: float) -> float:
    """Linear warmup to 1, then cosine decay to ``min_ratio`` at ``total``."""
    if it < warmup:
        return (it + 1) / warmup
    t = min(1.0, (it - warmup) / max(1, total - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))


class CsvLog:
    """Append-only CSV that, on resume, drops rows written after the checkpoint being resumed."""

    def __init__(self, path: Path, fields: list[str], keep_until: int | None):
        self.path, self.fields = path, fields
        rows = []
        if path.exists() and keep_until is not None:
            with open(path, newline="", encoding="utf-8") as f:
                rows = [r for r in csv.DictReader(f) if int(float(r["iter"])) <= keep_until]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

    def write(self, row: dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore", lineterminator="\n").writerow(
                {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
            )


class Trainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        run_dir: str | Path,
        train_sources: list[PairsDataset],
        val_set: PairsDataset | None = None,
        should_stop: Callable[[], bool] = lambda: False,
        progress: Callable[[str], None] | None = None,
        device: str | torch.device | None = None,
    ):
        self.cfg = merge(DEFAULTS, cfg)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.train_sources, self.val_set = train_sources, val_set
        self.should_stop, self.progress = should_stop, progress or (lambda text: log.info(text))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = bool(self.cfg["amp"]) and self.device.type == "cuda"

    # ------------------------------------------------------------------ setup

    def _build(self) -> None:
        c = self.cfg
        seed_everything(c["seed"])
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self.model = build_model(c["model"], **c["model_overrides"]).to(self.device)
        self.ema = EMA(self.model, c["ema"])
        o = c["optim"]
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=o["lr"], betas=tuple(o["betas"]), weight_decay=o["weight_decay"]
        )
        min_ratio = o["min_lr"] / o["lr"]
        self.total_iters = None if c["iters"] == "auto" else int(c["iters"])
        if self.total_iters is None:
            if not c["budget_minutes"]:
                raise ValueError('iters "auto" needs budget_minutes')
            if o["warmup"] < 2 * c["probe_iters"] + 10:
                raise ValueError("warmup must outlast the speed probe (the schedule length is set after it)")
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda it: lr_factor(it, o["warmup"], self._total(), min_ratio)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.loss_fn = TrainLoss(**c["loss"]).to(self.device)
        self.iteration, self.best = 0, {"psnr": -math.inf, "iter": None}
        self.input_psnr: float | None = None
        self.probe: dict[str, Any] = {"eager_it_per_s": None, "compiled_it_per_s": None, "choice": None,
                                      "error": None}  # fmt: skip

    def _total(self) -> int:
        """Schedule length; very long until an "auto" length is decided (warmup outlasts that)."""
        return self.total_iters or 10**9

    def _plan_iters(self, it_per_s: float) -> int:
        """Iterations that fill ``budget_minutes``, counting validation time, rounded down to
        ``iters_round`` (and at least one ``iters_round`` past the warmup)."""
        c = self.cfg
        step = c["iters_round"]
        per_iter = 1.0 / it_per_s + c["val_seconds"] / c["val_every"]
        return max(c["optim"]["warmup"] + step, int(60 * c["budget_minutes"] / per_iter) // step * step)

    def _resume(self) -> bool:
        path = self.run_dir / "last.pt"
        if not path.exists():
            return False
        # Load to CPU: load_state_dict copies weights and optimiser state to the model's device itself,
        # and the saved random-generator states must stay CPU byte tensors (torch.set_rng_state and
        # torch.cuda.set_rng_state_all refuse GPU tensors: "RNG state must be a torch.ByteTensor").
        ck = load_checkpoint(path, map_location="cpu")
        self.model.load_state_dict(ck["model"])
        self.ema.load_state_dict(ck["ema"])
        self.opt.load_state_dict(ck["optimizer"])
        self.sched.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.iteration, self.best, self.input_psnr = ck["iteration"], ck["best"], ck.get("input_psnr")
        self.total_iters = ck.get("total_iters", self.total_iters)
        self.probe = ck.get("probe", self.probe)
        set_rng_state(ck["rng"])
        log.info("resumed from %s at iteration %d", path, self.iteration)
        return True

    def _checkpoint(self) -> None:
        save_checkpoint(
            self.run_dir / "last.pt",
            {
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict(),
                "scaler": self.scaler.state_dict(),
                "iteration": self.iteration,
                "best": self.best,
                "input_psnr": self.input_psnr,
                "total_iters": self.total_iters,
                "probe": self.probe,
                "rng": rng_state(),
                "config": self.cfg,
                "spec": self.model.spec.as_dict(),
            },
        )

    # ------------------------------------------------------------- validation

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """EMA weights on the validation crops: mean uint8 PSNR/SSIM, plus a sample grid."""
        c, model = self.cfg, self.ema.model
        n = min(len(self.val_set), c["val_max"])
        t0 = time.perf_counter()
        scores, inputs, grid = [], [], []
        for start in range(0, n, c["val_batch"]):
            pairs = [self.val_set[i] for i in range(start, min(n, start + c["val_batch"]))]
            moire = torch.stack([p[0] for p in pairs]).to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
                out = model(moire)
            out = out.float().clamp(0, 1).cpu()
            for (m, g), o in zip(pairs, out, strict=True):
                o8, g8 = to_uint8(o.permute(1, 2, 0).numpy()), to_uint8(g.permute(1, 2, 0).numpy())
                scores.append((psnr(o8, g8), ssim(o8, g8)))
                if self.input_psnr is None:
                    inputs.append(psnr(to_uint8(m.permute(1, 2, 0).numpy()), g8))
                if len(grid) < c["sample_grid"]:
                    grid.append(np.hstack([to_uint8(m.permute(1, 2, 0).numpy()), o8, g8]))
        if self.input_psnr is None:
            self.input_psnr = float(np.mean(inputs))
        if grid:
            write_image(
                self.run_dir / "samples" / f"val_{self.iteration:06d}.jpg", np.vstack(grid), quality=85
            )
        s = np.array(scores)
        return {
            "psnr": float(s[:, 0].mean()),
            "ssim": float(s[:, 1].mean()),
            "input_psnr": self.input_psnr,
            "seconds": time.perf_counter() - t0,
        }

    def _after_validation(self, val: dict[str, float], val_log: CsvLog) -> None:
        val_log.write({"iter": self.iteration, **val})
        self._tb("val/psnr", val["psnr"])
        self._tb("val/ssim", val["ssim"])
        if val["psnr"] > self.best["psnr"]:
            self.best = {"psnr": val["psnr"], "ssim": val["ssim"], "iter": self.iteration}
            save_checkpoint(
                self.run_dir / "best.pt",
                {"model": self.ema.model.state_dict(), "spec": self.model.spec.as_dict(),
                 "model_name": self.cfg["model"], "iteration": self.iteration, "val": val},
            )  # fmt: skip
        self.progress(
            f"validation at {self.iteration}: PSNR {val['psnr']:.2f} dB (input {val['input_psnr']:.2f}), "
            f"SSIM {val['ssim']:.4f}; best {self.best['psnr']:.2f} at {self.best['iter']}"
        )

    # ------------------------------------------------------------------ train

    def _tb(self, tag: str, value: float) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, self.iteration)

    def fit(self) -> dict[str, Any]:
        """Train until ``iters`` or until ``should_stop()``; returns a summary with ``state``."""
        c = self.cfg
        self._build()
        resumed = self._resume()
        keep = self.iteration if resumed else None
        train_log = CsvLog(self.run_dir / "train_log.csv", TRAIN_FIELDS, keep)
        val_log = CsvLog(self.run_dir / "val_log.csv", VAL_FIELDS, keep)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(
                str(self.run_dir / "tb"), purge_step=self.iteration if resumed else None
            )
        except Exception:  # noqa: BLE001 - TensorBoard is optional
            self.writer = None

        start_iter, t_start = self.iteration, time.monotonic()
        state = "done"
        if self.total_iters is None or self.iteration < self.total_iters:
            data = MixedCrops(self.train_sources, c["crop"], c["seed"], c["source_weights"])
            end = self.total_iters or 10**8 // c["batch"]  # "auto": the loop decides where to stop
            loader = iter(train_loader(data, c["batch"], self.iteration, end, c["workers"]))
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            last_ckpt = last_progress = last_log_t = time.monotonic()
            wait, parts_sum, n_parts, grad_norm = 0.0, {}, 0, 0.0
            self.model.train()
            fwd, probe = self._start_probe()
            while self.total_iters is None or self.iteration < self.total_iters:
                if self.should_stop():
                    state = "partial"
                    break
                t0 = time.monotonic()
                moire, gt = next(loader)
                wait += time.monotonic() - t0
                moire, gt = moire.to(self.device, non_blocking=True), gt.to(self.device, non_blocking=True)
                try:
                    parts, grad_norm = self._step(fwd, moire, gt)
                except Exception as e:  # noqa: BLE001 - a failed compile must not end the run
                    if fwd is self.model:
                        raise
                    log.exception("torch.compile failed; continuing without it")
                    self.probe["error"] = f"{type(e).__name__}: {e}"[:500]
                    fwd = self.model
                    parts, grad_norm = self._step(fwd, moire, gt)
                    probe = self._finish_probe("eager")
                self.iteration += 1
                for k, v in parts.items():
                    parts_sum[k] = parts_sum.get(k, 0.0) + v
                n_parts += 1
                if probe is not None:
                    fwd, probe = self._advance_probe(fwd, probe)

                if self.iteration % c["log_every"] == 0 or self.iteration == self.total_iters:
                    now = time.monotonic()
                    floats = self.loss_fn.as_floats({k: v / n_parts for k, v in parts_sum.items()})
                    row = {
                        "iter": self.iteration,
                        "loss": floats["total"],
                        **{k: floats.get(k, "") for k in ("charbonnier", "fft", "fft_share")},
                        "lr": self.opt.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm),
                        "it_per_s": n_parts / max(now - last_log_t, 1e-9),
                        "data_wait": wait / max(now - last_log_t, 1e-9),
                        "gpu_mem_gb": torch.cuda.max_memory_allocated(self.device) / 1e9
                        if self.device.type == "cuda"
                        else 0.0,
                        "elapsed_min": (now - t_start) / 60,
                    }
                    train_log.write(row)
                    for k in ("loss", "fft_share", "lr", "it_per_s", "data_wait"):
                        self._tb(f"train/{k}", row[k] if row[k] != "" else 0.0)
                    parts_sum, n_parts, wait, last_log_t = {}, 0, 0.0, now
                    if now - last_progress > 60 * c["progress_minutes"]:
                        self.progress(
                            f"iteration {self.iteration}/{self.total_iters or '?'}: loss {row['loss']:.4f}, "
                            f"{row['it_per_s']:.2f} it/s"
                        )
                        last_progress = now
                if self.val_set is not None and self.iteration % c["val_every"] == 0:
                    self.model.eval()
                    self._after_validation(self.validate(), val_log)
                    self.model.train()
                if time.monotonic() - last_ckpt > 60 * c["ckpt_minutes"]:
                    self._checkpoint()
                    last_ckpt = time.monotonic()

        final = None
        if state == "done" and self.val_set is not None:
            done_val = self.iteration % c["val_every"] == 0 and self.iteration > start_iter
            if not done_val:  # make sure the last iteration is validated
                self.model.eval()
                final = self.validate()
                self._after_validation(final, val_log)
            else:
                final = self._last_val(val_log)
        self._checkpoint()
        if self.writer is not None:
            self.writer.close()
        seconds = time.monotonic() - t_start
        return {
            "state": state,
            "iteration": self.iteration,
            "iters": self.total_iters,
            "resumed": resumed,
            "iters_this_run": self.iteration - start_iter,
            "seconds_this_run": round(seconds, 1),
            "it_per_s": round((self.iteration - start_iter) / max(seconds, 1e-9), 3),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated(self.device) / 1e9, 2)
            if self.device.type == "cuda"
            else 0.0,
            "best": self.best,
            "final_val": final,
            "input_psnr": self.input_psnr,
            "model": c["model"],
            "params_m": round(count_params(self.model) / 1e6, 3),
            "crop": c["crop"],
            "batch": c["batch"],
            "amp": self.amp,
            "device": str(self.device),
            "speed_probe": self.probe,
        }

    # ------------------------------------------------------------ one step

    def _step(
        self, fwd, moire: torch.Tensor, gt: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        c = self.cfg
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            pred = fwd(moire)
        loss, parts = self.loss_fn(pred.float(), gt)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c["clip"])
        self.scaler.step(self.opt)
        self.scaler.update()
        self.sched.step()
        self.ema.update(self.model)
        return parts, grad_norm

    # ---------------------------------------------------------- speed probe
    # "auto" settings are decided on the first iterations of the first run (real training steps):
    #   compile "auto": time eager steps, then compiled ones; keep compiled only if more than 5% faster;
    #   iters "auto":   set the schedule length from the measured speed and budget_minutes.
    # Timing uses the second half of each window (the first half includes warm-up and compilation).

    def _start_probe(self):
        c, p = self.cfg, self.probe
        compiled = p["choice"] == "compiled" or (p["choice"] is None and c["compile"] is True)
        fwd = torch.compile(self.model) if compiled else self.model
        if p["choice"] is None and (c["compile"] == "auto" or self.total_iters is None):
            return fwd, {"phase": "compiled" if compiled else "eager", "start": self.iteration, "mark": None}
        if p["choice"] is None:
            p["choice"] = "compiled" if compiled else "eager"
        return fwd, None

    def _advance_probe(self, fwd, probe):
        c, p, k = self.cfg, self.probe, self.iteration - probe["start"]
        half = c["probe_iters"] // 2
        if k == half:
            self._sync()
            probe["mark"] = time.monotonic()
        if k < c["probe_iters"]:
            return fwd, probe
        self._sync()
        speed = round((c["probe_iters"] - half) / max(time.monotonic() - probe["mark"], 1e-9), 3)
        if probe["phase"] == "eager":
            p["eager_it_per_s"] = speed
            if c["compile"] == "auto":
                return torch.compile(self.model), {"phase": "compiled", "start": self.iteration, "mark": None}
            return fwd, self._finish_probe("eager")
        p["compiled_it_per_s"] = speed
        eager = p["eager_it_per_s"]
        if eager is not None and speed <= 1.05 * eager:
            return self.model, self._finish_probe("eager")
        return fwd, self._finish_probe("compiled")

    def _finish_probe(self, choice: str) -> None:
        p = self.probe
        p["choice"] = choice
        if self.total_iters is None:
            speed = p["compiled_it_per_s"] if choice == "compiled" else p["eager_it_per_s"]
            self.total_iters = self._plan_iters(speed or 1.0)
        error = f" (compile failed: {p['error'][:120]})" if p["error"] else ""
        self.progress(
            f"speed probe: eager {p['eager_it_per_s']} it/s, compiled {p['compiled_it_per_s']} it/s "
            f"-> {choice}; schedule {self.total_iters} iterations{error}"
        )
        return None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _last_val(self, val_log: CsvLog) -> dict[str, float] | None:
        with open(val_log.path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return {k: float(v) for k, v in rows[-1].items()} if rows else None
