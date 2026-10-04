# Job queue

Heavy work (dataset preparation, GPU training, evaluation) runs on Google Colab through
[`notebooks/colab_runner.ipynb`](../notebooks/colab_runner.ipynb). The notebook clones this public repo,
mounts Google Drive and runs:

```bash
python -m screenclean jobs run --job auto --drive-root /content/drive/MyDrive/screenclean
```

## Job specs

Each job is one file, `jobs/queue/NNNN_<name>.yaml`:

```yaml
id: 0009_train_sc_base          # must match the file name (without .yaml)
task: train                     # a registered task name
runtime: gpu                    # gpu | cpu
config: configs/train/sc_base_uhdm.yaml
depends_on: [0008_sanity_gpu]   # skipped until these jobs are done
requires: [real_captures/raw]   # skipped until these Drive folders hold files (optional)
max_minutes: 150                # checkpoint and stop gracefully before this
resume: true
attempt: 1                      # bump after fixing a failed job so `auto` picks it up again
drive_subdir: runs/0009_train_sc_base
notes: "free text"
```

## States

Job states live on Drive in `job_status/<id>.json`, not in the repo:

| State | Meaning |
|---|---|
| `running` | started; if the session dropped, the next run resumes it |
| `done` | finished; its results zip is in `results_zips/<id>.zip` |
| `partial` | time budget reached; run the notebook again to continue |
| `failed` | error; `auto` skips it until the spec's `attempt` is increased |

While a job runs, tasks write a short `progress` note (with `progress_utc`) into that status file after
each step. Drive shows it live. `job.log` stays open until the job ends, so Drive may show an old copy of it.

`auto` picks the first job (sorted by file name) that isn't `done`, didn't fail at its current
`attempt`, whose `depends_on` jobs are all `done` and whose `requires` folders on Drive hold files.
A job for the session's runtime comes first: on a CPU runtime the next CPU job runs, on a GPU
runtime the next GPU job. A GPU runtime never runs a CPU job; it lists the CPU jobs that are ready.

## Results

When a job ends as `done` or `failed`, the runner packs a small zip (at most 10 MB):
`status.json`, `summary.json`, `log_tail.txt`, `config_resolved.yaml`, `env.json` and small artifacts.
The notebook's last cell downloads it. The zip is then unpacked into `results/jobs/<id>/` with:

```bash
python -m screenclean results ingest path/to/<id>.zip
```
