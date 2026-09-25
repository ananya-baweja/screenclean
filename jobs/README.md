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

`auto` picks the first job (sorted by file name) that isn't `done`, didn't fail at its current
`attempt`, and whose `depends_on` jobs are all `done`.

## Results

When a job ends as `done` or `failed`, the runner packs a small zip (at most 10 MB):
`status.json`, `summary.json`, `log_tail.txt`, `config_resolved.yaml`, `env.json` and small artifacts.
The notebook's last cell downloads it. The zip is then unpacked into `results/jobs/<id>/` with:

```bash
python -m screenclean results ingest path/to/<id>.zip
```
