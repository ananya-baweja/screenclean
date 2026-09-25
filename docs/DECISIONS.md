# Design decisions

One entry per decision: date, the decision, why, and the alternatives considered.

## 2026-09-25: Heavy compute runs as queued jobs on Google Colab

- **Decision:** GPU training, dataset preparation and evaluation run on the Colab free tier through one
  runner notebook that executes jobs from `jobs/queue/`. Checkpoints and datasets live on Google Drive.
  Each finished job returns a small results zip that is committed under `results/jobs/<id>/`.
- **Why:** no local GPU. A queue keeps every run reproducible (spec, config, git SHA and environment are
  saved with each run), and resumable jobs survive Colab disconnects.
- **Alternatives:** Kaggle notebooks (different paths and quotas); paid cloud GPUs (cost).

## 2026-09-25: No access tokens anywhere

- **Decision:** the repo is public. Colab clones it without logging in and uses the normal Google sign-in
  for Drive. Nothing in the code or CI reads an API token or secret.
- **Why:** simpler and safer. There are no credentials to leak, rotate or store in notebooks.
- **Alternatives:** GitHub or Kaggle tokens stored as Colab secrets (rejected: extra risk for little gain).
  The Kaggle mirror of UHDM needs an API token, so the official download links are used instead.

## 2026-09-25: Images are float32 RGB in [0, 1] internally

- **Decision:** HWC numpy arrays and BCHW torch tensors, RGB order. OpenCV's BGR is converted only at I/O
  boundaries, and EXIF orientation is always applied when a photo is read.
- **Why:** one convention avoids silent channel-order and scaling bugs. Phone photos are often stored
  rotated, with an orientation tag.
