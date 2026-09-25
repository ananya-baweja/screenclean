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
  The Kaggle mirror of UHDM needs an API token, so it is not used.

## 2026-09-25: Images are float32 RGB in [0, 1] internally

- **Decision:** HWC numpy arrays and BCHW torch tensors, RGB order. OpenCV's BGR is converted only at I/O
  boundaries, and EXIF orientation is always applied when a photo is read.
- **Why:** one convention avoids silent channel-order and scaling bugs. Phone photos are often stored
  rotated, with an orientation tag.

## 2026-09-25: UHDM is read from a pinned public mirror, one image at a time

- **Decision:** the dataset job reads UHDM from a public Hugging Face mirror of the full release
  (`leegwang/uhdm`, one 45 GB zip, pinned to revision `9b06328`). It reads only the zip's table of contents
  up front, then fetches each image with an HTTP range request and checks its CRC-32. Nothing is extracted to disk.
- **Why:** the official Google Drive files were over their shared download quota (plain downloads refused),
  and the single-file training archive the original download script points to has been replaced by 18
  byte-split parts. The mirror was checked against the official archives: identical file sizes for spot-checked
  images, and 1.2 MB of image bytes compared byte for byte. It has 4,500 train and 500 test pairs, with no
  unmatched files. Random access also means a resumed job only fetches the images it still needs, and Colab's
  local disk size doesn't matter.
- **Alternatives:** official Drive files with "retry tomorrow" (can stall for days); the Kaggle mirror (needs an
  API token); streaming all 18 parts through `tar` (needs the quota, and a restart re-downloads everything).

## 2026-09-25: UHDM `test` split for evaluation; `test_origin` not used

- **Decision:** evaluate on `test` (500 pairs). `dev100` is a fixed sample of 100 of them (sorted keys, seed 0)
  used during development; the full 500 are used once for the final numbers.
- **Why:** `test` is what the paper's evaluation code uses. `test_origin` is kept out so there is exactly one test set.

## 2026-09-25: Validation = 200 held-out training images (seed 0)

- **Decision:** sort the 4,500 training keys and sample 200 with seed 0 as validation (2 fixed 512 crops each).
  Training uses the other 4,300, with 2 random 512 crops at full scale plus 1 at half scale per image.
- **Why:** model selection and tuning need data that isn't the test set. Fixed crops make validation numbers
  comparable across runs. The half-scale crop shows the model coarser moiré at the same crop size.

## 2026-09-25: Datasets stored as plain tar shards, read in place

- **Decision:** crops are stored as ~500 MB uncompressed tar files with a `manifest.json` (sizes, sha1, source
  images). Training copies the shards to local disk and reads samples by byte offset, without extracting them.
- **Why:** copying a few large files over the Drive mount is far faster than thousands of small ones.
  Reading in place avoids an extraction step and doubling the disk use.

## 2026-09-25: Classical baselines are tuned on validation images only

- **Decision:** chroma low-pass and the two FFT notch filters get their settings from a small grid search
  on 20 held-out validation images at full resolution, scored by mean PSNR. The chosen settings are saved
  with the job that picked them, and the evaluation reports where each method's settings came from.
- **Why:** a fair baseline needs reasonable settings, but choosing them on the test images would inflate
  their scores.
- **Note:** a first check on real UHDM photos showed the classical filters change PSNR by only a few
  hundredths of a dB. Much of the difference between a UHDM moiré photo and its ground truth is global
  colour and brightness shift, which frequency filters can't correct.

## 2026-09-25: Notch filtering uses the periodic + smooth decomposition

- **Decision:** before the FFT, each channel is split into a periodic part and a smooth part (Moisan, 2011).
  Only the periodic part is notch-filtered; the smooth part is added back unchanged.
- **Why:** the FFT treats the image as if it wrapped around, so the jump between opposite edges creates a
  bright cross in the spectrum that looks like "peaks". Without the decomposition, the filter damaged clean
  images (34 dB); with it, a clean image passes through almost untouched (65 dB in the unit test).

## 2026-09-25: ESDNet reference: official weights, fp16, cached on Drive

- **Decision:** the reference is the authors' lightweight ESDNet with their UHDM checkpoint, code pinned
  at commit `fa70a92`. Inference follows their test script (pad to a multiple of 32 with their padding colour,
  full resolution, first output), in fp16 on a T4, with tiled inference only if memory runs out (counted in
  the results). The checkpoint is downloaded once and cached on Drive.
- **Why:** a published, pretrained model on the same data is the honest bar for our model. The larger
  ESDNet-L checkpoint was not downloadable (Drive quota) when this was set up, and the authors' public demo
  weights were trained on several datasets combined, so they are not used.
