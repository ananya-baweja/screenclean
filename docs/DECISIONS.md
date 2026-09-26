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

## 2026-09-25: Rendered-page text is generated from project word lists

- **Decision:** the 1,500-line corpus (`src/screenclean/render/corpus_en.txt`) is produced by
  `tools/make_corpus.py` from templates and word lists written for this project: sentences, slide
  bullets, numbers, prices, dates, e-mail addresses and URLs on the reserved `example.*` domains, code
  and error messages. It is split 80/10/10 by line (seed 0); benchmark pages use only `test` lines.
- **Why:** no copyright or privacy questions, full control over the mix of content that people photograph
  on screens, and synthetic training pages can never contain benchmark text.

## 2026-09-25: Only matplotlib's bundled fonts by default

- **Decision:** pages use DejaVu Sans / Serif / Mono and STIX from matplotlib's package. System fonts are
  only used if passed explicitly.
- **Why:** the same fonts exist on every machine (laptop, CI, Colab), so pages are reproducible.

## 2026-09-25: The benchmark pages were checked for readability before any photo

- **Decision:** every line of the 40 clean capture pages was read with Tesseract 5 (single-line mode on
  a crop). CER: 0.10% at 16–24 px, 0.19% above 24 px, 0.30% below 16 px (the "hard" bucket). The report is
  `capture_kit/ocr_check.json`.
- **Why:** if clean pages were hard to read, OCR errors on the photos would come from the layout, not from
  moiré, and the benchmark would be unfair.

## 2026-09-25: Moiré simulator settings were measured, not guessed

- **Decision:** camera scale (sensor pixels per display pixel) 0.9–3.0 log-uniform, with half of the samples
  within about 4% of 1 or 2; Gaussian lens blur σ 0.2–0.7 px; phone-style noise reduction (colour σ 1–3 px,
  brightness σ 0–0.7 px); auto-exposure never brightens a dark screen by more than about 2.9× (35% floor).
- **Why:** a sweep on a flat white screen showed almost no moiré for the first-guess ranges (scale 0.7–1.6,
  blur σ up to 1.2): the subpixel pattern repeats about once per sensor pixel, and a σ ≈ 0.8 px blur removes it.
  Moiré contrast peaked near scale 2 (0.20 at σ 0.2). Comparing residual spectra with real UHDM crops then showed
  too much fine colour stripe (fixed by noise reduction: colour residual 2.5× → 1.3× UHDM) and too little
  low-frequency banding (improved by sampling near the resonances, where wide bands form).
- **Known gap:** UHDM's brightness residual is higher at low frequencies and follows a natural-image 1/f² shape,
  which suggests it comes from small misalignments in UHDM's pairs rather than from moiré. The simulator isn't
  tuned to match it. The real test is transfer: a model trained on synthetic pairs only is evaluated on real photos (P8).

## 2026-09-25: Synthetic ground-truth text = whole visible words

- **Decision:** for every synthetic crop, the JSON record keeps, per line, the run of whole words that is fully
  inside the crop (text + corner positions + photographed font size), measured with the same font the page used.
- **Why:** crops cut most lines at their edges. Keeping only complete lines left 17 of 24 test crops without any
  text; visible whole words give about 15 words per crop for OCR scoring.

## 2026-09-26: Screen detection is classical, accepts both polarities, and peels bezels

- **Decision:** the detector (P5.1) snaps each side of a candidate quadrilateral to the strongest straight edge
  nearby (median brightness step along the whole side, so a faint but long edge wins), accepts "inside
  brighter" and "inside darker" screens, and looks just inside the device outline for a display behind a thin
  ring without text (the bezel). Candidates must have the real shape of a screen (aspect 0.4-2.6).
- **Why:** on dark-mode pages the bezel is often *brighter* than the page, because it reflects the room. The
  first version assumed a bright screen and found none of the dark code pages.
- **Result** (200 synthetic photos, `results/checks/screen_detect.json`): 97% found on light pages, 58% on
  dark-mode pages, 83% overall; the P5.1 target was 90%. When found, the median corner error is 0.02% of the
  image diagonal; 0.85 s per photo on a laptop CPU. Most dark-mode misses have a page edge that is truly
  invisible (a step of 0-3 grey levels); then the scan uses the whole photo.
- **Next steps for the gap:** the web app lets users drag the corners (P11); a small learned corner model trained
  on these synthetic photos is the planned stretch goal. Tuning more rules on the same 200 photos stopped helping
  (the last three changes moved the score by -1.5 to +0.5 points).

## 2026-09-26: Synthetic photos place the screen by moving it in 3D

- **Decision:** `simulate/scene.py` keeps the camera's optical axis at the image centre and moves the screen
  sideways in 3D to place it in the frame.
- **Why:** the first version shifted the image instead, which is like an off-centre lens. The aspect-ratio
  estimate (which assumes a centred optical axis, as in real phones) was then off by 2.5% (median) and up to 83%.
  After the fix it is exact on true corners, and 0.2% (median) with 2 px of corner noise.

## 2026-09-26: Clean before warping, only the screen's area

- **Decision:** the scan pipeline removes moiré on the original photo, restricted to the screen's bounding box
  plus 32 px, and then warps. Where the warp shrinks the image, it blurs first by σ = 0.5·√(1/s² − 1).
- **Why:** warping resamples, and resampling a moiré photo can fold it into new patterns; cleaners are built for
  photos as cameras take them. The bounding box saves time on large photos.

## 2026-09-26: Searchable PDFs with reportlab; Tesseract runs in CI

- **Decision:** the PDF is the page image plus an invisible text layer (render mode 3), made with reportlab (BSD)
  and the DejaVu Sans font from matplotlib, so non-ASCII text survives. CI installs Tesseract from Ubuntu (5.3;
  5.5 on the laptop) so OCR tests run there too, with tolerant thresholds.
- **Why:** AGPL libraries such as PyMuPDF are avoided (plan C11). Standard PDF fonts only cover Latin-1.

## 2026-09-26: Scan without the classical moiré filter by default (until the model is ready)

- **Decision:** `scan` uses `--cleaner none` by default; `--cleaner fft_notch_local` still runs the tuned
  local notch filter. OCR uses Tesseract with local (Sauvola) thresholding.
- **Why:** on 14 simulated photos of capture-kit pages (`results/checks/scan_demo.json`), the notch filter
  lowered word F1 from 0.75 to 0.70 and raised CER from 0.36 to 0.43. It rescued a table (word F1 0.54 → 0.90)
  but damaged code (0.64 → 0.14): regular line spacing and monospace columns are periodic, so a notch filter
  removes them as if they were moiré. The filter was tuned for PSNR on UHDM (+0.31 dB), not for text.
  Tesseract's default global threshold also failed on photographed pages (one dark element such as a
  corner marker sets it and lighter coloured text breaks apart); Sauvola raised word F1 from 0.68 to 0.80.
- **Consequence:** this is the case for the learned model. P8 measures OCR with and without it on the same pages.

## 2026-09-26: ScreenCleanNet details (P6)

- **Decision:** `scnet_base` follows the planned diagram (4.65 M parameters, 52 GFLOPs at 512×512); `scnet_tiny`
  has 0.79 M. The last convolution starts at zero, so the untrained model returns its input.
- **Why:** starting as the identity means training starts at the input's quality and only learns corrections;
  NAFNet's zero-started β/γ already make every block start as the identity.
- **Note on the ablation:** Haar DWT + 1×1 conv spans exactly the same functions as a 2×2 stride-2 conv (unit
  test), and 1×1 conv + inverse Haar equals 1×1 conv + PixelShuffle. `plain_unet` therefore tests the
  starting point and training dynamics of the wavelet parametrisation (plus dilation), not extra capacity.

## 2026-09-26: Training samples are fixed by their number

- **Decision:** sample k of a run takes its source, pair, crop and flips from a generator seeded with
  (seed, k); the loader walks k = iteration × batch onwards.
- **Why:** Colab jobs stop at a time budget and resume from `last.pt`. With this, a resumed run sees exactly the
  data an uninterrupted one would (the CPU test gets the same loss curve for "stop at 20, resume to 50" as for one
  run of 50), independent of DataLoader workers, and sources can be mixed by weight for the text fine-tune.

## 2026-09-26: ONNX export test uses both exporters

- **Decision:** the fast test exports the Haar layers with the legacy TorchScript exporter (about 1 s); a slow
  test uses the new `torch.export`-based exporter (about 20 s), which is PyTorch's default since 2.9.
- **Why:** CI stays fast, and the export path that P10 will use is still checked.

