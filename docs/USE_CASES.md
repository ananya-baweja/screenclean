# Use cases and target metrics

## The problem

People photograph screens because it's faster than asking for the file. The photos come out with
moiré (wavy rainbow bands caused by the camera sampling the screen's pixel grid), glare and a skewed
angle. They are hard to read, and OCR tools misread them.

## User stories

1. **Students and anyone in a lecture or meeting.** "I photograph slides during a lecture. At the end I
   want one clean, searchable PDF of my notes."
2. **People sharing what's on a screen.** "I photograph an error message, dashboard or code on someone's
   monitor. I want the text as copyable text (for a support ticket or a web search), not a blurry photo."
3. **Accessibility.** "My screen reader or text-to-speech app should be able to read text from a photo of
   a screen." Clean text makes that much more reliable.

## What the product does

| Mode | Input | Output |
|---|---|---|
| Single photo | one photo of a screen | before/after view, the straightened clean page, the extracted text with a copy button |
| Slides to notes | several slide photos | one searchable PDF (clean pages + hidden text layer) and a Markdown/TXT file |
| API | the same over HTTP | JSON (corners, text lines, boxes), PNG or PDF |

## How we'll know it's useful

These are measured on a benchmark of real phone photos of screens (slides, monitors, TVs) showing
pages with known text:

| Metric | Target |
|---|---|
| OCR character error rate (CER): no cleaning → ScreenClean | a clear drop, reported with a 95% bootstrap confidence interval |
| Screen-detection success rate and mean corner error | reported as measured |
| Time per photo on a normal laptop CPU | a few seconds at most |
| Before/after examples from everyday situations | slides, monitor, TV |

Image-quality scores (PSNR / SSIM / LPIPS) on the UHDM test set are reported as well. The main goal
is readable text for people.

## A first worked example: slides to searchable notes

The pipeline runs today on a laptop CPU:

```bash
python -m screenclean scan lecture_photos/ --pdf lecture.pdf --md lecture.md
```

For each photo it finds the screen, straightens it into an upright page, evens out the lighting, and reads
the text with Tesseract. The result is one searchable PDF (a page per photo, with a hidden text layer for
search and copy) and a Markdown file with the text of every slide.

Until real photos are collected (P9), it was tried on **synthetic photos**: the 14 first capture-kit pages
shown on a simulated screen in a simulated room and photographed by a simulated phone (tilt, lens blur, Bayer
sensor, moiré, noise, JPEG). What we learned:

| What | Result |
|---|---|
| Screen found (200 synthetic photos) | 97% on light pages, 58% on dark-mode pages; corners within 0.02% of the image diagonal |
| Words read correctly (word F1, 14 pages) | 0.75 without moiré removal, 0.70 with the classical notch filter |
| Character error rate (14 pages) | 0.36 without, 0.43 with (CER also counts reading order, so tables and chat layouts score worse than they read) |
| Time per photo (laptop CPU) | about 0.9 s to find the screen; about 7 s in total with OCR |

![Scan demo](figures/scan_demo.jpg)

Lessons for the product:

- **Dark mode is the hard case** for finding the screen: the bezel often reflects more light than a dark page,
  so the page edge can be invisible. Then the whole photo is used; the web app will let users drag the corners.
- **A classical moiré filter is not enough for text.** It fixed one table but damaged code, whose regular
  lines look like moiré to it. This is why a learned model is trained next, and judged by OCR, not only PSNR.
- **OCR needs local thresholding** on photos of screens; a global threshold broke on coloured text.

## Out of scope

Getting around screenshot or screen-recording restrictions on any system. The product is built for
photos people already take.
