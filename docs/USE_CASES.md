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

## Out of scope

Getting around screenshot or screen-recording restrictions on any system. The product is built for
photos people already take.
