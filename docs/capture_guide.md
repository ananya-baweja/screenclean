# Capture guide: photographing the test pages

This benchmark measures how well ScreenClean reads text from **real phone photos of screens**.
You show the pages in `capture_kit/` on a screen and photograph them with a phone. The corner
markers let the software find out which page each photo shows, so there is nothing to label.

**40 photos: 20 pages on each of 2 screens, one photo per page.** About 20 minutes of shooting.

## Checklist

**Setup (each screen)**

- [ ] Get the latest version of the repository (in VS Code: Source Control → **Pull**).
- [ ] Open `capture_kit/index.html` in Chrome (double-click it, or drag it into a browser window).
- [ ] Press **F** for full-screen.
- [ ] Set screen brightness to about **70%**.
- [ ] Turn off Night Light / True Tone / blue-light filters.
- [ ] Silence notifications. **Only the capture pages should be on screen**, nothing personal.

**Screens and phone**

- [ ] **Screen 1:** your laptop. **Screen 2:** any other screen: an external or lab monitor, a TV, a
      friend's laptop or a tablet. (Only one screen available? Shoot all 40 pages on it.)
- [ ] Your own phone is enough. If a friend's phone is at hand, it can take screen 2 instead.
- [ ] Each screen + phone combination gets its own folder, named `screen-<name>__phone-<name>`,
      for example `screen-hp-laptop__phone-redmi12`. Use short names, lowercase, no spaces.

**The plan: one photo per page, in blocks of 5 from the same position**

| Screen 1 pages | Screen 2 pages | Position |
|---|---|---|
| 1–5 | 21–25 | **straight on**, about 40 cm away |
| 6–10 | 26–30 | at an **angle** of 15–30° |
| 11–15 | 31–35 | **farther** away, about 70 cm |
| 16–20 | 36–40 | **2× zoom** |

Keep to this plan: the software reads the position from the page number.

**Let the slideshow turn the pages**

- [ ] Screen 1 starts at page 1. On screen 2, type **21** and press **Enter** to jump to page 21.
- [ ] Press **A**: the pages now turn by themselves every 6 seconds, without any text on screen.
      **+** gives you more time per page, **−** less. Press **A** again to stop (for example while you
      move to the next position). The arrow keys **←** / **→** still work.
- [ ] Take one photo of each page while it is shown.

For every photo:

- [ ] Use the normal photo mode (no portrait, night or document mode), **no flash**, and hold steady.
- [ ] Make sure **all 4 corner markers** are fully inside the frame.
- [ ] **Moiré must be visible.** You should see rippling or rainbow patterns in the camera preview. If
      you don't, move a little closer or farther until you do (stay roughly at the planned position).
- [ ] Missed a page or took a blurry photo? Stop with **A**, go back with **←**, and retake it. Extra
      photos of a page are fine; delete the bad one.

## After shooting

- [ ] Don't edit, crop or filter the photos. Keep the **original JPEG** files.
  On iPhone, set Settings → Camera → Formats → **Most Compatible** first, so photos are JPEG, not HEIC.
- [ ] Upload each folder to Google Drive under `MyDrive/screenclean/real_captures/raw/`, for example
  `MyDrive/screenclean/real_captures/raw/screen-hp-laptop__phone-redmi12/`.

## Why these rules

| Rule | Reason |
|---|---|
| 40 photos | Enough to measure differences of about 2 percentage points in character error rate (95% confidence); four times as many photos would only halve that, at four times the effort and Colab time. |
| All 4 markers in the frame | The markers identify the page and map the photo onto it (at least 3 are needed). |
| Visible moiré | That's what the benchmark measures; a photo without moiré doesn't test anything. |
| Different positions | Moiré changes with angle and scale; real use has all of them. |
| Two screens | Pixel layouts differ; results should hold beyond one device. |
| Original JPEGs | Editing or re-saving changes the pixels the methods will be scored on. |
