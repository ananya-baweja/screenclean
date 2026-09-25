# Capture guide: photographing the test pages

This benchmark measures how well ScreenClean reads text from **real phone photos of screens**.
You show the pages in `capture_kit/` on a screen and photograph them with a phone. The corner
markers let the software find out which page each photo shows, so there is nothing to label.

About 1 to 1.5 hours in total.

## Checklist

**Setup (each screen)**

- [ ] Get the latest version of the repository (in VS Code: Source Control → **Pull**).
- [ ] Open `capture_kit/index.html` in Chrome (double-click it, or drag it into a browser window).
- [ ] Press **F** for full-screen. Use **→** / **←** (or click) to change pages.
- [ ] Set screen brightness to about **70%**.
- [ ] Turn off Night Light / True Tone / blue-light filters.
- [ ] Silence notifications. **Only the capture pages should be on screen**, nothing personal.

**Devices**

- [ ] Use every screen you can: your laptop, an external or lab monitor, a friend's laptop, a tablet, a TV.
- [ ] Use every phone you can borrow (at least your own).
- [ ] Each screen + phone combination gets its own folder, named `screen-<name>__phone-<name>`,
      for example `screen-hp-laptop__phone-redmi12`. Use short names, lowercase, no spaces.

**Per page: 3–4 photos, one of each**

- [ ] straight on, about **40 cm** away
- [ ] at an **angle** of 15–30°
- [ ] **farther** away, about 70 cm
- [ ] with **2× zoom**

For every photo:

- [ ] Use the normal photo mode (no portrait, night or document mode), **no flash**, and hold steady.
- [ ] Make sure **all 4 corner markers** are fully inside the frame.
- [ ] **Moiré must be visible.** You should see rippling or rainbow patterns in the camera preview. If
      you don't, move a little closer or farther, or change the zoom until you do.

**Target:** 120–160 photos in total (for example 40 pages × 3–4 photos, spread over your screen/phone
combinations). You don't need every page on every combination.

## After shooting

- [ ] Don't edit, crop or filter the photos. Keep the **original JPEG** files.
  On iPhone, set Settings → Camera → Formats → **Most Compatible** first, so photos are JPEG, not HEIC.
- [ ] Upload each folder to Google Drive under `MyDrive/screenclean/real_captures/raw/`, for example
  `MyDrive/screenclean/real_captures/raw/screen-hp-laptop__phone-redmi12/`.

## Why these rules

| Rule | Reason |
|---|---|
| All 4 markers in the frame | The markers identify the page and map the photo onto it (at least 3 are needed). |
| Visible moiré | That's what the benchmark measures; a photo without moiré doesn't test anything. |
| Different angles and distances | Moiré changes with angle and scale; real use has all of them. |
| Several screens and phones | Pixel layouts and camera processing differ; results should hold across devices. |
| Original JPEGs | Editing or re-saving changes the pixels the methods will be scored on. |
