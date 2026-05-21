# Tutorial PDF Plan — napari-slic Annotation Tool

Each section lists what to write and what screenshot(s) to capture.

---

## 1. Introduction (no screenshot needed)

- One paragraph: what the tool is for (labelling wound images)
- One paragraph: what a superpixel is — use the puzzle-piece analogy
- Mention the two tasks: borders and tissues

---

## 2. Installation

**Screenshots:**
- Terminal showing `uv sync` running and completing successfully

**Text:** Paste the two commands:
```
uv sync
uv run batch_annotator.py
```

---

## 3. Launching the Tool

**Screenshots:**
1. The native folder picker dialog that appears on launch (empty, before selecting anything)
2. The folder picker with the target folder selected/highlighted
3. The napari window immediately after it opens — showing the raw image and superpixel edges, both annotation layers hidden, panel on the right

**Text:** Explain that no folder argument is needed — the dialog appears automatically.

---

## 4. The Interface Overview

**Screenshot:**
- Full napari window, annotated with numbered callouts pointing to:
  1. The image canvas
  2. The superpixel edge overlay (black lines)
  3. The right-hand Annotation Controls panel
  4. The status bar at the top of the panel ("No class selected")
  5. The layer list on the left (napari's built-in panel)
  6. The image counter / filename label (batch mode)

**Text:** Brief description of each numbered area.

---

## 5. Showing a Layer

**Screenshots:**
1. Panel close-up — both layers hidden, both buttons say "Show"
2. Panel close-up — after clicking **Show** on BORDERS: button now says "Hide", TISSUES still says "Show"
3. Canvas showing the borders layer now visible (all transparent / white at this point since nothing is painted yet)

**Text:** Explain radio behaviour — showing one layer hides the other.

---

## 6. Selecting a Class

**Screenshots:**
1. Panel close-up — clicking the "1: Flat Wound Border" button (highlight the button)
2. Status bar close-up — showing it has turned blue with "✎ BORDERS → 1: Flat Wound Border"

**Text:** Explain that the coloured status bar always shows what will be painted next.

---

## 7. Painting Superpixels

**Screenshots:**
1. Cursor hovering over a superpixel before clicking
2. After a single click — one superpixel filled with blue
3. After a drag stroke — several adjacent superpixels filled
4. A wider view showing a partially annotated image

**Text:** Explain click to paint one, click-and-drag to paint many in one stroke.

---

## 8. Erasing (Reverting to Background)

**Screenshots:**
1. Panel close-up — clicking "0: Background" button; status bar turns white
2. Canvas before and after clicking a painted superpixel — it goes transparent

**Text:** Background (class 0) is the eraser. Select it, then click any painted superpixel to clear it.

---

## 9. Overwriting a Superpixel with a Different Class

**Screenshots:**
1. A superpixel painted blue (Flat Wound Border)
2. Same superpixel after selecting Lilac (Punched Out Border) and clicking — now lilac

**Text:** Just select the new class and click. No need to erase first.

---

## 10. Working on the Second Layer

**Screenshots:**
1. Clicking **Show** on TISSUES — borders layer hides, tissues layer appears
2. Status bar after selecting "3: Granulation" — shows orange
3. Canvas with some granulation painted in orange

**Text:** Emphasise that data on the hidden layer is safe — hiding just changes the view.

---

## 11. Undoing a Mistake

**Screenshots:**
1. Before: a wrongly painted superpixel
2. After pressing Ctrl-Z: the superpixel reverted

**Text:** Ctrl-Z undoes the last stroke on the active layer (up to 20 strokes).

---

## 12. Navigating Between Images (Batch Mode)

**Screenshots:**
1. Image counter label — e.g. "○ 1 / 5 — apple.jpg"
2. Clicking **Next ▶** — new image loads, counter updates to "○ 2 / 5 — apple-2.jpg"
3. Counter showing "✓" checkmark after the image has been saved

**Text:** Navigation auto-saves before switching. ✓ means that image's annotations are on disk.

---

## 13. Saving and Exporting

**Screenshots:**
1. Clicking **💾 Save** — console/terminal showing the save confirmation message
2. Clicking **📤 Export This** — terminal showing export output
3. File explorer showing the `masks/` folder with the generated `.npy` and `.png` files
4. One of the exported `_borders.png` and one `_tissues.png` side by side

**Text:** Explain the difference between Save (internal project state) and Export (final mask files).

---

## 14. Output Files Explained

**Screenshots:**
- Side-by-side: original image | borders PNG | tissues PNG

**Text:** Table of file names and what each contains:

| File | Contents |
|---|---|
| `.annotations/<stem>_borders.npy` | Auto-saved border annotations (raw, restorable) |
| `.annotations/<stem>_tissues.npy` | Auto-saved tissue annotations (raw, restorable) |
| `masks/<stem>_borders.npy` | Exported border mask — integer array, values 0–2 |
| `masks/<stem>_borders.png` | Colour image of border annotations |
| `masks/<stem>_tissues.npy` | Exported tissue mask — integer array, values 0, 3–5 |
| `masks/<stem>_tissues.png` | Colour image of tissue annotations |

---

## 15. Tips & Common Mistakes (no screenshots needed)

- **Painting on the wrong layer** — check the status bar colour before painting
- **Can't paint anything** — make sure you have clicked a class button first (status bar says "No class selected" until you do)
- **Accidentally hid a layer** — click **Show** to bring it back; data is never lost by hiding
- **Exported but not saved** — Export also saves; you can use either button
- **Re-opening a folder** — previous annotations load automatically; no need to re-annotate

---

## Suggested PDF Structure

1. Cover page (tool name, date, contact)
2. Introduction
3. Installation
4. Quick-start walkthrough (sections 3–9 above, combined into one flow)
5. Working with two layers (section 10)
6. Batch workflow — navigation, saving, exporting (sections 12–13)
7. Output files reference (section 14)
8. Tips & common mistakes (section 15)
