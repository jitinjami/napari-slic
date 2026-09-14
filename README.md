# napari-slic — Interactive Superpixel Annotation Tool

A point-and-click tool for labelling wound images. It automatically divides
your image into small, colour-coherent patches called **superpixels**, then
lets you colour each patch with a tissue or border label by clicking on it.

---

## What is a superpixel?

Imagine your image is split into hundreds of puzzle pieces — each piece covers
a region of similar colour and texture. Those pieces are superpixels.

Instead of labelling pixel-by-pixel, you label one whole piece at a time. This
makes annotation much faster: a single click colours an entire region.

---

## Installation

You need [**uv**](https://docs.astral.sh/uv/) (a fast Python package manager).
Once you have it, run this once inside the project folder:

```bash
uv sync
```

This installs every dependency automatically. You do not need to create a
virtual environment manually.

---

## Running the tool

The tool works on a **folder of images**. To annotate a single image, put it in
a folder of its own.

```bash
uv run batch_annotator.py                    # opens a folder picker dialog
uv run batch_annotator.py path/to/folder     # or pass the folder directly
```

If no folder is given, a native **folder picker dialog** appears so you can
browse to the right location without touching the terminal.

The right-hand panel gains extra controls:

| Control | What it does |
|---|---|
| `○ / ✓  1 / 12 — filename` | Image counter; `✓` means annotations are saved |
| **◀ Prev / Next ▶** | Navigate images (auto-saves first) |
| **💾 Save** | Manually save current image's annotations |
| **📤 Export This** | Export current image's masks to `masks/` |
| **📤 Export All** | Export every image that has saved annotations |

### Where files go

```
<folder>/
├── .annotations/          ← auto-saved project state (hidden folder)
│   ├── image_001_segs_550.npy    ← cached superpixels
│   ├── image_001_borders.npy     ← your border annotations
│   ├── image_001_tissues.npy     ← your tissue annotations
│   └── ...
└── masks/                 ← exported masks (created on demand)
    ├── image_001_borders.npy / .png
    └── image_001_tissues.npy / .png
```

> Re-opening the same folder later restores your previous annotations
> automatically.

---

## Speeding up large datasets *(optional)*

Working out the superpixels for an image takes a moment — roughly 0.2 s for a
small photo, 1.5 s for a large one. `batch_annotator.py` does this for you
automatically and caches the result, so you normally do not need to think
about it.

If you are preparing a **whole dataset for other people to annotate**, run this
once first:

```bash
uv run batch_precompute.py path/to/dataset
```

It walks every subfolder, computes the superpixels for every image using all
your CPU cores, and writes them into each folder's `.annotations/`. A few
hundred images take well under a minute instead of several minutes.

Then hand over the dataset folder **including the hidden `.annotations/`
folders** — the annotator opens every image instantly, with no waiting.

Already-computed images are skipped, so it is safe to re-run at any time.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `1` – `9` | Select the nth non-background class (across all layers, in config order) |
| `0` | Select Background on the active layer |
| `Ctrl-Z` | Undo the last paint stroke on the active layer (up to 20 strokes) |
| `←` / `→` | Prev / Next image *(batch mode only)* — auto-saves first |

---

## Custom classes

Classes are defined in one place: **`config.py`**. Edit the `_DEFAULT_LAYERS`
list near the top of that file — no other file needs changing.

```python
_DEFAULT_LAYERS = [
    {
        "name": "tissue_type",
        "classes": [
            {"id": 0, "label": "Background", "color": [255, 255, 255]},
            {"id": 1, "label": "Tumor",      "color": [220,  50,  50]},
            {"id": 2, "label": "Stroma",     "color": [ 50, 180, 100]},
        ],
    },
]
```

Colours are **RGB**, 0–255. `cell_size` is set just above, in the same file.

Rules:
- You can have **any number of layers** (each gets its own saved mask).
- Class `id` values can be **any integers**.
- Class `id: 0` is always treated as transparent background.
- Key `1` maps to the first non-background class in layer 1, key `2` to the
  second, and so on across all layers up to `9`.

---

## What you see when it opens

When the tool starts, a [napari](https://napari.org) window appears with:

| Layer | What it shows |
|---|---|
| **image** | Your original photo |
| **superpixel_edges** | Thin black lines showing where each superpixel ends |
| **borders** | Annotation overlay — hidden by default, toggle with **Show/Hide** |
| **tissues** | Annotation overlay — hidden by default, toggle with **Show/Hide** |

On the **right side** you'll see the *Annotation Controls* panel.

### Showing and hiding layers

Both annotation layers start **hidden** so you see only the raw image and
superpixel edges. Each layer section has a **Show / Hide** button next to its
header. Clicking **Show** makes that layer visible and automatically hides the
other — so at most one layer is visible at a time, keeping the view clean.

---

## The seven label classes

There are two separate labelling tasks. Each has its own set of classes and its
own layer:

### Border labels (painted on the *borders* layer)

| Button colour | Class name | What it means |
|---|---|---|
| White | 0 · Background | Not part of the wound border |
| Blue | 1 · Flat Wound Border | A flat, well-defined wound edge |
| Lilac | 2 · Punched Out Border | A raised or undermined wound edge |

### Tissue labels (painted on the *tissues* layer)

| Button colour | Class name | What it means |
|---|---|---|
| White | 0 · Background | Not part of the wound tissue |
| Orange | 3 · Granulation | Healing red/pink granulation tissue |
| Green | 4 · Slough | Yellow/white dead tissue |
| Black | 5 · Necrosis | Dark dead tissue |
| Burgundy | 6 · Unhealthy Granulation | Poor-quality, non-healing granulation |

---

## Step-by-step: how to annotate

1. **Show a layer** — click **Show** next to *BORDERS* or *TISSUES* in the
   right panel. The overlay appears; the other layer is hidden automatically.

2. **Pick a class** — click one of the coloured buttons.
   The status bar at the top of the panel turns that class's colour so you
   always know what will be painted next.

3. **Click a superpixel** — click anywhere inside a superpixel on the image.
   The entire superpixel fills with the class colour instantly.

4. **Paint multiple superpixels at once** — click and drag across several
   superpixels to fill them all in one stroke.

5. **Overwrite a superpixel** — pick a different class button and click the
   superpixel. It is repainted with the new class immediately.

6. **Erase a superpixel** — click the white **"0: Background"** button, then
   click the superpixel. It becomes transparent again.

7. **Switch layers** — click **Show** on the other layer. The current layer
   hides automatically so you can work on the new one without colour overlap.

8. **Start over** — click **↺ Reset Borders** or **↺ Reset Tissues** to wipe
   all annotations on that layer.

9. **Save your work** — your annotations are saved automatically whenever you
   move to another image, and **💾 Save** saves on demand. Use **📤 Export This**
   or **📤 Export All** to write finished masks to `masks/`. See
   [Where files go](#where-files-go).

   > **Always save before closing the window** — closing does not autosave.

---

## Adjusting superpixel behaviour

| Setting | Where | Default | What it does |
|---|---|---|---|
| `cell_size` | `config.py` | `550` | Pixels per superpixel; smaller → more, finer superpixels |
| `_SIGMA` | `batch_annotator.py` / `batch_precompute.py` | `1.0` | Blur before segmentation; higher → smoother boundaries |

The tool uses **SLICO** (parameter-free SLIC), which adapts compactness per
cluster automatically — no tuning needed.

The number of superpixels is set automatically based on image size
(`image_height × image_width ÷ cell_size`), so larger images get more superpixels.

> Changing `cell_size` invalidates cached superpixels — the cache filename
> embeds it, so new files are generated on the next run.

---

## File structure

```
napari-slic/
├── batch_annotator.py  # the annotation tool — run this on a folder of images
├── batch_precompute.py # optional: pre-compute superpixels for a whole dataset
├── config.py           # class definitions (edit this to change classes)
├── pyproject.toml      # dependencies (managed by uv)
└── README.md           # this file
```

---

## How it works under the hood *(optional reading)*

> You don't need to read this to use the tool. It explains *why* the tool is
> built the way it is, for the curious.

### Direct superpixel painting (no fill tool)

The tool does **not** use napari's built-in fill/paint tools. Instead, it
listens to raw mouse press and drag events at the viewer level:

```
mouse press  → find the superpixel under the cursor → paint all its pixels
mouse drag   → repeat for every new superpixel the cursor enters
mouse release → end stroke
```

"Paint all its pixels" means finding every pixel `p` where
`segments[p] == slic_id` and writing the active `class_id` directly into the
annotation array. Because the entire superpixel is written in one NumPy mask
operation, it is impossible for a neighbouring superpixel to be accidentally
modified — the mask is always exact.

### What the annotation data stores

| Value | Meaning |
|---|---|
| `0` | Background (transparent) — the default for every pixel |
| `1–6` | The painted class ID |

All pixels start as `0`. Painting Background (`0`) over a coloured superpixel
restores it to transparent — Background acts as an eraser.

### SLICO segmentation

Superpixels are computed with the **SLICO** variant of the SLIC algorithm
(`slic_zero=True` in scikit-image). Unlike standard SLIC, SLICO adapts its
compactness parameter per cluster, so superpixels naturally follow colour
and texture boundaries without manual tuning.

### Export

The `.npy` files contain only values `{0, 1, 2, 3, 4, 5, 6}` — one integer per
pixel representing its class ID. `0` means background (unannotated or
explicitly cleared).
