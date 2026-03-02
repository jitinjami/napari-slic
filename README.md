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

### Single image

```bash
uv run annotator.py path/to/image.png
uv run annotator.py path/to/image.png --cell-size 300   # finer superpixels
uv run annotator.py path/to/image.png --config my_labels.yaml
```

---

## Batch mode — annotate a whole folder

```bash
uv run batch_annotator.py path/to/folder
uv run batch_annotator.py path/to/folder --cell-size 300
uv run batch_annotator.py path/to/folder --config my_labels.yaml
```

If a `config.yaml` file is found **inside the folder**, it is used automatically
without needing `--config`.

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
├── .annotations/          ← auto-saved project state
│   ├── image_001_borders.npy
│   ├── image_001_tissues.npy
│   └── ...
└── masks/                 ← exported masks (created on demand)
    ├── image_001_borders.npy / .png
    └── image_001_tissues.npy / .png
```

> Re-opening the same folder later restores your previous annotations
> automatically.

---

## Keyboard shortcuts

| Key | Action |
|---|---|
| `1` – `9` | Select the nth non-background class (across all layers, in config order) |
| `0` | Select Background on the active layer |
| `Ctrl-Z` | Undo the last paint stroke on the active layer (up to 20 strokes) |
| `←` / `→` | Prev / Next image *(batch mode only)* — auto-saves first |

---

## Custom classes — config file

Copy `config.example.yaml` to your image folder (or anywhere), edit it, and
pass it with `--config`:

```yaml
cell_size: 300   # smaller = more superpixels

layers:
  - name: tissue_type
    classes:
      - id: 0
        label: Background
        color: [255, 255, 255]
      - id: 1
        label: Tumor
        color: [220, 50, 50]
      - id: 2
        label: Stroma
        color: [50, 180, 100]
```

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
| **borders** | A transparent overlay — paint border classes here |
| **tissues** | A transparent overlay — paint tissue classes here |

On the **right side** you'll see the *Annotation Controls* panel with coloured
buttons and an export button.

---

## The six label classes

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

---

## Step-by-step: how to annotate

1. **Pick a class** — click one of the coloured buttons on the right.
   The status bar at the top of the panel turns that class's colour so you
   always know what will be painted next.

2. **Click a superpixel** — click anywhere inside a superpixel on the image.
   The entire superpixel fills with the class colour instantly.

3. **Paint multiple superpixels at once** — click and drag across several
   superpixels to fill them all in one stroke.

4. **Change your mind** — to reassign a superpixel, just pick a different
   class button and click it again. Only that one superpixel is affected,
   even if its neighbours are already painted the same colour.

5. **Un-paint a superpixel** — click the white **"0: Background"** button,
   then click the superpixel. It goes back to white (unpainted).

6. **Start over** — click **↺ Reset Borders** or **↺ Reset Tissues** to wipe
   all annotations on that layer and go back to a clean slate.

7. **Save your work** — click **Export Annotations**. Four files are written
   next to your input image:

   | File | Contents |
   |---|---|
   | `borders.npy` | NumPy array of border class IDs (values 0–2) |
   | `borders.png` | Colour image of border annotations |
   | `tissues.npy` | NumPy array of tissue class IDs (values 0, 3–5) |
   | `tissues.png` | Colour image of tissue annotations |

---

## Adjusting superpixel behaviour

Open `annotator.py` and change these values near the top:

| Constant | Default | What it does |
|---|---|---|
| `COMPACTNESS` | `20` | Higher → more square/regular superpixels; lower → superpixels follow colour boundaries more closely |
| `SIGMA` | `1.0` | Amount of blur applied before segmentation; higher → smoother boundaries |

The number of superpixels is set automatically based on image size
(`image_height × image_width ÷ 550`), so larger images get more superpixels.

---

## File structure

```
napari-slic/
├── annotator.py       # the whole tool — run this
├── pyproject.toml     # list of Python packages used (managed by uv)
└── README.md          # this file
```

Output files are saved in the **same folder as your input image**.

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
| `-1` | Not yet annotated (shown transparent) |
| `0` | Explicitly set to Background |
| `1–5` | The painted class ID |

### Export / decoding

Unpainted pixels (`-1`) become background (`0`) on export:

```python
ann = np.where(raw >= 0, raw, 0)
```

The final `.npy` files only ever contain values `{0, 1, 2, 3, 4, 5}`.
# napari-slic
