"""
Napari Interactive Superpixel Annotation Tool
==============================================
Usage:
    uv run annotator.py [image_path]

If no image path is given, IMAGE_PATH below is used.
Install deps with:  uv sync

Controls:
    - Click a class button in the right panel to select a class.
    - Left-click (or drag) on the image to paint whole superpixels.
    - Clicking a painted superpixel a second time re-assigns it; only
      that superpixel is ever touched, never its neighbours.
    - Click "Export" to save borders and tissues as PNG + NPY.
"""

import sys
from pathlib import Path

import napari
import numpy as np
from magicgui import widgets as mw
from napari.utils.colormaps import Colormap
from skimage import io
from skimage.color import gray2rgb
from skimage.segmentation import find_boundaries, slic

# ─── Configurable constants ────────────────────────────────────────────────────
IMAGE_PATH  = "sample.png"
COMPACTNESS = 20
SIGMA       = 1.0
# ───────────────────────────────────────────────────────────────────────────────

# ─── Class palette (RGB 0-255) ─────────────────────────────────────────────────
PALETTE: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),   # Background          – White
    1: ( 12,  80, 155),   # Flat Wound Border   – Blue
    2: (228, 183, 229),   # Punched Out Border  – Lilac
    3: (238,  83,   0),   # Granulation         – Orange
    4: (  0, 157,  99),   # Slough              – Green
    5: (  0,   0,   0),   # Necrosis            – Black
}
BORDER_CLASSES = {0: "Background", 1: "Flat Wound Border", 2: "Punched Out Border"}
TISSUE_CLASSES = {0: "Background", 3: "Granulation", 4: "Slough", 5: "Necrosis"}
# ───────────────────────────────────────────────────────────────────────────────


def load_image(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 2:
        img = gray2rgb(img)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    return img[:, :, :3]


def run_slic(img: np.ndarray) -> np.ndarray:
    h, w, _ = img.shape
    n_segments = max(1, int((h * w) / 550))
    segments = slic(
        img,
        n_segments=n_segments,
        compactness=COMPACTNESS,
        sigma=SIGMA,
        start_label=0,
    ).astype(np.int32)
    return segments


def build_class_colormap(classes: dict[int, str]) -> dict:
    """
    Returns a direct label colormap dict mapping class_id → RGBA.

    The annotation data stores plain class IDs (0-5).
    Unpainted pixels are stored as -1 and shown as fully transparent,
    so the raw image shows through.
    """
    color_dict: dict = {
        None: (1.0, 1.0, 1.0, 0.0),   # catch-all → fully transparent
        -1:   (1.0, 1.0, 1.0, 0.0),   # unpainted → transparent
    }
    for class_id in classes:
        r, g, b = PALETTE[class_id]
        alpha = 0.0 if class_id == 0 else 0.65   # class 0 = background = transparent
        color_dict[class_id] = (r / 255, g / 255, b / 255, alpha)
    return color_dict


def make_button(text: str, rgb: tuple[int, int, int]) -> mw.PushButton:
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.144 * b
    fg = "#000000" if lum > 140 else "#ffffff"
    btn = mw.PushButton(text=text)
    btn.native.setStyleSheet(
        f"background-color: rgb({r},{g},{b}); color: {fg}; "
        f"font-weight: bold; padding: 5px 8px; border-radius: 4px;"
    )
    return btn


def main() -> None:
    image_path = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    img = load_image(image_path)
    segments = run_slic(img)
    H, W = segments.shape

    # ── Viewer ────────────────────────────────────────────────────────────────
    viewer = napari.Viewer(title="Superpixel Annotator")
    viewer.add_image(img, name="image")

    edges = find_boundaries(segments, mode="outer").astype(np.float32)
    black_edge_cmap = Colormap(
        name="black_edge",
        colors=[[0, 0, 0, 0], [0, 0, 0, 1]],
    )
    viewer.add_image(
        edges,
        name="superpixel_edges",
        colormap=black_edge_cmap,
        blending="translucent",
        opacity=0.85,
    )

    # ── Annotation layers ─────────────────────────────────────────────────────
    # Data stores plain class IDs: -1 = unpainted, 0 = background, 1-5 = classes.
    # We never use napari's fill tool; painting is done in mouse callbacks below.
    def make_ann_layer(name: str, classes: dict) -> napari.layers.Labels:
        data = np.full((H, W), -1, dtype=np.int32)
        from napari.utils import DirectLabelColormap
        layer = viewer.add_labels(
            data,
            name=name,
            opacity=1.0,
        )
        layer.colormap = DirectLabelColormap(color_dict=build_class_colormap(classes))
        layer.mode = "pan_zoom"   # disable all built-in drawing tools
        return layer

    borders_layer = make_ann_layer("borders", BORDER_CLASSES)
    tissues_layer = make_ann_layer("tissues", TISSUE_CLASSES)

    # ── State ─────────────────────────────────────────────────────────────────
    layer_class: dict[napari.layers.Labels, int] = {}

    # ── Painting helpers ──────────────────────────────────────────────────────
    def _paint_at_world(world_pos) -> None:
        """Paint the superpixel under world_pos on the currently active layer."""
        active = viewer.layers.selection.active
        if active not in (borders_layer, tissues_layer):
            return
        class_id = layer_class.get(active, None)
        if class_id is None:
            return   # no class selected yet

        coords = borders_layer.world_to_data(world_pos)
        if coords is None:
            return
        r, c = int(round(coords[-2])), int(round(coords[-1]))
        if not (0 <= r < H and 0 <= c < W):
            return

        slic_id = int(segments[r, c])
        mask = segments == slic_id          # every pixel of this superpixel

        new_data = active.data.copy()
        new_data[mask] = class_id           # store plain class ID
        active.data = new_data

    # ── Mouse drag callback (napari generator pattern) ────────────────────────
    # mouse_drag_callbacks fires on left-click and left-drag.
    # event.handled = True on each event suppresses pan_zoom so the drag
    # paints rather than pans.  When no class is selected we return early so
    # normal panning still works.
    def on_drag(viewer_obj, event):
        if event.button != 1:
            return
        active = viewer.layers.selection.active
        if active not in (borders_layer, tissues_layer):
            return
        if layer_class.get(active) is None:
            return  # no class chosen yet — allow normal panning
        event.handled = True              # suppress pan_zoom on press
        _paint_at_world(event.position)
        yield
        while event.type == "mouse_move":
            event.handled = True          # suppress pan_zoom on each move
            _paint_at_world(event.position)
            yield

    viewer.mouse_drag_callbacks.append(on_drag)


    # ── Widget ────────────────────────────────────────────────────────────────
    status_label = mw.Label(value="  No class selected  ")
    status_label.native.setStyleSheet(
        "background:#2a2a2a; color:#aaa; padding:6px; "
        "border-radius:4px; font-weight:bold; font-size:12px;"
    )

    def _activate(layer: napari.layers.Labels, class_id: int) -> None:
        layer_class[layer] = class_id
        viewer.layers.selection.active = layer
        layer_name = "BORDERS" if layer is borders_layer else "TISSUES"
        classes = BORDER_CLASSES if layer is borders_layer else TISSUE_CLASSES
        class_name = classes.get(class_id, "?")
        r, g, b = PALETTE[class_id]
        lum = 0.299 * r + 0.587 * g + 0.144 * b
        fg = "#000000" if lum > 140 else "#ffffff"
        status_label.value = f"  ✎  {layer_name}  →  {class_id}: {class_name}  "
        status_label.native.setStyleSheet(
            f"background-color:rgb({r},{g},{b}); color:{fg}; "
            f"padding:6px; border-radius:4px; font-weight:bold; font-size:12px;"
        )

    def _reset(layer: napari.layers.Labels) -> None:
        layer.data = np.full((H, W), -1, dtype=np.int32)
        layer.refresh()

    def _export(layer: napari.layers.Labels, name: str) -> None:
        out_dir = Path(image_path).parent
        raw = layer.data.copy()
        # Unpainted pixels (-1) become class 0 (background) on export
        ann = np.where(raw >= 0, raw, 0).astype(np.int32)
        npy_path = out_dir / f"{name}.npy"
        png_path = out_dir / f"{name}.png"
        np.save(npy_path, ann)
        h, w = ann.shape
        rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        for label_id, color in PALETTE.items():
            rgb[ann == label_id] = color
        io.imsave(str(png_path), rgb)
        print(f"[annotator] Saved -> {npy_path}  |  {png_path}")

    items: list[mw.Widget] = [status_label, mw.Label(value="")]

    items.append(mw.Label(value="─── BORDERS ───"))
    for lid, name in BORDER_CLASSES.items():
        btn = make_button(f"{lid}: {name}", PALETTE[lid])
        btn.changed.connect(lambda _, l=lid: _activate(borders_layer, l))
        items.append(btn)
    reset_borders = mw.PushButton(text="↺ Reset Borders")
    reset_borders.native.setStyleSheet(
        "color:#cc4444; font-weight:bold; padding:4px;"
    )
    reset_borders.changed.connect(lambda _: _reset(borders_layer))
    items.append(reset_borders)

    items.append(mw.Label(value=""))
    items.append(mw.Label(value="─── TISSUES ───"))
    for lid, name in TISSUE_CLASSES.items():
        btn = make_button(f"{lid}: {name}", PALETTE[lid])
        btn.changed.connect(lambda _, l=lid: _activate(tissues_layer, l))
        items.append(btn)
    reset_tissues = mw.PushButton(text="↺ Reset Tissues")
    reset_tissues.native.setStyleSheet(
        "color:#cc4444; font-weight:bold; padding:4px;"
    )
    reset_tissues.changed.connect(lambda _: _reset(tissues_layer))
    items.append(reset_tissues)

    items.append(mw.Label(value=""))
    export_btn = mw.PushButton(text="Export Annotations")
    export_btn.changed.connect(lambda _: (
        _export(borders_layer, "borders"),
        _export(tissues_layer, "tissues"),
        print("[annotator] Export complete."),
    ))
    items.append(export_btn)

    panel = mw.Container(widgets=items, label="")
    viewer.window.add_dock_widget(panel, area="right", name="Annotation Controls")
    napari.run()


if __name__ == "__main__":
    main()
