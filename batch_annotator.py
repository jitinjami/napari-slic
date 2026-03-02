"""
Napari Batch Superpixel Annotation Tool
========================================
Usage:
    uv run batch_annotator.py path/to/folder [--cell-size N] [--config path.yaml]

Reads all images in the folder and lets you annotate them one by one.
Navigation auto-saves the current image's annotations before switching.

All annotation state is saved to:
    <folder>/.annotations/<stem>_<layer>.npy

Exports (PNG + NPY) go to:
    <folder>/masks/<stem>_<layer>.npy  /  <stem>_<layer>.png

Keyboard shortcuts:
    0           — select Background on the active layer
    1 – 9       — select the nth non-background class (across all layers, in order)
    Left / Right — previous / next image (auto-saves)
    Ctrl-Z      — undo the last paint stroke on the active layer
"""

import argparse
import os
from pathlib import Path

import napari
import numpy as np
from magicgui import widgets as mw
from napari.utils import DirectLabelColormap
from napari.utils.colormaps import Colormap
from skimage import io
from skimage.color import gray2rgb
from skimage.segmentation import find_boundaries, slic

from config import build_layer_specs, build_palette, load_config

# ─── Image / SLIC helpers (self-contained, no dependency on annotator.py) ─────
_COMPACTNESS = 20
_SIGMA       = 1.0


def load_image(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 2:
        img = gray2rgb(img)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    return img[:, :, :3]


def run_slic(img: np.ndarray, cell_size: int = 550) -> np.ndarray:
    h, w, _ = img.shape
    n_segments = max(1, int((h * w) / cell_size))
    return slic(
        img,
        n_segments=n_segments,
        compactness=_COMPACTNESS,
        sigma=_SIGMA,
        start_label=0,
    ).astype(np.int32)


def build_class_colormap(classes: dict[int, str], palette: dict) -> dict:
    color_dict: dict = {
        None: (1.0, 1.0, 1.0, 0.0),
        -1:   (1.0, 1.0, 1.0, 0.0),
    }
    for class_id in classes:
        r, g, b = palette[class_id]
        alpha = 0.0 if class_id == 0 else 0.65
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


# ─── Configurable ─────────────────────────────────────────────────────────────
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SAVE_DIR   = ".annotations"
MASKS_DIR  = "masks"
MAX_UNDO   = 20
# ──────────────────────────────────────────────────────────────────────────────


# ── File helpers ───────────────────────────────────────────────────────────────

def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _anno_path(folder: Path, stem: str, layer_name: str) -> Path:
    d = folder / SAVE_DIR
    d.mkdir(exist_ok=True)
    return d / f"{stem}_{layer_name}.npy"


def _load_anno(path: Path, shape: tuple) -> np.ndarray:
    if path.exists():
        return np.load(str(path)).astype(np.int32)
    return np.full(shape, -1, dtype=np.int32)


def _save_all(ann_layers, layer_specs, folder: Path, stem: str) -> None:
    for (layer_name, _), ann_layer in zip(layer_specs, ann_layers):
        path = _anno_path(folder, stem, layer_name)
        np.save(str(path), ann_layer.data)
    names = ", ".join(f"{n}" for n, _ in layer_specs)
    print(f"[batch] Saved {stem} → {names}")


def _export_one(ann_layers, layer_specs, palette: dict,
                folder: Path, stem: str) -> None:
    masks_dir = folder / MASKS_DIR
    masks_dir.mkdir(exist_ok=True)
    for (layer_name, _), ann_layer in zip(layer_specs, ann_layers):
        raw = ann_layer.data.copy()
        ann = np.where(raw >= 0, raw, 0).astype(np.int32)
        npy_path = masks_dir / f"{stem}_{layer_name}.npy"
        png_path = masks_dir / f"{stem}_{layer_name}.png"
        np.save(str(npy_path), ann)
        h, w = ann.shape
        rgb = np.full((h, w, 3), 255, dtype=np.uint8)
        for label_id, color in palette.items():
            rgb[ann == label_id] = color
        io.imsave(str(png_path), rgb)
    print(f"[batch] Exported {stem} → {masks_dir}/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Napari batch superpixel annotation tool"
    )
    parser.add_argument("folder", nargs="?", default="./test/",
                        help="Folder containing images to annotate (default: current directory)")
    parser.add_argument("--cell-size", type=int, default=None,
                        help="Pixels per superpixel (default: 550 or from config)")
    parser.add_argument("--config", default=None,
                        help="Path to a YAML config file (classes, colors, cell_size)")
    args = parser.parse_args()

    folder = Path(os.path.normpath(os.path.abspath(os.path.expanduser(args.folder))))
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        raise SystemExit(1)

    images = find_images(folder)
    if not images:
        print(f"No images found in {folder}")
        raise SystemExit(1)

    # Auto-discover config.yaml in the folder if --config not given
    config_path = args.config
    if config_path is None and (folder / "config.yaml").exists():
        config_path = folder / "config.yaml"
        print(f"[batch] Using config: {config_path}")

    cfg         = load_config(config_path)
    cell_size   = args.cell_size if args.cell_size is not None else cfg["cell_size"]
    palette     = build_palette(cfg)
    layer_specs = build_layer_specs(cfg)

    n = len(images)
    print(f"[batch] Found {n} image(s) in {folder}")

    # ── Mutable state ──────────────────────────────────────────────────────────
    state: dict = {"idx": 0, "segments": None, "H": 0, "W": 0}

    def cur_stem() -> str:
        return images[state["idx"]].stem

    def _load_image_state(idx: int):
        img = load_image(str(images[idx]))
        segs = run_slic(img, cell_size=cell_size)
        H, W = segs.shape
        state["segments"] = segs
        state["H"] = H
        state["W"] = W
        return img, segs, H, W

    img, segs, H, W = _load_image_state(0)

    # ── Viewer ────────────────────────────────────────────────────────────────
    viewer = napari.Viewer(title="Batch Superpixel Annotator")
    img_layer = viewer.add_image(img, name="image")

    edges = find_boundaries(segs, mode="outer").astype(np.float32)
    black_edge_cmap = Colormap(name="black_edge", colors=[[0,0,0,0],[0,0,0,1]])
    edges_layer = viewer.add_image(
        edges, name="superpixel_edges",
        colormap=black_edge_cmap, blending="translucent", opacity=0.85,
    )

    # ── Annotation layers ─────────────────────────────────────────────────────
    def _make_ann_layer(layer_name: str, classes: dict,
                        stem: str, shape: tuple) -> napari.layers.Labels:
        path = _anno_path(folder, stem, layer_name)
        data = _load_anno(path, shape)
        layer = viewer.add_labels(data, name=layer_name, opacity=1.0)
        layer.colormap = DirectLabelColormap(
            color_dict=build_class_colormap(classes, palette)
        )
        layer.mode = "pan_zoom"
        return layer

    ann_layers = [
        _make_ann_layer(name, classes, cur_stem(), (H, W))
        for name, classes in layer_specs
    ]

    # ── State ─────────────────────────────────────────────────────────────────
    layer_class: dict[napari.layers.Labels, int] = {}
    _undo: dict[napari.layers.Labels, list[np.ndarray]] = {l: [] for l in ann_layers}

    def _push_undo(layer: napari.layers.Labels) -> None:
        stack = _undo[layer]
        stack.append(layer.data.copy())
        if len(stack) > MAX_UNDO:
            stack.pop(0)

    def _clear_undo() -> None:
        for stack in _undo.values():
            stack.clear()

    def _do_undo() -> None:
        active = viewer.layers.selection.active
        if active not in ann_layers:
            return
        stack = _undo[active]
        if stack:
            active.data = stack.pop()
            print(f"[undo] {active.name} — {len(stack)} step(s) remaining")

    # ── Painting ──────────────────────────────────────────────────────────────
    def _paint_at_world(world_pos) -> None:
        active = viewer.layers.selection.active
        if active not in ann_layers:
            return
        class_id = layer_class.get(active)
        if class_id is None:
            return
        coords = ann_layers[0].world_to_data(world_pos)
        if coords is None:
            return
        r, c = int(round(coords[-2])), int(round(coords[-1]))
        if not (0 <= r < state["H"] and 0 <= c < state["W"]):
            return
        slic_id = int(state["segments"][r, c])
        mask = state["segments"] == slic_id
        new_data = active.data.copy()
        new_data[mask] = class_id
        active.data = new_data

    def on_drag(viewer_obj, event):
        if event.button != 1:
            return
        active = viewer.layers.selection.active
        if active not in ann_layers:
            return
        if layer_class.get(active) is None:
            return
        _push_undo(active)
        event.handled = True
        _paint_at_world(event.position)
        yield
        while event.type == "mouse_move":
            event.handled = True
            _paint_at_world(event.position)
            yield

    viewer.mouse_drag_callbacks.append(on_drag)

    # ── Navigation ────────────────────────────────────────────────────────────
    def _autosave() -> None:
        _save_all(ann_layers, layer_specs, folder, cur_stem())

    def _goto(new_idx: int) -> None:
        _autosave()
        _clear_undo()        # don't carry undo history across images
        state["idx"] = new_idx
        img, segs, H, W = _load_image_state(new_idx)

        img_layer.data   = img
        edges_layer.data = find_boundaries(segs, mode="outer").astype(np.float32)

        stem = cur_stem()
        for (layer_name, _), ann_layer in zip(layer_specs, ann_layers):
            path = _anno_path(folder, stem, layer_name)
            ann_layer.data = _load_anno(path, (H, W))

        viewer.reset_view()
        _update_nav()

    # ── Widget ────────────────────────────────────────────────────────────────
    status_label = mw.Label(value="  No class selected  ")
    status_label.native.setStyleSheet(
        "background:#2a2a2a; color:#aaa; padding:6px; "
        "border-radius:4px; font-weight:bold; font-size:12px;"
    )

    nav_label = mw.Label(value="")
    nav_label.native.setStyleSheet("color:#ccc; font-size:11px; padding:2px 0;")

    def _update_nav() -> None:
        idx  = state["idx"]
        name = images[idx].name
        # Check if any layer file for this image exists
        saved = all(
            _anno_path(folder, images[idx].stem, ln).exists()
            for ln, _ in layer_specs
        )
        mark = "✓" if saved else "○"
        nav_label.value = f"  {mark}  {idx + 1} / {n}  —  {name}"

    _update_nav()

    def _activate(layer: napari.layers.Labels, class_id: int) -> None:
        layer_class[layer] = class_id
        viewer.layers.selection.active = layer
        layer.mode = "pan_zoom"
        layer_idx = ann_layers.index(layer)
        layer_name, classes = layer_specs[layer_idx]
        class_name = classes.get(class_id, "?")
        r, g, b = palette[class_id]
        lum = 0.299 * r + 0.587 * g + 0.144 * b
        fg = "#000000" if lum > 140 else "#ffffff"
        status_label.value = f"  ✎  {layer_name.upper()}  →  {class_id}: {class_name}  "
        status_label.native.setStyleSheet(
            f"background-color:rgb({r},{g},{b}); color:{fg}; "
            f"padding:6px; border-radius:4px; font-weight:bold; font-size:12px;"
        )

    def _reset(layer: napari.layers.Labels) -> None:
        _push_undo(layer)
        H, W = state["H"], state["W"]
        layer.data = np.full((H, W), -1, dtype=np.int32)
        layer.refresh()

    items: list[mw.Widget] = [nav_label, status_label, mw.Label(value="")]

    # Class buttons — one section per layer
    for (layer_name, classes), ann_layer in zip(layer_specs, ann_layers):
        items.append(mw.Label(value=f"─── {layer_name.upper()} ───"))
        for cls_id, cls_label in classes.items():
            btn = make_button(f"{cls_id}: {cls_label}", palette[cls_id])
            btn.changed.connect(
                lambda _, l=ann_layer, c=cls_id: _activate(l, c)
            )
            items.append(btn)
        reset_btn = mw.PushButton(text=f"↺ Reset {layer_name.title()}")
        reset_btn.native.setStyleSheet("color:#cc4444; font-weight:bold; padding:4px;")
        reset_btn.changed.connect(lambda _, l=ann_layer: _reset(l))
        items.append(reset_btn)
        items.append(mw.Label(value=""))

    # Navigation
    items.append(mw.Label(value="─── NAVIGATION ───"))
    prev_btn = mw.PushButton(text="◀ Prev")
    prev_btn.native.setStyleSheet("font-weight:bold; padding:4px 10px;")
    prev_btn.changed.connect(lambda _: _goto((state["idx"] - 1) % n))
    next_btn = mw.PushButton(text="Next ▶")
    next_btn.native.setStyleSheet("font-weight:bold; padding:4px 10px;")
    next_btn.changed.connect(lambda _: _goto((state["idx"] + 1) % n))
    items.append(mw.Container(widgets=[prev_btn, next_btn], layout="horizontal", label=""))
    items.append(mw.Label(value=""))

    # Save / export
    save_btn = mw.PushButton(text="💾 Save")
    save_btn.changed.connect(lambda _: (_autosave(), _update_nav()))

    export_this_btn = mw.PushButton(text="📤 Export This")
    export_this_btn.changed.connect(
        lambda _: _export_one(ann_layers, layer_specs, palette, folder, cur_stem())
    )

    def _export_all() -> None:
        count = 0
        for img_path in images:
            stem = img_path.stem
            # check all layers are saved
            if all(_anno_path(folder, stem, ln).exists() for ln, _ in layer_specs):
                # build fake layer list from saved npy files
                saved_data = [
                    _load_anno(_anno_path(folder, stem, ln), (1, 1))
                    for ln, _ in layer_specs
                ]
                # use real layers for current image, saved arrays for others
                if stem == cur_stem():
                    _export_one(ann_layers, layer_specs, palette, folder, stem)
                else:
                    class FakeLyr:
                        def __init__(self, data): self.data = data
                    fakes = [FakeLyr(d) for d in saved_data]
                    _export_one(fakes, layer_specs, palette, folder, stem)
                count += 1
        print(f"[batch] Export all done — {count} image(s) exported.")

    export_all_btn = mw.PushButton(text="📤 Export All")
    export_all_btn.changed.connect(lambda _: _export_all())

    items += [save_btn, export_this_btn, export_all_btn]

    panel = mw.Container(widgets=items, label="")
    viewer.window.add_dock_widget(panel, area="right", name="Batch Controls")

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    _shortcuts: list[tuple[napari.layers.Labels, int]] = []
    for (_, classes), ann_layer in zip(layer_specs, ann_layers):
        for cls_id in sorted(classes):
            if cls_id != 0:
                _shortcuts.append((ann_layer, cls_id))

    for key_num, (ann_layer, cls_id) in enumerate(_shortcuts[:9], start=1):
        viewer.bind_key(
            str(key_num),
            lambda _, l=ann_layer, c=cls_id: _activate(l, c),
            overwrite=True,
        )

    def _bg_shortcut(_):
        active = viewer.layers.selection.active
        target = active if active in ann_layers else ann_layers[0]
        _activate(target, 0)

    viewer.bind_key("0", _bg_shortcut, overwrite=True)
    viewer.bind_key("Control-Z", lambda _: _do_undo(), overwrite=True)
    viewer.bind_key("Left",  lambda _: _goto((state["idx"] - 1) % n), overwrite=True)
    viewer.bind_key("Right", lambda _: _goto((state["idx"] + 1) % n), overwrite=True)

    napari.run()


if __name__ == "__main__":
    main()
