"""
Napari Disagreement Corrector
=============================
Third-pass adjudication tool. Two annotators were merged into one set: where
they agreed the class was kept, where they disagreed the superpixel carries a
negative code saying WHICH two classes were in conflict (not who said what).

This tool shows one conflict type at a time and lets the adjudicator settle it
superpixel by superpixel.

Usage:
    uv run batch_corrector.py path/to/dataset

The folder is searched recursively, so a whole <patient>/<wound>/ tree works.

Reads (never modified):
    <folder>/.annotations/<stem>_<layer>.npy        merged, with negative codes
    <folder>/.annotations/<stem>_segs_*.npy         superpixel map

Writes:
    <folder>/.annotations/<stem>_<layer>_resolved.npy   plain class IDs

Re-opening the folder picks up each image exactly where it was left.

Keyboard shortcuts:
    Tab          — next unresolved conflict on this image
    1 / 2        — choose the first / second class of the active conflict
    Left / Right — previous / next image (auto-saves)
    Ctrl-Z       — undo the last decision
"""

import argparse
import os
from itertools import combinations
from pathlib import Path

import napari
import numpy as np
from magicgui import widgets as mw
from napari.utils import DirectLabelColormap
from napari.utils.colormaps import Colormap
from skimage import io
from skimage.color import gray2rgb
from skimage.segmentation import find_boundaries

from config import build_layer_specs, build_palette, load_config

# ─── Configurable ─────────────────────────────────────────────────────────────
IMAGE_EXTS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SAVE_DIR    = ".annotations"
SEG_MARKER  = "_segs_"
MAX_UNDO    = 20

# napari's Labels layer is happiest with non-negative integers, so the negative
# codes are shifted into a high positive band purely for display. The arrays on
# disk stay signed.  -1 -> 101, -10 -> 110
CODE_BASE = 100

# The conflict being resolved gets NO fill — only a cyan outline drawn around
# each of its superpixels, so the photo underneath stays fully visible. Once a
# superpixel is decided it picks up its class colour and the outline drops away.
ACTIVE_RGBA  = (0.00, 0.00, 0.00, 0.00)   # transparent — the outline marks it
PENDING_RGBA = (0.55, 0.55, 0.55, 0.15)   # grey wash — other conflicts, still open
OUTLINE_RGBA = (0.00, 1.00, 1.00, 1.00)   # cyan outline for the active conflict
SETTLED_A    = 0.30                       # agreed / already-decided, as context
# ──────────────────────────────────────────────────────────────────────────────


# ─── Pure helpers (no napari) ─────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 2:
        img = gray2rgb(img)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    return img[:, :, :3]


def find_images(root: Path) -> list[Path]:
    """Every annotated image under root, recursively, in folder order."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS or p.parent.name == SAVE_DIR:
            continue
        if segs_path(p) is not None:
            out.append(p)
    return out


def ann_dir(img_path: Path) -> Path:
    return img_path.parent / SAVE_DIR


def segs_path(img_path: Path):
    hits = sorted(ann_dir(img_path).glob(f"{img_path.stem}{SEG_MARKER}*.npy"))
    return hits[0] if hits else None


def merged_path(img_path: Path, layer: str) -> Path:
    return ann_dir(img_path) / f"{img_path.stem}_{layer}.npy"


def resolved_path(img_path: Path, layer: str) -> Path:
    return ann_dir(img_path) / f"{img_path.stem}_{layer}_resolved.npy"


def build_codes(layer_specs) -> dict:
    """{layer: {code: (class_a, class_b)}} — mirrors how the merge assigned them:
    every unordered pair of that layer's class IDs, in sorted order, numbered
    -1, -2, ... Direction was never recorded, so neither is it here."""
    codes = {}
    for name, classes in layer_specs:
        pairs = combinations(sorted(classes), 2)
        codes[name] = {-(i + 1): pair for i, pair in enumerate(pairs)}
    return codes


def to_display(signed: np.ndarray) -> np.ndarray:
    return np.where(signed < 0, CODE_BASE - signed, signed).astype(np.int32)


def to_signed(display: np.ndarray) -> np.ndarray:
    return np.where(display >= CODE_BASE, CODE_BASE - display, display).astype(np.int32)


def superpixel_values(seg: np.ndarray, data: np.ndarray, n_sp: int) -> np.ndarray:
    """One value per superpixel. Annotations are uniform inside a superpixel."""
    vals = np.zeros(n_sp, dtype=np.int32)
    vals[seg.ravel()] = data.ravel()
    return vals


def count_conflicts(seg, signed, n_sp, layer_codes) -> dict:
    """{code: number of superpixels still carrying it} — superpixels, not pixels."""
    vals = superpixel_values(seg, signed, n_sp)
    counts = {}
    for code in layer_codes:
        n = int((vals == code).sum())
        if n:
            counts[code] = n
    return counts


def load_layer(img_path: Path, layer: str):
    """(working_signed, n_conflicts_originally) or (None, 0) if absent."""
    src = merged_path(img_path, layer)
    if not src.exists():
        return None, 0
    merged = np.load(str(src)).astype(np.int32)
    done = resolved_path(img_path, layer)
    working = np.load(str(done)).astype(np.int32) if done.exists() else merged.copy()
    return working, merged


def make_button(text: str, rgb, enabled: bool = True) -> mw.PushButton:
    r, g, b = rgb
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    fg = "#000000" if lum > 140 else "#ffffff"
    btn = mw.PushButton(text=text)
    btn.native.setStyleSheet(
        f"background-color: rgb({r},{g},{b}); color: {fg}; "
        f"font-weight: bold; padding: 5px 8px; border-radius: 4px;"
    )
    btn.enabled = enabled
    return btn


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve two-annotator disagreements, one superpixel at a time"
    )
    parser.add_argument("folder", nargs="?", default=None,
                        help="Dataset folder (searched recursively; picker if omitted)")
    args = parser.parse_args()

    if args.folder is None:
        import tkinter as tk
        from tkinter import filedialog
        _tk = tk.Tk(); _tk.withdraw(); _tk.attributes("-topmost", True)
        chosen = filedialog.askdirectory(title="Select dataset folder",
                                         initialdir=Path.home())
        _tk.destroy()
        if not chosen:
            raise SystemExit(0)
        args.folder = chosen

    root = Path(os.path.normpath(os.path.abspath(os.path.expanduser(args.folder))))
    if not root.is_dir():
        print(f"Not a directory: {root}")
        raise SystemExit(1)

    cfg         = load_config(None)
    palette     = build_palette(cfg)
    layer_specs = build_layer_specs(cfg)
    codes       = build_codes(layer_specs)
    class_name  = {cid: lbl for _, cls in layer_specs for cid, lbl in cls.items()}

    print(f"[corrector] Scanning {root} ...")
    images = find_images(root)
    if not images:
        print(f"No annotated images found under {root}")
        raise SystemExit(1)
    n = len(images)
    print(f"[corrector] {n} image(s)")

    state: dict = {
        "idx": 0, "seg": None, "n_sp": 0, "H": 0, "W": 0,
        "work": {},      # layer -> signed working array
        "total": {},     # layer -> conflicts present in the merged file
        "counts": {},    # layer -> {code: remaining superpixels}
        "active": None,  # (layer, code)
        "paint": None,   # chosen class id
    }

    def cur() -> Path:
        return images[state["idx"]]

    def rel(p: Path) -> str:
        try:
            return str(p.relative_to(root))
        except ValueError:
            return p.name

    # ── Load one image's data ────────────────────────────────────────────────
    def load_state(idx: int):
        img_path = images[idx]
        img = load_image(str(img_path))
        seg = np.load(str(segs_path(img_path))).astype(np.int32)
        n_sp = int(seg.max()) + 1
        state.update({"seg": seg, "n_sp": n_sp,
                      "H": seg.shape[0], "W": seg.shape[1],
                      # inner boundary of every superpixel; intersected with the
                      # active-code mask this outlines each contested superpixel
                      # individually, which is what she clicks on.
                      "inner": find_boundaries(seg, mode="inner")})
        state["work"], state["total"], state["counts"] = {}, {}, {}
        for layer, _ in layer_specs:
            working, merged = load_layer(img_path, layer)
            if working is None:
                continue
            state["work"][layer] = working
            state["total"][layer] = sum(
                count_conflicts(seg, merged, n_sp, codes[layer]).values())
            state["counts"][layer] = count_conflicts(seg, working, n_sp, codes[layer])
        return img, seg

    img, seg = load_state(0)

    # ── Viewer ───────────────────────────────────────────────────────────────
    viewer = napari.Viewer(title="Disagreement Corrector")
    img_layer = viewer.add_image(img, name="image")

    edge_cmap = Colormap(name="black_edge", colors=[[0, 0, 0, 0], [0, 0, 0, 1]])
    edges_layer = viewer.add_image(
        find_boundaries(seg, mode="outer").astype(np.float32),
        name="superpixel_edges", colormap=edge_cmap,
        blending="translucent", opacity=0.35,
    )

    def colormap_for(layer: str, active_code):
        classes = dict(layer_specs)[layer]
        d = {None: (1.0, 1.0, 1.0, 0.0)}
        for cid in classes:
            r, g, b = palette[cid]
            d[cid] = (r / 255, g / 255, b / 255, 0.0 if cid == 0 else SETTLED_A)
        for code in codes[layer]:
            d[CODE_BASE - code] = ACTIVE_RGBA if code == active_code else PENDING_RGBA
        return DirectLabelColormap(color_dict=d)

    ann_layers: dict = {}
    for layer, _ in layer_specs:
        if layer not in state["work"]:
            continue
        lyr = viewer.add_labels(to_display(state["work"][layer]),
                                name=layer, opacity=1.0)
        lyr.colormap = colormap_for(layer, None)
        lyr.mode = "pan_zoom"
        lyr.visible = False
        ann_layers[layer] = lyr

    outline_layer = viewer.add_image(
        np.zeros(seg.shape, dtype=np.float32), name="active_conflict",
        colormap=Colormap(name="cyan_edge", colors=[[0, 1, 1, 0], list(OUTLINE_RGBA)]),
        blending="translucent", opacity=1.0,
    )

    def _refresh_outline() -> None:
        if state["active"] is None:
            outline_layer.data = np.zeros((state["H"], state["W"]), dtype=np.float32)
            return
        layer, code = state["active"]
        mask = state["work"][layer] == code
        outline_layer.data = (state["inner"] & mask).astype(np.float32)

    _undo: dict = {name: [] for name in ann_layers}

    # ── Panel widgets ────────────────────────────────────────────────────────
    nav_label = mw.Label(value="")
    nav_label.native.setStyleSheet("color:#ccc; font-size:11px; padding:2px 0;")

    progress_label = mw.Label(value="")
    progress_label.native.setStyleSheet("color:#9c9; font-size:11px; padding:2px 0;")

    status_label = mw.Label(value="  Pick a conflict below  ")
    status_label.native.setStyleSheet(
        "background:#2a2a2a; color:#aaa; padding:6px; "
        "border-radius:4px; font-weight:bold; font-size:12px;"
    )

    conflict_btns: dict = {}
    resolve_a = mw.PushButton(text="")
    resolve_b = mw.PushButton(text="")
    resolve_a.visible = resolve_b.visible = False

    def _style_resolve(btn, cid):
        r, g, b = palette[cid]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        fg = "#000000" if lum > 140 else "#ffffff"
        btn.text = f"{cid}: {class_name[cid]}"
        btn.native.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); color: {fg}; "
            f"font-weight: bold; padding: 8px; border-radius: 4px;"
        )
        btn.visible = True

    def _update_nav() -> None:
        remaining = sum(sum(c.values()) for c in state["counts"].values())
        mark = "✓" if remaining == 0 else "○"
        nav_label.value = f"  {mark}  {state['idx'] + 1} / {n}  —  {rel(cur())}"
        parts = []
        for layer, _ in layer_specs:
            if layer not in state["total"]:
                continue
            total = state["total"][layer]
            left = sum(state["counts"].get(layer, {}).values())
            pct = 100.0 * (total - left) / total if total else 100.0
            parts.append(f"{layer}: {total - left}/{total} ({pct:.0f}%)")
        progress_label.value = "  " + "   ".join(parts)

    def _refresh_conflicts() -> None:
        for (layer, code), btn in conflict_btns.items():
            left = state["counts"].get(layer, {}).get(code, 0)
            btn.visible = left > 0
            if left:
                a, b = codes[layer][code]
                mark = "▸ " if state["active"] == (layer, code) else "   "
                btn.text = f"{mark}{code}  {class_name[a]} / {class_name[b]}   ({left})"
        _update_nav()

    def _activate(layer: str, code: int) -> None:
        if state["counts"].get(layer, {}).get(code, 0) == 0:
            return
        state["active"] = (layer, code)
        state["paint"] = None
        for name, lyr in ann_layers.items():
            lyr.visible = (name == layer)
            lyr.colormap = colormap_for(name, code if name == layer else None)
        viewer.layers.selection.active = ann_layers[layer]
        ann_layers[layer].mode = "pan_zoom"
        a, b = codes[layer][code]
        _style_resolve(resolve_a, a)
        _style_resolve(resolve_b, b)
        status_label.value = f"  {code}  {class_name[a]} / {class_name[b]}  —  choose a class  "
        status_label.native.setStyleSheet(
            "background:#2a2a2a; color:#6ee; padding:6px; "
            "border-radius:4px; font-weight:bold; font-size:12px;"
        )
        _refresh_conflicts()
        _refresh_outline()

    def _choose(cid: int) -> None:
        if state["active"] is None:
            return
        state["paint"] = cid
        r, g, b = palette[cid]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        fg = "#000000" if lum > 140 else "#ffffff"
        layer, code = state["active"]
        status_label.value = f"  ✎  {code}  →  {cid}: {class_name[cid]}  "
        status_label.native.setStyleSheet(
            f"background-color:rgb({r},{g},{b}); color:{fg}; "
            f"padding:6px; border-radius:4px; font-weight:bold; font-size:12px;"
        )

    resolve_a.changed.connect(
        lambda _: _choose(codes[state["active"][0]][state["active"][1]][0])
        if state["active"] else None)
    resolve_b.changed.connect(
        lambda _: _choose(codes[state["active"][0]][state["active"][1]][1])
        if state["active"] else None)

    # ── Painting: one superpixel of the active conflict at a time ────────────
    def _push_undo(layer: str) -> None:
        stack = _undo[layer]
        stack.append(state["work"][layer].copy())
        if len(stack) > MAX_UNDO:
            stack.pop(0)

    def _paint_at(world_pos) -> bool:
        if state["active"] is None or state["paint"] is None:
            return False
        layer, code = state["active"]
        lyr = ann_layers[layer]
        coords = lyr.world_to_data(world_pos)
        if coords is None:
            return False
        r, c = int(round(coords[-2])), int(round(coords[-1]))
        if not (0 <= r < state["H"] and 0 <= c < state["W"]):
            return False
        work = state["work"][layer]
        # Only superpixels still carrying the active code may change. Agreed
        # regions and already-settled decisions are untouchable.
        if int(work[r, c]) != code:
            return False
        sid = int(state["seg"][r, c])
        new = work.copy()
        new[state["seg"] == sid] = state["paint"]
        state["work"][layer] = new
        lyr.data = to_display(new)
        state["counts"][layer][code] -= 1
        if state["counts"][layer][code] <= 0:
            del state["counts"][layer][code]
            state["active"] = None
            resolve_a.visible = resolve_b.visible = False
            status_label.value = "  conflict cleared — pick the next one  "
            lyr.colormap = colormap_for(layer, None)
        _refresh_outline()
        return True

    def on_drag(viewer_obj, event):
        if event.button != 1 or state["active"] is None or state["paint"] is None:
            return
        _push_undo(state["active"][0])
        event.handled = True
        changed = _paint_at(event.position)
        yield
        while event.type == "mouse_move":
            event.handled = True
            changed |= _paint_at(event.position)
            yield
        if changed:
            _refresh_conflicts()

    viewer.mouse_drag_callbacks.append(on_drag)

    def _do_undo() -> None:
        if state["active"] is None:
            actives = [l for l in ann_layers if _undo[l]]
            if not actives:
                return
            layer = actives[-1]
        else:
            layer = state["active"][0]
        stack = _undo[layer]
        if not stack:
            return
        restored = stack.pop()
        state["work"][layer] = restored
        ann_layers[layer].data = to_display(restored)
        state["counts"][layer] = count_conflicts(
            state["seg"], restored, state["n_sp"], codes[layer])
        _refresh_conflicts()
        _refresh_outline()
        print(f"[undo] {layer} — {len(stack)} step(s) remaining")

    # ── Save / navigate ──────────────────────────────────────────────────────
    def _save() -> None:
        for layer, work in state["work"].items():
            np.save(str(resolved_path(cur(), layer)), work)
        left = sum(sum(c.values()) for c in state["counts"].values())
        print(f"[corrector] Saved {rel(cur())} — {left} conflict(s) left")

    def _goto(new_idx: int) -> None:
        _save()
        for stack in _undo.values():
            stack.clear()
        state["idx"] = new_idx
        state["active"] = None
        state["paint"] = None
        img, seg = load_state(new_idx)
        img_layer.data = img
        edges_layer.data = find_boundaries(seg, mode="outer").astype(np.float32)
        for layer, lyr in ann_layers.items():
            if layer in state["work"]:
                lyr.data = to_display(state["work"][layer])
                lyr.colormap = colormap_for(layer, None)
            lyr.visible = False
        resolve_a.visible = resolve_b.visible = False
        status_label.value = "  Pick a conflict below  "
        status_label.native.setStyleSheet(
            "background:#2a2a2a; color:#aaa; padding:6px; "
            "border-radius:4px; font-weight:bold; font-size:12px;")
        viewer.reset_view()
        _refresh_conflicts()
        _refresh_outline()

    def _next_conflict() -> None:
        order = [(l, c) for l, _ in layer_specs
                 for c in sorted(codes.get(l, {}))
                 if state["counts"].get(l, {}).get(c, 0) > 0]
        if not order:
            return
        if state["active"] in order:
            nxt = order[(order.index(state["active"]) + 1) % len(order)]
        else:
            nxt = order[0]
        _activate(*nxt)

    # ── Assemble the panel ───────────────────────────────────────────────────
    items: list = [nav_label, progress_label, status_label, mw.Label(value="")]

    for layer, _ in layer_specs:
        if layer not in ann_layers:
            continue
        items.append(mw.Label(value=f"─── {layer.upper()} ───"))
        for code in sorted(codes[layer], reverse=True):
            btn = mw.PushButton(text="")
            btn.native.setStyleSheet(
                "text-align:left; padding:4px 8px; font-size:11px;")
            btn.changed.connect(lambda _, l=layer, c=code: _activate(l, c))
            btn.visible = False
            conflict_btns[(layer, code)] = btn
            items.append(btn)
        items.append(mw.Label(value=""))

    items.append(mw.Label(value="─── RESOLVE TO ───"))
    items += [resolve_a, resolve_b, mw.Label(value="")]

    items.append(mw.Label(value="─── NAVIGATION ───"))
    prev_btn = mw.PushButton(text="◀ Prev")
    prev_btn.native.setStyleSheet("font-weight:bold; padding:4px 10px;")
    prev_btn.changed.connect(lambda _: _goto((state["idx"] - 1) % n))
    next_btn = mw.PushButton(text="Next ▶")
    next_btn.native.setStyleSheet("font-weight:bold; padding:4px 10px;")
    next_btn.changed.connect(lambda _: _goto((state["idx"] + 1) % n))
    items.append(mw.Container(widgets=[prev_btn, next_btn],
                              layout="horizontal", label=""))

    save_btn = mw.PushButton(text="💾 Save")
    save_btn.changed.connect(lambda _: (_save(), _update_nav()))
    items.append(save_btn)

    edges_btn = mw.PushButton(text="Toggle superpixel edges")
    edges_btn.native.setStyleSheet("padding:2px 8px; font-size:11px;")
    edges_btn.changed.connect(
        lambda _: setattr(edges_layer, "visible", not edges_layer.visible))
    items.append(edges_btn)

    panel = mw.Container(widgets=items, label="")
    viewer.window.add_dock_widget(panel, area="right", name="Corrector")

    _refresh_conflicts()

    # ── Keyboard ─────────────────────────────────────────────────────────────
    viewer.bind_key("Tab", lambda _: _next_conflict(), overwrite=True)
    viewer.bind_key(
        "1", lambda _: _choose(codes[state["active"][0]][state["active"][1]][0])
        if state["active"] else None, overwrite=True)
    viewer.bind_key(
        "2", lambda _: _choose(codes[state["active"][0]][state["active"][1]][1])
        if state["active"] else None, overwrite=True)
    viewer.bind_key("Control-Z", lambda _: _do_undo(), overwrite=True)
    viewer.bind_key("Left",  lambda _: _goto((state["idx"] - 1) % n), overwrite=True)
    viewer.bind_key("Right", lambda _: _goto((state["idx"] + 1) % n), overwrite=True)

    napari.run()


if __name__ == "__main__":
    main()
