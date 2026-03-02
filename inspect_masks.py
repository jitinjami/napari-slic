"""
inspect_masks.py — Inspect exported annotation masks from batch_annotator.py

Usage:
    uv run inspect_masks.py [test_dir] [--config path.yaml]

Reads every *.npy file in <test_dir> (default: ./test/), prints the unique
class IDs, min/max, and saves a side-by-side RGB reconstruction next to each
.npy as <stem>_rgb_check.png.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from skimage import io

from config import build_palette, load_config

# ── Default palette copied from annotator.py (used if no config given) ─────────
_DEFAULT_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (255, 255, 255),   # Background          – White
    1: ( 12,  80, 155),   # Flat Wound Border   – Blue
    2: (228, 183, 229),   # Punched Out Border  – Lilac
    3: (238,  83,   0),   # Granulation         – Orange
    4: (  0, 157,  99),   # Slough              – Green
    5: (  0,   0,   0),   # Necrosis            – Black
}


def inspect(npy_path: Path, palette: dict) -> None:
    ann = np.load(str(npy_path)).astype(np.int32)

    unique   = sorted(np.unique(ann).tolist())
    min_cls  = int(ann.min())
    max_cls  = int(ann.max())

    # Map class IDs → human-readable names using the palette keys
    known    = {k for k in palette}
    unknown  = [c for c in unique if c not in known]

    print(f"\n{'─'*60}")
    print(f"  File    : {npy_path.name}")
    print(f"  Shape   : {ann.shape}")
    print(f"  Min ID  : {min_cls}")
    print(f"  Max ID  : {max_cls}")
    print(f"  Unique  : {unique}")
    if unknown:
        print(f"  ⚠ IDs not in palette: {unknown}")

    # Class pixel counts
    for cls_id in unique:
        count = int((ann == cls_id).sum())
        pct   = 100.0 * count / ann.size
        color = palette.get(cls_id, (128, 128, 128))
        print(f"    class {cls_id:2d}  →  {count:7d} px  ({pct:5.1f}%)  "
              f"RGB={color}")

    # Reconstruct RGB mask
    h, w = ann.shape
    rgb = np.full((h, w, 3), 200, dtype=np.uint8)   # mid-grey for unknowns
    for cls_id, color in palette.items():
        rgb[ann == cls_id] = color

    out_path = npy_path.with_name(npy_path.stem + "_rgb_check.png")
    io.imsave(str(out_path), rgb)
    print(f"  Saved   : {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect annotation .npy files")
    parser.add_argument("test_dir", nargs="?", default="./test/",
                        help="Folder containing exported .npy masks (default: ./test)")
    parser.add_argument("--config", default=None,
                        help="YAML config file (for palette). Uses built-in defaults if omitted.")
    args = parser.parse_args()

    base_dir = Path(os.path.normpath(os.path.abspath(os.path.expanduser(args.test_dir))))
    test_dir = base_dir / "masks"
    if not test_dir.is_dir():
        print(f"No 'masks' folder found inside {base_dir}")
        print(f"  (looked for: {test_dir})")
        raise SystemExit(1)

    if args.config:
        cfg     = load_config(args.config)
        palette = build_palette(cfg)
        print(f"[inspect] Using palette from {args.config}")
    else:
        palette = _DEFAULT_PALETTE
        print("[inspect] Using built-in default palette")

    npy_files = sorted(test_dir.glob("*.npy"))
    if not npy_files:
        print(f"No .npy files found in {test_dir}")
        raise SystemExit(0)

    print(f"[inspect] Found {len(npy_files)} mask(s) in {test_dir}\n")
    for npy_path in npy_files:
        inspect(npy_path, palette)

    print(f"\n[inspect] Done. RGB reconstructions saved next to each .npy.")


if __name__ == "__main__":
    main()
