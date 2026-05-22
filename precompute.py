"""
precompute.py — Pre-compute and cache superpixels for a folder of images.

Run this once before distributing a folder to annotators. It generates
  <folder>/.annotations/<stem>_segs_<cell_size>.npy
for every image. When batch_annotator.py opens the same folder it loads
these files instantly instead of recomputing.

Usage:
    uv run precompute.py path/to/folder
"""

import argparse
import os
from pathlib import Path

import numpy as np
from skimage.color import gray2rgb
from skimage import io
from skimage.segmentation import slic

from config import load_config

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SAVE_DIR   = ".annotations"
_SIGMA     = 1.0


def find_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_image(path: str) -> np.ndarray:
    img = io.imread(path)
    if img.ndim == 2:
        img = gray2rgb(img)
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    return img[:, :, :3]


def run_slic(img: np.ndarray, cell_size: int) -> np.ndarray:
    h, w, _ = img.shape
    n_segments = max(1, int((h * w) / cell_size))
    return slic(
        img,
        n_segments=n_segments,
        sigma=_SIGMA,
        slic_zero=True,
        start_label=0,
    ).astype(np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute superpixel cache for a folder of images"
    )
    parser.add_argument("folder", help="Folder containing images")
    args = parser.parse_args()

    folder = Path(os.path.normpath(os.path.abspath(os.path.expanduser(args.folder))))
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        raise SystemExit(1)

    cfg       = load_config(None)
    cell_size = cfg["cell_size"]

    images = find_images(folder)
    if not images:
        print(f"No images found in {folder}")
        raise SystemExit(1)

    seg_dir = folder / SAVE_DIR
    seg_dir.mkdir(exist_ok=True)

    print(f"[precompute] {len(images)} image(s) in {folder}")
    print(f"[precompute] cell_size={cell_size}  →  cache dir: {seg_dir}")

    for i, img_path in enumerate(images, 1):
        cache_path = seg_dir / f"{img_path.stem}_segs_{cell_size}.npy"
        if cache_path.exists():
            print(f"  [{i}/{len(images)}] {img_path.name}  (cached — skipping)")
            continue
        print(f"  [{i}/{len(images)}] {img_path.name}  computing...", end="", flush=True)
        img = load_image(str(img_path))
        segs = run_slic(img, cell_size=cell_size)
        np.save(str(cache_path), segs)
        print(" done")

    print(f"[precompute] All done. Distribute the folder including {SAVE_DIR}/")


if __name__ == "__main__":
    main()
