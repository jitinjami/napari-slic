"""
batch_precompute.py — Pre-compute and cache superpixels for every
image-containing subfolder under a root directory, using all CPU cores.

Run this before handing a dataset to annotators. It writes
    <folder>/.annotations/<stem>_segs_<cell_size>.npy
for every image, so batch_annotator.py loads them instantly instead of
recomputing. Already-cached images are skipped, so re-running is cheap.

Usage:
    uv run batch_precompute.py path/to/root
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from skimage import io
from skimage.color import gray2rgb
from skimage.segmentation import slic

from config import load_config

# Kept self-contained so the worker processes never import napari/Qt.
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


def find_image_folders(root: Path) -> list[Path]:
    folders = []
    for dirpath, _, filenames in os.walk(root):
        d = Path(dirpath)
        if any(f.lower().endswith(tuple(IMAGE_EXTS)) for f in filenames):
            folders.append(d)
    return sorted(folders)


def _compute_one(args: tuple) -> tuple[str, bool]:
    """Worker: compute and cache segments for one image. Returns (name, was_cached)."""
    img_path_str, cell_size = args
    img_path   = Path(img_path_str)
    cache_path = img_path.parent / SAVE_DIR / f"{img_path.stem}_segs_{cell_size}.npy"
    if cache_path.exists():
        return img_path.name, True
    img  = load_image(img_path_str)
    segs = run_slic(img, cell_size=cell_size)
    cache_path.parent.mkdir(exist_ok=True)
    np.save(str(cache_path), segs)
    return img_path.name, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute superpixels for all image folders under a root directory"
    )
    parser.add_argument("root", help="Root folder to search for image subfolders")
    args = parser.parse_args()

    root = Path(os.path.normpath(os.path.abspath(os.path.expanduser(args.root))))
    if not root.is_dir():
        print(f"Not a directory: {root}")
        raise SystemExit(1)

    cell_size = load_config(None)["cell_size"]

    folders = find_image_folders(root)
    if not folders:
        print(f"No image-containing folders found under {root}")
        raise SystemExit(0)

    all_images = [p for folder in folders for p in find_images(folder)]
    if not all_images:
        raise SystemExit(0)

    workers = os.cpu_count() or 4
    print(f"[batch_precompute] {len(all_images)} image(s) across {len(folders)} folder(s)")
    print(f"[batch_precompute] cell_size={cell_size}  workers={workers}\n")

    tasks = [(str(p), cell_size) for p in all_images]
    total = len(tasks)
    done  = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_compute_one, t): t for t in tasks}
        for future in as_completed(futures):
            name, cached = future.result()
            done += 1
            status = "cached" if cached else "computed"
            print(f"  [{done}/{total}] {name}  ({status})")

    print(f"\n[batch_precompute] Done.")


if __name__ == "__main__":
    main()
