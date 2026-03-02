"""
Shared configuration loader for annotator.py and batch_annotator.py.

If a config YAML is supplied (via --config or auto-discovery), it overrides
the built-in wound-annotation defaults.  Unknown keys are ignored.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# ── Built-in defaults (wound annotation) ──────────────────────────────────────
DEFAULT_CELL_SIZE = 550   # pixels per superpixel

_DEFAULT_LAYERS = [
    {
        "name": "borders",
        "classes": [
            {"id": 0, "label": "Background",         "color": [255, 255, 255]},
            {"id": 1, "label": "Flat Wound Border",  "color": [12,  80,  155]},
            {"id": 2, "label": "Punched Out Border", "color": [228, 183, 229]},
        ],
    },
    {
        "name": "tissues",
        "classes": [
            {"id": 0, "label": "Background",  "color": [255, 255, 255]},
            {"id": 3, "label": "Granulation", "color": [238,  83,   0]},
            {"id": 4, "label": "Slough",      "color": [  0, 157,  99]},
            {"id": 5, "label": "Necrosis",    "color": [  0,   0,   0]},
        ],
    },
]


def load_config(path: str | Path | None) -> dict:
    """Load a YAML config or return the built-in defaults."""
    cfg: dict = {"cell_size": DEFAULT_CELL_SIZE, "layers": _DEFAULT_LAYERS}
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg.update(user)
    return cfg


def build_palette(cfg: dict) -> dict[int, tuple[int, int, int]]:
    """Return {class_id: (R, G, B)} extracted from config."""
    palette: dict[int, tuple[int, int, int]] = {}
    for layer in cfg["layers"]:
        for cls in layer["classes"]:
            palette[int(cls["id"])] = tuple(int(v) for v in cls["color"])
    return palette


def build_layer_specs(cfg: dict) -> list[tuple[str, dict[int, str]]]:
    """Return [(layer_name, {class_id: label}), ...] in config order."""
    result = []
    for layer_cfg in cfg["layers"]:
        classes = {int(c["id"]): c["label"] for c in layer_cfg["classes"]}
        result.append((layer_cfg["name"], classes))
    return result
