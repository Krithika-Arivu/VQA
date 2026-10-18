r"""
check_dataset.py -- read-only sanity check of dataset_v2 before training.
(Nothing is modified.) Run from the cockpit2 folder:

    python process/check_dataset.py --data_dir dataset_v2

Checks: labels.csv columns, 500 rows per phase, every row has its image, no image
without a row, no stray files (e.g. "... - Copy.png"), image size, value ranges.
"""
import argparse
import os

import pandas as pd
from PIL import Image

PHASES = ["engine_off", "taxi", "takeoff_roll", "climb", "level_flight"]
NEED = ["image", "phase", "pitch", "roll", "airspeed", "altitude", "heading", "vs", "turn", "rpm"]
# (min, max) per phase from the capture script's sampling ranges (+ small margin)
RANGES = {
    "engine_off":   {"airspeed": (0, 0), "vs": (0, 0), "rpm": (0, 0)},
    "taxi":         {"airspeed": (0, 25), "vs": (0, 0), "rpm": (800, 1400)},
    "takeoff_roll": {"airspeed": (25, 65), "vs": (0, 0), "rpm": (2250, 2450)},
    "climb":        {"airspeed": (65, 95), "vs": (300, 1000), "rpm": (2250, 2450)},
    "level_flight": {"airspeed": (85, 125), "vs": (-150, 150), "rpm": (2100, 2450)},
}

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
args = ap.parse_args()

df = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
problems = []
miss_cols = [c for c in NEED if c not in df.columns]
if miss_cols:
    raise SystemExit(f"labels.csv is missing columns: {miss_cols}")

print("rows per phase:", df["phase"].value_counts().reindex(PHASES, fill_value=0).to_dict())
for ph in PHASES:
    n = int((df["phase"] == ph).sum())
    if n != 500:
        problems.append(f"{ph}: {n} rows (expected 500)")
if df.duplicated(["phase", "image"]).any():
    problems.append(f"{int(df.duplicated(['phase', 'image']).sum())} duplicate label rows")

sizes = set()
for ph in PHASES:
    folder = os.path.join(args.data_dir, ph)
    if not os.path.isdir(folder):
        problems.append(f"folder missing: {folder}")
        continue
    on_disk = {f for f in os.listdir(folder) if f.lower().endswith(".png")}
    in_csv = set(df.loc[df["phase"] == ph, "image"])
    if in_csv - on_disk:
        problems.append(f"{ph}: {len(in_csv - on_disk)} label rows without an image, e.g. {sorted(in_csv - on_disk)[:3]}")
    extra = on_disk - in_csv
    if extra:
        problems.append(f"{ph}: {len(extra)} PNGs not in labels.csv (ignored by training, but remove them), "
                        f"e.g. {sorted(extra)[:3]}")
    for f in sorted(in_csv & on_disk)[:5]:
        sizes.add(Image.open(os.path.join(folder, f)).size)
    for k, (lo, hi) in RANGES[ph].items():
        v = df.loc[df["phase"] == ph, k]
        if v.min() < lo - 1e-6 or v.max() > hi + 1e-6:
            problems.append(f"{ph}: {k} outside [{lo}, {hi}] (min {v.min()}, max {v.max()})")

print("image sizes seen:", sizes)
if len(sizes) != 1:
    problems.append(f"expected one image size, got {sizes}")

if problems:
    print("\nPROBLEMS:")
    for p in problems:
        print("  -", p)
    raise SystemExit(1)
print("\nOK: dataset looks consistent -- ready for training.")
