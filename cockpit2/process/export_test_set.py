r"""
export_test_set.py
Copy the 375 TEST images listed in model_v2/split.csv into a separate folder,
with a labels file containing only those images. Read-only on dataset_v2.

  python process\export_test_set.py --data_dir dataset_v2 --out dataset_v2_test

Output:
  dataset_v2_test/<phase>/<image>.png   (75 per phase)
  dataset_v2_test/labels_test.csv       (same columns as labels.csv, test rows only)
  dataset_v2_test/README.txt            (where the split came from)
"""
import argparse
import os
import shutil
import sys

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True, help="dataset_v2 folder (labels.csv + <phase>/ folders)")
ap.add_argument("--split_csv", default=None, help="default: <data_dir>/model_v2/split.csv")
ap.add_argument("--out", required=True, help="new folder to create, e.g. dataset_v2_test")
args = ap.parse_args()

split_csv = args.split_csv or os.path.join(args.data_dir, "model_v2", "split.csv")
if os.path.abspath(args.out).startswith(os.path.abspath(args.data_dir) + os.sep) or \
        os.path.abspath(args.out) == os.path.abspath(args.data_dir):
    sys.exit("--out must be outside the dataset folder (never write into dataset_v2).")
if os.path.exists(args.out) and os.listdir(args.out):
    sys.exit(f"{args.out} already exists and is not empty -- choose a new folder.")

split = pd.read_csv(split_csv)
test = split[split["split"] == "test"]
labels = pd.read_csv(os.path.join(args.data_dir, "labels.csv")).drop_duplicates(["phase", "image"])
test_labels = labels.merge(test[["image", "phase"]], on=["image", "phase"], how="inner")

if len(test_labels) != len(test):
    sys.exit(f"Only {len(test_labels)} of {len(test)} test images found in labels.csv -- "
             f"is this the same dataset the split was made from?")

missing = []
for _, r in test_labels.iterrows():
    src = os.path.join(args.data_dir, r["phase"], r["image"])
    if not os.path.exists(src):
        missing.append(src)
        continue
    dst_dir = os.path.join(args.out, r["phase"])
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, r["image"]))

if missing:
    sys.exit(f"{len(missing)} test images missing, e.g. {missing[0]}")

test_labels.to_csv(os.path.join(args.out, "labels_test.csv"), index=False)
with open(os.path.join(args.out, "README.txt"), "w") as f:
    f.write("Cockpit v2 TEST split (split_seed 42, stratified 70/15/15 by phase).\n"
            f"Source split: {os.path.abspath(split_csv)}\n"
            "Same 375 images every run 1-6 was evaluated on. These images were never used for\n"
            "training or early stopping, but runs 3-6 were designed after looking at errors on\n"
            "them, so they are not a fully unseen hold-out set.\n")
print("copied per phase:", test_labels["phase"].value_counts().to_dict())
print(f"total {len(test_labels)} -> {os.path.abspath(args.out)}")
