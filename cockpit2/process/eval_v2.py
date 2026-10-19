r"""
eval_v2.py
Evaluate ANY cockpit v2 checkpoint (runs 1-6) without training. Read-only on the dataset.

  # run 3 on the validation set (to compare with run 6 under the decision rule)
  python -u process\eval_v2.py --ckpt "dataset_v2\model_v2-perfect run3\best_model.pt" ^
      --data_dir dataset_v2 --split val --split_csv "dataset_v2\model_v2-perfect run3\split.csv" ^
      --out results\run3_val

  # run 6 on val + the decision rule against run 3
  python -u process\eval_v2.py --ckpt dataset_v2\model_v2\best_model.pt --data_dir dataset_v2 ^
      --split val --out results\run6_val --compare_to results\run3_val\metrics.json

  # FINAL model on the fresh hold-out set (once): every frame is used
  python -u process\eval_v2.py --ckpt <final>\best_model.pt --data_dir dataset_holdout ^
      --split all --out results\final_holdout --train_split_csv dataset_v2\model_v2\split.csv ^
      --train_labels dataset_v2\labels.csv

Outputs in --out: metrics.json, predictions.csv, per_phase.csv (shortcut check),
decision.txt (only with --compare_to).
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gui"))
from cockpit_model_v2 import SCALARS, decode, eval_transform, load_checkpoint  # noqa: E402
from metrics_v2 import circular_mean_deg, decide, evaluate, per_phase, print_block  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data_dir", required=True, help="folder with labels.csv and <phase>/ image folders")
ap.add_argument("--split", choices=["val", "test", "train", "all"], required=True,
                help="'all' = every frame (use for the hold-out set)")
ap.add_argument("--split_csv", default=None, help="default: <data_dir>/model_v2/split.csv")
ap.add_argument("--train_split_csv", default=None,
                help="for --split all on another dataset: split.csv of the TRAINING dataset (for baselines)")
ap.add_argument("--train_labels", default=None, help="labels.csv of the training dataset (with --train_split_csv)")
ap.add_argument("--out", required=True, help="new folder for the results")
ap.add_argument("--compare_to", default=None, help="metrics.json of the reference model on the SAME images")
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--workers", type=int, default=0)
args = ap.parse_args()

if os.path.exists(args.out) and os.listdir(args.out):
    sys.exit(f"{args.out} already exists and is not empty -- choose a new folder.")
data_abs = os.path.abspath(args.data_dir)
if os.path.abspath(args.out) == data_abs or os.path.abspath(args.out).startswith(data_abs + os.sep):
    sys.exit("--out must be outside the dataset folder.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, ckpt = load_checkpoint(args.ckpt, DEVICE)
norm = ckpt["norm"]
print(f"device {DEVICE} | ckpt {args.ckpt} | epoch {ckpt.get('epoch')} | seed {ckpt.get('seed')} | "
      f"needles {ckpt.get('needles') or ckpt.get('rpm_fine') and ['rpm'] or 'none'} | "
      f"needle_use {ckpt.get('needle_use', 'n/a (all heads used)')}")


def load_labels(d):
    df = pd.read_csv(os.path.join(d, "labels.csv"))
    df["path"] = [os.path.join(d, p, im) for p, im in zip(df["phase"], df["image"])]
    miss = ~df["path"].map(os.path.exists)
    if miss.any():
        print(f"WARNING: {miss.sum()} label rows have no image file -- dropped")
    return df[~miss].drop_duplicates("path").reset_index(drop=True)


df = load_labels(args.data_dir)
train_rows = None
if args.split != "all":
    sp = pd.read_csv(args.split_csv or os.path.join(args.data_dir, "model_v2", "split.csv"))
    df = df.merge(sp[["image", "phase", "split"]], on=["image", "phase"], how="inner")
    train_rows = df[df["split"] == "train"]
    df = df[df["split"] == args.split].reset_index(drop=True)
elif args.train_split_csv and args.train_labels:
    tl = pd.read_csv(args.train_labels).drop_duplicates(["image", "phase"])
    sp = pd.read_csv(args.train_split_csv)
    train_rows = tl.merge(sp[sp["split"] == "train"][["image", "phase"]], on=["image", "phase"])
if df.empty:
    sys.exit("No images selected -- check --split / --split_csv.")
print(f"{args.split}: {len(df)} images | per phase {df['phase'].value_counts().to_dict()}")

# baselines: the TRAIN means (stored in the checkpoint's norm stats = exactly the train mean)
base_mean = {k: norm[k]["mean"] for k in SCALARS}
if train_rows is not None and len(train_rows):
    base_heading = circular_mean_deg(train_rows["heading"])
    base_phase = train_rows["phase"].value_counts().idxmax()
else:
    base_heading, base_phase = 0.0, df["phase"].value_counts().idxmax()
    print("NOTE: no training split given -> heading baseline = constant 0 deg, "
          "phase baseline = most common phase in THIS set")


class Images(Dataset):
    def __init__(self, paths):
        self.p, self.tf = list(paths), eval_transform()

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB"))


preds = []
with torch.no_grad():
    for x in DataLoader(Images(df["path"]), batch_size=args.batch_size, num_workers=args.workers):
        preds += decode(model(x.to(DEVICE)), norm)

metrics, rows = evaluate(preds, df, base_mean, base_heading, base_phase)
metrics.update({"ckpt": os.path.abspath(args.ckpt), "split": args.split,
                "needle_use": ckpt.get("needle_use")})
table = per_phase(rows)
os.makedirs(args.out, exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(args.out, "predictions.csv"), index=False)
table.to_csv(os.path.join(args.out, "per_phase.csv"), index=False)
with open(os.path.join(args.out, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print_block(f"{args.split.upper()} ({len(df)} images)", metrics, table)

if args.compare_to:
    ref = json.load(open(args.compare_to))
    if ref.get("n") != metrics["n"] or ref.get("split") != metrics["split"]:
        sys.exit(f"--compare_to was computed on a different set (n={ref.get('n')}, split={ref.get('split')}) "
                 f"-- compare on the same images only.")
    ok, lines = decide(metrics, ref)
    text = "\n".join([f"new: {args.ckpt}", f"ref: {ref.get('ckpt')}", f"set: {args.split} ({metrics['n']})", ""]
                     + lines + ["", f"VERDICT: {'PASS -> new model is the final model' if ok else 'FAIL -> keep the reference model'}"])
    open(os.path.join(args.out, "decision.txt"), "w").write(text + "\n")
    print("\n================ DECISION RULE (fixed 2026-10-19, before run 6) ================")
    print(text)
print(f"\nsaved: {os.path.abspath(args.out)}")