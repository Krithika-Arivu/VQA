r"""
train_v2.py
Train + evaluate CockpitNetV2 on cockpit dataset v2 (capture_synthetic_v2.py).

  python train_v2.py --data_dir "C:\...\vqa-demo\cockpit\dataset_v2"
  python train_v2.py --data_dir ... --smoke        # 2-minute pipeline check on a few images

Outputs (in <data_dir>/model_v2/):
  best_model.pt        weights + norm stats + config (everything inference needs)
  split.csv            which image went to train / val / test (reproducible)
  test_metrics.json    per-gauge MAE / within-tolerance %, phase accuracy, BASELINES
  test_predictions.csv per-image predictions vs labels (for error analysis)
  phase_confusion.csv  learned phase head confusion matrix (test set)

Why the evaluation looks like this (lessons from v1):
  * baselines are reported next to every metric (v1 never showed that e.g. 58% of
    roll labels were one bin, so "92.8%" had no reference point)
  * errors are in real units (MAE) + % within an aviation-meaningful tolerance,
    not only bin accuracy (v1's pitch-vs-roll gap was mostly bin width)
  * the split is stratified by phase, frames are i.i.d. random states, and the
    test set is never used for model selection (early stopping uses val)
"""

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gui"))
from cockpit_model_v2 import (PHASES, SCALARS, UNITS, CockpitNetV2, decode,  # noqa: E402
                              eval_transform, train_transform, IMG_H, IMG_W, alt_fine_target,
                              ALT_SHORT_PERIOD)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True, help="dataset_v2 folder (contains labels.csv and <phase>/ folders)")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--patience", type=int, default=6, help="early stopping on val loss")
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--phase_weight", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=42, help="weights init / data order seed")
ap.add_argument("--split_seed", type=int, default=42,
                help="train/val/test split seed (keep 42 to compare runs on the SAME test images)")
ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest on Windows)")
ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline end-to-end")
ap.add_argument("--alt_fine", type=int, default=1, choices=[0, 1],
                help="1 = extra altimeter long-hand angle head (run 2); 0 = regression only (run 1)")
ap.add_argument("--alt_short", type=int, default=1, choices=[0, 1],
                help="1 = extra altimeter short-hand angle head (run 3); 0 = off (runs 1-2)")
args = ap.parse_args()

# ---------------------------------------------------------------- reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = os.path.join(args.data_dir, "model_v2")
os.makedirs(OUT, exist_ok=True)
print(f"device: {DEVICE} | input {IMG_W}x{IMG_H} | seed {args.seed} | split_seed {args.split_seed} | alt_fine {args.alt_fine} | alt_short {args.alt_short}")

# ---------------------------------------------------------------- data + split
df = pd.read_csv(os.path.join(args.data_dir, "labels.csv"))
df["path"] = [os.path.join(args.data_dir, p, im) for p, im in zip(df["phase"], df["image"])]
missing = ~df["path"].map(os.path.exists)
if missing.any():
    print(f"WARNING: {missing.sum()} label rows have no image file -- dropped")
    df = df[~missing].reset_index(drop=True)
df = df.drop_duplicates("path").reset_index(drop=True)
print("frames per phase:", df["phase"].value_counts().to_dict())

if args.smoke:
    # groupby().sample keeps the "phase" column (groupby().apply drops it in pandas >= 3.0)
    df = df.groupby("phase").sample(n=12, random_state=0).reset_index(drop=True)
    args.epochs, args.patience = 2, 2

# stratified 70/15/15 split by phase (frames are independent random states)
rng = np.random.RandomState(args.split_seed)
split = pd.Series("train", index=df.index)
for _, g in df.groupby("phase"):
    idx = g.index.to_numpy().copy()
    rng.shuffle(idx)
    n = len(idx)
    n_val, n_test = int(round(0.15 * n)), int(round(0.15 * n))
    split[idx[:n_val]] = "val"
    split[idx[n_val:n_val + n_test]] = "test"
df["split"] = split
df[["image", "phase", "split"]].to_csv(os.path.join(OUT, "split.csv"), index=False)
tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

norm = {k: {"mean": float(tr[k].mean()), "std": float(tr[k].std() or 1.0)} for k in SCALARS}


class PanelDataset(Dataset):
    def __init__(self, frame, tf):
        self.f, self.tf = frame, tf

    def __len__(self):
        return len(self.f)

    def __getitem__(self, i):
        r = self.f.iloc[i]
        x = self.tf(Image.open(r["path"]).convert("RGB"))
        reg = torch.tensor([(r[k] - norm[k]["mean"]) / norm[k]["std"] for k in SCALARS], dtype=torch.float32)
        h = math.radians(r["heading"])
        return x, reg, torch.tensor([math.sin(h), math.cos(h)], dtype=torch.float32), \
            torch.tensor(PHASES.index(r["phase"]), dtype=torch.long), \
            torch.tensor(alt_fine_target(r["altitude"]), dtype=torch.float32), \
            torch.tensor(alt_fine_target(r["altitude"], ALT_SHORT_PERIOD), dtype=torch.float32)


g = torch.Generator()
g.manual_seed(args.seed)
dl_tr = DataLoader(PanelDataset(tr, train_transform()), batch_size=args.batch_size, shuffle=True,
                   num_workers=args.workers, generator=g)
dl_va = DataLoader(PanelDataset(va, eval_transform()), batch_size=args.batch_size, num_workers=args.workers)
dl_te = DataLoader(PanelDataset(te, eval_transform()), batch_size=args.batch_size, num_workers=args.workers)

# ---------------------------------------------------------------- model
model = CockpitNetV2(pretrained=True, alt_fine=bool(args.alt_fine), alt_short=bool(args.alt_short)).to(DEVICE)
huber = nn.SmoothL1Loss()
ce = nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)


def run(loader, train):
    model.train(train)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, reg, hd, ph, af, ash in loader:
            x, reg, hd, ph = x.to(DEVICE), reg.to(DEVICE), hd.to(DEVICE), ph.to(DEVICE)
            af, ash = af.to(DEVICE), ash.to(DEVICE)
            out = model(x)
            loss = huber(out["reg"], reg) + huber(out["heading"], hd) + args.phase_weight * ce(out["phase"], ph)
            if "alt_fine" in out:
                loss = loss + huber(out["alt_fine"], af)
            if "alt_short" in out:
                loss = loss + huber(out["alt_short"], ash)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * x.size(0)
            n += x.size(0)
    return tot / max(n, 1)


best, bad_epochs, ckpt = float("inf"), 0, os.path.join(OUT, "best_model.pt")
for ep in range(1, args.epochs + 1):
    tl, vl = run(dl_tr, True), run(dl_va, False)
    sched.step(vl)
    flag = ""
    if vl < best:
        best, bad_epochs, flag = vl, 0, "  <- best (saved)"
        torch.save({"model_state_dict": model.state_dict(), "norm": norm, "epoch": ep, "val_loss": vl,
                    "img_size": [IMG_H, IMG_W], "phases": PHASES, "scalars": SCALARS,
                    "dataset": "v2", "seed": args.seed, "split_seed": args.split_seed, "alt_fine": bool(args.alt_fine),
                    "alt_short": bool(args.alt_short)}, ckpt)
    else:
        bad_epochs += 1
    print(f"epoch {ep:>2}  train {tl:.4f}  val {vl:.4f}  lr {opt.param_groups[0]['lr']:.1e}{flag}")
    if bad_epochs >= args.patience:
        print(f"early stopping: val loss did not improve for {args.patience} epochs")
        break

# ---------------------------------------------------------------- test evaluation
model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["model_state_dict"])
model.eval()
preds = []
with torch.no_grad():
    for x, *_ in dl_te:
        preds += decode(model(x.to(DEVICE)), norm)

TOL = {"pitch": 1.0, "roll": 2.0, "airspeed": 5.0, "altitude": 100.0, "vs": 100.0,
       "turn": 0.2, "rpm": 100.0, "heading": 5.0, "altitude_coarse": 100.0, "altitude_shorthand": 100.0}
UNITS = dict(UNITS, altitude_coarse="ft", altitude_shorthand="ft")
# diagnostics: altitude_coarse = plain regression (run 1's method);
#              altitude_shorthand = coarse corrected by the short hand only (before the long hand)
GAUGES = SCALARS + ["heading"] + (["altitude_coarse"] if (args.alt_fine or args.alt_short) else []) \
    + (["altitude_shorthand"] if args.alt_short else [])
TRUE_COL = {k: k for k in GAUGES}
TRUE_COL["altitude_coarse"] = TRUE_COL["altitude_shorthand"] = "altitude"


def err(k, p, t):
    if k == "heading":
        d = abs(p - t) % 360
        return min(d, 360 - d)
    return abs(p - t)


rows, metrics = [], {"n_test": len(te), "per_gauge": {}, "note": "baseline = always predict the TRAIN mean "
                     "(heading: train circular mean); phase baseline = most common train phase"}
for k in GAUGES:
    e = np.array([err(k, p[k], t) for p, t in zip(preds, te[TRUE_COL[k]])])
    if k == "heading":
        mean_h = math.degrees(math.atan2(np.sin(np.radians(tr[k])).mean(), np.cos(np.radians(tr[k])).mean())) % 360
        base = np.array([err(k, mean_h, t) for t in te[k]])
    else:
        base = np.array([err(k, tr[TRUE_COL[k]].mean(), t) for t in te[TRUE_COL[k]]])
    metrics["per_gauge"][k] = {"unit": UNITS[k], "MAE": round(float(e.mean()), 3),
                               f"within_{TOL[k]:g}": round(float((e <= TOL[k]).mean() * 100), 1),
                               "baseline_MAE": round(float(base.mean()), 3),
                               f"baseline_within_{TOL[k]:g}": round(float((base <= TOL[k]).mean() * 100), 1)}

true_ph, pred_ph = list(te["phase"]), [p["phase"] for p in preds]
acc = float(np.mean([a == b for a, b in zip(true_ph, pred_ph)]))
maj = tr["phase"].value_counts().idxmax()
metrics["phase"] = {"accuracy": round(acc * 100, 1),
                    "baseline_majority_accuracy": round(float((te["phase"] == maj).mean() * 100), 1)}
cm = pd.crosstab(pd.Series(true_ph, name="true"), pd.Series(pred_ph, name="pred")).reindex(
    index=PHASES, columns=PHASES, fill_value=0)
cm.to_csv(os.path.join(OUT, "phase_confusion.csv"))

for p, (_, t) in zip(preds, te.iterrows()):
    row = {"image": t["image"], "true_phase": t["phase"], "pred_phase": p["phase"],
           "phase_confidence": round(p["phase_confidence"], 3)}
    for k in GAUGES:
        row[f"true_{k}"], row[f"pred_{k}"] = t[TRUE_COL[k]], round(p[k], 2)
    rows.append(row)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)
with open(os.path.join(OUT, "test_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n================ TEST SET (never used for training or model selection) ================")
print(f"{'gauge':<9} {'MAE':>9} {'within tol':>12} | {'baseline MAE':>12} {'baseline within':>16}")
for k, m in metrics["per_gauge"].items():
    wk = [c for c in m if c.startswith("within_")][0]
    bk = [c for c in m if c.startswith("baseline_within_")][0]
    print(f"{k:<9} {m['MAE']:>7.2f} {m['unit']:<6}{m[wk]:>5.1f}% (±{wk[7:]}) | "
          f"{m['baseline_MAE']:>12.2f} {m[bk]:>15.1f}%")
print(f"phase     accuracy {metrics['phase']['accuracy']}%  "
      f"(baseline: always '{maj}' = {metrics['phase']['baseline_majority_accuracy']}%)")
print(cm.to_string())
print(f"\nsaved: {OUT}")