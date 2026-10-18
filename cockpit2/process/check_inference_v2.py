r"""
check_inference_v2.py -- end-to-end check of the INTEGRATION path (copilot_core_v2),
without the Gradio UI and without the Qwen router. Run from the cockpit2 folder:

    python process/check_inference_v2.py --data_dir dataset_v2

Part A: 25 test-set crops through CopilotSystemV2.run_perception -> must match the
        labels about as well as train_v2's test metrics (proves app == training).
Part B: every <phase>/_fullcheck_XXXX.jpg (half-size full screenshots saved during
        capture) -> auto-crop -> read -> compared with label row <phase>_XXXX.png.
        This tests the auto-crop on real full screenshots (note: these are
        half-resolution JPEGs, so expect slightly larger errors than Part A).
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gui"))
from copilot_core_v2 import CopilotSystemV2  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--model_dir", default=None, help="default: <data_dir>/model_v2")
ap.add_argument("--n", type=int, default=25)
args = ap.parse_args()

model_dir = args.model_dir or os.path.join(args.data_dir, "model_v2")
sysv2 = CopilotSystemV2(model_dir, reference_panel=os.path.join(args.data_dir, "reference_panel.png"),
                        use_llm=False)
print("loaded:", sysv2.model_info)
labels = pd.read_csv(os.path.join(args.data_dir, "labels.csv")).set_index("image")
G = ["pitch", "roll", "airspeed", "altitude", "vs", "turn", "rpm", "heading"]


def errors(pred, true):
    out = {}
    for k in G:
        e = abs(pred[k] - true[k])
        out[k] = min(e % 360, 360 - e % 360) if k == "heading" else e
    out["phase_ok"] = pred["phase"] == true["phase"]
    return out


def report(name, rows):
    if not rows:
        print(f"\n{name}: no images found")
        return
    df = pd.DataFrame(rows)
    print(f"\n{name}: {len(df)} images")
    print("  MAE: " + "  ".join(f"{k} {df[k].mean():.2f}" for k in G))
    print(f"  phase correct: {df['phase_ok'].mean() * 100:.0f}%   "
          f"panel match min {df['sim'].min():.2f}   crop modes {df['mode'].value_counts().to_dict()}")


split = pd.read_csv(os.path.join(model_dir, "split.csv"))
test = split[split.split == "test"].groupby("phase", group_keys=False).head(max(1, args.n // 5))
rows = []
for _, r in test.iterrows():
    pred, info = sysv2.run_perception(os.path.join(args.data_dir, r["phase"], r["image"]))
    rows.append({**errors(pred, labels.loc[r["image"]]), "sim": info["panel_similarity"] or 0,
                 "mode": info["crop_mode"]})
report("A) test-set crops via copilot_core_v2", rows)

rows = []
for path in sorted(glob.glob(os.path.join(args.data_dir, "*", "_fullcheck_*.jpg"))):
    phase = os.path.basename(os.path.dirname(path))
    idx = os.path.basename(path)[-8:-4]
    name = f"{phase}_{idx}.png"
    if name not in labels.index:
        continue
    pred, info = sysv2.run_perception(path)
    e = errors(pred, labels.loc[name])
    rows.append({**e, "sim": info["panel_similarity"] or 0, "mode": info["crop_mode"]})
    print(f"  {name:24s} alt {pred['altitude']:7.0f} (true {labels.loc[name, 'altitude']:7.0f})  "
          f"ias {pred['airspeed']:5.1f} ({labels.loc[name, 'airspeed']:5.1f})  "
          f"hdg {pred['heading']:5.1f} ({labels.loc[name, 'heading']:5.1f})  "
          f"phase {pred['phase']} {'OK' if e['phase_ok'] else 'WRONG'}  match {(info['panel_similarity'] or 0):.2f}")
report("B) _fullcheck screenshots (auto-crop, half-res JPEG)", rows)
print("\nNote: _fullcheck frames of climb/level_flight from the FIRST (VSI-lag) capture were moved to the "
      "backup, so only frames from the current dataset are compared.")
