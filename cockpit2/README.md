# cockpit2: Visual Co-Pilot cockpit module (dataset v2, CockpitNetV2)

This module reads a Cessna 172 instrument panel from a FlightGear screenshot and answers
questions about it: altitude, airspeed, heading, bank, pitch/climb, vertical speed,
engine RPM and flight phase.

It replaces the v1 cockpit module (`cockpit/`), which is kept unchanged for comparison.
The v1 images were captured with `--fdm=null`, where FlightGear's instruments don't
move, so the gauges didn't show the labelled values. v2 recaptures every frame with
the gauges driven to the label values, and each value is read back before and after
the screenshot.

```
cockpit2/
├── capture/
│   ├── capture_synthetic_v2.py   # dataset capture from FlightGear (v2.2, includes the VSI-lag fix)
│   └── vsi_probe.py              # diagnostic that found the VSI needle lag (why climb/level_flight were recaptured)
├── gui/
│   ├── cockpit_model_v2.py       # model + transforms, shared by training AND inference
│   └── copilot_core_v2.py        # inference + question answering (used by ../integrated.py)
├── process/
│   ├── check_dataset.py          # read-only dataset check (run before training)
│   ├── train_v2.py               # training + validation/test evaluation
│   ├── metrics_v2.py             # shared metrics, per-phase "shortcut" check, model decision rule
│   ├── eval_v2.py                # evaluate any checkpoint on val/test/hold-out; --compare_to applies the rule
│   ├── export_test_set.py        # copies the 375 test images into a separate folder (for sharing)
│   └── check_inference_v2.py     # checks the app path (auto-crop + model) against the labels
├── dataset_v2/                   # only small files are in git (see "What is not in git")
│   ├── reference_panel.png       # needed at runtime: "is this the training view?" check
│   ├── view_lock.json            # screen size + FlightGear camera view used for every capture
│   └── model_v2/                 # results of the final run (run 6)
│       ├── val_metrics.json      # validation metrics (the set the final model was chosen on)
│       ├── val_predictions.csv, val_per_phase.csv
│       ├── test_metrics.json     # per-gauge MAE, % within tolerance, baselines, phase accuracy
│       ├── test_predictions.csv, test_per_phase.csv
│       ├── phase_confusion.csv
│       └── split.csv             # which image was train / val / test
├── results/
│   ├── run3_val/                 # run 3 re-evaluated on the validation set (eval_v2.py)
│   └── run6_val/                 # run 6 on the validation set + decision.txt (PASS)
├── train_log*.txt                # console logs of training runs 1-6
├── requirements.txt
└── README.md
```

## What is not in git
| Item | Size | Where it lives |
|---|---|---|
| Dataset images (2,500 PNG crops, 936×729) + `labels.csv` | large | kept locally / shared by Drive |
| `dataset_v2/model_v2/best_model.pt` | ~45 MB | kept locally / shared separately (GitHub Release or Drive) |
| capture self-test sheets, `_fullcheck` screenshots, `vsi_probe/` images, backups | – | local only |

To run the app you need `best_model.pt` in `cockpit2/dataset_v2/model_v2/`.
To retrain or re-check the data you also need the full `dataset_v2/`.

## Setup (Windows, once)
```
cd cockpit2
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```
Install PyTorch **first**, using the command for your CUDA version from
https://pytorch.org/get-started/locally/ (CPU-only works too, but training is slow). Then:
```
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

## Run order (from the `cockpit2/` folder)
```
python process/check_dataset.py --data_dir dataset_v2            # must end with "OK"
python process/train_v2.py --data_dir dataset_v2 --smoke         # ~2 min pipeline check
python process/train_v2.py --data_dir dataset_v2 --needles rpm,airspeed,vs   # full training, final (run 6) config
python process/check_inference_v2.py --data_dir dataset_v2       # app path == training path?
python process/eval_v2.py --ckpt dataset_v2/model_v2/best_model.pt --data_dir dataset_v2 --split val --out results/my_val
```
Use `python -u` when piping output to a log file (e.g. `Tee-Object` in PowerShell), otherwise
nothing is printed until the end.

Without `--needles`, `train_v2.py` trains the run 3 configuration.
Checkpoints from every run (1-6) load with the current `gui/cockpit_model_v2.py`.

Training writes to `dataset_v2/model_v2/`. The split is fixed by `--split_seed 42`, so every
run uses the same 1750 / 375 / 375 train / val / test images. `--seed` changes only weight
init and data order.

The app is started from the repo root with `python integrated.py`. Cockpit v2 is the default;
see `python integrated.py --help`.

## Capturing (only needed to rebuild the dataset)
FlightGear launch options:
`--aircraft=c172p --airport=KSJC --fdm=null --heading=180 --timeofday=noon --httpd=5500`,
full screen, default cockpit view. The script was written for a 2880×1800 screen, and
`view_lock.json` makes it refuse to run on a different size.
```
python capture/capture_synthetic_v2.py --selftest          # check every gauge first
python capture/capture_synthetic_v2.py --phase engine_off  # repeat for each phase, 500 frames each
```
How it works:
- The gauge display properties are written every frame by a Nasal loop. The airspeed
  needle reads `/fdm/jsbsim/velocities/vias-kts`.
- The outside view is randomised separately, so the phase can't be learned from the window.
- Each frame is rejected if a read-back value is off or if the panel doesn't match
  `reference_panel.png`.

## Results (final model = run 6)
The final model was chosen on the **validation** set (375 frames) by a rule fixed before
run 6 was trained: every gauge and phase within 1 point of run 3's "% within tolerance", and
RPM ≥ 95%. Run 6 passed (`results/run6_val/decision.txt`).

Test set (375 frames = 75 per phase). Source: `dataset_v2/model_v2/test_metrics.json`.
The baseline is "always predict the training mean".

| Gauge (tolerance) | MAE | within tolerance | baseline within |
|---|---|---|---|
| pitch (±1°) | 0.17° | 99.7% | 19.5% |
| roll (±2°) | 0.42° | 98.9% | 74.7% |
| airspeed (±5 kt) | 1.02 kt | 98.7% | 2.4% |
| altitude (±100 ft) | 3.06 ft | 100% | 0% |
| vertical speed (±100 fpm) | 9.87 fpm | 100% | 7.5% |
| turn rate (±0.2) | 0.02 | 100% | 65.6% |
| RPM (±100) | 12.7 | 100% | 0% |
| heading (±5°) | 0.94° | 100% | 2.1% |
| phase | – | 100% accuracy | 20% (majority class) |

Validation, run 3 → run 6 (% within tolerance): RPM 87.7 → 99.7, airspeed 94.4 → 99.2,
pitch 100 → 99.5, roll 99.7 → 98.9; altitude, vertical speed, turn, heading and phase 100 in both.

The app check (`check_inference_v2.py`, 25 test crops through `copilot_core_v2`) gives
RPM MAE 12.4 and airspeed 1.07 kt, matching training.

Run history (test set, % within tolerance):
- **Run 1** used regression only. Altitude was 42.1% within ±100 ft, because the altimeter
  hands weren't read precisely.
- **Run 2** added an angle head for the altimeter's long hand (98.1%).
- **Run 3** added an angle head for the short hand. Altitude reached 100%, but RPM dropped
  to 86.7%.
- **Run 4** repeated run 3 with seed 7: RPM 83.2%, so the drop is real, not seed noise.
  Inside each phase the model output the phase's typical RPM instead of reading the needle
  (level_flight: prediction spread 34 rpm vs true 99 rpm).
- **Run 5** gave the tachometer its own needle-angle head: RPM 100%, but airspeed showed the
  same shortcut and fell to 85.1%.
- **Run 6** gave RPM, airspeed and vertical speed needle-angle heads. On validation, the
  needle reading was kept for RPM and airspeed; for vertical speed the plain regression
  was slightly better (9.15 vs 9.64 fpm MAE, both 100%), so the **needle heads are used
  for RPM and airspeed**. A dedicated needle-angle output fixed the shortcut for both.

## Limitations (please read before quoting the numbers)
- **In-distribution only.** The test images come from the same simulator, aircraft, camera
  view, screen resolution and crop as training. Real cockpits, other views and resolutions,
  and compressed images are untested.
- **Phase is 100% by construction.** The five phases have non-overlapping gauge ranges, so
  this shows correct gauge reading, not an understanding of real flight.
- **Slow airspeeds are less precise.** The C172 airspeed indicator has no markings below
  about 40 kt, so readings at taxi speeds are less precise.
- **Development test set.** Runs 3–6 were designed after looking at errors on the 375 test
  images, so they are no longer fully unseen; the final model was chosen on validation
  instead. A fresh hold-out set (new captures, evaluated once) will give the final figure.
- **Results come from one training seed per run** (run 3 was also repeated with seed 7).
  Differences of about one point between runs may be noise.
- **Small costs of run 6:** on validation, pitch has 2 frames out of tolerance (run 3: 0) and
  roll 4 (run 3: 1), mostly in climb and level flight.
- **Not comparable to v1's bin accuracies.** Those were measured on labels the images didn't show.