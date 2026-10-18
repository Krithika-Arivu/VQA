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
│   ├── train_v2.py               # training + test-set evaluation
│   └── check_inference_v2.py     # checks the app path (auto-crop + model) against the labels
├── dataset_v2/                   # only small files are in git (see "What is not in git")
│   ├── reference_panel.png       # needed at runtime: "is this the training view?" check
│   ├── view_lock.json            # screen size + FlightGear camera view used for every capture
│   └── model_v2/                 # results of the final run (run 3)
│       ├── test_metrics.json     # per-gauge MAE, % within tolerance, baselines, phase accuracy
│       ├── test_predictions.csv  # per-image prediction vs label (test set)
│       ├── phase_confusion.csv
│       └── split.csv             # which image was train / val / test
├── train_log.txt, train_log_run2.txt, train_log_run3.txt   # console logs of the three training runs
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
python process/train_v2.py --data_dir dataset_v2                 # full training (run 3 config is the default)
python process/check_inference_v2.py --data_dir dataset_v2       # app path == training path?
```
Training writes to `dataset_v2/model_v2/`. The split is fixed by `--split_seed 42`, so every
run is tested on the same 375 images. `--seed` changes only weight init and data order.

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

## Results (test set, 375 frames = 75 per phase; final model = run 3)
Source: `dataset_v2/model_v2/test_metrics.json`. The baseline is "always predict the training mean".

| Gauge (tolerance) | MAE | within tolerance | baseline within |
|---|---|---|---|
| pitch (±1°) | 0.21° | 100% | 19.5% |
| roll (±2°) | 0.39° | 99.7% | 74.7% |
| airspeed (±5 kt) | 1.75 kt | 96.3% | 2.4% |
| altitude (±100 ft) | 3.0 ft | 100% | 0% |
| vertical speed (±100 fpm) | 9.2 fpm | 100% | 7.5% |
| turn rate (±0.2) | 0.023 | 100% | 65.6% |
| RPM (±100) | 57.2 | 86.7% | 0% |
| heading (±5°) | 1.14° | 100% | 2.1% |
| phase | – | 100% accuracy | 20% (majority class) |

Run history:
- **Run 1** used regression only. Altitude was 42.1% within ±100 ft, because the altimeter
  hands weren't read precisely.
- **Run 2** added a long-hand angle head.
- **Run 3** added a short-hand angle head. Altitude reached 100% with no 1,000-ft misreads,
  but RPM dropped from 96.3% to 86.7% within ±100. The cause is not established yet: it
  could be loss balancing or seed-to-seed noise.

## Limitations (please read before quoting the numbers)
- **In-distribution only.** The test images come from the same simulator, aircraft, camera
  view, screen resolution and crop as training. Real cockpits, other views and resolutions,
  and compressed images are untested.
- **Phase is 100% by construction.** The five phases have non-overlapping gauge ranges, so
  this shows correct gauge reading, not an understanding of real flight.
- **Slow airspeeds are less precise.** The C172 airspeed indicator has no markings below
  about 40 kt, so readings at taxi speeds are less precise.
- **Results come from one training seed per run.** Small differences between runs may be noise.
- **Not comparable to v1's bin accuracies.** Those were measured on labels the images didn't show.