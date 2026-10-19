r"""
cockpit_model_v2.py
Shared model + preprocessing for cockpit dataset v2 (instrument-panel crops
captured with capture_synthetic_v2.py). Imported by BOTH process/train_v2.py and
the inference code, so training and inference can never drift apart (v1 had a
real bug where the CLI used letterbox while the model was trained with stretch).

Changes vs CockpitNet (v1):
  * input = the instrument-panel crop (six-pack + RPM, no window) resized to
    448x352 (keeps the crop's ~1.28 aspect ratio) instead of the whole 1918x570
    panel squashed to 224x224 -> the attitude indicator is ~100 px wide, not ~13
  * every gauge is a REGRESSION target (pitch, roll, airspeed, altitude, vertical
    speed, turn rate, rpm, heading as sin/cos) -> errors are reported in real
    units; bins (e.g. pitch ranges) are derived from the number, not learned
  * a learned 5-class flight-phase head replaces the hand-tuned rules
  * (run 2) altitude is ALSO read like the instrument: the altimeter's long hand
    makes one turn per 1000 ft, so an extra head predicts that hand as an angle
    (sin/cos, like heading) and decode() combines it with the coarse regression.
    Run 1 (regression only) gave 131 ft MAE and ~105 ft even on the ground,
    where altitude spans only 40 ft -> the fine hand position was not being read.
  * (run 3) the altimeter's SHORT hand (one turn per 10,000 ft) gets its own angle
    head too. Run 2: long-hand reading had median error 2 ft, but 7/375 frames were
    off by exactly 1000 ft because the coarse regression (MAE ~137 ft) missed by
    >500 ft. Decode is now a 3-step cascade: coarse number -> short hand (thousands)
    -> long hand (hundreds), each step only needing the previous within half a turn.
  * (runs 5-6) NEEDLE ANGLE HEADS. Runs 3-4 (two seeds): RPM ~57 MAE / 83-87% within
    100 rpm vs ~32 / 96-98% in runs 1-2. Inside each phase the RPM prediction barely
    moved (level_flight: prediction std 34 vs true std 99, error-vs-true correlation
    -0.93): the model output the phase's typical RPM instead of reading the (clearly
    visible) needle. Cause: each gauge in the shared regression is normalised by its
    std over ALL phases, so a 100 rpm mistake costs almost nothing. Run 5 gave the
    tachometer an angle head (sin/cos, one turn per period) -> RPM 11.5 / 100%, but
    airspeed then showed the same shortcut (taxi: prediction std 2.1 vs true 7.2,
    correlation -0.99; 85.1% within 5 kt vs 96.3%). Run 6 therefore gives EVERY
    needle gauge that has a wide range an angle head: rpm, airspeed, vertical speed.
    The periods are only encodings (> the gauge's range in the data); they don't
    need to equal the real dial sweep. decode() combines angle + coarse number the
    same way as for the altimeter.
"""

import math

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

PHASES = ["engine_off", "taxi", "takeoff_roll", "climb", "level_flight"]
# scalar regression targets (heading is handled separately as sin/cos)
SCALARS = ["pitch", "roll", "airspeed", "altitude", "vs", "turn", "rpm"]
UNITS = {"pitch": "deg", "roll": "deg", "airspeed": "kt", "altitude": "ft", "vs": "ft/min",
         "turn": "std-rate", "rpm": "rpm", "heading": "deg"}

IMG_H, IMG_W = 352, 448
ALT_PERIOD = 1000.0         # ft per revolution of the altimeter's long (hundreds) hand
ALT_SHORT_PERIOD = 10000.0  # ft per revolution of the short (thousands) hand
# needle-angle encodings: units per full turn. Must be > 2x the largest coarse-regression
# error (decode picks the turn from the coarse number) and cover the gauge's range.
NEEDLE_PERIODS = {"rpm": 4000.0, "airspeed": 250.0, "vs": 4000.0}
NON_NEGATIVE = {"rpm", "airspeed"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform():
    # Photometric only + a tiny shift. NO flips/rotations: they would change what
    # the gauges read (a mirrored attitude indicator shows the opposite bank).
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ColorJitter(brightness=0.35, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CockpitNetV2(nn.Module):
    def __init__(self, pretrained=True, alt_fine=True, alt_short=True, needles=()):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.dropout = nn.Dropout(0.2)
        self.reg = nn.Linear(dim, len(SCALARS))      # normalised scalar targets
        self.heading = nn.Linear(dim, 2)              # sin, cos
        self.phase = nn.Linear(dim, len(PHASES))
        self.alt_fine = nn.Linear(dim, 2) if alt_fine else None    # sin, cos of the long hand
        self.alt_short = nn.Linear(dim, 2) if alt_short else None  # sin, cos of the short hand
        self.needles = list(needles)                                 # e.g. ["rpm", "airspeed", "vs"]
        self.needle = nn.ModuleDict({k: nn.Linear(dim, 2) for k in self.needles})  # sin, cos per needle

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        out = {"reg": self.reg(f), "heading": self.heading(f), "phase": self.phase(f)}
        if self.alt_fine is not None:
            out["alt_fine"] = self.alt_fine(f)
        if self.alt_short is not None:
            out["alt_short"] = self.alt_short(f)
        for k in self.needles:
            out["needle_" + k] = self.needle[k](f)
        return out


def alt_fine_target(altitude_ft, period=ALT_PERIOD):
    """Altimeter hand angle as (sin, cos); period 1000 = long hand, 10000 = short hand."""
    a = 2 * math.pi * altitude_ft / period
    return math.sin(a), math.cos(a)


def combine_altitude(coarse_ft, s, c, period=ALT_PERIOD):
    """Exact position within one hand revolution (from the angle) + which revolution
    (from the coarse estimate): pick the value with that hand angle closest to the
    coarse estimate. Correct as long as the coarse estimate is within +-period/2."""
    frac = (math.atan2(s, c) / (2 * math.pi)) % 1.0 * period
    return period * round((coarse_ft - frac) / period) + frac


def load_checkpoint(path, device="cpu"):
    """Rebuild the model exactly as it was trained (works for run 1-6 files)."""
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model_state_dict"]
    needles = ckpt.get("needles") or (["rpm"] if ckpt.get("rpm_fine") else [])
    # run 5 stored its single tachometer head as "rpm_fine.*" -> rename to the run-6 layout
    state = {("needle.rpm." + k[len("rpm_fine."):] if k.startswith("rpm_fine.") else k): v
             for k, v in state.items()}
    model = CockpitNetV2(pretrained=False, alt_fine=ckpt.get("alt_fine", False),
                         alt_short=ckpt.get("alt_short", False), needles=needles)
    model.load_state_dict(state)
    set_needle_use(model, ckpt.get("needle_use"))
    return model.to(device).eval(), ckpt


def set_needle_use(model, needle_use):
    """Run 6: train_v2 checks on the VALIDATION set, per gauge, whether the needle-angle
    reading or the plain regression is better, and stores that as needle_use. A needle
    head that lost is simply not output any more, so decode() falls back to the
    regression for that gauge. The weights stay in the file (state dict unchanged)."""
    use = needle_use or {}
    model.needles = [k for k in model.needle.keys() if use.get(k, True)]


def decode(out, norm):
    """Model output (batch) -> list of dicts in real units + phase probabilities."""
    reg = out["reg"].detach().cpu()
    hs = out["heading"].detach().cpu()
    probs = torch.softmax(out["phase"].detach().cpu(), dim=1)
    af = out["alt_fine"].detach().cpu() if "alt_fine" in out else None
    ash = out["alt_short"].detach().cpu() if "alt_short" in out else None
    rows = []
    for i in range(reg.shape[0]):
        d = {k: float(reg[i, j]) * norm[k]["std"] + norm[k]["mean"] for j, k in enumerate(SCALARS)}
        d["altitude_coarse"] = d["altitude"]          # plain regression (run 1 method)
        est = d["altitude_coarse"]
        if ash is not None:                           # short hand -> thousands
            est = combine_altitude(est, float(ash[i, 0]), float(ash[i, 1]), ALT_SHORT_PERIOD)
            d["altitude_shorthand"] = est
        if af is not None:                            # long hand -> hundreds
            est = combine_altitude(est, float(af[i, 0]), float(af[i, 1]), ALT_PERIOD)
        d["altitude"] = est
        for k in NEEDLE_PERIODS:                      # needle angle -> exact reading
            if "needle_" + k in out:
                s, c = (float(v) for v in out["needle_" + k][i].detach().cpu())
                d[k + "_coarse"] = d[k]               # plain regression (runs 1-4 method)
                v = combine_altitude(d[k], s, c, NEEDLE_PERIODS[k])
                d[k] = max(0.0, v) if k in NON_NEGATIVE else v
        d["heading"] = math.degrees(math.atan2(float(hs[i, 0]), float(hs[i, 1]))) % 360
        p = probs[i]
        d["phase"] = PHASES[int(p.argmax())]
        d["phase_confidence"] = float(p.max())
        d["phase_probs"] = {ph: float(p[j]) for j, ph in enumerate(PHASES)}
        rows.append(d)
    return rows