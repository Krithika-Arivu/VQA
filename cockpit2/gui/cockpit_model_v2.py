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
    def __init__(self, pretrained=True, alt_fine=True, alt_short=True):
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

    def forward(self, x):
        f = self.dropout(self.backbone(x))
        out = {"reg": self.reg(f), "heading": self.heading(f), "phase": self.phase(f)}
        if self.alt_fine is not None:
            out["alt_fine"] = self.alt_fine(f)
        if self.alt_short is not None:
            out["alt_short"] = self.alt_short(f)
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
    """Rebuild the model exactly as it was trained (works for run 1 and run 2 files)."""
    ckpt = torch.load(path, map_location=device)
    model = CockpitNetV2(pretrained=False, alt_fine=ckpt.get("alt_fine", False),
                         alt_short=ckpt.get("alt_short", False))
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ckpt


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
        d["heading"] = math.degrees(math.atan2(float(hs[i, 0]), float(hs[i, 1]))) % 360
        p = probs[i]
        d["phase"] = PHASES[int(p.argmax())]
        d["phase_confidence"] = float(p.max())
        d["phase_probs"] = {ph: float(p[j]) for j, ph in enumerate(PHASES)}
        rows.append(d)
    return rows
