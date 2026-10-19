r"""
metrics_v2.py
Evaluation shared by train_v2.py (val + test) and eval_v2.py (any checkpoint, any split,
the hold-out set). One implementation, so every number in the report is computed the
same way. No torch needed here.

  * per gauge: MAE + % within an aviation tolerance, next to a constant-guess baseline
  * per phase: the "shortcut detector". Runs 3-5 lost accuracy because a gauge was
    guessed from the flight phase instead of read from its needle. The symptom inside
    one phase: predictions barely vary (pred std << true std) and the error is strongly
    negatively correlated with the true value (high values under-predicted, low values
    over-predicted). RPM in run 3 level_flight: std 34 vs 99, corr -0.93; airspeed in
    run 5 taxi: std 2.1 vs 7.2, corr -0.99.
"""

import math

import numpy as np
import pandas as pd

TOL = {"pitch": 1.0, "roll": 2.0, "airspeed": 5.0, "altitude": 100.0, "vs": 100.0,
       "turn": 0.2, "rpm": 100.0, "heading": 5.0}
UNITS = {"pitch": "deg", "roll": "deg", "airspeed": "kt", "altitude": "ft", "vs": "ft/min",
         "turn": "std-rate", "rpm": "rpm", "heading": "deg"}
MAIN = ["pitch", "roll", "airspeed", "altitude", "vs", "turn", "rpm", "heading"]
DIAG = ["altitude_coarse", "altitude_shorthand", "rpm_coarse", "airspeed_coarse", "vs_coarse"]
# heuristic flag thresholds for the shortcut detector (chosen from the two known cases above)
SHORTCUT_STD_RATIO, SHORTCUT_CORR = 0.6, -0.7


def true_col(k):
    """Diagnostic columns (altitude_coarse, rpm_coarse, ...) are scored against the real gauge."""
    if k.startswith("altitude"):
        return "altitude"
    return k[:-len("_coarse")] if k.endswith("_coarse") else k


def gauges_of(pred):
    """Main gauges + whichever diagnostics this checkpoint produces."""
    return MAIN + [k for k in DIAG if k in pred]


def err(k, p, t):
    if true_col(k) == "heading":
        d = abs(p - t) % 360
        return min(d, 360 - d)
    return abs(p - t)


def circular_mean_deg(deg):
    r = np.radians(np.asarray(deg, dtype=float))
    return math.degrees(math.atan2(np.sin(r).mean(), np.cos(r).mean())) % 360


def evaluate(preds, frame, base_mean, base_heading, base_phase):
    """preds: list of decode() dicts, aligned with frame rows (labels).
    base_mean: {gauge: constant} (the TRAIN mean); base_heading: constant heading;
    base_phase: constant phase guess. Returns (metrics dict, per-image rows)."""
    frame = frame.reset_index(drop=True)
    gauges = gauges_of(preds[0])
    metrics = {"n": len(frame), "per_gauge": {},
               "note": "baseline = always predict the TRAIN mean (heading: train circular mean); "
                       "phase baseline = always the most common train phase"}
    for k in gauges:
        tc, tol = true_col(k), TOL[true_col(k)]
        e = np.array([err(k, p[k], t) for p, t in zip(preds, frame[tc])])
        b = base_heading if tc == "heading" else base_mean[tc]
        base = np.array([err(k, b, t) for t in frame[tc]])
        metrics["per_gauge"][k] = {"unit": UNITS[tc], "MAE": round(float(e.mean()), 3),
                                   f"within_{tol:g}": round(float((e <= tol).mean() * 100), 1),
                                   "baseline_MAE": round(float(base.mean()), 3),
                                   f"baseline_within_{tol:g}": round(float((base <= tol).mean() * 100), 1)}
    true_ph, pred_ph = list(frame["phase"]), [p["phase"] for p in preds]
    metrics["phase"] = {"accuracy": round(float(np.mean([a == b for a, b in zip(true_ph, pred_ph)])) * 100, 1),
                        "baseline_majority_accuracy": round(float((frame["phase"] == base_phase).mean() * 100), 1),
                        "baseline_phase": base_phase}
    rows = []
    for p, (_, t) in zip(preds, frame.iterrows()):
        row = {"image": t["image"], "true_phase": t["phase"], "pred_phase": p["phase"],
               "phase_confidence": round(p["phase_confidence"], 3)}
        for k in gauges:
            row[f"true_{k}"], row[f"pred_{k}"] = t[true_col(k)], round(p[k], 2)
        rows.append(row)
    return metrics, rows


def within(metrics, k):
    m = metrics["per_gauge"][k]
    return m[[c for c in m if c.startswith("within_")][0]]


def per_phase(rows):
    """Shortcut detector: per phase and main gauge -> n, MAE, % within tol, pred std,
    true std, corr(signed error, true value), flag."""
    df = pd.DataFrame(rows)
    out = []
    for ph, g in df.groupby("true_phase", sort=False):
        for k in MAIN:
            t, p = g[f"true_{k}"].to_numpy(float), g[f"pred_{k}"].to_numpy(float)
            e = np.array([err(k, a, b) for a, b in zip(p, t)])
            rec = {"phase": ph, "gauge": k, "n": len(g), "MAE": round(float(e.mean()), 3),
                   "within": round(float((e <= TOL[k]).mean() * 100), 1),
                   "pred_std": np.nan, "true_std": np.nan, "corr": np.nan, "flag": ""}
            if k != "heading" and t.std() > 1e-6:
                rec["pred_std"], rec["true_std"] = round(float(p.std()), 3), round(float(t.std()), 3)
                if p.std() > 1e-9:
                    rec["corr"] = round(float(np.corrcoef(p - t, t)[0, 1]), 2)
                ratio = p.std() / t.std()
                if ratio < SHORTCUT_STD_RATIO and (np.isnan(rec["corr"]) or rec["corr"] < SHORTCUT_CORR):
                    rec["flag"] = "SHORTCUT?"
            out.append(rec)
    return pd.DataFrame(out)


def print_block(title, metrics, phase_table=None):
    print(f"\n================ {title} ================")
    print(f"{'gauge':<19} {'MAE':>9} {'within tol':>13} | {'baseline MAE':>12} {'baseline within':>16}")
    for k, m in metrics["per_gauge"].items():
        wk = [c for c in m if c.startswith("within_")][0]
        bk = [c for c in m if c.startswith("baseline_within_")][0]
        print(f"{k:<19} {m['MAE']:>9.2f} {m['unit']:<9}{m[wk]:>5.1f}% (±{wk[7:]}) | "
              f"{m['baseline_MAE']:>12.2f} {m[bk]:>15.1f}%")
    ph = metrics["phase"]
    print(f"phase               accuracy {ph['accuracy']}%  "
          f"(baseline: always '{ph['baseline_phase']}' = {ph['baseline_majority_accuracy']}%)")
    if phase_table is not None:
        flagged = phase_table[phase_table["flag"] != ""]
        print(f"shortcut check (per phase): {len(flagged)} flagged"
              + ("" if flagged.empty else "\n" + flagged.to_string(index=False)))


# ---------------------------------------------------------------- final-model decision rule
# Fixed on 2026-10-19, BEFORE run 6 was trained. Applied to the VALIDATION set, which has
# never been inspected (runs 3-6 were designed from TEST errors, so test can't decide).
RULE_GAUGES = MAIN
RULE_SLACK = 1.0      # percentage points of seed noise allowed per gauge (run 3 vs run 4 seed check)
RULE_RPM_MIN = 95.0


def decide(new, ref):
    """new/ref: metrics dicts from evaluate() on the same images. Returns (passed, lines)."""
    lines, ok = [], True
    for k in RULE_GAUGES:
        a, b = within(new, k), within(ref, k)
        good = a >= b - RULE_SLACK
        ok &= good
        lines.append(f"{k:<9} new {a:5.1f}%  ref {b:5.1f}%  {'ok' if good else 'WORSE'}")
    a, b = new["phase"]["accuracy"], ref["phase"]["accuracy"]
    good = a >= b - RULE_SLACK
    ok &= good
    lines.append(f"{'phase':<9} new {a:5.1f}%  ref {b:5.1f}%  {'ok' if good else 'WORSE'}")
    rpm_ok = within(new, "rpm") >= RULE_RPM_MIN
    ok &= rpm_ok
    lines.append(f"rpm >= {RULE_RPM_MIN:g}%: {'ok' if rpm_ok else 'NO'}")
    return ok, lines