r"""
capture_synthetic_v2.py
Cockpit dataset v2 for Visual Co-Pilot -- all 5 phases, fully scripted, no flying.

Same idea as the team's capture_phase_synthetic.py (FlightGear with --fdm=null,
state set directly), with the bug that broke dataset v1 fixed:

  v1 problem: under --fdm=null the instruments never update (no engine -> no
  vacuum gyros, no airflow -> no airspeed). Writing /orientation, /velocities
  moved the OUTSIDE VIEW only, so the model learned the window, not the gauges.

  v2 fix: a small loop INSIDE FlightGear (Nasal, every frame) writes each
  gauge's DISPLAYED value, so every needle shows exactly the label we log.
  Verified with drive_gauges_test.py (attitude, altimeter, heading, VSI, turn
  coordinator and RPM followed the targets). The C172 airspeed needle reads its
  own JSBSim property (/fdm/jsbsim/velocities/vias-kts, found with find_asi_property.py),
  which is written as well.

Other v1 problems fixed here:
  * labels = exactly what the gauges show (read back and verified per frame)
  * every frame is an independent random sample (no ramps / neighbours in time
    -> a random train/test split no longer leaks near-duplicates)
  * continuous headings 0-360 for every phase (v1: only 0/90/180/270)
  * realistic, OVERLAPPING ranges between phases (v1: phase = altitude band)
  * climbs show a real climb on the VSI (v1: vertical speed was always 0)
  * the outside-view attitude is randomised independently of the labels, so
    panel lighting/shadows carry no information about the answer
  * only the instrument panel is saved (six-pack + RPM, no window)

FlightGear launch (same as the old captures):
  --aircraft=c172p --airport=KSJC --fdm=null --heading=180 --timeofday=noon --httpd=5500
Full screen, normal cockpit view, don't touch the view or open menus.

  1) one-time check (~1 min):      python capture_synthetic_v2.py --selftest
  2) capture each phase:           python capture_synthetic_v2.py --phase climb --frames 400
     (re-running a phase appends; numbering continues)
Output: dataset_v2/<phase>/<phase>_NNNN.png  +  dataset_v2/labels.csv
"""

import argparse
import csv
import json
import math
import os
import random
import threading
import time

import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
from mss import mss
from PIL import Image

HTTP = "http://127.0.0.1:5500"   # not "localhost": avoids slow IPv6 lookups on Windows
PHASES = ["engine_off", "taxi", "takeoff_roll", "climb", "level_flight"]

ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=PHASES)
ap.add_argument("--frames", type=int, default=500)
ap.add_argument("--selftest", action="store_true", help="photograph each gauge at several values, then exit")
ap.add_argument("--out", default="dataset_v2")
ap.add_argument("--seed", type=int, default=None, help="default: different every run")
ap.add_argument("--settle", type=float, default=0.6, help="seconds after setting a state before the screenshot")
ap.add_argument("--crop", type=float, nargs=4, default=[0.365, 0.465, 0.690, 0.870],
                help="instrument-panel crop as screen fractions (verified on the 2880x1800 cockpit view)")
ap.add_argument("--start_delay", type=int, default=10)
ap.add_argument("--hold_timeout", type=float, default=20.0,
                help="max seconds to wait for lagged needles (VSI) to reach the label before rejecting")
args = ap.parse_args()
if not args.selftest and not args.phase:
    ap.error("give --phase <name> (or --selftest)")

S = requests.Session()
POOL = ThreadPoolExecutor(max_workers=4)

# label name -> FlightGear properties the panel draws from (all are written)
GAUGE_PROPS = {
    "pitch": ["/instrumentation/attitude-indicator/indicated-pitch-deg"],
    "roll": ["/instrumentation/attitude-indicator/indicated-roll-deg"],
    "airspeed": ["/instrumentation/airspeed-indicator/indicated-speed-kt",
                 "/fdm/jsbsim/velocities/vias-kts"],   # the C172 needle reads this (found by find_asi_property.py)
    "altitude": ["/instrumentation/altimeter/indicated-altitude-ft"],
    "heading": ["/instrumentation/heading-indicator/indicated-heading-deg"],
    "vs": ["/instrumentation/vertical-speed-indicator/indicated-speed-fpm",
           "/instrumentation/vertical-speed-indicator/indicated-speed-fpm-final"],
           # ^ the c172p VSI NEEDLE draws "-final", a lagged copy of indicated-speed-fpm
           #   (first-order lag, time constant ~2 s; found with vsi_probe.py, 2026-10-17).
           #   FlightGear's filter overwrites it every frame from its own internal state, so it
           #   can NOT be written (tested: --selftest still showed the lag). It is READ only:
           #   each frame waits until the needle has converged (see wait_until_held).
    "turn": ["/instrumentation/turn-indicator/indicated-turn-rate"],
    "rpm": ["/engines/engine/rpm"],
}
LABELS = list(GAUGE_PROPS)
# drawn properties we only READ BACK (never write): a FlightGear filter owns them
READ_ONLY = {"/instrumentation/vertical-speed-indicator/indicated-speed-fpm-final"}
# per-property tolerance overrides. 15 fpm on the VSI needle = ~1.4 deg of needle travel,
# far below what can be seen in the crop and below train_v2's 100 fpm evaluation tolerance.
PROP_TOL = {"/instrumentation/vertical-speed-indicator/indicated-speed-fpm-final": 15.0}
TOL = {"pitch": 0.2, "roll": 0.2, "airspeed": 0.5, "altitude": 2, "heading": 0.5,
       "vs": 5, "turn": 0.02, "rpm": 5}


# ---------------------------------------------------------------- FlightGear I/O
def run_nasal(code):
    body = {"name": "", "children": [{"name": "script", "index": 0, "value": code}]}
    S.post(HTTP + "/run.cgi?value=nasal", data=json.dumps(body), timeout=5)


_tls = threading.local()


def _session():
    # one persistent connection per thread (opening a new one per read can stall on Windows)
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def get(path):
    try:
        v = _session().get(HTTP + "/json" + path, timeout=(2, 3)).json().get("value")
        return float(v) if v is not None and not isinstance(v, bool) else v
    except Exception:
        return None


def nasal_map():
    pairs = []
    for k, paths in GAUGE_PROPS.items():
        for p in paths:
            if p in READ_ONLY:
                continue
            pairs.append(f'["{k}", "{p}"]')
    return "[" + ", ".join(pairs) + "]"


# Per-frame writer: copies /capture/gauge/<label> into every property listed.
# A generation counter makes a re-install replace (not duplicate) the loop.
WRITER = r'''
var g = "/capture/gauge/";
var pairs = %MAP%;
var gen = getprop(g ~ "gen"); if (gen == nil) gen = 0; gen += 1;   # (Nasal "or" returns 1/0)
setprop(g ~ "gen", gen);
var n = 0;
var loop = func {
    if (getprop(g ~ "gen") != gen) return;      # a newer writer took over
    n += 1; setprop(g ~ "ticks", n);
    if (getprop(g ~ "on") == 1)
        foreach (var pr; pairs) { var v = getprop(g ~ pr[0]); if (v != nil) setprop(pr[1], v); }
    settimer(loop, 0);
};
loop();
'''.replace("%MAP%", nasal_map())


def install_writer():
    run_nasal(WRITER)
    time.sleep(0.5)
    t1 = get("/capture/gauge/ticks"); time.sleep(1.0); t2 = get("/capture/gauge/ticks")
    if not t2 or (t1 is not None and t2 <= t1):
        raise SystemExit("Gauge writer is not running inside FlightGear (check --httpd=5500). Stopping.")
    print(f"gauge writer running ({t1:.0f} -> {t2:.0f} ticks/s)")


VIEW_LOCK = {}


def apply_state(labels, world):
    """One Nasal call: set gauge targets + the (nuisance) outside-view attitude + re-lock the view."""
    code = "".join(f'setprop("/capture/gauge/{k}", {float(labels[k])});' for k in LABELS)
    code += view_code(VIEW_LOCK)
    code += "".join(f'setprop("{p}", {float(v)});' for p, v in world.items())
    code += 'setprop("/capture/gauge/on", 1);'
    run_nasal(code)


def verify(labels):
    """Read every drawn property back; return the worst mismatches (empty = OK)."""
    jobs = [(k, p) for k, paths in GAUGE_PROPS.items() for p in paths]
    futs = [POOL.submit(get, p) for _, p in jobs]           # parallel: each read ~50 ms
    vals = []
    for fu in futs:
        try:
            vals.append(fu.result(timeout=5))
        except Exception:
            vals.append(None)                                  # counts as a mismatch, never hangs
    bad = {}
    for (k, p), v in zip(jobs, vals):
        if not isinstance(v, float) or abs(v - labels[k]) > PROP_TOL.get(p, TOL[k]):
            bad[p.rsplit("/", 1)[-1]] = v
    return bad


def wait_until_held(labels, timeout):
    """Poll verify() until every drawn property (incl. the lagged VSI needle) matches the
    labels, or the timeout passes. Returns the remaining mismatches (empty = OK)."""
    t_end = time.time() + timeout
    bad = verify(labels)
    while bad and time.time() < t_end:
        time.sleep(0.25)
        bad = verify(labels)
    return bad


def park_cursor():
    # Called before EVERY screenshot: a pointer hovering the taskbar makes Windows draw a
    # window-preview thumbnail over the panel (seen once in the self-test).
    try:
        import ctypes
        ctypes.windll.user32.SetCursorPos(5, 5)
    except Exception:
        pass


def grab(sct, mon):
    park_cursor()
    time.sleep(0.05)
    s = sct.grab(mon)
    full = Image.frombytes("RGB", s.size, s.rgb)
    W, H = full.size
    box = tuple(int(round(c * (W if i % 2 == 0 else H))) for i, c in enumerate(args.crop))
    return full, full.crop(box), box


def save_sheet(tiles):
    from PIL import ImageDraw
    w, h = 468, 365
    sheet = Image.new("RGB", (4 * w, ((len(tiles) + 3) // 4) * (h + 24)), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(tiles):
        x, y = (i % 4) * w, (i // 4) * (h + 24)
        sheet.paste(im.resize((w, h)), (x, y + 24))
        dr.text((x + 6, y + 4), lab, fill="black")
    sheet.save(os.path.join(args.out, "selftest_sheet.png"))


# ---------------------------------------------------------------- safety guards
VIEW_PROPS = ["/sim/current-view/view-number", "/sim/current-view/field-of-view",
              "/sim/current-view/heading-offset-deg", "/sim/current-view/pitch-offset-deg",
              "/sim/current-view/x-offset-m", "/sim/current-view/y-offset-m", "/sim/current-view/z-offset-m"]


# The cockpit view used for the verified self-test (view_lock.json of 2026-10-17 20:33).
KNOWN_GOOD_VIEW = {
    "/sim/current-view/view-number": 0.0,
    "/sim/current-view/field-of-view": 73.6,
    "/sim/current-view/heading-offset-deg": 0.0,
    "/sim/current-view/pitch-offset-deg": -12.0,
    "/sim/current-view/x-offset-m": -0.21,
    "/sim/current-view/y-offset-m": 0.273,
    "/sim/current-view/z-offset-m": 0.36,
}


def keep_awake_and_park_cursor():
    """Windows: stop the screen sleeping during a long capture, and move the mouse
    pointer to the top-left corner so it can never appear on the instrument panel."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
        ctypes.windll.user32.SetCursorPos(5, 5)
    except Exception:
        pass


def load_or_create_view_lock(screen_size):
    """The cockpit view + screen size are recorded on the first run and enforced on
    every later run, so every image in the dataset has the panel in the same place."""
    path = os.path.join(args.out, "view_lock.json")
    current = {p: get(p) for p in VIEW_PROPS}
    if not os.path.exists(path):
        # Always start from the KNOWN-GOOD cockpit view (verified on 2026-10-17, 2880x1800),
        # never from whatever state FlightGear happens to be in.
        os.makedirs(args.out, exist_ok=True)
        json.dump({"screen": list(screen_size), "view": KNOWN_GOOD_VIEW}, open(path, "w"), indent=2)
        print(f"saved standard view lock -> {path}")
        return dict(KNOWN_GOOD_VIEW)
    lock = json.load(open(path))
    if list(screen_size) != lock["screen"]:
        raise SystemExit(f"Screen is {screen_size} but the dataset was started at {lock['screen']}. "
                         f"Use the same screen/resolution (crops would not line up).")
    drift = {p: (current[p], v) for p, v in lock["view"].items()
             if v is not None and current.get(p) is not None and abs(float(current[p]) - float(v)) > 0.01}
    if drift:
        print(f"view differs from the locked view, restoring it: {drift}")
    return lock["view"]


def view_code(view):
    return "".join(f'setprop("{p}", {float(v)});' for p, v in view.items() if v is not None)


def small_gray(img):
    return np.asarray(img.convert("L").resize((96, 75)), dtype=np.float32)


def panel_similarity(crop, ref):
    """Normalised cross-correlation with the reference panel image (insensitive to
    brightness changes). ~1 = same panel layout; low = panel not visible/misaligned."""
    a, b = small_gray(crop), ref
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


# ---------------------------------------------------------------- state sampling
U = random.uniform


def turn_from_roll(roll, airspeed):
    """Coordinated-turn rate in 'standard rate' units (1.0 = 3 deg/s)."""
    v = max(airspeed, 40) * 0.5144                     # kt -> m/s
    rate_deg_s = math.degrees(9.81 * math.tan(math.radians(roll)) / v)
    rate = rate_deg_s / 3.0 + random.uniform(-0.15, 0.15)   # noise: TC must be read, not inferred from bank
    return max(-2.0, min(2.0, rate))


def sample(phase, field_elev):
    hdg = U(0, 360)
    if phase == "engine_off":
        d = dict(pitch=U(-1, 1), roll=U(-1, 1), airspeed=0.0, altitude=field_elev + U(-20, 20),
                 heading=hdg, vs=0.0, turn=0.0, rpm=0.0)
    elif phase == "taxi":
        d = dict(pitch=U(-1, 1), roll=U(-1, 1), airspeed=U(0, 25), altitude=field_elev + U(-20, 20),
                 heading=hdg, vs=0.0, turn=U(-0.6, 0.6), rpm=U(800, 1400))
    elif phase == "takeoff_roll":
        spd = U(25, 65)
        d = dict(pitch=U(0, 2) + (U(0, 4) if spd > 50 else 0), roll=U(-1, 1), airspeed=spd,
                 altitude=field_elev + U(-20, 20), heading=hdg, vs=0.0, turn=U(-0.2, 0.2),
                 rpm=U(2250, 2450))
    elif phase == "climb":
        roll = U(-20, 20) if random.random() < 0.5 else U(-3, 3)
        spd = U(65, 95)
        d = dict(pitch=U(3, 12), roll=roll, airspeed=spd, altitude=U(300, 9000), heading=hdg,
                 vs=U(300, 1000), turn=turn_from_roll(roll, spd), rpm=U(2250, 2450))
    else:  # level_flight
        roll = U(-25, 25) if random.random() < 0.5 else U(-3, 3)
        spd = U(85, 125)
        d = dict(pitch=U(-2, 3), roll=roll, airspeed=spd, altitude=U(1000, 9500), heading=hdg,
                 vs=U(-150, 150), turn=turn_from_roll(roll, spd), rpm=U(2100, 2450))
    d["heading"] %= 360
    return d


def nuisance_world():
    """Outside-view attitude, independent of the labels (affects only lighting)."""
    return {"/orientation/heading-deg": U(0, 360), "/orientation/pitch-deg": U(-5, 10),
            "/orientation/roll-deg": U(-20, 20), "/position/altitude-ft": 3000.0}


# ---------------------------------------------------------------- main
print(f"Click on the FlightGear window now -- starting in {args.start_delay}s. "
      f"Hands off, don't move the view.")
time.sleep(args.start_delay)
fm = get("/sim/flight-model")
print(f"flight model: {fm}")
install_writer()
field_elev = get("/position/ground-elev-ft") or 62.0
keep_awake_and_park_cursor()

with mss() as sct:
    mon = sct.monitors[1]
    VIEW_LOCK.update(load_or_create_view_lock((mon["width"], mon["height"])))
    run_nasal(view_code(VIEW_LOCK))          # restore the standard view right away
    time.sleep(1.5)
    ref_path = os.path.join(args.out, "reference_panel.png")

    if args.selftest:
        os.makedirs(args.out, exist_ok=True)
        base = dict(pitch=0, roll=0, airspeed=0, altitude=1000, heading=0, vs=0, turn=0, rpm=1000)
        tests = [("airspeed", [0, 15, 30, 50, 80, 120]), ("pitch", [-5, 0, 5, 10]), ("roll", [-20, 0, 20]),
                 ("altitude", [500, 3500]), ("heading", [0, 90, 225]), ("vs", [-500, 0, 800]),
                 ("rpm", [0, 1200, 2400])]
        tiles = []
        for name, values in tests:
            for v in values:
                st = dict(base, **{name: v})
                apply_state(st, nuisance_world() | {"/orientation/pitch-deg": 0, "/orientation/roll-deg": 0})
                time.sleep(args.settle + 0.4)
                bad = wait_until_held(st, args.hold_timeout)
                _, crop, _ = grab(sct, mon)
                tiles.append((f"{name}={v}", crop))
                if len(tiles) == 1:
                    crop.save(ref_path)                 # reference for the "panel visible" check
                print(f"  {name:>8} = {v:>6}: {'OK' if not bad else 'MISMATCH ' + str(bad)}", flush=True)
                save_sheet(tiles)                       # saved after every photo
        print(f"\nSaved {os.path.abspath(os.path.join(args.out, 'selftest_sheet.png'))}"
              f" -- send it to check every needle moved.")
        raise SystemExit(0)

    # ---- capture a phase
    if not os.path.exists(ref_path):
        raise SystemExit("Run  python capture_synthetic_v2.py --selftest  first "
                         "(it creates dataset_v2/reference_panel.png used to check every frame).")
    ref = small_gray(Image.open(ref_path))
    ref = ref - ref.mean()
    bad_view_streak = 0
    rng_seed = args.seed if args.seed is not None else int(time.time())
    random.seed(rng_seed)
    img_dir = os.path.join(args.out, args.phase)
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(args.out, "labels.csv")
    existing = [f for f in os.listdir(img_dir) if f.endswith(".png") and f.startswith(args.phase)]
    idx = max([int(f[-8:-4]) for f in existing], default=-1) + 1
    fields = ["image", "phase"] + LABELS + ["world_heading", "world_pitch", "world_roll",
                                            "seed", "crop_px", "panel_similarity", "unix_time"]
    new_file = not os.path.exists(csv_path)
    fcsv = open(csv_path, "a", newline="")
    wr = csv.DictWriter(fcsv, fieldnames=fields)
    if new_file:
        wr.writeheader()

    saved = rejected = 0
    print(f"Capturing {args.frames} '{args.phase}' frames (seed {rng_seed}) -> {img_dir}")
    while saved < args.frames:
        st = sample(args.phase, field_elev)
        world = nuisance_world()
        apply_state(st, world)
        time.sleep(args.settle)
        bad_before = wait_until_held(st, args.hold_timeout)   # VSI needle lags ~2 s: wait for it
        full, crop, box = grab(sct, mon)
        bad_after = verify(st)
        if bad_before or bad_after:
            rejected += 1
            print(f"\n  rejected (gauge value not held): {bad_before or bad_after}")
            continue
        sim = panel_similarity(crop, ref)
        if sim < 0.55 or np.asarray(crop.convert("L")).mean() < 8:
            rejected += 1
            bad_view_streak += 1
            print(f"\n  rejected: panel not visible/misaligned (similarity {sim:.2f}) -- is FlightGear "
                  f"in front, full screen, view untouched?")
            if bad_view_streak >= 5:
                raise SystemExit("Stopped: 5 frames in a row without the instrument panel. Fix the "
                                 "window/view and re-run the same command (it continues numbering).")
            keep_awake_and_park_cursor()
            time.sleep(2)
            continue
        bad_view_streak = 0
        name = f"{args.phase}_{idx:04d}.png"
        crop.save(os.path.join(img_dir, name))
        if idx % 100 == 0:
            full.resize((full.size[0] // 2, full.size[1] // 2)).save(
                os.path.join(img_dir, f"_fullcheck_{idx:04d}.jpg"))
        wr.writerow({"image": name, "phase": args.phase,
                     **{k: round(st[k], 3) for k in LABELS},
                     "world_heading": round(world["/orientation/heading-deg"], 2),
                     "world_pitch": round(world["/orientation/pitch-deg"], 2),
                     "world_roll": round(world["/orientation/roll-deg"], 2),
                     "seed": rng_seed, "crop_px": "%d %d %d %d" % box, "panel_similarity": round(sim, 3),
                     "unix_time": round(time.time(), 2)})
        fcsv.flush()
        saved += 1
        idx += 1
        print(f"\r[{args.phase}] {saved}/{args.frames} saved, {rejected} rejected", end="", flush=True)

    fcsv.close()
    print(f"\nDone: {saved} frames in {img_dir}, labels appended to {csv_path}")
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(1000, 400)
    except Exception:
        pass